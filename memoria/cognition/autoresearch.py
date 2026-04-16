"""Internal Autoresearch Loop: goal-directed hypothesis generation and evaluation.

Internalized version of Karpathy's autoresearch pattern. Instead of an external
agent editing code and running experiments, the cognitive state generates
hypotheses as provisional beliefs, tests them over N forward passes, and
promotes/evicts based on free energy improvement.

The loop:
  1. Telos identifies high-EFE goals (existing)
  2. HypothesisGenerator synthesizes candidate beliefs from goals + state
  3. Candidates are allocated as provisional (A1)
  4. Forward passes test them (natural)
  5. Provisional evaluation promotes or evicts (A1)
  6. HypothesisTracker records which goal-types produce successful hypotheses
  7. Success rates bias future hypothesis generation

Reference: Karpathy autoresearch (github.com/karpathy/autoresearch)
Reference: DiscoPOP (arXiv:2406.08414) — propose-evaluate-archive loop
Reference: BrainCL (arXiv:2504.14727) — wake/sleep staging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import EPSILON


class HypothesisGenerator(nn.Module):
    """Generate candidate beliefs (hypotheses) from active goals.

    For each active goal, synthesizes a belief vector that might help
    reduce free energy in the goal's direction. The hypothesis is a
    blend of the goal embedding, the current belief state summary, and
    a summary of *recent failures for this goal* (Meta-Harness-style
    conditioning on prior diagnostic experience, arXiv:2603.28052),
    transformed through a learned projection.

    The generator learns what kinds of beliefs help for what kinds of
    goals AND what directions have already been ruled out. New failure-
    conditioning weight columns are zero-initialized, so a freshly-built
    generator behaves exactly like the pre-failure-log version until the
    network learns to use them.
    """

    # Feature dimensions for the shared input to net/precision.
    # Layout: [goal_embed, belief_summary, failure_summary, progress, beta, failure_count_norm]
    # Old layout was [goal_embed, belief_summary, progress, beta].
    # The failure_summary and failure_count_norm columns are new and are
    # zero-initialized below so behavior at init matches the pre-#2 generator.

    def __init__(self, belief_dim: int):
        super().__init__()
        self.belief_dim = belief_dim

        # Extended input: +belief_dim (failure summary vector) +1 (failure count norm)
        in_dim = belief_dim * 3 + 3

        # Output: hypothesis belief vector (D)
        self.hypothesis_net = nn.Sequential(
            nn.Linear(in_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim),
        )

        # Precision head: how confident should the hypothesis be?
        # Low initial precision = tentative. Learns to calibrate.
        self.precision_head = nn.Sequential(
            nn.Linear(in_dim, belief_dim // 4),
            nn.ReLU(),
            nn.Linear(belief_dim // 4, 1),
            nn.Softplus(),
        )

        # Gate removed: hypothesis_gen runs under no_grad so a learned gate
        # could never update. The viable mask in run_autoresearch_step
        # (goal_success_ema > viable_goal_min_success MetaParam) already
        # does content-aware filtering. All goals that pass the viable
        # filter get hypotheses generated — no redundant gating.

        # Zero-init the failure-conditioning columns of the first linear in each
        # head. Input layout is:
        #   [0 : D]             goal_embed
        #   [D : 2D]            belief_summary
        #   [2D : 3D]            failure_summary      ← new (zero-init)
        #   [3D]                 progress             (old column 2D)
        #   [3D+1]               beta                 (old column 2D+1)
        #   [3D+2]               failure_count_norm   ← new (zero-init)
        # Zero-init means failure-conditioning signals have no effect at
        # t=0; the network learns to use them as training progresses.
        # (Same pattern as GoalRouter zero-init in the read path.)
        with torch.no_grad():
            for head in (self.hypothesis_net, self.precision_head):
                first_linear = head[0]
                first_linear.weight[:, belief_dim * 2 : belief_dim * 3].zero_()
                first_linear.weight[:, belief_dim * 3 + 2].zero_()

    def forward(
        self,
        goal_embeddings: Tensor,
        goal_progress: Tensor,
        beliefs: Tensor,
        active_mask: Tensor,
        beta: float,
        failure_summary: Tensor | None = None,
        failure_count: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate hypothesis beliefs from active goals.

        Args:
            goal_embeddings: [G, D] active goal embeddings
            goal_progress: [G] progress estimates per goal
            beliefs: [max_beliefs, D] full belief tensor
            active_mask: [max_beliefs] boolean
            beta: exploration/exploitation parameter
            failure_summary: optional [G, D] per-goal mean of recent failed
                angles. None → treated as zero vector (same as no prior
                failures). This is the Meta-Harness conditioning signal:
                the generator sees directions that already failed for each
                goal and can learn to push away from them.
            failure_count: optional [G] number of records in the per-goal
                failed buffer (capped at the buffer size). None → zeros.

        Returns:
            hypotheses: [G, D] candidate belief vectors
            precisions: [G] precision for each hypothesis
            goal_indices: [G] index of each goal (identity mapping)
        """
        G = goal_embeddings.shape[0]
        device = goal_embeddings.device

        if G == 0 or not active_mask.any():
            return (torch.zeros(0, self.belief_dim, device=device),
                    torch.zeros(0, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        # Belief state summary: precision-weighted mean of active beliefs
        active_beliefs = beliefs[active_mask]
        radii = active_beliefs.norm(dim=-1, keepdim=True).clamp(min=EPSILON)
        belief_summary = (active_beliefs * radii).sum(dim=0) / radii.sum()  # [D]
        belief_summary = belief_summary.unsqueeze(0).expand(G, -1)  # [G, D]

        # Failure conditioning (Meta-Harness): per-goal summary of recent
        # failed hypotheses. Zero when absent — network sees "no prior
        # failures on this goal" as zero signal.
        if failure_summary is None:
            failure_summary = torch.zeros(G, self.belief_dim, device=device)
        if failure_count is None:
            failure_count = torch.zeros(G, device=device)
        # Normalize failure count by buffer depth so the feature is bounded.
        # Buffer depth lives on the tracker; we clamp to [0, 1] defensively
        # in case callers pass raw counts.
        failure_count_norm = failure_count.clamp(0.0, 1.0).unsqueeze(-1)  # [G, 1]

        # Build input features
        beta_t = torch.full((G, 1), beta, device=device)
        progress_t = goal_progress.unsqueeze(-1)  # [G, 1]

        features = torch.cat([
            goal_embeddings,     # [G, D]
            belief_summary,      # [G, D]
            failure_summary,     # [G, D] — new
            progress_t,          # [G, 1]
            beta_t,              # [G, 1]
            failure_count_norm,  # [G, 1] — new
        ], dim=-1)  # [G, 3D+3]

        # No gate: all goals that pass the viable filter in
        # run_autoresearch_step get hypotheses generated. The viable mask
        # (goal_success_ema > viable_goal_min_success) already does
        # content-aware filtering; a learned gate here was dead code
        # because hypothesis_gen runs under no_grad.
        selected_features = features
        selected_indices = torch.arange(G, device=device)

        # Generate hypothesis direction
        raw_hypothesis = self.hypothesis_net(selected_features)  # [G, D]
        hypothesis_dir = F.normalize(raw_hypothesis, dim=-1, eps=EPSILON)

        # Generate precision (low = tentative, learns to calibrate)
        precision = self.precision_head(selected_features).squeeze(-1)  # [G]

        # Scale hypothesis by precision
        hypotheses = hypothesis_dir * precision.unsqueeze(-1)

        return hypotheses, precision, selected_indices


class HypothesisTracker(nn.Module):
    """Track which goals produce successful hypotheses and log their failures.

    Maintains per-goal success/failure counts and a ring buffer of recent
    failed hypothesis angles per goal. Goals with higher success rates get
    priority boosts. Failed angles are surfaced back to HypothesisGenerator
    so new proposals can learn to push away from recently-failed regions.

    This is the "log to results.tsv" step of the autoresearch loop,
    internalized as running statistics — extended with the Meta-Harness
    insight (arXiv:2603.28052 Table 3, Appendix A.2) that the richest
    signal for a proposer is raw access to prior failures, not a
    compressed success EMA.
    """

    def __init__(self, max_goals: int, belief_dim: int,
                 failed_buffer_depth: int):
        super().__init__()
        self.max_goals = max_goals
        self.belief_dim = belief_dim
        self.failed_buffer_depth = failed_buffer_depth

        # Per-goal hypothesis tracking
        self.register_buffer('hypothesis_count', torch.zeros(max_goals))
        self.register_buffer('hypothesis_promoted', torch.zeros(max_goals))
        self.register_buffer('hypothesis_evicted', torch.zeros(max_goals))

        # EMA of per-goal success rate (smoothed signal)
        self.register_buffer('goal_success_ema', torch.full((max_goals,), 0.5))

        # Belief → goal mapping: which goal generated which provisional belief
        # -1 = not a hypothesis belief
        self.register_buffer('belief_source_goal', torch.full((1,), -1, dtype=torch.long))
        self._belief_source_goal_size = 0

        # ── Failed-hypothesis log (Meta-Harness, arXiv:2603.28052) ──
        # Per-goal ring buffer of evicted belief angles. Newest write at
        # position `failed_write_idx[goal] % depth`. When the buffer wraps,
        # older entries are overwritten (LRU).
        self.register_buffer(
            'failed_angles',
            torch.zeros(max_goals, failed_buffer_depth, belief_dim),
        )
        # Eviction reason code per entry (0 = empty slot, else EVICT_*).
        self.register_buffer(
            'failed_reasons',
            torch.zeros(max_goals, failed_buffer_depth, dtype=torch.long),
        )
        # FE delta at eviction time (signed float).
        self.register_buffer(
            'failed_fe_deltas',
            torch.zeros(max_goals, failed_buffer_depth),
        )
        # Ring-buffer write pointer per goal.
        self.register_buffer(
            'failed_write_idx',
            torch.zeros(max_goals, dtype=torch.long),
        )
        # Number of valid entries per goal (caps at failed_buffer_depth).
        self.register_buffer(
            'failed_count',
            torch.zeros(max_goals, dtype=torch.long),
        )

    def _ensure_belief_buffer(self, max_beliefs: int, device: torch.device):
        """Lazily resize belief→goal buffer to match state size."""
        if self._belief_source_goal_size < max_beliefs:
            new_buf = torch.full((max_beliefs,), -1, dtype=torch.long, device=device)
            old_size = self.belief_source_goal.shape[0]
            if old_size > 0 and old_size <= max_beliefs:
                new_buf[:old_size] = self.belief_source_goal[:old_size].to(device)
            self.belief_source_goal = new_buf
            self._belief_source_goal_size = max_beliefs

    def record_hypothesis(self, belief_slot: int, goal_idx: int):
        """Record that a provisional belief was generated from a goal."""
        self._ensure_belief_buffer(max(belief_slot + 1, self._belief_source_goal_size),
                                   self.hypothesis_count.device)
        self.belief_source_goal[belief_slot] = goal_idx
        if goal_idx < self.max_goals:
            self.hypothesis_count[goal_idx] += 1

    def record_outcome(
        self,
        belief_slot: int,
        promoted: bool,
        reason_code: int = 0,
        fe_delta: float = 0.0,
        failed_angle: Tensor | None = None,
    ):
        """Record whether a hypothesis was promoted or evicted.

        Args:
            belief_slot: the belief slot that was evaluated
            promoted: True if promoted, False if evicted
            reason_code: 0 for promotion, EVICT_* for eviction reason
            fe_delta: FE(current) - FE(allocation), signed
            failed_angle: belief direction at eviction time. When provided
                and not promoted, pushed into the per-goal failed ring
                buffer so future HypothesisGenerator calls can condition
                on recent failures for the same goal.
        """
        if belief_slot >= len(self.belief_source_goal):
            return
        goal_idx = self.belief_source_goal[belief_slot].item()
        if goal_idx < 0 or goal_idx >= self.max_goals:
            return

        decay = 0.9  # EMA decay
        if promoted:
            self.hypothesis_promoted[goal_idx] += 1
            self.goal_success_ema[goal_idx] = (
                decay * self.goal_success_ema[goal_idx] + (1 - decay) * 1.0
            )
        else:
            self.hypothesis_evicted[goal_idx] += 1
            self.goal_success_ema[goal_idx] = (
                decay * self.goal_success_ema[goal_idx] + (1 - decay) * 0.0
            )
            # Push into the per-goal failed buffer if we have an angle.
            if failed_angle is not None and failed_angle.numel() == self.belief_dim:
                self._push_failure(goal_idx, failed_angle, reason_code, fe_delta)

        # Clear the mapping
        self.belief_source_goal[belief_slot] = -1

    def _push_failure(
        self,
        goal_idx: int,
        angle: Tensor,
        reason_code: int,
        fe_delta: float,
    ):
        """Write a failure record into the per-goal ring buffer."""
        with torch.no_grad():
            write_idx = int(self.failed_write_idx[goal_idx].item())
            self.failed_angles[goal_idx, write_idx] = angle.to(
                self.failed_angles.device
            )
            self.failed_reasons[goal_idx, write_idx] = int(reason_code)
            self.failed_fe_deltas[goal_idx, write_idx] = float(fe_delta)
            self.failed_write_idx[goal_idx] = (
                (write_idx + 1) % self.failed_buffer_depth
            )
            self.failed_count[goal_idx] = min(
                int(self.failed_count[goal_idx].item()) + 1,
                self.failed_buffer_depth,
            )

    def get_failure_summary(self, goal_indices: Tensor) -> tuple[Tensor, Tensor]:
        """Get per-goal failure conditioning for HypothesisGenerator.

        Returns a (summary, count_norm) pair where:
            summary: [G, D] — mean of stored failed angles per goal. A goal
                with zero failures yields a zero vector.
            count_norm: [G] — number of stored failures / buffer depth,
                clamped to [0, 1]. Lets the generator distinguish "no
                failures yet" from "failure buffer is saturated".

        The mean is the simplest pooling that exposes a directional signal
        without imposing structure. The generator has its own learned layer
        on top and can do richer pooling if needed.
        """
        device = self.failed_angles.device
        G = len(goal_indices)
        summary = torch.zeros(G, self.belief_dim, device=device)
        count_norm = torch.zeros(G, device=device)
        if G == 0:
            return summary, count_norm

        valid = (goal_indices >= 0) & (goal_indices < self.max_goals)
        for i in range(G):
            if not valid[i]:
                continue
            g = int(goal_indices[i].item())
            n = int(self.failed_count[g].item())
            if n == 0:
                continue
            # Mean only over populated slots (up to n, but ring-buffer may
            # have wrapped so take the full first `min(n, depth)` rows).
            n_active = min(n, self.failed_buffer_depth)
            summary[i] = self.failed_angles[g, :n_active].mean(dim=0)
            count_norm[i] = n_active / self.failed_buffer_depth
        return summary, count_norm

    def goal_priority_boost(self, goal_indices: Tensor) -> Tensor:
        """Get priority boost for goals based on hypothesis success rate.

        Goals with higher success rates get positive boost (productive research).
        Goals with low success rates get negative boost (unproductive, try other directions).

        Args:
            goal_indices: [G] global goal slot indices

        Returns:
            [G] priority adjustments in [-0.5, 0.5]
        """
        if len(goal_indices) == 0:
            return torch.zeros(0, device=self.goal_success_ema.device)

        # Clamp indices to valid range
        valid = goal_indices < self.max_goals
        boosts = torch.zeros(len(goal_indices), device=self.goal_success_ema.device)
        if valid.any():
            valid_idx = goal_indices[valid]
            # Map success rate [0,1] to boost [-0.5, 0.5]
            boosts[valid] = self.goal_success_ema[valid_idx] - 0.5

        return boosts


def run_autoresearch_step(
    state: CognitiveState,
    hypothesis_gen: HypothesisGenerator,
    tracker: HypothesisTracker,
    current_step: int,
    current_fe: float,
) -> dict:
    """Run one step of the internal autoresearch loop.

    Called from pass2 after goal generation. Generates hypothesis beliefs
    from active goals and allocates them as provisional.

    Args:
        state: cognitive state (modified in-place)
        hypothesis_gen: the hypothesis generator network
        tracker: hypothesis success tracker
        current_step: current training step
        current_fe: current free energy (stored for provisional evaluation)

    Returns:
        dict with stats
    """
    stats = {'hypotheses_generated': 0, 'hypotheses_gated_out': 0}

    goal_indices, goal_embeds, _goal_meta = state.get_active_goals()
    if len(goal_indices) == 0:
        return stats

    # Estimate progress for each goal
    progress = state.telos.estimate_progress(
        goal_embeds, state.beliefs.data, state.get_active_mask()
    )

    beta = state.meta.data[0].item()

    # Apply priority boost from hypothesis success history
    _boosts = tracker.goal_priority_boost(goal_indices)  # noqa: F841 — reserved for future goal priority modulation
    # Skip goals with very low success rate — don't waste slots.
    # Threshold is a learnable MetaParam (viable_goal_min_success, default 0.2).
    viable_threshold = state.meta_params.viable_goal_min_success.item()
    viable = tracker.goal_success_ema[goal_indices] > viable_threshold
    # But always try goals that haven't been tested yet (count == 0)
    untested = tracker.hypothesis_count[goal_indices] == 0
    viable = viable | untested

    if not viable.any():
        stats['hypotheses_gated_out'] = len(goal_indices)
        return stats

    viable_idx = viable.nonzero(as_tuple=False).squeeze(-1)
    viable_embeds = goal_embeds[viable_idx]
    viable_progress = progress[viable_idx]
    viable_global_indices = goal_indices[viable_idx]

    # Meta-Harness conditioning: pull per-goal failure summaries from the
    # tracker so the generator can push away from recently-failed directions.
    with torch.no_grad():
        failure_summary, failure_count_norm = tracker.get_failure_summary(
            viable_global_indices
        )

        hypotheses, _precisions, selected_local = hypothesis_gen(
            viable_embeds, viable_progress,
            state.beliefs.data, state.get_active_mask(), beta,
            failure_summary=failure_summary,
            failure_count=failure_count_norm,
        )

    if hypotheses.shape[0] == 0:
        stats['hypotheses_gated_out'] = len(goal_indices)
        return stats

    # ── PARL-style fair round-robin allocation ──
    # Instead of sequential first-come-first-served (which starves later goals),
    # distribute available slots proportionally across goals weighted by
    # hypothesis success history (priority boost from tracker).
    # Reference: PARL (arXiv:2602.02276) — prevents serial collapse in allocation
    tracker._ensure_belief_buffer(state.config.max_beliefs, state.beliefs.device)

    # Group hypotheses by source goal
    K = hypotheses.shape[0]
    goal_to_hyp: dict[int, list[int]] = {}  # goal_global_idx → [hypothesis indices]
    for i in range(K):
        goal_local_idx = selected_local[i].item()
        goal_global_idx = viable_global_indices[goal_local_idx].item()
        goal_to_hyp.setdefault(goal_global_idx, []).append(i)

    # Compute per-goal slot budgets using success-weighted fair allocation
    n_goals_with_hyps = len(goal_to_hyp)
    if n_goals_with_hyps == 0:
        stats['hypotheses_gated_out'] = len(goal_indices)
        return stats

    available_slots = state.config.max_beliefs - state.num_active_beliefs()
    if available_slots <= 0:
        stats['hypotheses_gated_out'] = len(goal_indices)
        return stats

    # Fair base: each goal gets at least floor(available / n_goals) slots
    # Remainder distributed by priority boost (success EMA)
    base_per_goal = max(1, available_slots // n_goals_with_hyps)
    remainder = available_slots - base_per_goal * n_goals_with_hyps

    # Sort goals by success EMA descending — higher success gets remainder first
    goal_order = sorted(
        goal_to_hyp.keys(),
        key=lambda g: tracker.goal_success_ema[g].item() if g < tracker.max_goals else 0.0,
        reverse=True,
    )

    goal_budgets: dict[int, int] = {}
    for rank, g in enumerate(goal_order):
        budget = min(base_per_goal + (1 if rank < remainder else 0), len(goal_to_hyp[g]))
        goal_budgets[g] = budget

    # Allocate in round-robin order across goals
    stats['per_goal_allocated'] = {}
    with torch.no_grad():
        for goal_idx in goal_order:
            budget = goal_budgets[goal_idx]
            allocated_for_goal = 0
            for hyp_i in goal_to_hyp[goal_idx][:budget]:
                slot = state.allocate_belief(
                    hypotheses[hyp_i],
                    source_type=0,  # observation (from internal generation)
                    step=current_step,
                    provisional=True,
                    current_fe=current_fe,
                )
                if slot >= 0:
                    tracker.record_hypothesis(slot, goal_idx)
                    stats['hypotheses_generated'] += 1
                    allocated_for_goal += 1
                else:
                    break  # state is full — stop all allocation
            stats['per_goal_allocated'][goal_idx] = allocated_for_goal
            if slot < 0:
                break  # propagate state-full signal

    stats['hypotheses_gated_out'] = len(goal_indices) - len(viable_idx)
    stats['goals_with_hypotheses'] = n_goals_with_hyps
    return stats
