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
    blend of the goal embedding and the current belief state summary,
    transformed through a learned projection.

    The generator learns what kinds of beliefs help for what kinds of goals.
    """

    def __init__(self, belief_dim: int):
        super().__init__()
        self.belief_dim = belief_dim

        # Input: goal_embed (D) + belief_state_summary (D) + goal_progress (1) + beta (1)
        # Output: hypothesis belief vector (D)
        self.hypothesis_net = nn.Sequential(
            nn.Linear(belief_dim * 2 + 2, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim),
        )

        # Precision head: how confident should the hypothesis be?
        # Low initial precision = tentative. Learns to calibrate.
        self.precision_head = nn.Sequential(
            nn.Linear(belief_dim * 2 + 2, belief_dim // 4),
            nn.ReLU(),
            nn.Linear(belief_dim // 4, 1),
            nn.Softplus(),
        )

        # Gate: should we even generate a hypothesis for this goal?
        # Starts mostly closed (bias=-1.0 → sigmoid ≈ 0.27).
        # Prevents flooding the state with bad hypotheses early in training.
        self.generate_gate = nn.Sequential(
            nn.Linear(belief_dim * 2 + 2, belief_dim // 4),
            nn.ReLU(),
            nn.Linear(belief_dim // 4, 1),
        )
        nn.init.constant_(self.generate_gate[-1].bias, -1.0)

    def forward(
        self,
        goal_embeddings: Tensor,
        goal_progress: Tensor,
        beliefs: Tensor,
        active_mask: Tensor,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate hypothesis beliefs from active goals.

        Args:
            goal_embeddings: [G, D] active goal embeddings
            goal_progress: [G] progress estimates per goal
            beliefs: [max_beliefs, D] full belief tensor
            active_mask: [max_beliefs] boolean
            beta: exploration/exploitation parameter

        Returns:
            hypotheses: [K, D] candidate belief vectors (K <= G, gated)
            precisions: [K] precision for each hypothesis
            goal_indices: [K] which goal generated each hypothesis
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

        # Build input features
        beta_t = torch.full((G, 1), beta, device=device)
        progress_t = goal_progress.unsqueeze(-1)  # [G, 1]

        features = torch.cat([
            goal_embeddings,     # [G, D]
            belief_summary,      # [G, D]
            progress_t,          # [G, 1]
            beta_t,              # [G, 1]
        ], dim=-1)  # [G, 2D+2]

        # Gate: should we generate for this goal?
        gate_logits = self.generate_gate(features).squeeze(-1)  # [G]
        gate = torch.sigmoid(gate_logits)

        # Only generate for goals where gate > 0.5
        generate_mask = gate > 0.5
        if not generate_mask.any():
            return (torch.zeros(0, self.belief_dim, device=device),
                    torch.zeros(0, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        selected_features = features[generate_mask]
        selected_indices = generate_mask.nonzero(as_tuple=False).squeeze(-1)

        # Generate hypothesis direction
        raw_hypothesis = self.hypothesis_net(selected_features)  # [K, D]
        hypothesis_dir = F.normalize(raw_hypothesis, dim=-1, eps=EPSILON)

        # Generate precision (low = tentative, learns to calibrate)
        precision = self.precision_head(selected_features).squeeze(-1)  # [K]

        # Scale hypothesis by precision
        hypotheses = hypothesis_dir * precision.unsqueeze(-1)

        return hypotheses, precision, selected_indices


class HypothesisTracker(nn.Module):
    """Track which goals produce successful hypotheses.

    Maintains per-goal success/failure counts. Goals with higher success
    rates get priority boosts, biasing future hypothesis generation toward
    productive research directions.

    This is the "log to results.tsv" step of the autoresearch loop,
    internalized as running statistics.
    """

    def __init__(self, max_goals: int):
        super().__init__()
        self.max_goals = max_goals

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

    def record_outcome(self, belief_slot: int, promoted: bool):
        """Record whether a hypothesis was promoted or evicted."""
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

        # Clear the mapping
        self.belief_source_goal[belief_slot] = -1

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
    # Skip goals with very low success rate (< 0.2 EMA) — don't waste slots
    viable = tracker.goal_success_ema[goal_indices] > 0.2
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

    with torch.no_grad():
        hypotheses, _precisions, selected_local = hypothesis_gen(
            viable_embeds, viable_progress,
            state.beliefs.data, state.get_active_mask(), beta,
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
