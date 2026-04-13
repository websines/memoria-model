"""SEAL-style cognitive controller with PARL staged reward shaping.

Instead of hardcoded heuristics deciding when to allocate, merge, prune, or
connect beliefs, a small learned controller makes these decisions using the
current cognitive state as input and belief_advantage as reward signal.

Action space (continuous, per-step decisions):
- allocate_rate: what fraction of candidate beliefs to allocate [0, 1]
- merge_threshold: similarity threshold for consolidation [0.8, 0.99]
- prune_threshold: minimum precision to keep a belief [0.01, 0.5]
- connect_rate: what fraction of proposed edges to accept [0, 1]
- goal_rate: how many goals to generate [0, 3]

The policy network outputs Beta distribution parameters (α, β) per action.
Actions are sampled from Beta(α, β) ∈ [0, 1] and scaled to their valid ranges.
REINFORCE uses the actual log-probability of the sampled action under the Beta
distribution, which provides an unbiased policy gradient estimator.

PARL staged reward: r_perf + r_parallel + r_finish. The r_parallel component
rewards diverse goal pursuit (prevents serial collapse), r_finish penalizes
abandoned goals (prevents spurious parallelism). Both auxiliary rewards are
annealed to zero over training so the final policy optimizes purely for r_perf.

Meta-Harness rich-history encoding (arXiv:2603.28052, Table 3): in addition
to the scalar state features, the controller ingests a rolling history of
the last HISTORY_DEPTH (action, outcome) tuples through a small GRU. The
GRU's output is projected and ADDED to the scalar features before the policy
head — the projection is zero-initialized so the pre-#1 controller behavior
is exactly preserved at t=0. As training progresses the projection learns
to surface patterns like "the last three prune actions correlated with
rising FE," letting the policy do the causal reasoning over its own history
that the Meta-Harness ablation shows is worth ~15 accuracy points.

Reference: Meta-Harness (Stanford, 2026, arXiv:2603.28052) — filesystem-based
    access to prior diagnostic experience beats compressed scalar feedback
Reference: SEAL (MIT, NeurIPS 2025, arXiv:2506.10943)
Reference: SEC — Self-Evolving Curriculum (arXiv:2505.14970)
Reference: PARL (Kimi K2.5, arXiv:2602.02276) — staged reward, serial collapse
Reference: D3PO (arXiv:2602.07764) — diversity regularization
Reference: GCR-PPO (arXiv:2509.14816) — per-objective gradient decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta


NUM_ACTIONS = 6
# Action indices
ACT_ALLOCATE_RATE = 0
ACT_MERGE_THRESHOLD = 1
ACT_PRUNE_THRESHOLD = 2
ACT_CONNECT_RATE = 3
ACT_GOAL_RATE = 4
ACT_STRATEGY_SCALE = 5

# Action names for dict output
ACTION_NAMES = [
    'allocate_rate', 'merge_threshold', 'prune_threshold',
    'connect_rate', 'goal_rate', 'strategy_scale',
]

# Per-step outcome vector dimensions:
#   [belief_advantage, d_fill, d_mean_radius, d_edge_count,
#    d_goal_diversity, d_goal_completion]
# These are exactly the signals compute_dense_reward already derives each
# step, minus r_perf/r_parallel/r_finish (which are linear combinations of
# the same raw deltas). Keeping them un-fused preserves the signal and lets
# the GRU learn its own weighting. This is structural — it matches the
# number of raw delta signals the outcome vector carries — not a tunable.
OUTCOME_DIM = 6


class CognitiveController(nn.Module):
    """Learned controller over pass2 structural actions.

    Input: cognitive state summary vector
    Output: continuous action values sampled from Beta distributions

    Trained via REINFORCE with belief_advantage as reward.
    The controller learns WHEN to allocate aggressively, WHEN to prune,
    WHEN to merge, based on what improves downstream performance.

    The policy outputs Beta(α, β) parameters per action. Actions are sampled
    stochastically during training (exploration) and use the mean during
    inference (exploitation). Log-probabilities are computed analytically
    from the Beta PDF for unbiased REINFORCE gradients.
    """

    def __init__(
        self,
        belief_dim: int,
        hidden_dim: int = 128,
        action_ranges: list[tuple[float, float]] | None = None,
        reward_ema_decay: float = 0.99,
        history_depth: int = 32,
    ):
        super().__init__()
        self.belief_dim = belief_dim
        self.reward_ema_decay = reward_ema_decay
        self.history_depth = history_depth

        # Action ranges: (min, max) for each action, configurable
        if action_ranges is None:
            action_ranges = [
                (0.0, 1.0),    # allocate_rate
                (0.80, 0.99),  # merge_threshold
                (0.01, 0.50),  # prune_threshold
                (0.0, 1.0),    # connect_rate
                (0.0, 3.0),    # goal_rate
                (0.0, 2.0),    # strategy_scale — multiplier on MetaParam perturbation_scale
            ]
        self.action_ranges = action_ranges

        # State encoder features:
        # - mean/std of active belief radii (2)
        # - belief fill ratio (1)
        # - active edge count / max (1)
        # - beta (1)
        # - mean surprise (1)
        # - active goal count / max (1)
        # - belief advantage EMA (1)
        # - goal hypothesis diversity (normalized entropy over per-goal hyp counts) (1)
        # - goal completion rate (completed / (completed + failed + active)) (1)
        self.state_dim = 10

        # Policy network: outputs 2 * NUM_ACTIONS values (α_raw, β_raw per action)
        # softplus(x) + 1.0 gives Beta params ≥ 1.0 (unimodal distributions)
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS * 2),
        )
        # Init bias so default Beta(α,β) is concentrated near 1.0:
        # softplus(3) + 1 ≈ 4.05 for α, softplus(0) + 1 ≈ 1.69 for β
        # Beta(4.05, 1.69) has mean ≈ 0.71 and moderate variance
        # → allocate_rate ≈ 0.71, connect_rate ≈ 0.71 (permissive defaults)
        with torch.no_grad():
            bias = self.policy[-1].bias
            bias[:NUM_ACTIONS] = 3.0   # α params → high (skew toward 1)
            bias[NUM_ACTIONS:] = 0.0   # β params → moderate

        # Value baseline (for variance reduction in REINFORCE)
        self.value_head = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Meta-Harness rich history encoder (#1) ──
        # GRU over the rolling (action, outcome) history. Hidden state is
        # projected through a zero-initialized linear layer and ADDED to
        # the scalar feature vector before the policy/value heads read it.
        # Zero-init of `history_proj` means the history contribution starts
        # at exactly zero, so at t=0 the controller behaves identically to
        # the pre-#1 version. As `history_proj` learns non-zero columns,
        # patterns in the action/outcome trajectory flow into the policy.
        hist_input_dim = NUM_ACTIONS + OUTCOME_DIM
        self.history_hidden = hidden_dim // 2
        self.history_gru = nn.GRU(
            input_size=hist_input_dim,
            hidden_size=self.history_hidden,
            batch_first=True,
        )
        self.history_proj = nn.Linear(self.history_hidden, self.state_dim)
        with torch.no_grad():
            self.history_proj.weight.zero_()
            self.history_proj.bias.zero_()

        # Ring buffer for history. We store NORMALIZED actions (the raw
        # Beta samples in (0, 1)) so the GRU sees a bounded input regardless
        # of the action_ranges configuration. Outcomes are the raw signed
        # deltas that compute_dense_reward already derives.
        self.register_buffer(
            'history_actions', torch.zeros(history_depth, NUM_ACTIONS),
        )
        self.register_buffer(
            'history_outcomes', torch.zeros(history_depth, OUTCOME_DIM),
        )
        # Total number of entries ever pushed. `history_total_writes % depth`
        # is the next write slot; min(total, depth) is the valid prefix.
        self.register_buffer(
            'history_total_writes', torch.zeros(1, dtype=torch.long),
        )
        # Most recent sampled action (normalized, pre-scaling). Staged here
        # by get_actions and consumed by compute_dense_reward when it commits
        # the (action, outcome) pair to the ring buffer.
        self.register_buffer('_pending_action', torch.zeros(NUM_ACTIONS))
        self.register_buffer('_has_pending_action', torch.zeros(1, dtype=torch.bool))

        # Running stats for reward normalization
        self.register_buffer('reward_mean', torch.tensor(0.0))
        self.register_buffer('reward_var', torch.tensor(1.0))
        self.register_buffer('belief_advantage_ema', torch.tensor(0.0))

        # SAC-style adaptive entropy temperature (Haarnoja 2018, arXiv:1812.05905)
        # Learns optimal entropy coefficient automatically. Replaces fixed 0.02.
        self.log_alpha_entropy = nn.Parameter(torch.tensor(0.0))
        self.target_entropy = -float(NUM_ACTIONS)  # -5.0

        # PPO clipping epsilon (Petrazzini & Antonelo 2021, arXiv:2111.02202)
        self.ppo_clip_eps = 0.2

        # Trajectory buffers for PPO
        self._saved_log_probs: list[Tensor] = []
        self._saved_old_log_probs: list[Tensor] = []
        self._saved_entropies: list[Tensor] = []
        self._saved_values: list[Tensor] = []
        self._saved_rewards: list[float] = []

        # Dense reward tracking (state-delta auxiliary signals)
        self._prev_fill_ratio: float = 0.0
        self._prev_mean_radius: float = 0.0
        self._prev_edge_count: float = 0.0

        # PARL staged reward tracking
        # Tracks per-goal hypothesis counts and completion for diversity/finish rewards
        self._prev_goal_diversity: float = 0.0
        self._prev_goal_completion_rate: float = 0.0

    def _get_beta_dist(self, features: Tensor) -> tuple[Beta, Tensor, Tensor]:
        """Compute Beta distribution parameters from state features.

        Returns:
            dist: Beta distribution for sampling
            alpha: [NUM_ACTIONS] concentration parameter α
            beta: [NUM_ACTIONS] concentration parameter β
        """
        raw = self.policy(features)  # [NUM_ACTIONS * 2]
        alpha_raw = raw[:NUM_ACTIONS]
        beta_raw = raw[NUM_ACTIONS:]
        # softplus + 1.0 ensures params ≥ 1.0 (unimodal Beta)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        return Beta(alpha, beta), alpha, beta

    def encode_state(self, state) -> Tensor:
        """Encode cognitive state into a fixed-size feature vector.

        The scalar features below are the legacy 10-dim snapshot used by
        prior versions of the controller. On top of those, the rolling
        history of (action, outcome) tuples is passed through a GRU and
        its projection is ADDED to the scalar features (Meta-Harness #1).
        The projection is zero-initialized, so a freshly-built controller
        behaves exactly like the pre-#1 version at t=0 and learns to use
        the history signal as training progresses.

        Args:
            state: CognitiveState instance

        Returns:
            [state_dim] feature vector
        """
        device = state.beliefs.device

        with torch.no_grad():
            radii = state.get_belief_radii()
            active = state.get_active_mask()
            active_radii = radii[active] if active.any() else torch.zeros(1, device=device)

            features = torch.zeros(self.state_dim, device=device)
            features[0] = active_radii.mean()
            features[1] = active_radii.std() if len(active_radii) > 1 else 0.0
            features[2] = active.float().mean()  # fill ratio
            features[3] = state.num_active_edges() / max(state.config.max_edges, 1)
            features[4] = state.beta
            features[5] = state.running_stats.mean_surprise.item()
            features[6] = state.num_active_goals() / max(state.config.max_goals, 1)
            features[7] = self.belief_advantage_ema.item()

            # PARL: goal hypothesis diversity — normalized entropy over per-goal hyp counts
            # High diversity = multiple goals being actively explored
            # Low diversity = serial collapse (one goal dominating)
            features[8] = self._compute_goal_diversity(state)

            # PARL: goal completion rate — fraction of goals reaching completed status
            features[9] = self._compute_goal_completion_rate(state)

        # Meta-Harness rich history (#1): add GRU-encoded trajectory signal.
        # Zero-init of history_proj means this adds zeros at t=0, preserving
        # the pre-#1 scalar-only behavior. The history_proj Linear is a real
        # nn.Parameter, so this term participates in gradients from the
        # policy/value heads.
        history_feature = self._encode_history(device)
        return features + history_feature

    def _encode_history(self, device: torch.device) -> Tensor:
        """Encode the rolling (action, outcome) history into state-dim features.

        Reads the ring buffer, re-orders entries into chronological order
        (oldest first), runs them through the GRU, and projects the final
        hidden state to state-dim via the zero-initialized `history_proj`.
        Returns a zero vector when no entries have been pushed yet.
        """
        total = int(self.history_total_writes.item())
        if total == 0:
            return torch.zeros(self.state_dim, device=device)

        n_valid = min(total, self.history_depth)
        if total < self.history_depth:
            # Buffer has not wrapped — entries [0, total) are in order.
            actions = self.history_actions[:n_valid]
            outcomes = self.history_outcomes[:n_valid]
        else:
            # Buffer wrapped. The current write slot points at the oldest
            # entry; roll so position 0 is the oldest.
            write_slot = total % self.history_depth
            actions = torch.cat(
                [self.history_actions[write_slot:], self.history_actions[:write_slot]],
                dim=0,
            )
            outcomes = torch.cat(
                [self.history_outcomes[write_slot:], self.history_outcomes[:write_slot]],
                dim=0,
            )

        # [1, T, NUM_ACTIONS + OUTCOME_DIM]
        seq = torch.cat([actions, outcomes], dim=-1).unsqueeze(0).to(device)
        _, h_n = self.history_gru(seq)
        # h_n: [1, 1, history_hidden] → [history_hidden]
        last_hidden = h_n.squeeze(0).squeeze(0)
        return self.history_proj(last_hidden)

    def _commit_history_entry(self, action_vec: Tensor, outcome_vec: Tensor):
        """Push a (normalized action, outcome) pair into the ring buffer.

        Called from compute_dense_reward once the outcome for the pending
        action has been derived. Uses `.data` assignment because buffers
        are not differentiable — they're summaries the GRU reads from, not
        gradient carriers.
        """
        with torch.no_grad():
            total = int(self.history_total_writes.item())
            slot = total % self.history_depth
            self.history_actions[slot] = action_vec.detach().to(
                self.history_actions.device
            )
            self.history_outcomes[slot] = outcome_vec.detach().to(
                self.history_outcomes.device
            )
            self.history_total_writes[0] = total + 1

    def _compute_goal_diversity(self, state) -> float:
        """Normalized entropy over per-goal hypothesis counts.

        Returns 0.0 when all hypotheses belong to one goal (serial collapse),
        1.0 when evenly distributed across all active goals (max diversity).
        """
        if not hasattr(state, 'hypothesis_tracker'):
            return 0.0
        tracker = state.hypothesis_tracker
        goal_indices, _, _ = state.get_active_goals()
        if len(goal_indices) <= 1:
            return 1.0  # one or zero goals — diversity is trivially maximal
        counts = tracker.hypothesis_count[goal_indices].float()
        total = counts.sum()
        if total < 1.0:
            return 1.0  # no hypotheses yet — neutral
        probs = counts / total
        probs = probs.clamp(min=1e-8)  # avoid log(0)
        entropy = -(probs * probs.log()).sum()
        max_entropy = torch.tensor(float(len(goal_indices))).log()
        return (entropy / max_entropy.clamp(min=1e-8)).item()

    def _compute_goal_completion_rate(self, state) -> float:
        """Fraction of non-empty goals that reached completed status."""
        statuses = state.goal_status_logits.argmax(dim=-1)  # [max_goals]
        non_empty = (statuses != 0)  # not empty
        if not non_empty.any():
            return 0.0
        completed = (statuses == 4)  # completed status index
        return completed.float().sum().item() / non_empty.float().sum().item()

    def get_actions(self, state) -> dict[str, float]:
        """Sample actions from the Beta policy.

        Returns a dict with named action values scaled to valid ranges.
        Also saves log_prob and value for REINFORCE update.
        """
        features = self.encode_state(state)

        # Build Beta distribution from policy network
        dist, alpha, beta_param = self._get_beta_dist(features)

        # Sample from Beta(α, β) ∈ (0, 1) — stochastic exploration
        # Clamp to avoid exact 0/1 (log_prob is -inf at boundaries)
        raw_samples = dist.rsample().clamp(1e-6, 1 - 1e-6)  # [NUM_ACTIONS]

        # Stage the normalized action for the history buffer. compute_dense_reward
        # will pair it with the derived outcome and commit the tuple to the
        # ring buffer.
        with torch.no_grad():
            self._pending_action.copy_(raw_samples.detach())
            self._has_pending_action[0] = True

        # Scale to action ranges
        actions = {}
        for i, (name, (lo, hi)) in enumerate(zip(ACTION_NAMES, self.action_ranges)):
            actions[name] = lo + (hi - lo) * raw_samples[i].item()

        # PARL metrics: expose goal diversity and completion for wandb logging
        actions['_goal_diversity'] = features[8].item()
        actions['_goal_completion_rate'] = features[9].item()

        # Save actual log-probability and entropy for REINFORCE + entropy bonus
        log_prob = dist.log_prob(raw_samples).sum()  # sum over independent actions
        entropy = dist.entropy().sum()  # Beta entropy (closed-form, cheap)
        value = self.value_head(features)

        # Cap trajectory length to prevent OOM from unbounded graph retention.
        # Each entry pins the full controller forward-pass graph in memory.
        # compute_loss() may only be called every ~1000 steps, so without a cap
        # the buffer would accumulate 1000+ graph snapshots.
        MAX_TRAJECTORY = 128
        if len(self._saved_log_probs) >= MAX_TRAJECTORY:
            # Detach oldest entries to free their computation graphs
            self._saved_log_probs[0] = self._saved_log_probs[0].detach()
            self._saved_values[0] = self._saved_values[0].detach()
            self._saved_entropies[0] = self._saved_entropies[0].detach()

        self._saved_log_probs.append(log_prob)
        self._saved_old_log_probs.append(log_prob.detach())  # PPO: detached copy
        self._saved_entropies.append(entropy)
        self._saved_values.append(value.squeeze())

        return actions

    def record_reward(self, belief_advantage: float):
        """Record a reward signal (belief_advantage delta)."""
        self._saved_rewards.append(belief_advantage)

        # Update EMA
        self.belief_advantage_ema.copy_(
            self.reward_ema_decay * self.belief_advantage_ema
            + (1 - self.reward_ema_decay) * belief_advantage
        )

    def compute_loss(self) -> Tensor:
        """Compute REINFORCE loss from saved trajectory.

        Call this periodically (e.g., every 100 steps) to update the controller.
        Clears the saved buffers after computing loss.

        Uses proper log π(a|s) from Beta distribution, giving unbiased
        policy gradient estimates: ∇J = E[∇log π(a|s) · advantage].

        Includes entropy bonus (H_coeff × H[π]) to prevent premature collapse
        when rewards are sparse. The Beta distribution's entropy has a closed-form
        expression, so this adds negligible cost.

        Returns:
            Scalar policy loss (add to main training loss at small weight)
        """
        if len(self._saved_rewards) == 0 or len(self._saved_log_probs) == 0:
            device = self.reward_mean.device
            return torch.tensor(0.0, device=device)

        device = self._saved_log_probs[0].device

        # Normalize rewards
        rewards = torch.tensor(self._saved_rewards, device=device)
        if len(rewards) > 1:
            reward_mean = rewards.mean()
            reward_std = rewards.std().clamp(min=1e-8)
            normalized = (rewards - reward_mean) / reward_std
        else:
            normalized = rewards

        # PPO clipped surrogate + value loss + adaptive entropy
        n = min(len(self._saved_log_probs), len(normalized))
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_bonus = torch.tensor(0.0, device=device)

        for i in range(n):
            advantage = normalized[i] - self._saved_values[i].detach()
            # PPO clipped surrogate (Petrazzini & Antonelo 2021)
            if i < len(self._saved_old_log_probs):
                ratio = torch.exp(self._saved_log_probs[i] - self._saved_old_log_probs[i])
                clipped = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps)
                policy_loss = policy_loss - torch.min(ratio * advantage, clipped * advantage)
            else:
                policy_loss = policy_loss - self._saved_log_probs[i] * advantage
            value_loss = value_loss + F.mse_loss(
                self._saved_values[i], normalized[i].detach()
            )
            if i < len(self._saved_entropies):
                entropy_bonus = entropy_bonus + self._saved_entropies[i]

        # SAC-style adaptive entropy (Haarnoja 2018)
        alpha_ent = self.log_alpha_entropy.exp().detach()
        total_loss = (policy_loss + 0.5 * value_loss - alpha_ent * entropy_bonus) / max(n, 1)

        # Alpha entropy loss: adjust temperature toward target entropy
        mean_entropy = entropy_bonus.detach() / max(n, 1)
        alpha_loss = -(self.log_alpha_entropy * (mean_entropy - self.target_entropy))
        total_loss = total_loss + alpha_loss

        # Clear buffers
        self._saved_log_probs.clear()
        self._saved_old_log_probs.clear()
        self._saved_entropies.clear()
        self._saved_values.clear()
        self._saved_rewards.clear()

        return total_loss

    def compute_dense_reward(
        self, state, belief_advantage: float,
        training_progress: float = 0.0,
    ) -> float:
        """Combine sparse belief_advantage with PARL staged reward signals.

        Three reward components (PARL, arXiv:2602.02276):
        - r_perf: belief_advantage + state deltas (always active)
        - r_parallel: rewards diverse goal pursuit (prevents serial collapse)
        - r_finish: penalizes abandoned goals (prevents spurious parallelism)

        r_parallel and r_finish are annealed to zero by (1 - training_progress)
        so the final policy optimizes purely for r_perf. Weights are learned
        MetaParams, not hardcoded constants.

        Args:
            state: CognitiveState
            belief_advantage: scalar advantage from forward pass
            training_progress: float in [0, 1], 0=start, 1=end of training

        Reference: SASR (ICLR 2025, arXiv:2408.03029) — dense auxiliary rewards
        Reference: PARL (arXiv:2602.02276) — staged reward shaping
        """
        with torch.no_grad():
            fill_ratio = state.get_active_mask().float().mean().item()
            radii = state.get_belief_radii()
            active = state.get_active_mask()
            mean_radius = radii[active].mean().item() if active.any() else 0.0
            edge_count = state.num_active_edges() / max(state.config.max_edges, 1)

        d_fill = fill_ratio - self._prev_fill_ratio
        d_radius = mean_radius - self._prev_mean_radius
        d_edges = edge_count - self._prev_edge_count

        self._prev_fill_ratio = fill_ratio
        self._prev_mean_radius = mean_radius
        self._prev_edge_count = edge_count

        # r_perf: performance reward (always active, never annealed)
        r_perf = belief_advantage + 0.1 * d_fill + 0.1 * d_radius + 0.05 * d_edges

        # ── PARL auxiliary rewards (annealed to zero) ──
        anneal = max(0.0, 1.0 - training_progress)  # 1.0 → 0.0 over training

        # r_parallel: reward for diverse goal pursuit
        # Positive when goal diversity increases (multiple goals being explored),
        # negative when diversity drops (serial collapse beginning).
        goal_diversity = self._compute_goal_diversity(state)
        d_diversity = goal_diversity - self._prev_goal_diversity
        self._prev_goal_diversity = goal_diversity

        w_parallel = state.meta_params.parl_parallel_reward_weight.item()
        r_parallel = w_parallel * d_diversity * anneal

        # r_finish: penalty for abandoned goals (spawned but never completed)
        # Positive when completion rate improves, negative when goals are abandoned.
        # Prevents the controller from gaming r_parallel by spawning dummy goals.
        completion_rate = self._compute_goal_completion_rate(state)
        d_completion = completion_rate - self._prev_goal_completion_rate
        self._prev_goal_completion_rate = completion_rate

        w_finish = state.meta_params.parl_finish_reward_weight.item()
        r_finish = w_finish * d_completion * anneal

        # ── Meta-Harness #1: commit (action, outcome) to rolling history ──
        # The outcome vector preserves the RAW state deltas the GRU will
        # learn over, not the scalar r_perf/r_parallel/r_finish reward.
        # Layout matches OUTCOME_DIM: [belief_advantage, d_fill, d_radius,
        # d_edges, d_diversity, d_completion]. If no action is pending
        # (e.g. compute_dense_reward called without a preceding get_actions),
        # skip the commit rather than push a stale action.
        if bool(self._has_pending_action.item()):
            outcome_vec = torch.tensor(
                [
                    belief_advantage,
                    d_fill,
                    d_radius,
                    d_edges,
                    d_diversity,
                    d_completion,
                ],
                device=self._pending_action.device,
            )
            self._commit_history_entry(self._pending_action.clone(), outcome_vec)
            with torch.no_grad():
                self._has_pending_action[0] = False

        return r_perf + r_parallel + r_finish
