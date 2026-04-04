"""SEAL-style cognitive controller: learned policy over pass2 structural actions.

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

Reference: SEAL (MIT, NeurIPS 2025, arXiv:2506.10943)
Reference: SEC — Self-Evolving Curriculum (arXiv:2505.14970)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta


NUM_ACTIONS = 5
# Action indices
ACT_ALLOCATE_RATE = 0
ACT_MERGE_THRESHOLD = 1
ACT_PRUNE_THRESHOLD = 2
ACT_CONNECT_RATE = 3
ACT_GOAL_RATE = 4

# Action names for dict output
ACTION_NAMES = ['allocate_rate', 'merge_threshold', 'prune_threshold', 'connect_rate', 'goal_rate']


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
    ):
        super().__init__()
        self.belief_dim = belief_dim
        self.reward_ema_decay = reward_ema_decay

        # Action ranges: (min, max) for each action, configurable
        if action_ranges is None:
            action_ranges = [
                (0.0, 1.0),    # allocate_rate
                (0.80, 0.99),  # merge_threshold
                (0.01, 0.50),  # prune_threshold
                (0.0, 1.0),    # connect_rate
                (0.0, 3.0),    # goal_rate
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
        self.state_dim = 8

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

        return features

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

        # Scale to action ranges
        actions = {}
        for i, (name, (lo, hi)) in enumerate(zip(ACTION_NAMES, self.action_ranges)):
            actions[name] = lo + (hi - lo) * raw_samples[i].item()

        # Save actual log-probability and entropy for REINFORCE + entropy bonus
        log_prob = dist.log_prob(raw_samples).sum()  # sum over independent actions
        entropy = dist.entropy().sum()  # Beta entropy (closed-form, cheap)
        value = self.value_head(features)

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

    def compute_dense_reward(self, state, belief_advantage: float) -> float:
        """Combine sparse belief_advantage with dense state-delta signals.

        Uses cognitive state feature deltas as auxiliary rewards to densify
        the sparse belief_advantage signal.

        Reference: SASR (ICLR 2025, arXiv:2408.03029)
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

        return belief_advantage + 0.1 * d_fill + 0.1 * d_radius + 0.05 * d_edges
