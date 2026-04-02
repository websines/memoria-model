"""SEAL-style cognitive controller: learned policy over pass2 structural actions.

Instead of hardcoded heuristics deciding when to allocate, merge, prune, or
connect beliefs, a small learned controller makes these decisions using the
current cognitive state as input and belief_advantage as reward signal.

Action space (discrete, per-step decisions):
- allocate_rate: what fraction of candidate beliefs to allocate [0, 1]
- merge_threshold: similarity threshold for consolidation [0.8, 0.99]
- prune_threshold: minimum precision to keep a belief [0.01, 0.5]
- connect_rate: what fraction of proposed edges to accept [0, 1]
- goal_rate: how many goals to generate [0, 3]

The controller outputs continuous action values. Pass2 uses them directly.

Reference: SEAL (MIT, NeurIPS 2025, arXiv:2506.10943)
Reference: SEC — Self-Evolving Curriculum (arXiv:2505.14970)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    Output: continuous action values scaled to valid ranges

    Trained via REINFORCE with belief_advantage as reward.
    The controller learns WHEN to allocate aggressively, WHEN to prune,
    WHEN to merge, based on what improves downstream performance.
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

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
            nn.Sigmoid(),  # output in [0, 1], scaled to action ranges
        )
        # Init bias so default actions are fully permissive
        # sigmoid(5) ≈ 0.993 → allocate_rate≈1.0, connect_rate≈1.0
        with torch.no_grad():
            self.policy[-2].bias.fill_(5.0)

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

        # Log-probs buffer for REINFORCE
        self._saved_log_probs: list[Tensor] = []
        self._saved_values: list[Tensor] = []
        self._saved_rewards: list[float] = []

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
        """Get action values from the controller.

        Returns a dict with named action values scaled to valid ranges.
        Also saves log_prob and value for REINFORCE update.
        """
        features = self.encode_state(state)

        # Policy output in [0, 1]
        raw_actions = self.policy(features)  # [NUM_ACTIONS]

        # Scale to action ranges
        actions = {}
        for i, (name, (lo, hi)) in enumerate(zip(ACTION_NAMES, self.action_ranges)):
            actions[name] = lo + (hi - lo) * raw_actions[i].item()

        # Save for REINFORCE
        value = self.value_head(features)
        # Log-prob proxy: penalize deviation from midpoint (encourages exploration)
        # Midpoint of sigmoid output is 0.5 — this is a structural constant (center of [0,1])
        log_prob = -((raw_actions - 0.5) ** 2).sum()
        self._saved_log_probs.append(log_prob)
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

        # REINFORCE with baseline
        n = min(len(self._saved_log_probs), len(normalized))
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)

        for i in range(n):
            advantage = normalized[i] - self._saved_values[i].detach()
            policy_loss = policy_loss - self._saved_log_probs[i] * advantage
            value_loss = value_loss + F.mse_loss(
                self._saved_values[i], normalized[i].detach()
            )

        total_loss = (policy_loss + 0.5 * value_loss) / max(n, 1)

        # Clear buffers
        self._saved_log_probs.clear()
        self._saved_values.clear()
        self._saved_rewards.clear()

        return total_loss
