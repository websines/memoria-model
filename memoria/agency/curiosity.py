"""D3: Curiosity-Driven Telos Generation — intrinsic motivation for exploration.

Two curiosity signals, both derived from existing Memoria infrastructure:

1. **Actor-side curiosity**: perplexity over generated output. High perplexity
   means the model is in novel territory → should explore.

2. **Critic-side curiosity**: variance of EFE estimates across candidate actions.
   High variance means the model is uncertain about *what to do* → should
   investigate before committing.

When either signal exceeds the learned curiosity_threshold (MetaParam),
the system generates exploration goals via Telos. The curiosity_weight
(MetaParam) controls how strongly exploration goals compete with pragmatic
goals.

No hardcoded thresholds — all behavioral constants from MetaParams.

Reference: CDE — Curiosity-Driven Exploration (arXiv:2509.09675)
Reference: IMAGINE — Intrinsic Motivation (arXiv:2505.17621)
Reference: CD-RLHF — Curiosity-Driven RLHF (arXiv:2501.11463)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.polar import EPSILON


class CuriosityModule(nn.Module):
    """Intrinsic motivation system for autonomous exploration.

    Computes actor-side and critic-side curiosity signals, combines them
    with learned weights, and generates exploration goal embeddings when
    curiosity exceeds the learned threshold.

    The exploration goal generator takes high-curiosity belief regions and
    produces goal embeddings that point toward information-gathering states.

    Args:
        belief_dim: dimension of belief vectors
    """

    def __init__(self, belief_dim: int):
        super().__init__()
        self.belief_dim = belief_dim

        # Actor-side: perplexity estimator from belief state
        # Maps current belief state → expected output entropy
        self.actor_net = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # entropy is non-negative
        )

        # Critic-side: EFE variance estimator
        # Maps state features → predicted EFE variance across actions
        self.critic_net = nn.Sequential(
            nn.Linear(belief_dim + 4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        # Exploration goal generator: high-curiosity beliefs → goal embeddings
        self.goal_generator = nn.Sequential(
            nn.Linear(belief_dim + 2, belief_dim),
            nn.GELU(),
            nn.Linear(belief_dim, belief_dim),
        )

        # Running EMA of curiosity signals (for normalization)
        self.register_buffer('_actor_ema', torch.tensor(1.0))
        self.register_buffer('_critic_ema', torch.tensor(1.0))
        self.register_buffer('_ema_decay', torch.tensor(0.95))

    def compute_actor_curiosity(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Compute actor-side curiosity: expected output entropy.

        High values = model is in novel territory, output is uncertain.

        Args:
            beliefs: [N, D] active belief vectors

        Returns:
            scalar curiosity signal (higher = more curious)
        """
        if beliefs.shape[0] == 0:
            return torch.tensor(0.0, device=beliefs.device)

        # Mean belief → expected entropy
        mean_belief = beliefs.mean(dim=0)
        raw_curiosity = self.actor_net(mean_belief.unsqueeze(0)).squeeze()

        # Normalize by running EMA
        normalized = raw_curiosity / (self._actor_ema + EPSILON)

        # Update EMA
        with torch.no_grad():
            self._actor_ema = (
                self._ema_decay * self._actor_ema
                + (1 - self._ema_decay) * raw_curiosity.detach()
            )

        return normalized

    def compute_critic_curiosity(
        self,
        beliefs: torch.Tensor,
        efe_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute critic-side curiosity: variance of EFE across actions.

        High values = model is uncertain about what to do.

        Args:
            beliefs: [N, D] active belief vectors
            efe_scores: [n_actions] optional EFE scores from action selector

        Returns:
            scalar curiosity signal
        """
        device = beliefs.device
        if beliefs.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        mean_belief = beliefs.mean(dim=0)

        # Contextual features
        if efe_scores is not None:
            efe_var = efe_scores.var()
            efe_range = efe_scores.max() - efe_scores.min()
            efe_mean = efe_scores.mean()
            n_actions = torch.tensor(float(len(efe_scores)), device=device)
        else:
            efe_var = torch.tensor(1.0, device=device)
            efe_range = torch.tensor(1.0, device=device)
            efe_mean = torch.tensor(0.0, device=device)
            n_actions = torch.tensor(1.0, device=device)

        features = torch.cat([
            mean_belief,
            efe_var.unsqueeze(0),
            efe_range.unsqueeze(0),
            efe_mean.unsqueeze(0),
            n_actions.unsqueeze(0),
        ])

        raw_curiosity = self.critic_net(features.unsqueeze(0)).squeeze()

        # Normalize
        normalized = raw_curiosity / (self._critic_ema + EPSILON)
        with torch.no_grad():
            self._critic_ema = (
                self._ema_decay * self._critic_ema
                + (1 - self._ema_decay) * raw_curiosity.detach()
            )

        return normalized

    def compute_combined_curiosity(
        self,
        beliefs: torch.Tensor,
        curiosity_weight: torch.Tensor,
        efe_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined curiosity signal.

        combined = curiosity_weight * (actor + critic) / 2

        Args:
            beliefs: [N, D] active beliefs
            curiosity_weight: from MetaParams.curiosity_weight
            efe_scores: optional EFE scores for critic computation

        Returns:
            scalar combined curiosity
        """
        actor = self.compute_actor_curiosity(beliefs)
        critic = self.compute_critic_curiosity(beliefs, efe_scores)
        return curiosity_weight * (actor + critic) / 2.0

    def generate_exploration_goals(
        self,
        state,
        curiosity_threshold: torch.Tensor,
        curiosity_weight: torch.Tensor,
        max_goals: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate exploration goals when curiosity exceeds threshold.

        Finds the highest-uncertainty belief regions and generates goal
        embeddings that point toward information-gathering states.

        Args:
            state: CognitiveState
            curiosity_threshold: from MetaParams.curiosity_threshold
            curiosity_weight: from MetaParams.curiosity_weight
            max_goals: maximum exploration goals to generate

        Returns:
            goal_embeddings: [K, D] exploration goal vectors (K ≤ max_goals)
            goal_priorities: [K] priority scores for each goal
        """
        device = state.beliefs.device
        active_mask = state.get_active_mask()
        n_active = active_mask.sum().item()

        if n_active == 0:
            return (
                torch.zeros(0, self.belief_dim, device=device),
                torch.zeros(0, device=device),
            )

        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        active_beliefs = state.beliefs.data[active_indices]

        # Combined curiosity signal
        curiosity = self.compute_combined_curiosity(active_beliefs, curiosity_weight)

        if curiosity < curiosity_threshold:
            return (
                torch.zeros(0, self.belief_dim, device=device),
                torch.zeros(0, device=device),
            )

        # Find highest-uncertainty beliefs
        variances = state.belief_precision_var[active_indices]
        n_goals = min(max_goals, n_active)
        top_var_indices = variances.topk(n_goals).indices

        # Generate exploration goals from high-uncertainty beliefs
        seed_beliefs = active_beliefs[top_var_indices]  # [K, D]
        seed_vars = variances[top_var_indices]            # [K]
        seed_curiosity = curiosity.expand(n_goals)        # [K]

        features = torch.cat([
            seed_beliefs,                                  # [K, D]
            seed_vars.unsqueeze(-1),                       # [K, 1]
            seed_curiosity.unsqueeze(-1),                  # [K, 1]
        ], dim=-1)

        goal_embeddings = self.goal_generator(features)    # [K, D]

        # Priority proportional to variance (more uncertain → more urgent)
        priorities = (variances[top_var_indices] / (variances.max() + EPSILON)).clamp(0, 1)
        # Scale by curiosity weight
        priorities = priorities * curiosity_weight

        return goal_embeddings, priorities


def run_curiosity_step(state, curiosity: CuriosityModule) -> dict:
    """Run one curiosity evaluation step.

    Called from pass2 or daemon loop. Computes curiosity signals and
    generates exploration goals if threshold is exceeded.

    Args:
        state: CognitiveState with curiosity module
        curiosity: CuriosityModule instance

    Returns:
        dict with curiosity stats and generated goals
    """
    active_mask = state.get_active_mask()
    if not active_mask.any():
        return {'actor_curiosity': 0.0, 'critic_curiosity': 0.0, 'goals_generated': 0}

    active_beliefs = state.beliefs.data[active_mask]

    actor = curiosity.compute_actor_curiosity(active_beliefs)
    critic = curiosity.compute_critic_curiosity(active_beliefs)

    threshold = state.meta_params.curiosity_threshold
    weight = state.meta_params.curiosity_weight

    goals, priorities = curiosity.generate_exploration_goals(
        state, threshold, weight,
    )

    return {
        'actor_curiosity': actor.item(),
        'critic_curiosity': critic.item(),
        'combined_curiosity': (weight * (actor + critic) / 2.0).item(),
        'threshold': threshold.item(),
        'goals_generated': goals.shape[0],
        'goal_priorities': priorities.tolist() if goals.shape[0] > 0 else [],
    }
