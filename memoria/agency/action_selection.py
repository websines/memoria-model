"""D2: EFE-based Action Selection — score and select actions via free energy.

Every possible action (respond, tool_call, search, wait, explore) is scored
by its Expected Free Energy decomposition:

    EFE(a) = -pragmatic(a) + w_epistemic * epistemic(a) + w_risk * risk(a)

The agent naturally:
- Gathers information when uncertain (epistemic dominates → search/explore)
- Executes plans when confident (pragmatic dominates → respond/tool_call)
- Avoids risky actions when fragile (risk dominates → wait)
- Explores when no goals are pressing (curiosity → explore)

Action selection uses Gumbel-Softmax for differentiable discrete choices,
with temperature from MetaParams.action_temperature.

Reference: CDE — Curiosity-Driven Exploration (arXiv:2509.09675)
Reference: IMAGINE — Intrinsic Motivation (arXiv:2505.17621)
Reference: Deep Active Inference for Long Horizons (arXiv:2505.19867)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.polar import EPSILON
from .daemon import ActionType


# Canonical action ordering (matches ActionType enum values)
ACTION_TYPES = [
    ActionType.RESPOND,
    ActionType.TOOL_CALL,
    ActionType.SEARCH,
    ActionType.WAIT,
    ActionType.EXPLORE,
    ActionType.CONSOLIDATE,
]
N_ACTIONS = len(ACTION_TYPES)


class ActionSelector(nn.Module):
    """EFE-based action selection network.

    Scores each candidate action by predicting its Expected Free Energy,
    then selects using Gumbel-Softmax for differentiable discrete choice.

    The network takes the current cognitive state features and predicts
    per-action EFE components (pragmatic, epistemic, risk), which are
    combined using learned weights from MetaParams.

    Args:
        belief_dim: dimension of belief/state features
        n_actions: number of possible actions
    """

    def __init__(self, belief_dim: int, n_actions: int = N_ACTIONS):
        super().__init__()
        self.belief_dim = belief_dim
        self.n_actions = n_actions

        # State encoder: belief stats → compressed state representation
        self.state_encoder = nn.Sequential(
            nn.Linear(belief_dim + 8, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Per-action EFE prediction heads
        # Each predicts (pragmatic, epistemic, risk) for one action type
        self.efe_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 3),  # [pragmatic, epistemic, risk]
            )
            for _ in range(n_actions)
        ])

        # Action embedding (learnable representation of each action type)
        self.action_embeddings = nn.Parameter(
            torch.randn(n_actions, 64) * 0.1,
        )

        # Side-loss buffer: (predicted_efe_components, observed_fe_delta) pairs.
        # Gives a supervised training signal — the net learns to predict the
        # EFE decomposition that actually matches observed FE changes.
        self._side_loss_pairs: list[tuple[torch.Tensor, float]] = []
        self._max_side_buffer = 64

    def extract_state_features(self, state) -> torch.Tensor:
        """Extract fixed-size feature vector from cognitive state.

        Returns:
            [belief_dim + 8] feature vector
        """
        device = state.beliefs.device
        active_mask = state.get_active_mask()
        n_active = active_mask.sum().item()

        if n_active > 0:
            active_beliefs = state.beliefs.data[active_mask]
            mean_belief = active_beliefs.mean(dim=0)   # [D]
        else:
            mean_belief = torch.zeros(state.config.belief_dim, device=device)

        # Contextual features
        max_b = max(state.config.max_beliefs, 1)
        max_e = max(state.config.max_edges, 1)
        max_g = max(state.config.max_goals, 1)

        context = torch.tensor([
            n_active / max_b,                                  # fill ratio
            state.num_active_edges() / max_e,                  # edge fill
            state.num_active_goals() / max_g,                  # goal fill
            state.meta.data[0].item(),                         # beta
            state.meta.data[1].item(),                         # surprise
            state.belief_precision_var[active_mask].mean().item() if n_active > 0 else 1.0,
            float(n_active > 0 and state.belief_provisional[active_mask].any().item()),
            state.meta.data[2].item() / 100.0,                # consolidation timer
        ], device=device)

        return torch.cat([mean_belief, context])

    def score_actions(
        self,
        state,
        efe_epistemic_weight: torch.Tensor,
        efe_risk_weight: torch.Tensor,
        action_risk_aversion: torch.Tensor,
    ) -> dict:
        """Score each action type by its predicted EFE.

        Lower EFE = better action (free energy is minimized).

        Args:
            state: CognitiveState
            efe_epistemic_weight: learned weight on epistemic term
            efe_risk_weight: learned weight on risk term
            action_risk_aversion: additional risk multiplier for actions

        Returns:
            dict with:
                efe_scores: [n_actions] EFE per action (lower = better)
                pragmatic: [n_actions] goal alignment per action
                epistemic: [n_actions] information gain per action
                risk: [n_actions] risk per action
        """
        features = self.extract_state_features(state)
        encoded = self.state_encoder(features.unsqueeze(0)).squeeze(0)  # [64]

        pragmatic = torch.zeros(self.n_actions, device=features.device)
        epistemic = torch.zeros(self.n_actions, device=features.device)
        risk = torch.zeros(self.n_actions, device=features.device)

        for i, head in enumerate(self.efe_heads):
            # Combine state encoding with action embedding
            action_input = encoded + self.action_embeddings[i]
            efe_components = head(action_input.unsqueeze(0)).squeeze(0)  # [3]
            pragmatic[i] = efe_components[0]
            epistemic[i] = efe_components[1]
            risk[i] = F.softplus(efe_components[2])  # risk is non-negative

        # Combined EFE: lower = better (we minimize free energy)
        efe_scores = (
            -pragmatic
            + efe_epistemic_weight * epistemic
            + efe_risk_weight * action_risk_aversion * risk
        )

        return {
            'efe_scores': efe_scores,
            'pragmatic': pragmatic,
            'epistemic': epistemic,
            'risk': risk,
        }

    def select_action(
        self,
        state,
        temperature: torch.Tensor,
        action_risk_aversion: torch.Tensor,
    ) -> tuple[int, dict]:
        """Select an action via Gumbel-Softmax over negative EFE.

        Actions with lower EFE (better) get higher selection probability.

        Args:
            state: CognitiveState
            temperature: Gumbel-Softmax temperature (from MetaParams)
            action_risk_aversion: risk multiplier (from MetaParams)

        Returns:
            (action_index, info_dict)
        """
        efe_info = self.score_actions(
            state,
            state.meta_params.efe_epistemic_weight,
            state.meta_params.efe_risk_weight,
            action_risk_aversion,
        )

        # Negate EFE for selection (lower EFE = higher logit)
        logits = -efe_info['efe_scores']

        # Gumbel-Softmax (hard=True for discrete selection)
        tau = temperature.clamp(min=0.01)
        probs = F.gumbel_softmax(logits, tau=tau.item(), hard=True)
        action_idx = probs.argmax().item()

        info = {
            'action_type': ACTION_TYPES[action_idx],
            'action_idx': action_idx,
            'efe_score': efe_info['efe_scores'][action_idx].item(),
            'pragmatic': efe_info['pragmatic'][action_idx].item(),
            'epistemic': efe_info['epistemic'][action_idx].item(),
            'risk': efe_info['risk'][action_idx].item(),
            'all_efe': efe_info['efe_scores'].detach(),
            'probs': F.softmax(logits, dim=-1).detach(),
        }

        return action_idx, info

    def record_efe_observation(
        self, state, action_idx: int, observed_fe_delta: float,
    ) -> None:
        """Supervise the EFE head for the chosen action.

        Target: observed FE change after the action was selected. The idea
        is that a well-calibrated EFE head's sum(−pragmatic + epistemic +
        risk) should approximate the true Δ(free-energy). Storing the
        prediction under autograd and the observed delta lets
        compute_side_loss train it by regression.
        """
        features = self.extract_state_features(state)
        encoded = self.state_encoder(features.unsqueeze(0)).squeeze(0)
        action_input = encoded + self.action_embeddings[action_idx]
        head = self.efe_heads[action_idx]
        predicted_components = head(action_input.unsqueeze(0)).squeeze(0)
        # Predicted Δ(FE) = sum with EFE sign conventions.
        predicted_delta = (
            -predicted_components[0]                  # −pragmatic
            + predicted_components[1]                 # epistemic
            + F.softplus(predicted_components[2])     # risk (≥0)
        )
        self._side_loss_pairs.append((predicted_delta, float(observed_fe_delta)))
        if len(self._side_loss_pairs) > self._max_side_buffer:
            self._side_loss_pairs = self._side_loss_pairs[-self._max_side_buffer:]

    def compute_side_loss(self) -> torch.Tensor:
        """MSE on predicted vs observed Δ(FE) for selected actions."""
        if not self._side_loss_pairs:
            return (
                sum(p.sum() * 0.0 for p in self.state_encoder.parameters())
                + sum(p.sum() * 0.0 for h in self.efe_heads for p in h.parameters())
                + self.action_embeddings.sum() * 0.0
            )
        preds = torch.stack([p for p, _ in self._side_loss_pairs])
        targets = torch.tensor(
            [t for _, t in self._side_loss_pairs],
            device=preds.device, dtype=preds.dtype,
        )
        loss = F.mse_loss(preds, targets)
        self._side_loss_pairs.clear()
        return loss


def select_action(state) -> tuple[ActionType, dict]:
    """Convenience: select an action using the state's action selector.

    Reads temperature and risk aversion from MetaParams.

    Args:
        state: CognitiveState with action_selector attribute

    Returns:
        (ActionType, info_dict)
    """
    selector = state.action_selector
    temperature = state.meta_params.action_temperature
    risk_aversion = state.meta_params.action_risk_aversion

    action_idx, info = selector.select_action(state, temperature, risk_aversion)
    return ACTION_TYPES[action_idx], info
