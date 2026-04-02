"""Differentiable Telos: learned goal generation, progress, and lifecycle.

Replaces hardcoded heuristics with four learned components:
1. RND surprise: prediction error of a frozen random projection → novelty signal
2. Goal generator: MLP that proposes goal embeddings from belief state summary
3. Progress estimator: EFE-based progress via cosine distance in belief space
4. Transition network: predicts Gumbel-Softmax status transitions

All operations participate in the computation graph. Gradients flow to
goal embeddings, belief vectors, and all module parameters.

Reference: Burda et al. 2019 (RND), Jang et al. 2017 (Gumbel-Softmax),
           Friston et al. (Expected Free Energy), Florensa et al. 2018 (Goal GAN)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.polar import EPSILON


# Status indices for the 6-dim Gumbel-Softmax vector
STATUS_EMPTY = 0
STATUS_PROPOSED = 1
STATUS_ACTIVE = 2
STATUS_STALLED = 3
STATUS_COMPLETED = 4
STATUS_FAILED = 5
NUM_STATUS = 6


class TelosModule(nn.Module):
    """Differentiable goal system.

    Components:
    - surprise_target: Frozen random projection (RND target network)
    - surprise_predictor: Trained to match target (prediction error = novelty)
    - goal_generator: belief_summary → goal_embedding candidates
    - transition_net: (goal_features) → status_logit_deltas for Gumbel-Softmax
    - progress_head: (belief_state, goal_embedding) → progress estimate

    All stored goal state (embeddings, status logits, EFE at creation) lives
    in the CognitiveState's goal region. This module provides the forward
    computation only.
    """

    def __init__(self, belief_dim: int, max_goals: int = 64, surprise_dim: int | None = None):
        super().__init__()
        self.belief_dim = belief_dim
        self.max_goals = max_goals
        self.surprise_dim = surprise_dim if surprise_dim is not None else belief_dim

        # --- RND Surprise ---
        # Target: frozen random projection of belief space.
        # Uses full belief_dim to avoid information loss (64 was too small for D=256).
        # Orthogonal init for stable, seed-independent random features.
        self.surprise_target = nn.Linear(belief_dim, self.surprise_dim, bias=False)
        nn.init.orthogonal_(self.surprise_target.weight)
        # Freeze target — never updated
        for p in self.surprise_target.parameters():
            p.requires_grad = False
        # Predictor: trained to match target. Prediction error = novelty.
        self.surprise_predictor = nn.Sequential(
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, self.surprise_dim),
        )

        # --- Goal Generator ---
        # Takes belief state summary → proposes goal embedding
        self.goal_generator = nn.Sequential(
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim),
        )

        # --- Transition Network ---
        # Takes goal features → status logit deltas
        # Input: goal_embedding (D) + current_status_probs (6) + progress (1) + surprise (1) = D+8
        self.transition_net = nn.Sequential(
            nn.Linear(belief_dim + NUM_STATUS + 2, belief_dim // 2),
            nn.ReLU(),
            nn.Linear(belief_dim // 2, NUM_STATUS),
        )

        # --- Progress Head ---
        # Estimates goal progress from current beliefs vs goal embedding
        # Uses EFE-inspired distance: how far are relevant beliefs from the goal?
        self.progress_head = nn.Sequential(
            nn.Linear(belief_dim * 2, belief_dim // 2),
            nn.ReLU(),
            nn.Linear(belief_dim // 2, 1),
            nn.Sigmoid(),  # progress in [0, 1]
        )

        # Gumbel-Softmax temperature (annealed during training)
        self.register_buffer('gumbel_tau', torch.tensor(1.0))

    def compute_surprise(self, beliefs: Tensor) -> Tensor:
        """Compute per-belief novelty via RND.

        Args:
            beliefs: [N, D] active belief vectors

        Returns:
            [N] surprise scores (higher = more novel)
        """
        if beliefs.shape[0] == 0:
            return torch.zeros(0, device=beliefs.device)

        with torch.no_grad():
            target = self.surprise_target(beliefs)  # [N, surprise_dim]
        predicted = self.surprise_predictor(beliefs)  # [N, surprise_dim]

        # MSE per belief = novelty signal
        surprise = (target - predicted).pow(2).mean(dim=-1)  # [N]
        return surprise

    def surprise_loss(self, beliefs: Tensor) -> Tensor:
        """Training loss for the surprise predictor.

        Minimizing this makes the predictor better at predicting the target
        for SEEN beliefs, so prediction error for NOVEL beliefs stays high.

        Args:
            beliefs: [N, D] active belief vectors

        Returns:
            Scalar loss
        """
        if beliefs.shape[0] == 0:
            return torch.tensor(0.0, device=beliefs.device)

        with torch.no_grad():
            target = self.surprise_target(beliefs)
        predicted = self.surprise_predictor(beliefs)
        return F.mse_loss(predicted, target)

    def generate_goals(
        self,
        beliefs: Tensor,
        active_mask: Tensor,
        beta: float,
        max_new: int = 3,
    ) -> tuple[Tensor, Tensor]:
        """Propose new goal embeddings from belief state.

        Uses the belief state summary + RND surprise to generate goals
        in high-novelty regions of belief space.

        Args:
            beliefs: [max_beliefs, D] full belief tensor
            active_mask: [max_beliefs] boolean mask of active beliefs
            beta: exploration/exploitation balance (high = more goals)
            max_new: maximum goals to propose

        Returns:
            goal_embeddings: [K, D] proposed goal embeddings (K <= max_new)
            goal_surprise: [K] surprise score for each proposed goal
        """
        if not active_mask.any():
            return torch.zeros(0, self.belief_dim, device=beliefs.device), \
                   torch.zeros(0, device=beliefs.device)

        active_beliefs = beliefs[active_mask]  # [N_active, D]

        # Compute surprise for each active belief
        surprise = self.compute_surprise(active_beliefs)  # [N_active]

        # Select top-k most surprising beliefs as seeds for goals
        k = min(max_new, len(active_beliefs))
        if k == 0:
            return torch.zeros(0, self.belief_dim, device=beliefs.device), \
                   torch.zeros(0, device=beliefs.device)

        top_surprise, top_idx = surprise.topk(k)
        seed_beliefs = active_beliefs[top_idx]  # [K, D]

        # Generate goal embeddings from seed beliefs
        goal_embeds = self.goal_generator(seed_beliefs)  # [K, D]

        # Normalize to unit sphere and scale by surprise (higher surprise = higher precision goal)
        goal_embeds = F.normalize(goal_embeds, dim=-1, eps=EPSILON)
        goal_embeds = goal_embeds * top_surprise.unsqueeze(-1).clamp(min=0.1)

        return goal_embeds, top_surprise

    def update_status(
        self,
        goal_embeddings: Tensor,
        status_logits: Tensor,
        progress: Tensor,
        beliefs: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        """Compute status transitions via Gumbel-Softmax.

        Args:
            goal_embeddings: [N_goals, D]
            status_logits: [N_goals, 6] current status logits
            progress: [N_goals] current progress estimates
            beliefs: [max_beliefs, D] full belief tensor
            active_mask: [max_beliefs] boolean

        Returns:
            new_status_logits: [N_goals, 6] updated status logits
        """
        if goal_embeddings.shape[0] == 0:
            return status_logits

        # Compute per-goal surprise (how novel is the goal region?)
        goal_surprise = self.compute_surprise(goal_embeddings)  # [N_goals]

        # Current status probabilities
        status_probs = F.gumbel_softmax(status_logits, tau=self.gumbel_tau.item(), hard=False)

        # Build transition input: goal_embed + current_status + progress + surprise
        transition_input = torch.cat([
            goal_embeddings,
            status_probs,
            progress.unsqueeze(-1),
            goal_surprise.unsqueeze(-1),
        ], dim=-1)  # [N_goals, D+8]

        # Predict status deltas
        logit_deltas = self.transition_net(transition_input)  # [N_goals, 6]

        # Apply deltas to current logits (residual connection for stability)
        new_logits = status_logits + logit_deltas

        return new_logits

    def estimate_progress(
        self,
        goal_embeddings: Tensor,
        beliefs: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        """Estimate goal progress from belief state (EFE-inspired).

        Progress = how close are current beliefs to the goal?
        High progress when beliefs in the goal's region are precise and aligned.

        Args:
            goal_embeddings: [N_goals, D]
            beliefs: [max_beliefs, D]
            active_mask: [max_beliefs] boolean

        Returns:
            [N_goals] progress in [0, 1]
        """
        if goal_embeddings.shape[0] == 0:
            return torch.zeros(0, device=beliefs.device)

        if not active_mask.any():
            return torch.zeros(goal_embeddings.shape[0], device=beliefs.device)

        active_beliefs = beliefs[active_mask]  # [N_active, D]

        # For each goal, find the most relevant belief (highest cosine sim)
        goal_angles = F.normalize(goal_embeddings, dim=-1, eps=EPSILON)
        belief_angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)

        # [N_goals, N_active] similarity matrix
        sim = goal_angles @ belief_angles.T

        # Soft-max over beliefs: weighted average of similarities
        # (attention-like: which beliefs are most relevant to each goal?)
        attn = F.softmax(sim * 10.0, dim=-1)  # [N_goals, N_active] sharp attention

        # Weighted belief representation for each goal
        belief_repr = attn @ active_beliefs  # [N_goals, D]

        # Progress head: (goal_embed, relevant_beliefs) → progress
        combined = torch.cat([goal_embeddings, belief_repr], dim=-1)  # [N_goals, 2D]
        progress = self.progress_head(combined).squeeze(-1)  # [N_goals]

        return progress

    def anneal_temperature(self, step: int, total_steps: int):
        """Anneal Gumbel-Softmax temperature from 1.0 → 0.1 over training."""
        progress = min(step / max(total_steps, 1), 1.0)
        tau = max(0.1, 1.0 * math.exp(-3.0 * progress))
        self.gumbel_tau.fill_(tau)
