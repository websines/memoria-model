"""Read path: retrieve relevant beliefs from cognitive state into transformer hidden stream.

Uses Hopfield-style content-addressable lookup (Ramsauer et al., 2020):
    retrieved = softmax(query @ keys.T / sqrt(d)) @ values

Where:
- query = current hidden state projected into belief space
- keys = belief angles (unit vectors, content direction)
- values = belief vectors (full cartesian, precision-weighted)
- Goal modulation biases attention toward goal-relevant beliefs

Only active beliefs (radius > 0) participate. Cost scales with active count.

Reference: hopfield-layers (github.com/ml-jku/hopfield-layers)
Reference: prototype-research/src/pipeline/scoring.rs (factor message fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import belief_is_active, angular_similarity, EPSILON


class ReadPath(nn.Module):
    """Content-addressable retrieval from belief region.

    Projects transformer hidden states into belief space, retrieves relevant
    beliefs via Hopfield attention, optionally modulated by active goals,
    and projects back to hidden space.
    """

    def __init__(self, hidden_dim: int, belief_dim: int, num_heads: int = 4, top_k: int = 32):
        """
        Args:
            hidden_dim: transformer hidden dimension
            belief_dim: belief representation dimension (D)
            num_heads: number of retrieval heads (parallel queries)
            top_k: max beliefs to attend over (sparse, for efficiency)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.num_heads = num_heads
        self.top_k = top_k

        # Project hidden state → query in belief space
        self.query_proj = nn.Linear(hidden_dim, belief_dim * num_heads, bias=False)

        # Project retrieved beliefs → hidden space
        self.output_proj = nn.Linear(belief_dim * num_heads, hidden_dim, bias=False)

        # Goal modulation: project goal embeddings to attention bias
        self.goal_gate = nn.Linear(belief_dim, num_heads, bias=False)

        # Learnable temperature for Hopfield attention (per head)
        self.log_temperature = nn.Parameter(torch.zeros(num_heads))

        # Initialize output projection to zero (residual-friendly, from nanogpt speedrun)
        nn.init.zeros_(self.output_proj.weight)

    def forward(
        self,
        hidden: Tensor,
        state: CognitiveState,
        goal_embeddings: Tensor | None = None,
        goal_priorities: Tensor | None = None,
    ) -> Tensor:
        """Retrieve relevant beliefs and project into hidden space.

        Args:
            hidden: [B, T, hidden_dim] transformer hidden states
            state: the cognitive state to read from
            goal_embeddings: [N_goals, D] active goal embeddings (optional)
            goal_priorities: [N_goals] goal priorities for weighting (optional)

        Returns:
            [B, T, hidden_dim] retrieved belief information to add to residual stream
        """
        B, T, H = hidden.shape
        device = hidden.device

        # Get active beliefs only (sparse)
        active_mask = state.get_active_mask()
        if not active_mask.any():
            # No beliefs → return zeros (model degrades to pure transformer)
            return torch.zeros_like(hidden)

        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        active_beliefs = state.beliefs.data[active_indices]  # [N_active, D]
        N_active = active_beliefs.shape[0]

        # Limit to top_k for efficiency if many active beliefs
        if N_active > self.top_k:
            # Pre-filter: pick top_k by rough relevance (mean hidden state dot product)
            mean_query = hidden.mean(dim=(0, 1))  # [H]
            rough_query = self.query_proj(mean_query).view(self.num_heads, self.belief_dim)
            rough_query = rough_query.mean(dim=0)  # [D]

            # Get belief angles for rough matching
            active_radii = active_beliefs.norm(dim=-1).clamp(min=EPSILON)
            active_angles = active_beliefs / active_radii.unsqueeze(-1)
            rough_scores = active_angles @ rough_query  # [N_active]

            _, topk_local = rough_scores.topk(self.top_k)
            active_indices = active_indices[topk_local]
            active_beliefs = active_beliefs[topk_local]
            N_active = self.top_k

        # Compute keys from belief angles, values from full belief vectors
        active_radii = active_beliefs.norm(dim=-1).clamp(min=EPSILON)
        active_angles = active_beliefs / active_radii.unsqueeze(-1)  # [N_active, D]

        keys = active_angles       # [N_active, D] — unit vectors (content)
        values = active_beliefs    # [N_active, D] — full vectors (precision-weighted)

        # Project hidden → query in belief space [B, T, num_heads * D]
        queries = self.query_proj(hidden)
        queries = queries.view(B, T, self.num_heads, self.belief_dim)  # [B, T, H_n, D]

        # Temperature per head
        temperature = self.log_temperature.exp().clamp(min=0.1)  # [num_heads]

        # Attention scores: [B, T, num_heads, N_active]
        # query @ keys.T / temperature
        scores = torch.einsum('bthd,nd->bthn', queries, keys)  # [B, T, H_n, N_active]
        scores = scores / (self.belief_dim ** 0.5)
        scores = scores * temperature.view(1, 1, self.num_heads, 1)

        # Goal modulation: bias scores toward goal-relevant beliefs
        if goal_embeddings is not None and len(goal_embeddings) > 0:
            goal_bias = self._compute_goal_bias(
                goal_embeddings, goal_priorities, active_angles
            )  # [num_heads, N_active]
            scores = scores + goal_bias.unsqueeze(0).unsqueeze(0)  # broadcast over B, T

        # Softmax attention
        attn = F.softmax(scores, dim=-1)  # [B, T, H_n, N_active]

        # Retrieve: weighted sum of values
        # values: [N_active, D] → retrieved: [B, T, H_n, D]
        retrieved = torch.einsum('bthn,nd->bthd', attn, values)

        # Concatenate heads and project to hidden space
        retrieved = retrieved.reshape(B, T, self.num_heads * self.belief_dim)
        output = self.output_proj(retrieved)  # [B, T, hidden_dim]

        return output

    def _compute_goal_bias(
        self,
        goal_embeddings: Tensor,
        goal_priorities: Tensor | None,
        belief_angles: Tensor,
    ) -> Tensor:
        """Compute goal-directed attention bias.

        Beliefs aligned with active goals get boosted.

        Args:
            goal_embeddings: [N_goals, D]
            goal_priorities: [N_goals] or None
            belief_angles: [N_active, D]

        Returns:
            [num_heads, N_active] attention bias
        """
        # Goal angles
        goal_radii = goal_embeddings.norm(dim=-1).clamp(min=EPSILON)
        goal_angles = goal_embeddings / goal_radii.unsqueeze(-1)

        # Similarity between each belief and each goal
        sim = belief_angles @ goal_angles.T  # [N_active, N_goals]

        # Weight by priority
        if goal_priorities is not None:
            sim = sim * goal_priorities.unsqueeze(0)  # broadcast priorities

        # Max similarity across goals (each belief gets its best goal match)
        max_goal_sim = sim.max(dim=-1).values  # [N_active]

        # Project through goal gate to get per-head bias
        # Use max_goal_sim as a scalar bias expanded per head
        bias = max_goal_sim.unsqueeze(0).expand(self.num_heads, -1)  # [num_heads, N_active]

        return bias
