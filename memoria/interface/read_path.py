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


class BeliefCache:
    """Pre-computed active belief data, shared across all interface layers per forward pass.

    Eliminates redundant get_active_mask() + indexing that each interface layer
    would otherwise compute independently. Also enables future DRAM offloading:
    since beliefs don't change during the forward pass, this cache can be
    pre-fetched from CPU RAM during early transformer layers.

    Reference: Engram (DeepSeek, arXiv:2601.07372) — hash addresses computed
    ahead of time enable async prefetch from host memory.
    """
    __slots__ = ('indices', 'beliefs', 'angles', 'radii', 'n_active')

    def __init__(
        self,
        indices: Tensor,    # [N_active] global indices of active beliefs
        beliefs: Tensor,    # [N_active, D] full cartesian vectors
        angles: Tensor,     # [N_active, D] unit vectors (content direction)
        radii: Tensor,      # [N_active] precision (magnitude)
    ):
        self.indices = indices
        self.beliefs = beliefs
        self.angles = angles
        self.radii = radii
        self.n_active = indices.shape[0]

    @staticmethod
    @torch.no_grad()
    def from_state(state: 'CognitiveState') -> 'BeliefCache':
        """Snapshot active beliefs from cognitive state.

        Called once per forward pass before the block loop.
        """
        active_mask = state.get_active_mask()
        if not active_mask.any():
            device = state.beliefs.device
            D = state.beliefs.shape[-1]
            return BeliefCache(
                indices=torch.empty(0, dtype=torch.long, device=device),
                beliefs=torch.empty(0, D, device=device),
                angles=torch.empty(0, D, device=device),
                radii=torch.empty(0, device=device),
            )
        indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        beliefs = state.beliefs[indices].clone()
        radii = beliefs.norm(dim=-1).clamp(min=EPSILON)
        angles = beliefs / radii.unsqueeze(-1)
        return BeliefCache(indices=indices, beliefs=beliefs, angles=angles, radii=radii)


class ReadPath(nn.Module):
    """Content-addressable retrieval from belief region.

    Projects transformer hidden states into belief space, retrieves relevant
    beliefs via Hopfield attention, optionally modulated by active goals,
    and projects back to hidden space.
    """

    def __init__(self, hidden_dim: int, belief_dim: int, num_heads: int = 4, top_k: int = 32,
                 read_gate_init_bias: float = 2.0):
        """
        Args:
            hidden_dim: transformer hidden dimension
            belief_dim: belief representation dimension (D)
            num_heads: number of retrieval heads (parallel queries)
            top_k: max beliefs to attend over (sparse, for efficiency)
            read_gate_init_bias: initial bias for read gate sigmoid (higher = more open)
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

        # Utility head: predicts next-token logits from retrieved beliefs alone.
        # Provides gradient signal for "are the retrieved beliefs useful for prediction?"
        # This indirectly trains the write path to store useful information.
        self.utility_head = nn.Linear(belief_dim * num_heads, hidden_dim, bias=False)

        # ── Read Gate (Engram-inspired context-aware gating) ──
        # Per-position scalar gate: "does this position even need beliefs?"
        # Positions where the transformer is already confident skip retrieval.
        # Starts mostly open (positive bias → ~88% sigmoid output), learns to close.
        # Reference: Engram's context-aware sigmoid gate with sqrt-sign activation.
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        nn.init.constant_(self.read_gate[-1].bias, read_gate_init_bias)

        # ── Local Convolution on Retrieved Beliefs (Engram-inspired ShortConv) ──
        # After Hopfield retrieval, a depthwise 1D conv lets adjacent positions'
        # belief retrievals interact. Without this, each position's retrieval is
        # completely independent — the conv adds local coherence.
        # Zero-init so it starts as a no-op and learns to mix.
        # Reference: Engram's ShortConv (depthwise dilated conv + SiLU).
        retrieval_dim = belief_dim * num_heads
        self.post_conv = nn.Conv1d(
            retrieval_dim, retrieval_dim,
            kernel_size=4, groups=retrieval_dim,  # depthwise
            padding=3, bias=False,                # causal padding
        )
        nn.init.zeros_(self.post_conv.weight)
        self.post_conv_act = nn.SiLU()

        # Initialize output projection to zero (residual-friendly, from nanogpt speedrun)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.utility_head.weight)

    def forward(
        self,
        hidden: Tensor,
        state: CognitiveState,
        goal_embeddings: Tensor | None = None,
        goal_priorities: Tensor | None = None,
        current_step: int = -1,
        belief_cache: BeliefCache | None = None,
        depth_bias: float = 0.0,
    ) -> Tensor:
        """Retrieve relevant beliefs and project into hidden space.

        Args:
            hidden: [B, T, hidden_dim] transformer hidden states
            state: the cognitive state to read from
            goal_embeddings: [N_goals, D] active goal embeddings (optional)
            goal_priorities: [N_goals] goal priorities for weighting (optional)
            belief_cache: pre-computed active belief data (avoids redundant computation)
            depth_bias: per-layer temperature bias for depth-conditioned retrieval

        Returns:
            [B, T, hidden_dim] retrieved belief information to add to residual stream
        """
        B, T, H = hidden.shape
        device = hidden.device

        # Use pre-computed cache if available, otherwise compute fresh
        if belief_cache is not None:
            active_indices = belief_cache.indices
            active_beliefs = belief_cache.beliefs
            active_angles = belief_cache.angles
            active_radii = belief_cache.radii
            N_active = belief_cache.n_active
        else:
            active_mask = state.get_active_mask()
            if not active_mask.any():
                zero_attn = torch.zeros(B, T, self.num_heads, 0, device=device)
                zero_retrieved = torch.zeros(B, T, self.num_heads, self.belief_dim, device=device)
                return torch.zeros_like(hidden), torch.zeros_like(hidden), [], zero_attn, zero_retrieved
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
            active_beliefs = state.beliefs[active_indices]
            active_radii = active_beliefs.norm(dim=-1).clamp(min=EPSILON)
            active_angles = active_beliefs / active_radii.unsqueeze(-1)
            N_active = active_beliefs.shape[0]

        if N_active == 0:
            zero_attn = torch.zeros(B, T, self.num_heads, 0, device=device)
            zero_retrieved = torch.zeros(B, T, self.num_heads, self.belief_dim, device=device)
            return torch.zeros_like(hidden), torch.zeros_like(hidden), [], zero_attn, zero_retrieved

        # Limit to top_k for efficiency if many active beliefs
        if N_active > self.top_k:
            mean_query = hidden.mean(dim=(0, 1))
            rough_query = self.query_proj(mean_query).view(self.num_heads, self.belief_dim)
            rough_query = rough_query.mean(dim=0)
            rough_scores = active_angles @ rough_query

            _, topk_local = rough_scores.topk(self.top_k)
            active_indices = active_indices[topk_local]
            active_beliefs = active_beliefs[topk_local]
            active_angles = active_angles[topk_local]
            active_radii = active_radii[topk_local]
            N_active = self.top_k

        keys = active_angles       # [N_active, D] — unit vectors (content)
        values = active_beliefs    # [N_active, D] — full vectors (precision-weighted)

        # ── Read Gate (compute FIRST, skip attention for closed positions) ──
        # At inference: positions with gate < threshold skip the expensive
        # Hopfield attention entirely. At training: soft gate for gradient flow.
        gate_raw = self.read_gate(hidden)  # [B, T, 1]
        gate_raw = gate_raw.abs().clamp_min(1e-6).sqrt() * gate_raw.sign()
        gate = torch.sigmoid(gate_raw)     # [B, T, 1]

        # At inference, identify which positions need retrieval
        if not self.training:
            open_mask = (gate.squeeze(-1) > 0.1)  # [B, T] — positions worth retrieving
            n_open = open_mask.sum().item()
            if n_open == 0:
                # All positions closed — skip everything
                zero_attn = torch.zeros(B, T, self.num_heads, N_active, device=device)
                zero_retrieved = torch.zeros(B, T, self.num_heads, self.belief_dim, device=device)
                return torch.zeros_like(hidden), torch.zeros_like(hidden), [], zero_attn, zero_retrieved

        # Project hidden → query in belief space [B, T, num_heads * D]
        queries = self.query_proj(hidden)
        queries = queries.view(B, T, self.num_heads, self.belief_dim)  # [B, T, H_n, D]

        # Temperature per head (with depth-conditioned bias)
        temperature = (self.log_temperature + depth_bias).exp().clamp(min=0.1)  # [num_heads]

        # Attention scores: [B, T, num_heads, N_active]
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

        # Track which beliefs were attended to (recency update)
        with torch.no_grad():
            mean_attn = attn.mean(dim=(0, 1, 2))  # [N_active]
            attended = (mean_attn > 1.0 / max(N_active, 1)).nonzero(as_tuple=False).squeeze(-1)
            if len(attended) > 0:
                state.touch_beliefs(active_indices[attended], max(current_step, 0))
            read_belief_indices = active_indices[attended].tolist() if len(attended) > 0 else []

        # Retrieve: weighted sum of values
        retrieved_per_head = torch.einsum('bthn,nd->bthd', attn, values)

        # Concatenate heads
        retrieved_flat = retrieved_per_head.reshape(B, T, self.num_heads * self.belief_dim)

        # ── Local Convolution on Retrieved Beliefs ──
        # Depthwise 1D conv: adjacent positions' retrievals interact.
        # Causal: padding=3 + truncate to T ensures no future leakage.
        conv_out = self.post_conv(retrieved_flat.transpose(1, 2))  # [B, C, T+pad]
        conv_out = conv_out[:, :, :T].transpose(1, 2)             # [B, T, C]
        retrieved_flat = retrieved_flat + self.post_conv_act(conv_out)

        # Project to hidden space and apply gate
        output = gate * self.output_proj(retrieved_flat)  # [B, T, hidden_dim]

        # Utility prediction: can the retrieved beliefs predict the next token?
        utility_logits = self.utility_head(retrieved_flat)  # [B, T, hidden_dim]

        return output, utility_logits, read_belief_indices, attn, retrieved_per_head

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
        # goal_gate: [D] → [num_heads], applied as a learned scaling per head
        gate_weights = torch.sigmoid(self.goal_gate(goal_angles.mean(dim=0)))  # [num_heads]
        bias = gate_weights.unsqueeze(-1) * max_goal_sim.unsqueeze(0)  # [num_heads, N_active]

        return bias
