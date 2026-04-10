"""State Interface Layer: the bridge between transformer and cognitive state.

Inserted every K transformer blocks. Bidirectional:
- Read path: beliefs → hidden stream (what I know shapes how I think)
- Write path: hidden stream → write candidates (new observations buffered for pass 2)

This layer participates in the forward pass. Gradients from L_token flow through
the read path projections and write path projections (shaping HOW the model reads/writes).
The cognitive state content itself is updated in pass 2 (not by L_token gradients).

Reference: Griffin (google-deepmind/recurrentgemma) — hybrid architecture pattern
Reference: TTT layers (test-time-training/ttt-lm-pytorch) — mutable state in forward pass
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..core.state import CognitiveState
from .read_path import ReadPath, BeliefCache
from .write_path import WritePath, WriteCandidate


class StateInterfaceLayer(nn.Module):
    """Combined read + write interface between transformer and cognitive state.

    In the forward pass:
    1. Read path retrieves relevant beliefs → adds to residual stream
    2. Write path projects hidden states → buffers write candidates

    Write candidates are returned (not committed) for pass 2 to process.
    """

    def __init__(
        self,
        hidden_dim: int,
        belief_dim: int,
        num_heads: int = 4,
        top_k: int = 32,
        layer_idx: int = 0,
        n_interfaces: int = 1,
        read_gate_init_bias: float = 2.0,
        parallel_goals: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_interfaces = n_interfaces

        self.read_path = ReadPath(
            hidden_dim=hidden_dim,
            belief_dim=belief_dim,
            num_heads=num_heads,
            top_k=top_k,
            read_gate_init_bias=read_gate_init_bias,
            parallel_goals=parallel_goals,
        )
        self.write_path = WritePath(
            hidden_dim=hidden_dim,
            belief_dim=belief_dim,
        )

        # Pre-norm before read (standard transformer convention)
        self.norm = nn.RMSNorm(hidden_dim)

        # Depth-conditioned retrieval: per-layer learnable temperature bias.
        # Early interfaces retrieve broadly (high temp), later ones focus (low temp).
        # Reference: Mamba HaltingHead's loop position scalar — depth encoding
        # improves retrieval quality by letting the model learn depth-specific behavior.
        self.depth_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden: Tensor,
        state: CognitiveState,
        current_step: int = -1,
        belief_cache: BeliefCache | None = None,
    ) -> tuple[Tensor, list[WriteCandidate], Tensor, list[int], Tensor, Tensor, Tensor]:
        """Forward pass through the state interface.

        Args:
            hidden: [B, T, hidden_dim] from previous transformer block
            state: cognitive state to read from / write to
            current_step: for recency tracking (-1 to skip)
            belief_cache: pre-computed active belief data (shared across all interfaces)

        Returns:
            hidden: [B, T, hidden_dim] with belief information added
            candidates: list of WriteCandidate for pass 2
            utility_logits: [B, T, hidden_dim] from utility head (for aux loss)
            read_indices: list of belief indices actually retrieved
            attn_weights: [B, T, num_heads, N_active] attention weights (for L_fe)
            retrieved: [B, T, num_heads, D] retrieved beliefs per head (for L_fe)
            obs_vectors: [B, T, D] observation projections (for L_fe)
        """
        normed = self.norm(hidden)

        # Get active goals for read path modulation
        goal_indices, goal_embeddings, goal_metadata = state.get_active_goals()
        goal_priorities = goal_metadata[:, 0] if len(goal_indices) > 0 else None

        # Read: retrieve beliefs, add to residual stream
        belief_info, utility_logits, read_indices, attn_weights, retrieved = self.read_path(
            normed, state,
            goal_embeddings=goal_embeddings if len(goal_indices) > 0 else None,
            goal_priorities=goal_priorities,
            current_step=current_step,
            belief_cache=belief_cache,
            depth_bias=self.depth_bias,
        )
        hidden = hidden + belief_info  # residual connection

        # Write: project hidden states, match against beliefs, buffer candidates
        candidates, obs_vectors = self.write_path(
            normed, state, layer_idx=self.layer_idx, belief_cache=belief_cache,
        )

        return hidden, candidates, utility_logits, read_indices, attn_weights, retrieved, obs_vectors
