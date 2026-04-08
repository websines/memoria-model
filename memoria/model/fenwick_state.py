"""Fenwick tree state manager for Log-Linear attention.

Manages O(log T) hierarchical hidden states for the Log-Linear DeltaProduct
layer. Each level covers an exponentially larger span of history:
  Level 0: current chunk (size 1 chunk)
  Level 1: last 1 chunk
  Level 2: last 2 chunks
  Level 3: last 4 chunks
  ...
  Level L: last 2^(L-1) chunks

The Fenwick (binary indexed tree) structure governs when states merge and
promote: at chunk index t, levels 0..lssb(t) merge into level lssb(t)+1,
where lssb(t) is the least significant set bit of t.

Reference: Log-Linear Attention (ICLR 2026, arXiv:2506.04761)
Reference: Fenwick tree (P. Fenwick, "A New Data Structure for Cumulative
           Frequency Tables", Software: Practice and Experience, 1994)
"""

import math
import torch
from torch import Tensor


# Maximum number of Fenwick tree levels (covers up to 2^14 = 16384 chunks;
# at chunk_size=64 that's ~1M tokens).
MAX_NUM_LEVELS = 15


def _lssb(t: int) -> int:
    """Least significant set bit position (0-indexed).

    lssb(1)=0, lssb(2)=1, lssb(3)=0, lssb(4)=2, lssb(6)=1, ...
    For t=0, returns 0 by convention (level 0 → level 1 merge).
    """
    if t == 0:
        return 0
    return (t & -t).bit_length() - 1


class FenwickStateTree:
    """Manages hierarchical hidden states for Log-Linear DeltaProduct.

    Usage:
        tree = FenwickStateTree(num_heads, head_k_dim, head_v_dim, device, dtype)

        for chunk_idx in range(num_chunks):
            # Get weighted initial state for this chunk
            init_state = tree.query(level_scales_chunk)

            # ... run DeltaProduct kernel on chunk with init_state ...

            # Update tree with the chunk's final state
            tree.update(chunk_idx, final_state, level_scales_chunk)

    The tree holds up to MAX_NUM_LEVELS state matrices per head.
    Each state is [num_heads, head_k_dim, head_v_dim] (the KV memory matrix).
    """

    def __init__(
        self,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 1,
    ):
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_levels = MAX_NUM_LEVELS
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        # State buffer: [num_levels, batch_size, num_heads, head_k_dim, head_v_dim]
        self.states = torch.zeros(
            self.num_levels, batch_size, num_heads, head_k_dim, head_v_dim,
            device=device, dtype=dtype,
        )
        # Track which levels have been written to (for correct merging)
        self.active = torch.zeros(self.num_levels, dtype=torch.bool, device=device)
        self._chunk_count = 0

    def reset(self):
        """Reset all states to zero. Call at the start of each sequence."""
        self.states.zero_()
        self.active.zero_()
        self._chunk_count = 0

    def query(
        self,
        level_scales: Tensor,
    ) -> Tensor:
        """Get weighted initial state for the next chunk.

        Args:
            level_scales: [batch_size, num_heads, num_levels] — data-dependent
                per-level weights (from softplus(L * l_proj(hidden))).

        Returns:
            init_state: [batch_size, num_heads, head_k_dim, head_v_dim] —
                weighted sum of active level states.
        """
        # level_scales: [B, H, L] → [L, B, H, 1, 1] for broadcasting
        weights = level_scales.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
        # states: [L, B, H, K, V] — detached from graph (values only,
        # gradients flow through weights/level_scales, not stored states)
        mask = self.active.view(self.num_levels, 1, 1, 1, 1).float()
        weighted = self.states.detach() * weights * mask
        init_state = weighted.sum(dim=0)  # [B, H, K, V]
        return init_state

    def update(
        self,
        chunk_idx: int,
        chunk_state: Tensor,
        level_scales: Tensor,
    ):
        """Update the Fenwick tree after processing a chunk.

        Follows the Fenwick tree update rule:
        1. Level 0 gets the new chunk_state
        2. Levels 1..lssb(chunk_idx) get merged into level lssb(chunk_idx)+1
        3. Levels 1..lssb(chunk_idx) are cleared

        Args:
            chunk_idx: 0-based index of the chunk just processed
            chunk_state: [batch_size, num_heads, head_k_dim, head_v_dim] —
                the DeltaProduct recurrent state after this chunk
            level_scales: [batch_size, num_heads, num_levels] — data-dependent
                level weights for this chunk (used for weighted merging)
        """
        # Detach states from computation graph — the kernel handles per-chunk
        # gradients internally. Fenwick states carry values for initial_state
        # seeding, not gradient paths across chunks. Level scale gradients
        # still flow through query() because level_scales retains grad_fn.
        chunk_state = chunk_state.detach()

        # Level 0 always gets the latest chunk state
        self.states[0] = chunk_state
        self.active[0] = True

        # Fenwick merge: determine how many levels to merge
        merge_level = _lssb(chunk_idx + 1)  # +1 because Fenwick is 1-indexed
        merge_level = min(merge_level, self.num_levels - 2)  # don't exceed buffer

        if merge_level > 0:
            # Merge levels 0..merge_level into level merge_level
            target_level = merge_level
            merged = torch.zeros_like(self.states[0])

            for l in range(merge_level):
                if self.active[l]:
                    merged = merged + self.states[l]

            self.states[target_level] = merged
            self.active[target_level] = True

            # Clear merged lower levels (except level 0 which always gets fresh state)
            for l in range(1, merge_level):
                self.states[l] = torch.zeros_like(self.states[l])
                self.active[l] = False

        self._chunk_count += 1

    def get_all_states(self) -> tuple[Tensor, Tensor]:
        """Return all level states and active mask.

        Returns:
            states: [num_levels, batch_size, num_heads, head_k_dim, head_v_dim]
            active: [num_levels] boolean mask
        """
        return self.states, self.active
