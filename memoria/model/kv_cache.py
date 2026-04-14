"""Incremental KV/state cache for speculative decoding inference.

Manages per-layer persistent state across spec_generate rounds to avoid
reprocessing the full sequence each round. Three cache types:

1. RecurrentCache (H/D layers): stores DeltaProduct recurrent state
   [B, H, K, V]. For LogLinear H layers, also stores the Fenwick tree
   (level states + active mask + chunk count). FLA's chunk_gated_delta_product
   accepts initial_state and returns final_state natively.

2. AttentionCache (L layers): stores assembled K and V tensors for MLA
   or CausalSelfAttention. New tokens' KV is appended each round.
   For MLA, K = cat(k_rope, k_nope), V = v_up(latent) — fully assembled,
   no recomputation needed from cache.

3. NoOpCache (M/S layers): Mamba and SlidingWindow layers. Mamba is
   recurrent (handled by FLA). SlidingWindow manages its own internal
   cache within forward(). No cross-round caching needed.

The cache supports speculative verify + selective commit for DDTree:
- checkpoint(): save all recurrent states (snapshot before tree verify)
- restore(): rollback to snapshot (discard speculative state)
- commit_attention(keep_indices): compact attention caches to accepted path

Reference: DDTree (Ringel & Romano — liranringel.github.io/ddtree/)
Reference: FLA initial_state/output_final_state — stateful recurrence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

import torch
from torch import Tensor

from .fenwick_state import FenwickStateTree


@dataclass
class RecurrentCache:
    """Cache for DeltaProduct / LogLinearDeltaProduct recurrent layers.

    Stores the recurrent state matrix [B, H, K, V] that the FLA kernel
    accepts as initial_state. For LogLinear layers, also stores the
    full Fenwick tree state for correct hierarchical initialization.

    Attributes:
        state: [B, H, K, V] float32 recurrent state matrix.
                None until first forward pass populates it.
        fenwick_states: [num_levels, B, H, K, V] Fenwick level states.
                        None for plain DeltaProduct (no hierarchy).
        fenwick_active: [num_levels] bool mask of active levels.
        fenwick_chunk_count: number of chunks processed so far.
        is_loglinear: whether this is a LogLinear layer (has Fenwick tree).
    """
    state: Tensor | None = None
    fenwick_states: Tensor | None = None
    fenwick_active: Tensor | None = None
    fenwick_chunk_count: int = 0
    is_loglinear: bool = False

    def save_fenwick(self, fenwick: FenwickStateTree) -> None:
        """Save Fenwick tree state from a LogLinear layer."""
        self.fenwick_states = fenwick.states.clone()
        self.fenwick_active = fenwick.active.clone()
        self.fenwick_chunk_count = fenwick._chunk_count
        self.is_loglinear = True

    def restore_fenwick(self, fenwick: FenwickStateTree) -> None:
        """Restore Fenwick tree state into a LogLinear layer."""
        if self.fenwick_states is not None:
            fenwick.states.copy_(self.fenwick_states)
            fenwick.active.copy_(self.fenwick_active)
            fenwick._chunk_count = self.fenwick_chunk_count


@dataclass
class AttentionCache:
    """Cache for MLA / CausalSelfAttention layers.

    Stores assembled K and V tensors. New tokens' KV is appended each round.
    For MLA, K already includes both rope and nope components — no latent
    recomputation needed.

    Shape: K, V are [B, T_cached, n_kv_head, head_dim] in BHSD layout
    (B=batch, H=heads, S=sequence, D=head_dim — transposed to BHSD before
    attention, but stored as BSHD for efficient append along S dim).

    Attributes:
        k: [B, T_cached, n_kv_head, head_dim] cached keys.
        v: [B, T_cached, n_kv_head, head_dim] cached values.
        seq_len: number of committed tokens in cache.
    """
    k: Tensor | None = None
    v: Tensor | None = None
    seq_len: int = 0

    def append(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Append new K/V and return full (cached + new) tensors.

        Does NOT commit to persistent cache — call commit() after
        tree walk to keep only accepted entries.

        Args:
            k_new: [B, T_new, n_kv_head, head_dim] new keys.
            v_new: [B, T_new, n_kv_head, head_dim] new values.

        Returns:
            (k_full, v_full): concatenation of cached + new.
        """
        if self.k is None:
            return k_new, v_new
        k_full = torch.cat([self.k, k_new], dim=1)
        v_full = torch.cat([self.v, v_new], dim=1)
        return k_full, v_full

    def commit(self, k_full: Tensor, v_full: Tensor, keep_new: int) -> None:
        """Commit new entries to persistent cache.

        Args:
            k_full: [B, T_old + T_new, H, D] full K tensor.
            v_full: [B, T_old + T_new, H, D] full V tensor.
            keep_new: number of new entries to keep (len(accepted_path)).
                      Entries are taken from the END of the full tensor.
        """
        # Keep all old entries + keep_new new entries
        total_keep = self.seq_len + keep_new
        self.k = k_full[:, :total_keep].contiguous()
        self.v = v_full[:, :total_keep].contiguous()
        self.seq_len = total_keep

    def commit_indices(self, k_new: Tensor, v_new: Tensor,
                       keep_indices: list[int]) -> None:
        """Commit only specific new entries (for DDTree accepted path).

        Args:
            k_new: [B, T_tree, H, D] new keys from tree verify.
            v_new: [B, T_tree, H, D] new values from tree verify.
            keep_indices: indices into the tree to keep (accepted path).
        """
        if len(keep_indices) == 0:
            return
        idx = torch.tensor(keep_indices, dtype=torch.long, device=k_new.device)
        k_accepted = k_new.index_select(1, idx)
        v_accepted = v_new.index_select(1, idx)
        if self.k is None:
            self.k = k_accepted.contiguous()
            self.v = v_accepted.contiguous()
        else:
            self.k = torch.cat([self.k, k_accepted], dim=1).contiguous()
            self.v = torch.cat([self.v, v_accepted], dim=1).contiguous()
        self.seq_len += len(keep_indices)


class IncrementalState:
    """Full model incremental state for spec_generate.

    Orchestrates per-layer caches for the hybrid HHHHL architecture.
    Supports speculative verify (checkpoint/restore) and selective
    commit (DDTree tree walk).

    Usage:
        state = IncrementalState(model)

        # Prefill: run full forward, populate caches
        state.init_from_prefill(model, input_ids)

        # Each generation round:
        state.checkpoint()           # snapshot recurrent states
        logits = model.incremental_forward(tree_tokens, state)  # speculative
        accepted = walk_tree(logits)

        state.restore()              # rollback recurrent states
        model.incremental_forward(accepted_tokens, state, commit=True)  # commit
    """

    def __init__(self):
        self.layer_caches: dict[int, RecurrentCache | AttentionCache] = {}
        self.seq_len: int = 0
        self._checkpoint: dict[int, RecurrentCache] | None = None

    def register_recurrent(self, layer_idx: int, is_loglinear: bool = False) -> None:
        """Register a recurrent layer (D or H type)."""
        cache = RecurrentCache(is_loglinear=is_loglinear)
        self.layer_caches[layer_idx] = cache

    def register_attention(self, layer_idx: int) -> None:
        """Register an attention layer (L type)."""
        self.layer_caches[layer_idx] = AttentionCache()

    def get_recurrent(self, layer_idx: int) -> RecurrentCache | None:
        """Get recurrent cache for a layer, or None if not registered."""
        cache = self.layer_caches.get(layer_idx)
        return cache if isinstance(cache, RecurrentCache) else None

    def get_attention(self, layer_idx: int) -> AttentionCache | None:
        """Get attention cache for a layer, or None if not registered."""
        cache = self.layer_caches.get(layer_idx)
        return cache if isinstance(cache, AttentionCache) else None

    def checkpoint(self) -> None:
        """Save snapshot of all recurrent states (before speculative verify).

        Only recurrent states need checkpointing — attention caches are
        append-only and can be compacted after the fact via commit_indices.
        Recurrent states are irreversibly modified by processing tree tokens,
        so we need a snapshot to rollback to.
        """
        self._checkpoint = {}
        for idx, cache in self.layer_caches.items():
            if isinstance(cache, RecurrentCache):
                snap = RecurrentCache(is_loglinear=cache.is_loglinear)
                if cache.state is not None:
                    snap.state = cache.state.clone()
                if cache.fenwick_states is not None:
                    snap.fenwick_states = cache.fenwick_states.clone()
                    snap.fenwick_active = cache.fenwick_active.clone()
                    snap.fenwick_chunk_count = cache.fenwick_chunk_count
                self._checkpoint[idx] = snap

    def restore(self) -> None:
        """Restore recurrent states from checkpoint (after speculative verify).

        Call this BEFORE the commit pass that replays accepted tokens.
        """
        if self._checkpoint is None:
            raise RuntimeError("No checkpoint to restore — call checkpoint() first")
        for idx, snap in self._checkpoint.items():
            cache = self.layer_caches[idx]
            assert isinstance(cache, RecurrentCache)
            if snap.state is not None:
                cache.state = snap.state.clone()
            else:
                cache.state = None
            if snap.fenwick_states is not None:
                cache.fenwick_states = snap.fenwick_states.clone()
                cache.fenwick_active = snap.fenwick_active.clone()
                cache.fenwick_chunk_count = snap.fenwick_chunk_count
        self._checkpoint = None

    @staticmethod
    def build_for_model(model) -> 'IncrementalState':
        """Build IncrementalState with correct cache types for each layer.

        Inspects the model's block types and registers the appropriate
        cache for each layer:
        - H (LogLinearDeltaProduct) → RecurrentCache(is_loglinear=True)
        - D (DeltaProduct) → RecurrentCache(is_loglinear=False)
        - L (MLA/CausalSelfAttention) → AttentionCache
        - M (Mamba), S (SlidingWindow) → no cache (internal state)
        """
        from .deltaproduct_layers import (
            DeltaProductBlock,
            LogLinearDeltaProductBlock,
        )
        from .transformer import (
            CausalSelfAttention,
            MLACausalSelfAttention,
        )

        state = IncrementalState()
        for i, block in enumerate(model.transformer.blocks):
            attn = block.attn
            if isinstance(attn, LogLinearDeltaProductBlock):
                state.register_recurrent(i, is_loglinear=True)
            elif isinstance(attn, DeltaProductBlock):
                state.register_recurrent(i, is_loglinear=False)
            elif isinstance(attn, (MLACausalSelfAttention, CausalSelfAttention)):
                state.register_attention(i)
            # Mamba, SlidingWindow: no cross-round cache needed
        return state
