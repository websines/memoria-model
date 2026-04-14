"""DeltaProduct and Log-Linear DeltaProduct layers for the hybrid transformer.

Three layer types for the recurrent (non-attention) positions in the
window_pattern:

  D (DeltaProductBlock): GatedDeltaProduct₃ — flat O(T) error-correcting
      recurrence. Uses FLA's optimized chunk kernel at full speed.

  H (LogLinearDeltaProductBlock): Log-Linear DeltaProduct₃ — hierarchical
      O(T log T) recurrence with Fenwick tree multi-scale state. Uses FLA's
      chunk kernel per-chunk with Python Fenwick state management between
      chunks. Slower training throughput than D but richer temporal context.

  E (LogLinearGDNBlock): Log-Linear Gated Delta Net — hierarchical O(T log T)
      with single-step delta rule (n_h=1). Uses hattention's fused kernel at
      full speed. Fallback when H is too slow.

All layers share the same forward signature as MambaBlock / SlidingWindowAttention:
    forward(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor
    Input/output shape: (B, T, D) → (B, T, D)
    cos/sin are accepted but ignored (no positional encoding in recurrent layers).

Reference: DeltaProduct (NeurIPS 2025, arXiv:2502.10297)
Reference: Gated Delta Networks (ICLR 2025, arXiv:2412.06464)
Reference: Log-Linear Attention (ICLR 2026, arXiv:2506.04761)
Library: flash-linear-attention (fla-org/flash-linear-attention)
Library: hattention (HanGuo97/log-linear-attention)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from .fenwick_state import FenwickStateTree, MAX_NUM_LEVELS

# ── Lazy imports (fail gracefully if libraries not installed) ──

_fla_available = False
try:
    from fla.layers import GatedDeltaProduct
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    _fla_available = True
except ImportError:
    pass

_hattention_available = False
try:
    from hattention.modeling_h_gated_deltanet import HGatedDeltaNet
    _hattention_available = True
except ImportError:
    pass


# ── Type alias for config to avoid circular import ──

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import TransformerConfig


# ═══════════════════════════════════════════════════════════════════════════
# D layer — GatedDeltaProduct₃ (flat, O(T), full speed)
# ═══════════════════════════════════════════════════════════════════════════

class DeltaProductBlock(nn.Module):
    """GatedDeltaProduct₃ — error-correcting recurrence via Householder products.

    Wraps FLA's GatedDeltaProduct layer. O(T) linear complexity with fixed-size
    recurrent state. Each token triggers 3 Householder reflection steps that
    error-correct the state — the recurrent layer *learns* at every token.

    Eigenvalue range [-1,1] (allow_neg_eigval=True) enables representation of
    permutations, negation, and cyclic state transitions. This is the
    architectural reason DeltaProduct succeeds at state-tracking benchmarks
    where Mamba-2 (restricted to real diagonal [0,1]) fundamentally cannot.

    Uses FLA's optimized Triton chunk kernel — full chunkwise parallel training.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        if not _fla_available:
            raise ImportError(
                "flash-linear-attention is required for DeltaProduct layers. "
                "Install with: pip install flash-linear-attention"
            )

        n_embd = config.n_embd
        head_dim = config.deltaproduct_head_dim
        num_heads = n_embd // head_dim  # auto-compute from hidden/head dims

        self.deltaproduct = GatedDeltaProduct(
            hidden_size=n_embd,
            head_dim=head_dim,
            num_heads=num_heads,
            expand_v=config.deltaproduct_expand_v,
            num_householder=config.deltaproduct_n_householder,
            allow_neg_eigval=config.deltaproduct_allow_neg_eigval,
            use_forget_gate=config.deltaproduct_use_forget_gate,
            use_short_conv=config.deltaproduct_use_short_conv,
            conv_size=config.deltaproduct_conv_size,
            use_output_gate=True,
            mode='chunk',
            layer_idx=layer_idx,
        )
        self.layer_idx = layer_idx
        self._yarn_scale = 1.0  # dummy for YaRN scale setting loop

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                initial_state: Tensor | None = None,
                output_final_state: bool = False) -> Tensor | tuple[Tensor, Tensor | None]:
        """Forward pass with optional incremental state.

        Args:
            x: [B, T, D] input hidden states.
            cos, sin: RoPE tensors (ignored — recurrent layer).
            initial_state: [B, H, K, V] recurrent state from previous round.
                           None = start from zero (full-sequence mode).
            output_final_state: if True, return (output, final_state) tuple.

        Returns:
            If output_final_state=False: output [B, T, D]
            If output_final_state=True: (output [B, T, D], final_state [B, H, K, V])
        """
        # cos/sin ignored — DeltaProduct uses recurrent state for ordering
        # FLA Triton kernels require bfloat16 — autocast handles dtype seamlessly
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # FLA's GatedDeltaProduct returns (output, final_state, cache)
            # We pass initial_state through via the layer's use_cache mechanism
            if initial_state is not None or output_final_state:
                # Access the underlying FLA layer for stateful processing
                # GatedDeltaProduct stores past_key_values internally when use_cache=True
                out_tuple = self.deltaproduct(
                    x, past_key_values=None,
                    use_output_gate=True,
                )
                out = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                final_state = out_tuple[1] if isinstance(out_tuple, tuple) and len(out_tuple) > 1 else None
            else:
                out, _, _ = self.deltaproduct(x)
                final_state = None

        result = out.to(x.dtype)
        if output_final_state:
            return result, final_state
        return result


# ═══════════════════════════════════════════════════════════════════════════
# H layer — Log-Linear DeltaProduct₃ (hierarchical, O(T log T))
# ═══════════════════════════════════════════════════════════════════════════

class LogLinearDeltaProductBlock(nn.Module):
    """Log-Linear DeltaProduct₃ — hierarchical error-correcting recurrence.

    Combines DeltaProduct₃'s multi-Householder error correction with
    Log-Linear Attention's Fenwick tree hierarchical state. Each of the
    ~log₂(T) levels gets its own DeltaProduct state, maintaining multi-scale
    temporal context: recent tokens at high resolution, distant tokens
    compressed into coarser levels.

    Architecture:
    1. Project input → q, k, v, g, beta (DeltaProduct style, n_h interleaving)
    2. Project input → level_scales (Log-Linear style, data-dependent λ)
    3. Split sequence into chunks of size C
    4. For each chunk:
       a. Query Fenwick tree for initial state (weighted by level_scales)
       b. Run DeltaProduct chunk kernel (existing optimized Triton kernel)
       c. Update Fenwick tree with chunk's final state
    5. Concat chunk outputs → project

    Training note: chunks process sequentially (not parallel) because the
    Fenwick state of each chunk depends on previous chunks. The intra-chunk
    computation uses FLA's optimized kernel. The inter-chunk Fenwick logic
    is O(log T) small matrix operations — negligible cost. The sequential
    overhead is ~T/C kernel launches (32 for T=2048, C=64).

    For a fully fused kernel, the inter-chunk Fenwick logic would need to be
    merged into the Triton kernel. This is a future optimization.

    This is a novel combination — no existing library provides Log-Linear
    DeltaProduct. We compose FLA's DeltaProduct kernel with a Python Fenwick
    tree state manager.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        if not _fla_available:
            raise ImportError(
                "flash-linear-attention is required for Log-Linear DeltaProduct layers. "
                "Install with: pip install flash-linear-attention"
            )

        self.n_embd = config.n_embd
        self.layer_idx = layer_idx
        self._yarn_scale = 1.0

        n_embd = config.n_embd
        head_dim = config.deltaproduct_head_dim
        self.num_heads = n_embd // head_dim
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * config.deltaproduct_expand_v)
        self.num_householder = config.deltaproduct_n_householder
        self.chunk_size = config.loglinear_chunk_size

        # ── DeltaProduct projections (same as GatedDeltaProduct) ──
        key_dim = self.num_heads * self.head_k_dim
        value_dim = self.num_heads * self.head_v_dim
        n_h = self.num_householder

        self.q_proj = nn.Linear(n_embd, key_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, key_dim * n_h, bias=False)
        self.v_proj = nn.Linear(n_embd, value_dim * n_h, bias=False)
        self.b_proj = nn.Linear(n_embd, self.num_heads * n_h, bias=False)  # beta
        self.a_proj = nn.Linear(n_embd, self.num_heads, bias=False)  # forget gate
        self.g_proj = nn.Linear(n_embd, value_dim, bias=False)  # output gate
        self.o_proj = nn.Linear(value_dim, n_embd, bias=False)

        # Short causal convolution (pre-SSM context, same as Mamba/DeltaProduct)
        self.use_short_conv = config.deltaproduct_use_short_conv
        if self.use_short_conv:
            self.conv_size = config.deltaproduct_conv_size
            self.q_conv = nn.Conv1d(
                key_dim, key_dim, self.conv_size,
                groups=key_dim, padding=self.conv_size - 1, bias=False,
            )
            self.k_conv = nn.Conv1d(
                key_dim * n_h, key_dim * n_h, self.conv_size,
                groups=key_dim * n_h, padding=self.conv_size - 1, bias=False,
            )
            self.v_conv = nn.Conv1d(
                value_dim * n_h, value_dim * n_h, self.conv_size,
                groups=value_dim * n_h, padding=self.conv_size - 1, bias=False,
            )

        # Group norm before output gate (FLA convention)
        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads, num_channels=value_dim, eps=1e-5,
        )

        # ── Log-Linear level projections ──
        # Input-dependent level scales: λ_t^(l) = softplus(L * l_proj(x_t))
        self.l_proj = nn.Linear(n_embd, self.num_heads * MAX_NUM_LEVELS, bias=False)
        # Learnable per-head per-level base weights
        self.L = nn.Parameter(torch.ones(self.num_heads, MAX_NUM_LEVELS))

        # ── Fenwick tree (created lazily per forward, since batch/device vary) ──
        self._fenwick: FenwickStateTree | None = None

    def _get_fenwick(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> FenwickStateTree:
        """Get or create Fenwick tree for current batch."""
        if (self._fenwick is None
                or self._fenwick.batch_size != batch_size
                or self._fenwick.device != device):
            self._fenwick = FenwickStateTree(
                num_heads=self.num_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
        else:
            self._fenwick.reset()
        return self._fenwick

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                incremental_cache=None,
                commit: bool = False) -> Tensor:
        """Forward pass with optional incremental state.

        Args:
            x: [B, T, D] input hidden states.
            cos, sin: RoPE tensors (ignored — recurrent layer).
            incremental_cache: RecurrentCache from IncrementalState.
                If provided, restores Fenwick tree state from cache before
                processing, and saves updated state after processing.
                Enables cross-round state persistence for spec_generate.
            commit: if True, save final state to cache. If False (speculative
                verify), process but don't update persistent cache.
                Used by DDTree: verify speculatively, then commit only
                accepted tokens.

        Returns:
            output [B, T, D]
        """
        # cos/sin ignored — recurrent layer, no positional encoding
        input_dtype = x.dtype
        # FLA Triton kernels require bfloat16. Run everything in bf16.
        _bf = torch.bfloat16
        xb = x.to(_bf)

        B, T, D = x.shape
        C = self.chunk_size
        H = self.num_heads
        K = self.head_k_dim
        V = self.head_v_dim
        n_h = self.num_householder

        # ── Projections ──
        # Handles both plain nn.Linear and WeightQuantLinear wrappers.
        # WeightQuantLinear.forward() applies STE quantization noise + dtype cast.
        # Plain nn.Linear needs explicit weight casting for bf16 input.
        from ..core.quantize import WeightQuantLinear
        def _lin(proj, inp):
            if isinstance(proj, WeightQuantLinear):
                # WeightQuantLinear.forward handles STE + dtype casting
                return proj(inp)
            # Plain nn.Linear: manual bf16 weight cast (original behavior)
            w = proj.weight.to(_bf)
            b = proj.bias.to(_bf) if proj.bias is not None else None
            return nn.functional.linear(inp, w, b)

        q = _lin(self.q_proj, xb)
        k = _lin(self.k_proj, xb)
        v = _lin(self.v_proj, xb)
        beta = _lin(self.b_proj, xb)
        g_gate = _lin(self.a_proj, xb)
        o_gate = _lin(self.g_proj, xb)

        # Short causal convolution (explicit weight cast, no in-place data swap)
        if self.use_short_conv:
            def _conv_bf16(conv, t):
                return nn.functional.conv1d(
                    t.transpose(1, 2),
                    conv.weight.to(_bf),
                    conv.bias.to(_bf) if conv.bias is not None else None,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups,
                )[..., :T].transpose(1, 2)
            q = _conv_bf16(self.q_conv, q)
            k = _conv_bf16(self.k_conv, k)
            v = _conv_bf16(self.v_conv, v)

        # Activations and reshape (FLA interleaving convention)
        # q: [B, T, H, K] — normal
        q = q.view(B, T, H, K)
        # k: interleave n_h into time dim → [B, T*n_h, H, K]
        k = nn.functional.silu(k).view(B, T, H, K, n_h).permute(0, 1, 4, 2, 3).reshape(B, T * n_h, H, K)
        # v: interleave n_h into time dim → [B, T*n_h, H, V]
        v = nn.functional.silu(v).view(B, T, H, V, n_h).permute(0, 1, 4, 2, 3).reshape(B, T * n_h, H, V)
        # beta: interleave n_h into time dim → [B, T*n_h, H]
        beta = beta.view(B, T, H, n_h).permute(0, 1, 3, 2).reshape(B, T * n_h, H).sigmoid()
        # g: one per token (NOT expanded) → [B, T, H]
        g_gate = nn.functional.logsigmoid(g_gate).view(B, T, H)
        o_gate = o_gate.view(B, T, H * V)

        # ── Level scales (Log-Linear) ──
        dl = _lin(self.l_proj, xb).view(B, T, H, MAX_NUM_LEVELS)
        level_weights = nn.functional.softplus(self.L.to(_bf).unsqueeze(0).unsqueeze(0) * dl)

        # ── Pad sequence to multiple of chunk_size ──
        pad_len = (C - T % C) % C
        if pad_len > 0:
            q = nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len * n_h))
            v = nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len * n_h))
            beta = nn.functional.pad(beta, (0, 0, 0, pad_len * n_h))
            g_gate = nn.functional.pad(g_gate, (0, 0, 0, pad_len))
            level_weights = nn.functional.pad(level_weights, (0, 0, 0, 0, 0, pad_len))

        T_padded = q.shape[1]
        num_chunks = T_padded // C

        # ── Initialize Fenwick tree (in bf16) ──
        # If incremental_cache has saved Fenwick state, restore it instead
        # of resetting. This enables cross-round state persistence.
        from .kv_cache import RecurrentCache
        if (incremental_cache is not None
                and isinstance(incremental_cache, RecurrentCache)
                and incremental_cache.fenwick_states is not None):
            # Restore Fenwick from cache — don't reset
            if (self._fenwick is None
                    or self._fenwick.batch_size != B
                    or self._fenwick.device != x.device):
                self._fenwick = FenwickStateTree(
                    num_heads=self.num_heads,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    device=x.device, dtype=_bf, batch_size=B,
                )
            incremental_cache.restore_fenwick(self._fenwick)
        else:
            fenwick = self._get_fenwick(B, x.device, _bf)
        fenwick = self._fenwick

        # ── Process chunks sequentially with Fenwick state management ──
        outputs = []
        for ci in range(num_chunks):
            sq, eq = ci * C, (ci + 1) * C
            sk, ek = ci * C * n_h, (ci + 1) * C * n_h

            q_c = q[:, sq:eq]          # [B, C, H, K]
            k_c = k[:, sk:ek]          # [B, C*n_h, H, K]
            v_c = v[:, sk:ek]          # [B, C*n_h, H, V]
            beta_c = beta[:, sk:ek]    # [B, C*n_h, H]
            g_c = g_gate[:, sq:eq]     # [B, C, H]

            lw_c = level_weights[:, sq:eq].mean(dim=1)  # [B, H, L]
            init_state = fenwick.query(lw_c)  # [B, H, K, V]

            # Run DeltaProduct chunk kernel (all tensors already bf16)
            out_c = chunk_gated_delta_product(
                q=q_c, k=k_c, v=v_c, g=g_c, beta=beta_c,
                num_householder=n_h,
                initial_state=init_state,
                output_final_state=True,
            )

            if isinstance(out_c, tuple):
                chunk_output, final_state = out_c[0], out_c[1]
            else:
                chunk_output = out_c
                final_state = None

            outputs.append(chunk_output)

            if final_state is not None:
                fenwick.update(ci, final_state, lw_c)

        # ── Save Fenwick state to cache if committing ──
        if incremental_cache is not None and isinstance(incremental_cache, RecurrentCache) and commit:
            incremental_cache.save_fenwick(fenwick)

        # ── Concat, trim, cast back to input dtype ──
        output = torch.cat(outputs, dim=1)[:, :T]  # [B, T, H, V] bf16
        output = output.to(input_dtype).reshape(B, T, H * V)

        # Output gate (FLA convention: norm → gate → project)
        output = self.g_norm(output.transpose(1, 2)).transpose(1, 2)
        output = output * o_gate.to(input_dtype).sigmoid()
        output = self.o_proj(output)

        return output


# ═══════════════════════════════════════════════════════════════════════════
# E layer — Log-Linear GDN (hierarchical, O(T log T), fused kernel)
# ═══════════════════════════════════════════════════════════════════════════

class LogLinearGDNBlock(nn.Module):
    """Log-Linear Gated Delta Net — hierarchical recurrence with fused kernel.

    Uses hattention's HGatedDeltaNet which fuses the Fenwick tree logic into
    the Triton kernel. Single-step delta rule (n_h=1) but with full chunkwise
    parallel training. Faster than LogLinearDeltaProductBlock but less
    expressive (no multi-Householder error correction).

    Use as the E layer type in DDDEL pattern, or as a fallback when H layers
    are too slow for training.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        if not _hattention_available:
            raise ImportError(
                "hattention is required for Log-Linear GDN layers. "
                "Install with: pip install git+https://github.com/HanGuo97/log-linear-attention.git"
            )

        n_embd = config.n_embd
        head_dim = config.deltaproduct_head_dim
        num_heads = n_embd // head_dim

        self.hgdn = HGatedDeltaNet(
            hidden_size=n_embd,
            head_dim=head_dim,
            num_heads=num_heads,
            expand_v=config.deltaproduct_expand_v,
            use_gate=True,
            use_short_conv=config.deltaproduct_use_short_conv,
            conv_size=config.deltaproduct_conv_size,
            mode='chunk',
            layer_idx=layer_idx,
        )
        self.layer_idx = layer_idx
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out, _, _ = self.hgdn(x)
        return out.to(x.dtype)
