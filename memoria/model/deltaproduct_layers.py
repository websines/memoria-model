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

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # cos/sin ignored — DeltaProduct uses recurrent state for ordering
        # FLA Triton kernels require bfloat16 — autocast handles dtype seamlessly
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out, _, _ = self.deltaproduct(x)
        return out.to(x.dtype)


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

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # cos/sin ignored — recurrent layer, no positional encoding
        # FLA Triton kernels require bfloat16 — autocast the entire forward
        input_dtype = x.dtype
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = self._forward_impl(x)
        return out.to(input_dtype)

    def _forward_impl(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        C = self.chunk_size
        H = self.num_heads
        K = self.head_k_dim
        V = self.head_v_dim
        n_h = self.num_householder

        # ── Projections ──
        q = self.q_proj(x)        # [B, T, H*K]
        k = self.k_proj(x)        # [B, T, H*K*n_h]
        v = self.v_proj(x)        # [B, T, H*V*n_h]
        beta = self.b_proj(x)     # [B, T, H*n_h]
        g_gate = self.a_proj(x)   # [B, T, H] — forget gate (log-space via sigmoid)
        o_gate = self.g_proj(x)   # [B, T, H*V] — output gate

        # Short causal convolution
        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[..., :T].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[..., :T].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[..., :T].transpose(1, 2)

        # Activations (FLA convention)
        q = q.view(B, T, H, K)
        k = nn.functional.silu(k).view(B, T, H, K * n_h)
        v = nn.functional.silu(v).view(B, T, H, V * n_h)
        beta = beta.view(B, T, H * n_h).sigmoid()
        g_gate = nn.functional.logsigmoid(g_gate).view(B, T, H)  # log-space forget gate
        o_gate = o_gate.view(B, T, H * V)

        # ── Level scales (Log-Linear) ──
        dl = self.l_proj(x)  # [B, T, H*L]
        dl = dl.view(B, T, H, MAX_NUM_LEVELS)
        # Data-dependent level weights: softplus(L * dl)
        level_weights = nn.functional.softplus(self.L.unsqueeze(0).unsqueeze(0) * dl)
        # [B, T, H, L]

        # ── Pad sequence to multiple of chunk_size ──
        pad_len = (C - T % C) % C
        if pad_len > 0:
            q = nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
            beta = nn.functional.pad(beta, (0, 0, 0, pad_len))
            g_gate = nn.functional.pad(g_gate, (0, 0, 0, pad_len))
            level_weights = nn.functional.pad(level_weights, (0, 0, 0, 0, 0, pad_len))

        T_padded = q.shape[1]
        num_chunks = T_padded // C

        # ── Initialize Fenwick tree ──
        fenwick = self._get_fenwick(B, x.device, x.dtype)

        # ── Process chunks sequentially with Fenwick state management ──
        outputs = []
        for ci in range(num_chunks):
            s, e = ci * C, (ci + 1) * C

            q_c = q[:, s:e]              # [B, C, H, K]
            k_c = k[:, s:e]              # [B, C, H, K*n_h]
            v_c = v[:, s:e]              # [B, C, H, V*n_h]
            beta_c = beta[:, s:e]        # [B, C, H*n_h]
            g_c = g_gate[:, s:e]         # [B, C, H]

            # Average level scales across chunk positions for Fenwick query
            lw_c = level_weights[:, s:e].mean(dim=1)  # [B, H, L]

            # Query Fenwick tree for initial state
            init_state = fenwick.query(lw_c)  # [B, H, K, V]

            # Run DeltaProduct chunk kernel
            out_c = chunk_gated_delta_product(
                q=q_c,
                k=k_c,
                v=v_c,
                g=g_c,
                beta=beta_c,
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

            # Update Fenwick tree with chunk's final state
            if final_state is not None:
                fenwick.update(ci, final_state, lw_c)

        # ── Concat and trim to original length ──
        output = torch.cat(outputs, dim=1)[:, :T]  # [B, T, H, V]

        # Reshape for output gating and projection
        output = output.reshape(B, T, H * V)

        # Output gate (FLA convention: norm → gate → project)
        output = self.g_norm(output.transpose(1, 2)).transpose(1, 2)
        output = output * o_gate.sigmoid()
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
