"""GPT Transformer blocks with hybrid attention (SSSL/Mamba pattern).

Attention types:
- S (Sliding window): local attention within a fixed window, O(T×W) cost.
  Uses FlashAttention-2 native window_size if available, otherwise blockwise fallback.
- L (Long/global): full causal attention with MLA (Multi-Head Latent Attention)
  for KV compression. O(T²) but with ~3-10x smaller KV cache.
- M (Mamba): Mamba-2 selective scan, O(T) linear recurrence. No KV cache.
  Replaces sliding-window attention for local context processing.

Position encoding: YaRN-scaled RoPE for 200K+ positions (attention layers only;
Mamba layers handle sequence ordering through recurrent state).

Also includes: QK-Norm, ReLU², value embeddings, per-layer residual scalars,
logit softcapping. Muon optimizer setup.

Reference: github.com/karpathy/autoresearch (train.py)
Reference: github.com/KellerJordan/modded-nanogpt
Reference: DeepSeek-V3 (MLA architecture)
Reference: YaRN (arxiv.org/abs/2309.00071)
Reference: Mamba-2 (github.com/state-spaces/mamba)
Reference: Jamba (arxiv.org/abs/2403.19887) — hybrid Mamba-Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import asdict

from .config import TransformerConfig


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def apply_rotary_emb_partial(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int) -> Tensor:
    """Apply RoPE to only the first rope_dim dimensions, leave rest unchanged.

    Used by MLA where only a subset of key dimensions get positional encoding.
    The rest come from the position-invariant latent decompression.
    """
    assert x.ndim == 4
    d = rope_dim // 2
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    x1, x2 = x_rope[..., :d], x_rope[..., d:]
    y1 = x1 * cos[..., :d] + x2 * sin[..., :d]
    y2 = x1 * (-sin[..., :d]) + x2 * cos[..., :d]
    x_rotated = torch.cat([y1, y2], -1)

    return torch.cat([x_rotated, x_pass], -1)


# ── Sliding Window Attention (S layers) ──

# Try to import FlashAttention-2 for native sliding window support
_flash_attn_available = False
try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except ImportError:
    pass


class SlidingWindowAttention(nn.Module):
    """Local attention within a fixed window. O(T × W) cost.

    KV compression via RotorQuant (PlanarQuant 2D Givens rotation + Lloyd-Max
    centroids). Falls back to PolarQuant (uniform scalar) if unavailable.
    10x KV memory reduction at 3-bit.

    Training: STE quantization-aware training (QAT) injects quantization noise
    into K,V so the model learns representations robust to compression.
    Inference: actual quantized blockwise attention for long sequences.

    Uses FlashAttention-2's native window_size parameter when available.
    Falls back to blockwise processing that never materializes a T×T mask.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.window_size = config.sliding_window_size
        self.layer_idx = layer_idx

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # KV compression: RotorQuant (PlanarQuant/IsoQuant) with PolarQuant fallback
        from ..core.quantize import QuantizedKVCache, _make_quantizer, ste_quantize
        self.kv_cache = QuantizedKVCache(self.head_dim, bits=3)
        # STE quantizer for training-time QAT (separate instance, same config)
        self._kv_ste_quantizer = _make_quantizer(self.head_dim, bits=3)
        self._ste_quantize = ste_quantize

        # YaRN attention temperature scaling
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # QAT: during training, inject quantization noise via STE so the model
        # learns K,V representations that are robust to 3-bit compression.
        # Gradient flows through as identity (straight-through estimator).
        if self.training:
            k = self._ste_quantize(k, self._kv_ste_quantizer)
            v = self._ste_quantize(v, self._kv_ste_quantizer)

        # GQA: expand kv heads if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        if _flash_attn_available and x.is_cuda:
            # FlashAttention-2: native sliding window for sequences within window
            y = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(self.window_size - 1, 0),
                softmax_scale=self._yarn_scale / math.sqrt(self.head_dim),
            )
        else:
            # Quantized blockwise sliding window:
            # 1. Quantize K,V to 3-bit (10x smaller)
            # 2. Process in chunks, dequantizing only the window per chunk
            # 3. Never materializes a T×T mask. Memory is O(W²) per chunk.
            q = q.transpose(1, 2)  # [B, H, T, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = self._quantized_blockwise_attention(q, k, v, T)
            y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _quantized_blockwise_attention(self, q: Tensor, k: Tensor, v: Tensor, T: int) -> Tensor:
        """Quantized blockwise sliding window attention.

        For short sequences (T <= W): standard causal attention, no quantization.
        For long sequences (T > W):
          1. Quantize K,V to 3-bit immediately, free full-precision copies
          2. For each chunk, dequantize ONLY the window slice (no full materialization)
          3. Compute attention on the small dequantized window

        Peak KV memory: O(T × 1 byte) quantized + O(W × D × 4 bytes) per chunk.
        At 200K with W=4K, D=128: ~25MB quantized + ~4MB window = ~29MB vs ~300MB unquantized.
        """
        W = self.window_size
        scale = self._yarn_scale / math.sqrt(self.head_dim)

        if T <= W:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        B, H, _, D = q.shape
        out = torch.zeros_like(q)

        # Step 1: Compress K,V via RotorQuant/PolarQuant
        compressed = self.kv_cache.compress(k, v)
        del k, v

        # Step 2: Process in chunks, dequantizing only the window per chunk
        for start in range(0, T, W):
            end = min(start + W, T)
            q_chunk = q[:, :, start:end]

            kv_start = max(0, start - W + 1)

            # Dequantize only the needed slice — no full-tensor materialization
            k_window, v_window = self.kv_cache.decompress_slice(compressed, kv_start, end)

            if kv_start == 0 and end <= W:
                attn_out = F.scaled_dot_product_attention(
                    q_chunk, k_window, v_window, is_causal=True, scale=scale,
                )
            else:
                q_pos = torch.arange(start, end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(kv_start, end, device=q.device).unsqueeze(0)
                dist = q_pos - k_pos
                valid = (dist >= 0) & (dist < W)
                mask = torch.where(valid, 0.0, float('-inf')).to(q.dtype)
                attn_out = F.scaled_dot_product_attention(
                    q_chunk, k_window, v_window, attn_mask=mask, scale=scale,
                )

            out[:, :, start:end] = attn_out

        return out


# ── DeltaProduct / Log-Linear layers (D, H, E layers) ──

_fla_available = False
_hattention_available = False
try:
    from .deltaproduct_layers import (
        DeltaProductBlock, LogLinearDeltaProductBlock, LogLinearGDNBlock,
        _fla_available as _fla_avail, _hattention_available as _hattention_avail,
    )
    _fla_available = _fla_avail
    _hattention_available = _hattention_avail
except ImportError:
    pass

# ── Mamba-2 Selective Scan (M layers — legacy) ──

_mamba_available = False
try:
    from mamba_ssm import Mamba2
    _mamba_available = True
except ImportError:
    pass


class MambaBlock(nn.Module):
    """Mamba-2 selective scan block. O(T) linear recurrence — no KV cache.

    Drop-in replacement for SlidingWindowAttention on S layers.
    Input/output shape: (B, T, D) → (B, T, D), identical to attention blocks.

    Mamba handles sequence ordering through its recurrent state — no positional
    encoding needed. The cos/sin args in forward() are accepted but ignored
    so the Block class can call all attention types with the same signature.

    Parameter count: ~3 × expand × d_model² per block.

    Reference: Mamba-2 (arxiv.org/abs/2405.21060)
    Reference: Jamba (arxiv.org/abs/2403.19887) — 7:1 Mamba:Attention hybrid
    Reference: Nemotron-H (NVIDIA) — 5:1 Mamba:Attention hybrid
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        if not _mamba_available:
            raise ImportError(
                "mamba-ssm is required for Mamba layers. "
                "Install with: pip install mamba-ssm --no-build-isolation"
            )
        self.mamba = Mamba2(
            d_model=config.n_embd,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )
        self.layer_idx = layer_idx
        # Dummy attribute for compatibility with YaRN scale setting loop
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # cos/sin ignored — Mamba uses recurrent state for sequence ordering
        return self.mamba(x)


# ── MLA: Multi-Head Latent Attention (L layers) ──

class MLACausalSelfAttention(nn.Module):
    """Multi-Head Latent Attention with decoupled RoPE (DeepSeek V3 style).

    Compresses KV into a shared low-rank latent before caching:
        x → c_kv_compress → latent [B, T, d_latent]   (small, cached)
        latent → k_up → K_nope [B, T, n_kv_head, head_dim - d_rope]
        latent → v_up → V [B, T, n_kv_head, head_dim]

    The RoPE component is computed separately (small, per-position):
        x → c_k_rope → K_rope [B, T, n_kv_head, d_rope]

    Final key: K = cat(K_rope_with_RoPE, K_nope)

    Supports three attention modes:
    - Full causal (mla_window_size=0, dsa_enabled=False): O(T²) — short contexts
    - Windowed (mla_window_size>0, dsa_enabled=False): O(T×W) — long context fallback
    - DSA sparse (dsa_enabled=True): O(T×K) — belief-conditioned sparse global attention
      Lightning Indexer scores all tokens, selects top-k, MLA runs on sparse subset.
      Indexer keys RotorQuant-compressed via STE QAT. Beliefs bias token selection.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.d_latent = config.mla_latent_dim
        self.d_rope = config.mla_rope_dim
        self.d_nope = self.head_dim - self.d_rope
        self.layer_idx = layer_idx
        self.mla_window_size = getattr(config, 'mla_window_size', 0)

        assert self.d_rope <= self.head_dim, (
            f"mla_rope_dim ({self.d_rope}) must be <= head_dim ({self.head_dim})"
        )
        assert self.d_latent > 0, "MLACausalSelfAttention requires mla_latent_dim > 0"

        # Query projection (standard, full head_dim per head)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)

        # KV compression: x → latent (shared across heads, position-invariant)
        self.c_kv_compress = nn.Linear(self.n_embd, self.d_latent, bias=False)

        # KV decompression: latent → K_nope, V (per kv head)
        self.k_up = nn.Linear(self.d_latent, self.n_kv_head * self.d_nope, bias=False)
        self.v_up = nn.Linear(self.d_latent, self.n_kv_head * self.head_dim, bias=False)

        # RoPE component: x → K_rope (small, per kv head, gets positional encoding)
        self.c_k_rope = nn.Linear(self.n_embd, self.n_kv_head * self.d_rope, bias=False)

        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # DSA: Belief-Conditioned Lightning Indexer
        self.dsa_enabled = getattr(config, 'dsa_enabled', False)
        if self.dsa_enabled:
            from ..core.quantize import LightningIndexer
            # belief_dim=0 at init; set later by MemoriaModel once state is available
            self.indexer = LightningIndexer(
                hidden_dim=self.n_embd,
                index_dim=config.dsa_index_dim,
                n_heads=config.dsa_index_heads,
                bits=config.dsa_index_bits,
                belief_dim=0,
                belief_lambda=config.dsa_belief_lambda,
            )
            self.dsa_top_k = config.dsa_top_k
            self.dsa_top_k_ratio = config.dsa_top_k_ratio
            # Store last KL loss for training loop to pick up
            self._last_dsa_kl_loss = None

        # RotorQuant KV compression for windowed MLA at long context (non-DSA fallback)
        if self.mla_window_size > 0 and not self.dsa_enabled:
            from ..core.quantize import QuantizedKVCache, _make_quantizer, ste_quantize
            self.mla_kv_cache = QuantizedKVCache(self.head_dim, bits=3)
            self._mla_ste_quantizer = _make_quantizer(self.head_dim, bits=3)
            self._ste_quantize = ste_quantize

        # YaRN attention temperature scaling
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                beliefs: Tensor | None = None,
                attn_mask: Tensor | None = None,
                kv_cache=None,
                commit: bool = False) -> Tensor:
        B, T, C = x.size()

        # Query: standard projection + full RoPE
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)

        # KV compression → latent
        latent = self.c_kv_compress(x)  # [B, T, d_latent]

        # Decompress: latent → K_nope (no positional encoding)
        k_nope = self.k_up(latent).view(B, T, self.n_kv_head, self.d_nope)

        # RoPE component: small per-position keys
        k_rope = self.c_k_rope(x).view(B, T, self.n_kv_head, self.d_rope)

        # Apply RoPE to k_rope and to the rope portion of q
        # cos/sin are [1, T, 1, head_dim/2] — slice to d_rope/2 for k_rope
        cos_rope = cos[..., :self.d_rope // 2]
        sin_rope = sin[..., :self.d_rope // 2]
        k_rope = apply_rotary_emb(k_rope, cos_rope, sin_rope)

        # Apply RoPE to query: only first d_rope dims get rotated
        q = apply_rotary_emb_partial(q, cos, sin, self.d_rope)

        # Assemble full key: [rope_dims | nope_dims]
        k_new = torch.cat([k_rope, k_nope], dim=-1)  # [B, T, n_kv_head, head_dim]

        # Decompress values from latent
        v_new = self.v_up(latent).view(B, T, self.n_kv_head, self.head_dim)

        # Incremental KV cache: prepend cached K/V from previous rounds.
        # K already has RoPE applied; V is position-invariant. Both are
        # fully assembled and ready for attention — no latent recomputation.
        k, v = k_new, v_new
        if kv_cache is not None:
            from .kv_cache import AttentionCache
            if isinstance(kv_cache, AttentionCache) and kv_cache.k is not None:
                k, v = kv_cache.append(k_new, v_new)
                if commit:
                    kv_cache.commit_indices(k_new, v_new, list(range(T)))

        # QK norm
        q, k = norm(q), norm(k)

        # DSA: belief-conditioned sparse attention
        if self.dsa_enabled:
            y = self._dsa_attention(x, q, k, v, beliefs)
        else:
            # Non-DSA path: windowed or full causal
            # QAT: during training with windowed MLA, inject quantization noise
            if self.training and self.mla_window_size > 0:
                k = self._ste_quantize(k, self._mla_ste_quantizer)
                v = self._ste_quantize(v, self._mla_ste_quantizer)

            # GQA: expand kv heads if needed
            if self.n_kv_head < self.n_head:
                rep = self.n_head // self.n_kv_head
                k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
                v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

            W = self.mla_window_size
            # DDTree verify: when attn_mask is provided, force SDPA path
            # (FlashAttention-2 does not support the tree attention pattern).
            if attn_mask is not None:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    scale=self._yarn_scale / math.sqrt(self.head_dim),
                )
                y = y.transpose(1, 2)
            elif W <= 0 or T <= W:
                # Full causal attention
                if _flash_attn_available and x.is_cuda and W > 0:
                    y = flash_attn_func(
                        q, k, v,
                        causal=True,
                        window_size=(W - 1, 0) if W > 0 else (-1, -1),
                        softmax_scale=self._yarn_scale / math.sqrt(self.head_dim),
                    )
                else:
                    q = q.transpose(1, 2)  # [B, H, T, D]
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)
                    y = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True,
                        scale=self._yarn_scale / math.sqrt(self.head_dim),
                    )
                    y = y.transpose(1, 2)
            else:
                # Windowed MLA with RotorQuant KV compression for long context
                if _flash_attn_available and x.is_cuda:
                    y = flash_attn_func(
                        q, k, v,
                        causal=True,
                        window_size=(W - 1, 0),
                        softmax_scale=self._yarn_scale / math.sqrt(self.head_dim),
                    )
                else:
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)
                    y = self._windowed_attention(q, k, v, T)
                    y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _dsa_attention(
        self,
        x: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beliefs: Tensor | None,
    ) -> Tensor:
        """Belief-conditioned sparse attention via Lightning Indexer.

        1. Indexer scores tokens chunk-by-chunk (O(T×C) memory, never T²)
        2. Select top-k tokens per query position
        3. Gather sparse KV at selected indices
        4. Run MLA attention on sparse subset with causal masking
        5. KL alignment loss only in dense path (short context)

        During training at short context (T <= effective_k), runs full
        attention and computes KL loss for indexer alignment. At long context,
        sparse path runs without KL (no dense target available).
        """
        from ..core.quantize import gather_sparse_kv

        B, T, _, _ = q.shape
        scale = self._yarn_scale / math.sqrt(self.head_dim)

        # Determine top-k for current context
        if self.training:
            effective_k = max(1, int(T * self.dsa_top_k_ratio))
        else:
            effective_k = min(self.dsa_top_k, T)

        # GQA: expand kv heads if needed (before sparse gather)
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        if effective_k >= T:
            # Dense path: full causal attention + KL alignment for indexer training
            q_t = q.transpose(1, 2)  # [B, H, T, D]
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=scale,
            )
            y = y.transpose(1, 2)

            # KL alignment: train indexer against dense attention (only here,
            # where the true dense target is available)
            if self.training:
                # Dense scores for KL target (affordable at short T)
                indexer_scores = self.indexer.compute_dense_scores(x, beliefs)
                with torch.no_grad():
                    attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
                    causal = torch.triu(
                        torch.full((T, T), float('-inf'), device=q.device, dtype=q.dtype),
                        diagonal=1,
                    )
                    attn_weights = torch.softmax(attn_weights + causal, dim=-1)
                self._last_dsa_kl_loss = self.indexer.compute_kl_loss(
                    indexer_scores, attn_weights
                )
        else:
            # Sparse path: chunk-based top-k selection, no T² matrix
            indices, _ = self.indexer.compute_topk(x, effective_k, beliefs)  # [B, T, k]

            # Gather sparse KV: [B, T, k, H, D]
            k_sparse, v_sparse = gather_sparse_kv(k, v, indices)

            # Per-query attention with causal masking on gathered tokens
            BT = B * T
            q_r = q.reshape(BT, 1, self.n_head, self.head_dim).transpose(1, 2)  # [BT, H, 1, D]
            k_r = k_sparse.reshape(BT, effective_k, self.n_head, self.head_dim).transpose(1, 2)  # [BT, H, k, D]
            v_r = v_sparse.reshape(BT, effective_k, self.n_head, self.head_dim).transpose(1, 2)  # [BT, H, k, D]

            # Causal mask: mask out gathered keys that are from future positions.
            # indices: [B, T, k] — the global position of each gathered key.
            # query position t should only attend to keys where indices[b,t,j] <= t.
            query_pos = torch.arange(T, device=q.device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            future_mask = indices > query_pos  # [B, T, k] — True for future keys
            future_mask = future_mask.reshape(BT, effective_k)  # [BT, k]
            # Expand for SDPA: [BT, 1, 1, k] broadcast to [BT, H, 1, k]
            attn_mask = torch.zeros(BT, 1, 1, effective_k, device=q.device, dtype=q.dtype)
            attn_mask.masked_fill_(future_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

            y = F.scaled_dot_product_attention(q_r, k_r, v_r, attn_mask=attn_mask, scale=scale)
            y = y.transpose(1, 2).reshape(B, T, self.n_head, self.head_dim)

            # No KL loss in sparse path — the indexer was trained via dense-path
            # KL during phase 1/early training. Computing KL here would be
            # self-referential (training indexer to match its own selection).
            self._last_dsa_kl_loss = None

        return y

    def _windowed_attention(self, q: Tensor, k: Tensor, v: Tensor, T: int) -> Tensor:
        """Windowed MLA with RotorQuant KV compression (CPU/non-flash fallback).

        Same blockwise approach as SlidingWindowAttention but for MLA layers.
        Compresses K,V to 3-bit, processes in window-sized chunks.
        """
        W = self.mla_window_size
        scale = self._yarn_scale / math.sqrt(self.head_dim)

        if T <= W:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        B, H, _, D = q.shape
        out = torch.zeros_like(q)

        compressed = self.mla_kv_cache.compress(k, v)
        del k, v

        for start in range(0, T, W):
            end = min(start + W, T)
            q_chunk = q[:, :, start:end]
            kv_start = max(0, start - W + 1)

            k_window, v_window = self.mla_kv_cache.decompress_slice(
                compressed, kv_start, end
            )

            if kv_start == 0 and end <= W:
                attn_out = F.scaled_dot_product_attention(
                    q_chunk, k_window, v_window, is_causal=True, scale=scale,
                )
            else:
                q_pos = torch.arange(start, end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(kv_start, end, device=q.device).unsqueeze(0)
                dist = q_pos - k_pos
                valid = (dist >= 0) & (dist < W)
                mask = torch.where(valid, 0.0, float('-inf')).to(q.dtype)
                attn_out = F.scaled_dot_product_attention(
                    q_chunk, k_window, v_window, attn_mask=mask, scale=scale,
                )

            out[:, :, start:end] = attn_out

        return out


# ── Standard Causal Attention (fallback) ──

class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # YaRN attention temperature scaling
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                attn_mask: Tensor | None = None,
                kv_cache=None,
                commit: bool = False) -> Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k_new = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v_new = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k_new = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k_new, cos, sin)
        q, k_new = norm(q), norm(k_new)

        # Incremental KV cache: prepend cached K/V from previous rounds.
        # In incremental mode, x contains only NEW tokens, and cos/sin are
        # already sliced to the correct positions. The cached KV already has
        # RoPE applied from when those positions were first processed.
        k, v = k_new, v_new
        if kv_cache is not None:
            from .kv_cache import AttentionCache
            if isinstance(kv_cache, AttentionCache) and kv_cache.k is not None:
                k_full, v_full = kv_cache.append(k_new, v_new)
                k, v = k_full, v_full
                if commit:
                    kv_cache.commit_indices(k_new, v_new, list(range(T)))

        # GQA: expand kv heads if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, k.shape[1], self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, v.shape[1], self.n_head, self.head_dim)

        q = q.transpose(1, 2)  # [B, H, T_q, D]
        k = k.transpose(1, 2)  # [B, H, T_kv, D] (T_kv >= T_q when cache is used)
        v = v.transpose(1, 2)

        # When using KV cache (incremental mode), T_q != T_kv so is_causal
        # can't be used — build a causal mask manually or use attn_mask.
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                scale=self._yarn_scale / math.sqrt(self.head_dim),
            )
        elif kv_cache is not None and k.shape[2] > T:
            # Incremental: Q has T tokens, KV has T_cached + T tokens.
            # Build causal mask: each query at position p attends to KV at positions ≤ p.
            T_kv = k.shape[2]
            T_cached = T_kv - T
            # Query positions are T_cached..T_cached+T-1 (absolute)
            # KV positions are 0..T_kv-1
            q_pos = torch.arange(T_cached, T_cached + T, device=x.device).unsqueeze(1)
            kv_pos = torch.arange(T_kv, device=x.device).unsqueeze(0)
            causal = (kv_pos <= q_pos).to(dtype=x.dtype)
            causal = causal.masked_fill(causal == 0, torch.finfo(x.dtype).min)
            causal = causal.masked_fill(causal == 1, 0.0)
            causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T_kv]
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal,
                scale=self._yarn_scale / math.sqrt(self.head_dim),
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                scale=self._yarn_scale / math.sqrt(self.head_dim),
            )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² from nanogpt speedrun
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with attention type dispatch based on window_pattern.

    M (Mamba)   → MambaBlock (selective scan, O(T) linear)
    S (Sliding) → SlidingWindowAttention (local, O(T×W))
    L (Long)    �� MLACausalSelfAttention (global, MLA-compressed KV)
    Fallback    → CausalSelfAttention (standard full attention)
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        pattern = config.window_pattern
        attn_type = pattern[layer_idx % len(pattern)] if pattern else 'F'

        if attn_type == 'H' and _fla_available:
            self.attn = LogLinearDeltaProductBlock(config, layer_idx)
        elif attn_type == 'D' and _fla_available:
            self.attn = DeltaProductBlock(config, layer_idx)
        elif attn_type == 'E' and _hattention_available:
            self.attn = LogLinearGDNBlock(config, layer_idx)
        elif attn_type in ('H', 'D', 'E') and _fla_available:
            self.attn = DeltaProductBlock(config, layer_idx)
        elif attn_type == 'M' and _mamba_available:
            self.attn = MambaBlock(config, layer_idx)
        elif attn_type in ('M', 'S', 'H', 'D', 'E'):
            self.attn = SlidingWindowAttention(config, layer_idx)
        elif attn_type == 'L' and config.mla_latent_dim > 0:
            self.attn = MLACausalSelfAttention(config, layer_idx)
        elif attn_type == 'L':
            # L layer but MLA disabled ��� use standard full attention
            self.attn = CausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)

        self.mlp = MLP(config)
        self._attn_type = attn_type

    def forward(self, x: Tensor, x0: Tensor, cos: Tensor, sin: Tensor,
                resid_lambda: Tensor, x0_lambda: Tensor,
                beliefs: Tensor | None = None,
                attn_mask: Tensor | None = None,
                layer_cache=None,
                commit: bool = False) -> Tensor:
        """Transformer block with optional incremental state.

        Args:
            layer_cache: per-layer cache from IncrementalState. Type depends on
                layer: RecurrentCache for H/D, AttentionCache for L, None for M/S.
            commit: if True, persist state changes to cache. If False (speculative
                verify), process without updating persistent cache.
        """
        # Per-layer residual scaling (from autoresearch)
        x = resid_lambda * x + x0_lambda * x0

        # Dispatch attention with appropriate kwargs based on layer type
        from .deltaproduct_layers import DeltaProductBlock, LogLinearDeltaProductBlock

        if isinstance(self.attn, LogLinearDeltaProductBlock):
            # H layer: recurrent with Fenwick tree — pass incremental cache
            x = x + self.attn(
                norm(x), cos, sin,
                incremental_cache=layer_cache, commit=commit,
            )
        elif isinstance(self.attn, DeltaProductBlock):
            # D layer: recurrent without Fenwick — pass initial/final state
            x = x + self.attn(norm(x), cos, sin)
        elif isinstance(self.attn, MLACausalSelfAttention) and self.attn.dsa_enabled:
            # L layer with DSA: belief-conditioned sparse attention + KV cache
            x = x + self.attn(
                norm(x), cos, sin, beliefs=beliefs,
                attn_mask=attn_mask, kv_cache=layer_cache, commit=commit,
            )
        elif isinstance(self.attn, (CausalSelfAttention, MLACausalSelfAttention)):
            # L layer (standard): attention + KV cache
            x = x + self.attn(
                norm(x), cos, sin,
                attn_mask=attn_mask, kv_cache=layer_cache, commit=commit,
            )
        else:
            # Mamba, SlidingWindow, etc. — no cross-round cache
            x = x + self.attn(norm(x), cos, sin)

        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    """GPT-style transformer backbone with hybrid SSSL attention.

    Does NOT include state interface layers — those are added in MemoriaModel.
    This is the pure language model component.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer residual scalars (from autoresearch)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Initialize rotary embedding frequencies (lazy: cos/sin computed on demand)
        head_dim = config.n_embd // config.n_head
        self._init_rotary(
            head_dim=head_dim,
            base=config.rope_base,
            original_max_len=config.sequence_len,
            scaling=config.rope_scaling,
            scaling_factor=config.rope_scaling_factor,
        )

        # Set YaRN temperature scale on all attention layers
        for block in self.blocks:
            block.attn._yarn_scale = self._yarn_scale

        # Softcap for logits
        self.softcap = 15.0

        # Track max cached RoPE length for lazy extension
        self._rope_cached_len = 0

    def _init_rotary(
        self,
        head_dim: int,
        base: int = 10000,
        original_max_len: int = 2048,
        scaling: str = "yarn",
        scaling_factor: float = 100.0,
    ):
        """Initialize RoPE inverse frequencies with optional YaRN scaling.

        Lazy design: only stores inv_freq here. cos/sin buffers are computed
        on first forward call and extended as needed, so model construction
        doesn't allocate memory for 200K positions upfront.

        YaRN (Yet another RoPE extensioN) combines:
        1. NTK-aware base frequency scaling
        2. Per-dimension interpolation ramp (high freqs extrapolate, low freqs interpolate)
        3. Attention temperature correction

        Reference: arxiv.org/abs/2309.00071
        """
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        base_freqs = 1.0 / (base ** (channel_range / head_dim))

        if scaling == "yarn" and scaling_factor > 1.0:
            # YaRN parameters
            beta_fast = 32   # high frequency boundary
            beta_slow = 1    # low frequency boundary

            # Wavelength of each frequency dimension
            wavelengths = 2 * math.pi / base_freqs
            # How many times each wavelength fits in the original context
            ratios = original_max_len / wavelengths

            # Smooth interpolation ramp between beta_slow and beta_fast
            # ramp=0 → low freq (interpolate), ramp=1 → high freq (extrapolate)
            ramp = ((ratios - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)

            # NTK-aware scaled base: increases base frequency for interpolation
            base_scaled = base * (scaling_factor ** (head_dim / (head_dim - 2)))
            inv_freq_scaled = 1.0 / (base_scaled ** (channel_range / head_dim))

            # Blend: high-freq dims keep original, low-freq dims use scaled
            inv_freq = ramp * base_freqs + (1.0 - ramp) * inv_freq_scaled

            # YaRN attention temperature correction
            self._yarn_scale = 0.1 * math.log(scaling_factor) + 1.0
        else:
            inv_freq = base_freqs
            self._yarn_scale = 1.0

        # Store inv_freq as buffer (small: just head_dim/2 floats)
        self.register_buffer('_inv_freq', inv_freq, persistent=False)
        # Initialize empty cos/sin buffers (will be lazily filled)
        self.register_buffer('cos', torch.empty(1, 0, 1, len(inv_freq)), persistent=False)
        self.register_buffer('sin', torch.empty(1, 0, 1, len(inv_freq)), persistent=False)

    def _ensure_rope(self, seq_len: int):
        """Lazily extend RoPE cos/sin buffers to cover seq_len positions.

        Only recomputes when seq_len exceeds the current cached length.
        Grows in chunks (2x current or seq_len, whichever is larger) to
        avoid recomputing on every slight increase.
        """
        if seq_len <= self._rope_cached_len:
            return

        # Grow to at least 2x current or seq_len + buffer
        new_len = max(seq_len + 256, self._rope_cached_len * 2)
        new_len = min(new_len, self.config.max_position)  # never exceed max_position
        new_len = max(new_len, seq_len)  # but always cover what's requested

        t = torch.arange(new_len, dtype=torch.float32, device=self._inv_freq.device)
        freqs = torch.outer(t, self._inv_freq)
        self.cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
        self.sin = freqs.sin().unsqueeze(0).unsqueeze(2)
        self._rope_cached_len = new_len

    @torch.no_grad()
    def init_weights(self):
        """Custom weight initialization (from autoresearch)."""
        s = 3**0.5 * self.config.n_embd**-0.5
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.blocks:
            attn = block.attn
            if isinstance(attn, (DeltaProductBlock, LogLinearDeltaProductBlock, LogLinearGDNBlock)):
                # DeltaProduct/Log-Linear layers use their own internal init — skip
                pass
            elif isinstance(attn, MambaBlock):
                # Mamba-2 uses its own internal initialization — skip
                pass
            elif isinstance(attn, MLACausalSelfAttention):
                # MLA: init compression/decompression near-orthogonal
                nn.init.uniform_(attn.c_q.weight, -s, s)
                nn.init.uniform_(attn.c_kv_compress.weight, -s, s)
                nn.init.uniform_(attn.c_k_rope.weight, -s, s)
                # Up-projections: small init to start near-identity
                nn.init.normal_(attn.k_up.weight, mean=0.0, std=0.01)
                nn.init.normal_(attn.v_up.weight, mean=0.0, std=0.01)
                nn.init.zeros_(attn.c_proj.weight)
                # DSA: init Lightning Indexer projections
                if attn.dsa_enabled:
                    nn.init.uniform_(attn.indexer.q_proj.weight, -s, s)
                    nn.init.uniform_(attn.indexer.k_proj.weight, -s, s)
                    nn.init.normal_(attn.indexer.head_gate.weight, mean=0.0, std=0.01)
            else:
                nn.init.uniform_(attn.c_q.weight, -s, s)
                nn.init.uniform_(attn.c_k.weight, -s, s)
                nn.init.uniform_(attn.c_v.weight, -s, s)
                nn.init.zeros_(attn.c_proj.weight)

            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

    def forward_blocks(self, idx: Tensor) -> list[Tensor]:
        """Run through transformer blocks, returning hidden states after each block.

        Used by MemoriaModel to insert state interface layers between blocks.

        Args:
            idx: [B, T] token indices

        Returns:
            List of [B, T, n_embd] hidden states, one per block
        """
        B, T = idx.size()
        self._ensure_rope(T)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = self.wte(idx)
        x = norm(x)
        x0 = x

        hiddens = []
        for i, block in enumerate(self.blocks):
            x = block(x, x0, cos, sin, self.resid_lambdas[i], self.x0_lambdas[i])
            hiddens.append(x)

        return hiddens

    def head(self, x: Tensor) -> Tensor:
        """Apply LM head to final hidden state.

        Args:
            x: [B, T, n_embd] final hidden state

        Returns:
            [B, T, vocab_size] logits
        """
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor:
        """Standard forward pass (no state interface, for baseline comparison).

        Args:
            idx: [B, T] token indices
            targets: [B, T] target token indices (optional)

        Returns:
            logits or loss
        """
        hiddens = self.forward_blocks(idx)
        logits = self.head(hiddens[-1])

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits
