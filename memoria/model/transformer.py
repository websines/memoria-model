"""GPT Transformer blocks with hybrid attention (SSSL pattern).

Attention types:
- S (Sliding window): local attention within a fixed window, O(T×W) cost.
  Uses FlashAttention-2 native window_size if available, otherwise blockwise fallback.
- L (Long/global): full causal attention with MLA (Multi-Head Latent Attention)
  for KV compression. O(T²) but with ~3-10x smaller KV cache.

Position encoding: YaRN-scaled RoPE for 200K+ positions.

Also includes: QK-Norm, ReLU², value embeddings, per-layer residual scalars,
logit softcapping. Muon optimizer setup.

Reference: github.com/karpathy/autoresearch (train.py)
Reference: github.com/KellerJordan/modded-nanogpt
Reference: DeepSeek-V3 (MLA architecture)
Reference: YaRN (arxiv.org/abs/2309.00071)
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

    KV is quantized to 3-bit via PolarQuant (random rotation + uniform scalar
    quantization). This reduces KV memory by ~10x and improves speed on
    memory-bandwidth-bound attention.

    Uses FlashAttention-2's native window_size parameter if available.
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

        # KV compression: TurboQuant (7.7x savings) with our PolarQuant fallback
        self._use_turboquant = False
        try:
            from turboquant import TurboQuantKVCache
            self.kv_cache = TurboQuantKVCache(head_dim=self.head_dim, bit_width=3)
            self._use_turboquant = True
        except ImportError:
            from ..core.quantize import QuantizedKVCache
            self.kv_cache = QuantizedKVCache(self.head_dim, bits=3)

        # YaRN attention temperature scaling
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # GQA: expand kv heads if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        if _flash_attn_available and x.is_cuda and T <= self.window_size:
            # FlashAttention-2: native sliding window for sequences within window
            y = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(self.window_size - 1, 0),
                softmax_scale=self._yarn_scale / math.sqrt(self.head_dim),
            )
        else:
            # Quantized blockwise sliding window:
            # 1. Quantize K,V to 3-bit (10x smaller) — peak KV memory drops from O(T×D) to O(T×1byte)
            # 2. Process in chunks of window_size, dequantizing only the window needed per chunk
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
          2. For each chunk, dequantize ONLY the W-sized window (element-wise, no matmul)
          3. Compute attention on the small dequantized window

        No rotation matrix: keys are QK-normed so dimensions already have equal
        variance — rotation is redundant. Dequant is pure element-wise ops.

        Peak KV memory: O(T × 1 byte) quantized + O(W × D × 4 bytes) per chunk window.
        At 200K with W=4K, D=128: ~25MB quantized + ~4MB window = ~29MB vs ~300MB unquantized.
        """
        W = self.window_size
        scale = self._yarn_scale / math.sqrt(self.head_dim)

        if T <= W:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        B, H, _, D = q.shape
        out = torch.zeros_like(q)

        # Step 1: Compress K,V — TurboQuant (7.7x) or PolarQuant (3.9x) fallback
        if self._use_turboquant:
            compressed = self.kv_cache.compress(k, v)
        else:
            compressed = self.kv_cache.compress(k, v)
        # Free full-precision K,V
        del k, v

        # Step 2: Process in chunks, dequantizing only the window per chunk
        for start in range(0, T, W):
            end = min(start + W, T)
            q_chunk = q[:, :, start:end]  # [B, H, chunk_len, D]

            # KV window boundaries
            kv_start = max(0, start - W + 1)

            # Dequantize ONLY the window slice
            if self._use_turboquant:
                # TurboQuant: decompress full then slice (API doesn't support slicing compressed)
                # TODO: upstream slice support to turboquant-torch
                k_full = self.kv_cache.decompress_keys(compressed)
                v_full = self.kv_cache.decompress_values(compressed)
                k_window = k_full[:, :, kv_start:end]
                v_window = v_full[:, :, kv_start:end]
                del k_full, v_full
            else:
                k_window = self.kv_cache.quantizer.dequantize(
                    compressed['k_codes'][:, :, kv_start:end],
                    compressed['k_scale'][:, :, kv_start:end],
                )
                v_window = self.kv_cache.quantizer.dequantize(
                    compressed['v_codes'][:, :, kv_start:end],
                    compressed['v_scale'][:, :, kv_start:end],
                )

            if kv_start == 0 and end <= W:
                # First chunk: standard causal
                attn_out = F.scaled_dot_product_attention(
                    q_chunk, k_window, v_window, is_causal=True, scale=scale,
                )
            else:
                # Small causal+window mask for this chunk: [chunk_len, window_len]
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

    This decoupling allows the latent to be position-invariant (cacheable
    without storing per-position RoPE'd keys), while the small K_rope
    component carries positional information.
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

        # YaRN attention temperature scaling
        self._yarn_scale = 1.0

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
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
        k = torch.cat([k_rope, k_nope], dim=-1)  # [B, T, n_kv_head, head_dim]

        # Decompress values from latent
        v = self.v_up(latent).view(B, T, self.n_kv_head, self.head_dim)

        # QK norm
        q, k = norm(q), norm(k)

        # GQA: expand kv heads if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        # Standard causal attention (full context for L layers)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            scale=self._yarn_scale / math.sqrt(self.head_dim),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


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

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # GQA: expand kv heads if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
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

    S (Sliding) → SlidingWindowAttention (local, O(T×W))
    L (Long)    → MLACausalSelfAttention (global, MLA-compressed KV)
    Fallback    → CausalSelfAttention (standard full attention)
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        pattern = config.window_pattern
        attn_type = pattern[layer_idx % len(pattern)] if pattern else 'F'

        if attn_type == 'S':
            self.attn = SlidingWindowAttention(config, layer_idx)
        elif attn_type == 'L' and config.mla_latent_dim > 0:
            self.attn = MLACausalSelfAttention(config, layer_idx)
        elif attn_type == 'L':
            # L layer but MLA disabled — use standard full attention
            self.attn = CausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)

        self.mlp = MLP(config)
        self._attn_type = attn_type

    def forward(self, x: Tensor, x0: Tensor, cos: Tensor, sin: Tensor,
                resid_lambda: Tensor, x0_lambda: Tensor) -> Tensor:
        # Per-layer residual scaling (from autoresearch)
        x = resid_lambda * x + x0_lambda * x0
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
            if isinstance(attn, MLACausalSelfAttention):
                # MLA: init compression/decompression near-orthogonal
                nn.init.uniform_(attn.c_q.weight, -s, s)
                nn.init.uniform_(attn.c_kv_compress.weight, -s, s)
                nn.init.uniform_(attn.c_k_rope.weight, -s, s)
                # Up-projections: small init to start near-identity
                nn.init.normal_(attn.k_up.weight, mean=0.0, std=0.01)
                nn.init.normal_(attn.v_up.weight, mean=0.0, std=0.01)
                nn.init.zeros_(attn.c_proj.weight)
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
