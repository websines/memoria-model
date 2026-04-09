"""Quantization via RotorQuant block-diagonal rotations + CAGE weight QAT + DSA.

Primary backend: RotorQuant (PlanarQuant for 3-bit, IsoQuant for 4-bit)
  - Block-diagonal 2D/4D rotations: O(d) FMAs, fully parallelizable
  - Lloyd-Max MSE-optimal centroids for the rotated coordinate distribution
  - QJL residual correction (Stage 2) for inner-product-preserving compression

Fallback: PolarQuantizer (no external deps)
  - Random orthogonal rotation + uniform scalar quantization
  - Works on CPU without triton/scipy

Training integration via STE (Straight-Through Estimator):
  - Forward: quantize → dequantize (injects quantization noise)
  - Backward: gradient passes through unmodified (identity)
  - Result: model learns representations robust to quantization noise,
    so inference-time KV compression and checkpoint compression are ~lossless

Weight QAT via CAGE (Curvature-Aware Gradient Estimation):
  - STE noise injection on weight matrices during forward pass
  - CAGE post-optimizer correction nudges weights toward quantization grid:
    weight -= lr * lambda * (weight - Q(weight))
  - Phase-aligned schedule: silent phase 1, ramp phase 2, full phase 3

DSA (Belief-Conditioned Sparse Attention) via Lightning Indexer:
  - Lightweight learned indexer scores all tokens, selects top-k for full MLA
  - Indexer keys compressed via RotorQuant STE QAT (same pipeline as KV/weights)
  - Belief conditioning: active beliefs bias scores toward relevant tokens
  - KL alignment loss trains indexer against dense attention distribution
  - Replaces windowed MLA with sparse global MLA at long context

Reference: RotorQuant — scrya.com/rotorquant.pdf (March 2026)
Reference: TurboQuant — arxiv.org/abs/2504.19874 (ICLR 2026)
Reference: CAGE — arxiv.org/abs/2510.18784 (IST-DASLab 2025)
Reference: DeepSeek-V3.2 DSA — arxiv.org/abs/2512.02556 (DeepSeek 2025)
Reference: NSA — arxiv.org/abs/2502.11089 (ACL 2025)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

# ── Backend detection ──

_rotorquant_available = False
try:
    from turboquant import PlanarQuantMSE, IsoQuantMSE, PlanarQuantProd, IsoQuantProd
    _rotorquant_available = True
except ImportError:
    pass

_triton_available = False
try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    pass


# ── STE (Straight-Through Estimator) ──

class _STEQuantizeFn(torch.autograd.Function):
    """Quantize in forward, identity gradient in backward."""

    @staticmethod
    def forward(ctx, x: Tensor, quantizer: nn.Module) -> Tensor:
        with torch.no_grad():
            x_hat, _ = quantizer(x)
        return x_hat

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


class _STEPolarFn(torch.autograd.Function):
    """STE for the PolarQuantizer fallback (different API)."""

    @staticmethod
    def forward(ctx, x: Tensor, quantizer) -> Tensor:
        with torch.no_grad():
            codes, scale = quantizer.quantize(x)
            x_hat = quantizer.dequantize(codes, scale)
        return x_hat

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


def ste_quantize(x: Tensor, quantizer: nn.Module) -> Tensor:
    """Simulate quantization with straight-through gradient.

    During training: injects quantization noise so the model learns
    representations robust to compression. Gradient flows through as identity.
    During eval or no_grad: just quantize→dequantize directly.
    """
    if not x.requires_grad:
        if isinstance(quantizer, PolarQuantizer):
            codes, scale = quantizer.quantize(x)
            return quantizer.dequantize(codes, scale)
        x_hat, _ = quantizer(x)
        return x_hat

    if isinstance(quantizer, PolarQuantizer):
        return _STEPolarFn.apply(x, quantizer)
    return _STEQuantizeFn.apply(x, quantizer)


# ── Fallback: PolarQuantizer (no external deps) ──

class PolarQuantizer(nn.Module):
    """Quantize vectors via optional rotation + uniform scalar quantization.

    Two modes:
    - rotate=True: Random orthogonal rotation before quantization. Spreads
      information uniformly across dimensions. For unnormalized vectors.
    - rotate=False: Direct quantization without rotation. For QK-normed or
      RMS-normed data where dimensions already have similar variance.
    """

    def __init__(self, dim: int, bits: int = 3, rotate: bool = False):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.levels = 2 ** bits

        if rotate:
            R, _ = torch.linalg.qr(torch.randn(dim, dim))
            self.register_buffer('R', R)
        self.rotate = rotate

    @torch.no_grad()
    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize to low-bit codes + per-vector scale."""
        if self.rotate:
            x = x @ self.R

        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm = x / scale
        half_levels = (self.levels - 1) / 2.0
        codes = ((x_norm + 1.0) * half_levels).round().clamp(0, self.levels - 1).to(torch.uint8)
        return codes, scale

    @torch.no_grad()
    def dequantize(self, codes: Tensor, scale: Tensor) -> Tensor:
        """Reconstruct from codes + scale."""
        half_levels = (self.levels - 1) / 2.0
        x = codes.float() / half_levels - 1.0
        x = x * scale
        if self.rotate:
            x = x @ self.R.T
        return x


# ── RotorQuant wrapper ──

def _make_quantizer(dim: int, bits: int, device: str = 'cpu') -> nn.Module:
    """Create the best available quantizer for the given config.

    3-bit: PlanarQuantMSE (2D Givens rotation, 256 FMAs)
    4-bit: IsoQuantMSE (4D quaternion rotation, 512 FMAs)
    Fallback: PolarQuantizer (uniform scalar, no external deps)
    """
    if _rotorquant_available:
        if bits <= 3:
            return PlanarQuantMSE(d=dim, bits=bits, device=device)
        else:
            return IsoQuantMSE(d=dim, bits=bits, mode='fast', device=device)
    return PolarQuantizer(dim, bits, rotate=False)


# ── Quantized KV Cache ──

class QuantizedKVCache(nn.Module):
    """Sliding window KV cache with RotorQuant compression.

    Stores K and V in quantized form (3-4 bit) instead of float32/bf16.
    On attention computation, dequantizes the relevant window on the fly.

    Memory at 4K window, 128 head_dim, 3-bit:
    - float32: 4096 × 128 × 4 bytes = 2 MB per K or V per head
    - 3-bit:   ~0.2 MB per K or V per head (10x savings)
    """

    def __init__(self, head_dim: int, bits: int = 3, device: str = 'cpu'):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self._use_rotorquant = _rotorquant_available
        self.quantizer = _make_quantizer(head_dim, bits, device)

    def compress(self, k: Tensor, v: Tensor) -> dict:
        """Compress K and V tensors.

        Args:
            k, v: [B, H, T, D] tensors

        Returns:
            dict with compressed data (format depends on backend)
        """
        if self._use_rotorquant:
            # RotorQuant: returns (x_hat, indices_dict)
            _, k_data = self.quantizer.quantize(k)
            _, v_data = self.quantizer.quantize(v)
            return {'k': k_data, 'v': v_data, 'backend': 'rotorquant'}
        else:
            k_codes, k_scale = self.quantizer.quantize(k)
            v_codes, v_scale = self.quantizer.quantize(v)
            return {
                'k_codes': k_codes, 'k_scale': k_scale,
                'v_codes': v_codes, 'v_scale': v_scale,
                'backend': 'polar',
            }

    def decompress_slice(self, cache: dict, start: int, end: int) -> tuple[Tensor, Tensor]:
        """Decompress a slice [start:end] along the T dimension.

        Only dequantizes the requested window — no full-tensor materialization.
        """
        if cache['backend'] == 'rotorquant':
            k_slice = {k_: v[..., start:end, :] if v.dim() >= 3 else v
                       for k_, v in cache['k'].items() if k_ != '_norms'}
            v_slice = {k_: v[..., start:end, :] if v.dim() >= 3 else v
                       for k_, v in cache['v'].items() if k_ != '_norms'}
            # Norms need T-dimension slicing
            if '_norms' in cache['k']:
                k_slice['_norms'] = cache['k']['_norms'][..., start:end]
            if '_norms' in cache['v']:
                v_slice['_norms'] = cache['v']['_norms'][..., start:end]
            k = self.quantizer.dequantize(k_slice)
            v = self.quantizer.dequantize(v_slice)
        else:
            k = self.quantizer.dequantize(
                cache['k_codes'][..., start:end, :],
                cache['k_scale'][..., start:end, :],
            )
            v = self.quantizer.dequantize(
                cache['v_codes'][..., start:end, :],
                cache['v_scale'][..., start:end, :],
            )
        return k, v

    def decompress(self, cache: dict) -> tuple[Tensor, Tensor]:
        """Decompress full K and V from cache."""
        if cache['backend'] == 'rotorquant':
            k = self.quantizer.dequantize(cache['k'])
            v = self.quantizer.dequantize(cache['v'])
        else:
            k = self.quantizer.dequantize(cache['k_codes'], cache['k_scale'])
            v = self.quantizer.dequantize(cache['v_codes'], cache['v_scale'])
        return k, v


# ── Quantized Belief Store ──

class QuantizedBeliefStore:
    """Quantized storage for belief vectors.

    Beliefs are naturally polar (radius = precision, angle = content).
    We quantize the angle (unit vector) and store radius in full precision.
    Uses RotorQuant when available for better MSE than uniform quantization.

    Memory at 16K beliefs, 256-dim, 3-bit:
    - float32: 16384 × 256 × 4 bytes = 16 MB
    - 3-bit angles + float32 radii: ~1.6 MB (10x savings)
    """

    def __init__(self, belief_dim: int, bits: int = 3, device: str = 'cpu'):
        self.belief_dim = belief_dim
        self.bits = bits
        self._use_rotorquant = _rotorquant_available
        self.quantizer = _make_quantizer(belief_dim, bits, device)

    @torch.no_grad()
    def compress_beliefs(self, beliefs: Tensor) -> dict:
        """Compress belief vectors, preserving polar structure.

        Args:
            beliefs: [N, D] belief vectors (radius encodes precision)

        Returns:
            dict with compressed angles, radii, and active mask
        """
        radii = beliefs.norm(dim=-1, keepdim=True)
        active = (radii.squeeze(-1) > 1e-10)
        safe_radii = radii.clamp(min=1e-10)
        angles = beliefs / safe_radii

        if self._use_rotorquant:
            _, indices_data = self.quantizer.quantize(angles)
            return {
                'backend': 'rotorquant',
                'indices': indices_data,
                'radii': radii,
                'active': active,
            }
        else:
            angle_codes, angle_scale = self.quantizer.quantize(angles)
            return {
                'backend': 'polar',
                'angle_codes': angle_codes,
                'angle_scale': angle_scale,
                'radii': radii,
                'active': active,
            }

    @torch.no_grad()
    def decompress_beliefs(self, compressed: dict) -> Tensor:
        """Reconstruct belief vectors from compressed form."""
        if compressed.get('backend') == 'rotorquant':
            angles = self.quantizer.dequantize(compressed['indices'])
        else:
            angles = self.quantizer.dequantize(
                compressed['angle_codes'], compressed['angle_scale']
            )
        angles = torch.nn.functional.normalize(angles, dim=-1, eps=1e-10)
        beliefs = angles * compressed['radii']

        if 'active' in compressed:
            beliefs[~compressed['active']] = 0.0

        return beliefs


# ── Weight QAT (Quantization-Aware Training for model weights) ──

class WeightQuantLinear(nn.Module):
    """Drop-in nn.Linear replacement with STE weight quantization.

    During training: quantizes weight row-wise via RotorQuant, dequantizes,
    then uses the noisy weight for the matmul. STE gradient flows through.
    During eval: uses full-precision weight (quantize at deployment time).

    The quantizer treats each row of weight [out_features, in_features] as
    an in_features-dimensional vector and applies block-diagonal rotation +
    Lloyd-Max optimal centroids.
    """

    def __init__(self, linear: nn.Linear, bits: int = 4):
        super().__init__()
        self.linear = linear
        self.bits = bits
        self.quantizer = _make_quantizer(linear.in_features, bits)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            w_hat = ste_quantize(self.linear.weight, self.quantizer)
            # Cast to input dtype for mixed-precision (bf16 inputs, fp32 weights)
            if w_hat.dtype != x.dtype:
                w_hat = w_hat.to(x.dtype)
            bias = self.linear.bias
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            return nn.functional.linear(x, w_hat, bias)
        return self.linear(x)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @weight.setter
    def weight(self, value: Tensor):
        self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features


def apply_weight_qat(model: nn.Module, bits: int = 4, mlp_bits: int = 0) -> list[str]:
    """Patch eligible nn.Linear modules with WeightQuantLinear wrappers.

    Quantizes all nn.Linear in transformer blocks (H-block projections, MLA
    projections, MLP layers). Skips: embeddings, lm_head, interface layers,
    cognitive state, DFlash, and all small cognitive modules.

    Args:
        model: MemoriaModel or PretrainedMemoriaModel
        bits: default bit-width for weight quantization
        mlp_bits: bit-width for MLP layers (0 = use bits)

    Returns:
        list of patched module paths for logging
    """
    if bits == 0:
        return []

    if mlp_bits == 0:
        mlp_bits = bits

    # Only quantize modules under transformer.blocks
    # This naturally skips: wte, lm_head, interfaces, state, dflash_head, etc.
    patched = []

    transformer = getattr(model, 'transformer', None)
    if transformer is None:
        return []

    blocks = getattr(transformer, 'blocks', None)
    if blocks is None:
        return []

    for block_idx, block in enumerate(blocks):
        # Collect targets first to avoid mutating module tree during iteration
        targets = []
        for name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Determine bits: MLP layers get mlp_bits, everything else gets bits
            is_mlp = 'c_fc' in name or ('c_proj' in name and 'mlp' in name)
            # Also check if parent is MLP class
            parts = name.rsplit('.', 1)
            if len(parts) == 2 and parts[0].endswith('mlp'):
                is_mlp = True
            target_bits = mlp_bits if is_mlp else bits

            # Skip tiny layers (conv projections, etc.) — not worth the overhead
            if module.in_features < 64 or module.out_features < 64:
                continue

            targets.append((name, module, target_bits))

        # Now patch collected targets
        modules_dict = dict(block.named_modules())
        for name, module, target_bits in targets:
            parts = name.rsplit('.', 1)
            parent_name = parts[0] if len(parts) == 2 else ''
            child_name = parts[-1]

            parent = modules_dict[parent_name] if parent_name else block

            wrapped = WeightQuantLinear(module, bits=target_bits)
            setattr(parent, child_name, wrapped)

            path = f"blocks.{block_idx}.{name}"
            patched.append(f"{path} ({target_bits}-bit)")

    # ── DFlash KV injection: quantize modules annotated with _qat_bits ──
    # KV injection projections are marked with _qat_bits = 3 for aggressive
    # quantization. Draft accuracy is less critical since the verifier filters.
    dflash_head = getattr(model, 'dflash_head', None)
    if dflash_head is not None:
        dflash_targets = []
        for name, module in dflash_head.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, '_qat_bits'):
                target_bits = module._qat_bits
                dflash_targets.append((name, module, target_bits))

        dflash_modules = dict(dflash_head.named_modules())
        for name, module, target_bits in dflash_targets:
            parts = name.rsplit('.', 1)
            parent_name = parts[0] if len(parts) == 2 else ''
            child_name = parts[-1]
            parent = dflash_modules[parent_name] if parent_name else dflash_head
            wrapped = WeightQuantLinear(module, bits=target_bits)
            setattr(parent, child_name, wrapped)
            patched.append(f"dflash_head.{name} ({target_bits}-bit)")

    return patched


@torch.no_grad()
def cage_step(
    model: nn.Module,
    lr: float,
    cage_lambda: float,
):
    """CAGE post-optimizer correction: push weights toward quantization grid.

    For each WeightQuantLinear in the model, compute:
        e = weight - Q(weight)    # quantization error
        weight -= lr * λ * e      # nudge toward grid

    This is the decoupled CAGE variant — optimizer-agnostic, works with
    both Muon and AdamW parameter groups.

    Args:
        model: model containing WeightQuantLinear modules
        lr: current learning rate (from scheduler)
        cage_lambda: current CAGE lambda (0 during silence, ramps to lambda_base)
    """
    if cage_lambda <= 0:
        return

    for module in model.modules():
        if not isinstance(module, WeightQuantLinear):
            continue

        w = module.linear.weight
        q = module.quantizer

        # Quantize → dequantize to get grid-snapped weight
        if isinstance(q, PolarQuantizer):
            codes, scale = q.quantize(w.data)
            w_hat = q.dequantize(codes, scale)
        elif _rotorquant_available:
            w_hat, _ = q(w.data)
        else:
            continue

        # CAGE correction: nudge weight toward quantization grid
        e = w.data - w_hat
        w.data.add_(e, alpha=-lr * cage_lambda)


def get_cage_lambda(step: int, config) -> float:
    """Compute CAGE lambda for the current step, phase-aligned.

    Phase 1 (0 → phase1_steps): λ = 0 (silent — model learns language freely)
    Phase 2 (phase1_steps → + phase2_steps): λ ramps 0 → λ_base
    Phase 3 (remainder): λ = λ_base (full correction)

    If cage_silence_ratio > 0, uses that instead of phase alignment.
    """
    tc = config.training
    lambda_base = tc.cage_lambda_base

    if tc.cage_silence_ratio > 0:
        # Manual silence ratio (CAGE default style)
        total_steps = getattr(tc, '_total_steps', 10000)
        silence_steps = int(total_steps * tc.cage_silence_ratio)
        if step < silence_steps:
            return 0.0
        ramp_steps = total_steps - silence_steps
        if ramp_steps <= 0:
            return lambda_base
        progress = min(1.0, (step - silence_steps) / ramp_steps)
        return lambda_base * progress

    # Phase-aligned schedule (default)
    phase1_end = tc.phase1_steps
    phase2_end = phase1_end + tc.phase2_steps

    if step < phase1_end:
        return 0.0
    elif step < phase2_end:
        progress = (step - phase1_end) / tc.phase2_steps
        return lambda_base * progress
    else:
        return lambda_base


# ── DSA: Belief-Conditioned Lightning Indexer ──

# ── Triton kernel for fused chunk scoring ──
# Fuses: dot product + ReLU + head-weighted sum + belief bias + causal mask
# into a single kernel launch per chunk. Eliminates 5 separate kernel launches
# and reduces global memory traffic (q, w loaded once instead of per-op).

if _triton_available:
    @triton.jit
    def _fused_chunk_score_kernel(
        Q, K, W, BB, OUT,
        T,                          # sequence length (runtime)
        CHUNK_START,                # start index of current key chunk (runtime)
        CHUNK_C: tl.constexpr,      # chunk size (compile-time)
        H: tl.constexpr,            # number of indexer heads
        D: tl.constexpr,            # indexer dim per head
        HAS_BB: tl.constexpr,       # whether belief bias is provided
        BELIEF_LAMBDA,              # belief bias scaling (runtime float)
    ):
        """Fused indexer chunk scoring: one program per (b, t) query.

        Computes for each query position t and key chunk [CHUNK_START, CHUNK_START+C):
            score[c] = Σ_h w[h] * ReLU(q[h] · k[chunk_start+c, h]) + λ*bb[chunk_start+c]
        with causal masking (future keys → -inf).

        Tensors assumed contiguous:
            Q: [B*T, H, D]  K: [B*T, H, D]  W: [B*T, H]
            BB: [B*T] (flattened belief bias)  OUT: [B*T, CHUNK_C]
        """
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T

        c_off = tl.arange(0, CHUNK_C)
        key_pos = CHUNK_START + c_off      # global key positions
        valid = (key_pos < T) & (key_pos <= t)  # in-range + causal

        HD = H * D
        d_off = tl.arange(0, D)
        acc = tl.zeros([CHUNK_C], dtype=tl.float32)

        for h in range(H):
            # q[pid, h, :] → [D]
            q_vec = tl.load(Q + pid * HD + h * D + d_off).to(tl.float32)

            # k[b*T + key_pos, h, :] → [C, D]
            k_base = (b * T + key_pos)[:, None] * HD + h * D + d_off[None, :]
            k_mat = tl.load(K + k_base, mask=valid[:, None], other=0.0).to(tl.float32)

            dot = tl.sum(k_mat * q_vec[None, :], axis=1)  # [C]
            dot = tl.maximum(dot, 0.0)                      # ReLU

            w_val = tl.load(W + pid * H + h).to(tl.float32)
            acc += w_val * dot

        if HAS_BB:
            bb_idx = b * T + key_pos
            bb_vals = tl.load(BB + bb_idx, mask=(key_pos < T), other=0.0).to(tl.float32)
            acc += BELIEF_LAMBDA * bb_vals

        acc = tl.where(valid, acc, float('-inf'))
        tl.store(OUT + pid * CHUNK_C + c_off, acc)


class LightningIndexer(nn.Module):
    """Learned sparse token selector for MLA with belief conditioning.

    Scores all tokens via small projections, selects top-k for full attention.
    Indexer keys are RotorQuant-compressed via STE QAT — same pipeline as
    KV cache and weight quantization.

    Belief conditioning: active belief vectors are projected into indexer space.
    Their similarity to each token's indexer key adds an additive bias to the
    score, so MLA preferentially attends to tokens relevant to cognitive state.

    Score computation (per indexer head j):
        I(t, s) = Σ_j w(t,j) · ReLU( q_I(t,j) · k_I(s,j) )
                  + λ · max_b sim(k_I(s), belief_proj(b))

    Chunk-based scoring: to avoid materializing a [B, T, T] matrix, scores are
    computed in chunks of C keys at a time. Per-query top-k is maintained across
    chunks via torch.topk merging. Memory: O(T × C × H) instead of O(T² × H).

    Reference: DeepSeek-V3.2 (arxiv.org/abs/2512.02556)
    Reference: NSA (arxiv.org/abs/2502.11089, ACL 2025)
    """

    # Chunk size for scoring — balances memory vs kernel efficiency.
    # At d_I=32, H=4: chunk of 256 uses [B, T, 4, 256] = 0.25M elements per batch.
    SCORE_CHUNK_SIZE = 256

    def __init__(
        self,
        hidden_dim: int,
        index_dim: int = 32,
        n_heads: int = 4,
        bits: int = 3,
        belief_dim: int = 0,
        belief_lambda: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.index_dim = index_dim
        self.n_heads = n_heads
        self.bits = bits
        self.belief_lambda = belief_lambda

        # Query and key projections into small indexer space
        self.q_proj = nn.Linear(hidden_dim, n_heads * index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * index_dim, bias=False)

        # Per-head learned weights: w(t, j) — data-dependent head gating
        self.head_gate = nn.Linear(hidden_dim, n_heads, bias=False)

        # RotorQuant for indexer keys (STE QAT during training)
        self.quantizer = _make_quantizer(index_dim, bits)

        # Belief conditioning projections (if beliefs exist)
        self.has_beliefs = belief_dim > 0
        if self.has_beliefs:
            self.belief_proj = nn.Linear(belief_dim, index_dim, bias=False)

    def compute_topk(
        self,
        hidden: Tensor,
        top_k: int,
        beliefs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute indexer scores chunk-by-chunk and return top-k indices per query.

        Instead of materializing the full [B, T, T] score matrix, processes keys
        in chunks of SCORE_CHUNK_SIZE and maintains a running top-k per query.
        Memory: O(B × T × max(C, k) × H) instead of O(B × T² × H).

        Dispatches to Triton kernel when available (fuses dot product + ReLU +
        weighted sum + belief bias + causal mask into one kernel per chunk),
        falls back to PyTorch on CPU or when Triton is not installed.

        Args:
            hidden: [B, T, D] — hidden states from transformer
            top_k: number of tokens to select per query position
            beliefs: [N_active, D_belief] — active belief vectors (optional)

        Returns:
            indices: [B, T, k] — selected token indices per query
            scores_at_indices: [B, T, k] — scores at those indices
        """
        B, T, _ = hidden.shape
        device = hidden.device
        C = self.SCORE_CHUNK_SIZE
        k = min(top_k, T)

        # Project to indexer space
        q = self.q_proj(hidden).view(B, T, self.n_heads, self.index_dim)
        k_all = self.k_proj(hidden).view(B, T, self.n_heads, self.index_dim)

        # STE quantize indexer keys (RotorQuant noise during training)
        k_flat = k_all.reshape(-1, self.index_dim)
        k_flat = ste_quantize(k_flat, self.quantizer)
        k_all = k_flat.view(B, T, self.n_heads, self.index_dim)

        # Per-head gating: w(t, j) via softmax
        w = torch.softmax(self.head_gate(hidden), dim=-1)  # [B, T, n_heads]

        # Belief bias: precompute per-token bias [B, T] (column bias, shared by all queries)
        belief_bias = None
        if self.has_beliefs and beliefs is not None and beliefs.shape[0] > 0:
            belief_bias = self._compute_belief_bias(k_all, beliefs)  # [B, T]

        # Dispatch: Triton on CUDA, PyTorch fallback on CPU
        if _triton_available and hidden.is_cuda:
            return self._topk_triton(q, k_all, w, belief_bias, B, T, k, C, device)
        return self._topk_pytorch(q, k_all, w, belief_bias, B, T, k, C, device, hidden.dtype)

    def _topk_triton(
        self, q: Tensor, k_all: Tensor, w: Tensor,
        belief_bias: Tensor | None, B: int, T: int, k: int, C: int, device,
    ) -> tuple[Tensor, Tensor]:
        """Chunk-based top-k with fused Triton scoring kernel."""
        BT = B * T

        # Flatten to [B*T, H, D] contiguous for the kernel
        q_flat = q.reshape(BT, self.n_heads, self.index_dim).contiguous()
        k_flat = k_all.reshape(BT, self.n_heads, self.index_dim).contiguous()
        w_flat = w.reshape(BT, self.n_heads).contiguous()
        bb_flat = belief_bias.reshape(BT).contiguous() if belief_bias is not None else torch.empty(1, device=device)

        # Output buffer (reused each chunk — kernel writes all C positions)
        chunk_scores = torch.empty(BT, C, device=device, dtype=torch.float32)

        # Running top-k
        best_scores = torch.full((BT, k), float('-inf'), device=device, dtype=torch.float32)
        best_indices = torch.zeros(BT, k, dtype=torch.long, device=device)

        # Chunk indices buffer (padded to C, clamped to T-1 for invalid positions;
        # those positions have -inf scores so they'll never win the top-k)
        grid = (BT,)

        for chunk_start in range(0, T, C):
            _fused_chunk_score_kernel[grid](
                q_flat, k_flat, w_flat, bb_flat, chunk_scores,
                T, chunk_start,
                CHUNK_C=C,
                H=self.n_heads,
                D=self.index_dim,
                HAS_BB=(belief_bias is not None),
                BELIEF_LAMBDA=self.belief_lambda,
            )

            # Indices for this chunk (padded to C, clamped for out-of-range)
            chunk_end = min(chunk_start + C, T)
            C_actual = chunk_end - chunk_start
            chunk_idx = torch.arange(chunk_start, chunk_start + C, device=device).clamp(max=T - 1)
            chunk_idx = chunk_idx.unsqueeze(0).expand(BT, -1)

            # Merge with running top-k
            merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
            merged_indices = torch.cat([best_indices, chunk_idx], dim=-1)
            topk_scores, topk_pos = merged_scores.topk(k, dim=-1)
            best_scores = topk_scores
            best_indices = merged_indices.gather(-1, topk_pos)

        return best_indices.view(B, T, k), best_scores.view(B, T, k)

    def _topk_pytorch(
        self, q: Tensor, k_all: Tensor, w: Tensor,
        belief_bias: Tensor | None, B: int, T: int, k: int, C: int, device, dtype,
    ) -> tuple[Tensor, Tensor]:
        """Chunk-based top-k with pure PyTorch ops (CPU / no-Triton fallback)."""
        best_scores = torch.full((B, T, k), float('-inf'), device=device, dtype=dtype)
        best_indices = torch.zeros(B, T, k, dtype=torch.long, device=device)

        for chunk_start in range(0, T, C):
            chunk_end = min(chunk_start + C, T)
            C_actual = chunk_end - chunk_start

            k_chunk = k_all[:, chunk_start:chunk_end]
            scores_per_head = torch.einsum('bthd,bchd->bthc', q, k_chunk)
            scores_per_head = torch.relu(scores_per_head)
            chunk_scores = (w.unsqueeze(-1) * scores_per_head).sum(dim=2)

            if belief_bias is not None:
                chunk_scores = chunk_scores + self.belief_lambda * belief_bias[:, chunk_start:chunk_end].unsqueeze(1)

            key_positions = torch.arange(chunk_start, chunk_end, device=device)
            query_positions = torch.arange(T, device=device)
            causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            chunk_scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))

            chunk_indices = torch.arange(chunk_start, chunk_end, device=device)
            chunk_indices = chunk_indices.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

            merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
            merged_indices = torch.cat([best_indices, chunk_indices], dim=-1)
            topk_scores, topk_pos = merged_scores.topk(k, dim=-1)
            best_scores = topk_scores
            best_indices = merged_indices.gather(-1, topk_pos)

        return best_indices, best_scores

    def _compute_belief_bias(
        self,
        k: Tensor,
        beliefs: Tensor,
    ) -> Tensor:
        """Compute per-token belief relevance bias.

        Args:
            k: [B, T, n_heads, d_I] — indexer keys
            beliefs: [N_active, D_belief] — active belief vectors

        Returns:
            bias: [B, T] — max belief similarity per token
        """
        b_proj = self.belief_proj(beliefs)  # [N_active, d_I]
        k_avg = k.mean(dim=2)  # [B, T, d_I]
        sim = torch.einsum('btd,nd->btn', k_avg, b_proj)  # [B, T, N_active]
        return sim.max(dim=-1).values  # [B, T]

    def compute_dense_scores(
        self,
        hidden: Tensor,
        beliefs: Tensor | None = None,
    ) -> Tensor:
        """Compute full [B, T, T] score matrix (for KL alignment at short context).

        Only called during training when T is small enough that dense is affordable.

        Args:
            hidden: [B, T, D]
            beliefs: [N_active, D_belief] (optional)

        Returns:
            scores: [B, T, T] — full indexer score matrix with causal mask
        """
        B, T, _ = hidden.shape

        q = self.q_proj(hidden).view(B, T, self.n_heads, self.index_dim)
        k = self.k_proj(hidden).view(B, T, self.n_heads, self.index_dim)

        k_flat = k.reshape(-1, self.index_dim)
        k_flat = ste_quantize(k_flat, self.quantizer)
        k = k_flat.view(B, T, self.n_heads, self.index_dim)

        w = torch.softmax(self.head_gate(hidden), dim=-1)
        scores_per_head = torch.einsum('bthd,bshd->bths', q, k)
        scores_per_head = torch.relu(scores_per_head)
        scores = (w.unsqueeze(-1) * scores_per_head).sum(dim=2)

        if self.has_beliefs and beliefs is not None and beliefs.shape[0] > 0:
            belief_bias = self._compute_belief_bias(k, beliefs)
            scores = scores + self.belief_lambda * belief_bias.unsqueeze(1)

        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal_mask

        return scores

    def compute_kl_loss(
        self,
        indexer_scores: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """KL alignment loss: train indexer to match dense attention distribution.

        Args:
            indexer_scores: [B, T, T] — raw indexer scores (logits)
            attn_weights: [B, H, T, T] — dense attention weights from full MLA

        Returns:
            KL divergence loss (scalar)
        """
        # Aggregate dense attention across heads (L1-normalized as in DSA paper)
        target = attn_weights.sum(dim=1)  # [B, T, T]
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

        # Replace -inf with large negative finite value for log_softmax
        scores_safe = indexer_scores.clone()
        scores_safe[scores_safe == float('-inf')] = -1e9

        log_pred = torch.log_softmax(scores_safe, dim=-1)
        nonzero = target > 1e-10
        kl_elem = torch.zeros_like(target)
        kl_elem[nonzero] = target[nonzero] * (
            target[nonzero].log() - log_pred[nonzero]
        )
        return kl_elem.sum(dim=-1).mean()


def gather_sparse_kv(
    k: Tensor,
    v: Tensor,
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather K,V at selected token indices for sparse attention.

    Args:
        k: [B, T, H, D] — full key tensor
        v: [B, T, H, D] — full value tensor
        indices: [B, T_q, top_k] — selected indices per query

    Returns:
        k_sparse: [B, T_q, top_k, H, D]
        v_sparse: [B, T_q, top_k, H, D]
    """
    B, T, H, D = k.shape
    T_q = indices.shape[1]
    top_k = indices.shape[2]

    # Flatten indices for gather: [B, T_q * top_k]
    flat_idx = indices.reshape(B, -1)  # [B, T_q * top_k]

    # Expand for multi-head gather: [B, T_q*top_k, H, D]
    flat_idx_exp = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)

    # Gather from full tensors
    k_gathered = k.gather(1, flat_idx_exp)  # [B, T_q*top_k, H, D]
    v_gathered = v.gather(1, flat_idx_exp)

    # Reshape to per-query groups
    k_sparse = k_gathered.view(B, T_q, top_k, H, D)
    v_sparse = v_gathered.view(B, T_q, top_k, H, D)

    return k_sparse, v_sparse
