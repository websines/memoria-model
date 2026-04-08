"""Quantization via RotorQuant block-diagonal rotations + CAGE weight QAT.

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

Reference: RotorQuant — scrya.com/rotorquant.pdf (March 2026)
Reference: TurboQuant — arxiv.org/abs/2504.19874 (ICLR 2026)
Reference: CAGE — arxiv.org/abs/2510.18784 (IST-DASLab 2025)
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
