"""KV cache and belief quantization via RotorQuant block-diagonal rotations.

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

Reference: RotorQuant — scrya.com/rotorquant.pdf (March 2026)
Reference: TurboQuant — arxiv.org/abs/2504.19874 (ICLR 2026)
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
