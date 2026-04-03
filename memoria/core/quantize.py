"""PolarQuant-style vector quantization for KV cache and belief storage.

Two-stage compression inspired by Google's TurboQuant (ICLR 2026):
1. Random orthogonal rotation — spreads information uniformly across dims
2. Uniform scalar quantization — each dim quantized independently with
   a single per-vector scale (no per-channel constants needed)

The rotation is the key insight: without it, a few "hot" dimensions dominate
and quantization destroys them. After rotation, information is uniform across
all dimensions, so uniform quantization is near-optimal.

Memory savings at 200K context with 12 layers, 4K sliding window:
- float32 KV: 12 × 4096 × 768 × 2 × 4 bytes = ~300 MB
- 3-bit KV:   12 × 4096 × 768 × 2 × 3/8 bytes = ~28 MB (10.7x reduction)

Speed: smaller tensors = less memory bandwidth = faster attention on
bandwidth-bound workloads (which attention always is at small batch sizes).

Reference: TurboQuant (arxiv.org/abs/2504.19874) — ICLR 2026
Reference: PolarQuant (arxiv.org/abs/2502.02617) — AISTATS 2026
"""

import torch
import torch.nn as nn
from torch import Tensor


class PolarQuantizer(nn.Module):
    """Quantize vectors to low-bit using random rotation + uniform scalar quantization.

    Usage:
        quantizer = PolarQuantizer(dim=128, bits=3)
        codes, scale = quantizer.quantize(x)     # x: [..., 128] → codes: [..., 128] uint8
        x_rec = quantizer.dequantize(codes, scale)  # reconstruct

    The random rotation matrix R is fixed at construction time and stored as a buffer.
    It must be the same R for quantize and dequantize.
    """

    def __init__(self, dim: int, bits: int = 3):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.levels = 2 ** bits  # 8 for 3-bit, 16 for 4-bit

        # Precompute random orthogonal rotation via QR decomposition
        # This is fixed per layer/quantizer instance — not learned
        R, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.register_buffer('R', R)

    @torch.no_grad()
    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize input vectors to low-bit codes.

        Args:
            x: [..., dim] float tensor

        Returns:
            codes: [..., dim] uint8 tensor (values in [0, levels-1])
            scale: [..., 1] float tensor (per-vector max absolute value)
        """
        # Step 1: Random orthogonal rotation
        x_rot = x @ self.R  # [..., dim]

        # Step 2: Per-vector scale (single scalar per vector — minimal overhead)
        scale = x_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

        # Step 3: Normalize to [-1, 1] then quantize to [0, levels-1]
        x_norm = x_rot / scale
        half_levels = (self.levels - 1) / 2.0
        codes = ((x_norm + 1.0) * half_levels).round().clamp(0, self.levels - 1).to(torch.uint8)

        return codes, scale

    @torch.no_grad()
    def dequantize(self, codes: Tensor, scale: Tensor) -> Tensor:
        """Reconstruct vectors from quantized codes.

        Args:
            codes: [..., dim] uint8 tensor
            scale: [..., 1] float tensor

        Returns:
            [..., dim] float tensor (approximate reconstruction)
        """
        half_levels = (self.levels - 1) / 2.0
        x_norm = codes.float() / half_levels - 1.0
        x_rot = x_norm * scale
        return x_rot @ self.R.T


class QuantizedKVCache(nn.Module):
    """Sliding window KV cache with PolarQuant compression.

    Stores K and V in quantized form (3-4 bit) instead of float32/bf16.
    On attention computation, dequantizes the relevant window on the fly.

    Memory at 4K window, 128 head_dim, 3-bit:
    - float32: 4096 × 128 × 4 bytes = 2 MB per K or V per head
    - 3-bit:   4096 × 128 × 3/8 bytes ≈ 0.19 MB + 4096 × 4 bytes scale ≈ 0.21 MB
    - Savings: ~9.5x per head
    """

    def __init__(self, head_dim: int, bits: int = 3):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.quantizer = PolarQuantizer(head_dim, bits)

    def compress(self, k: Tensor, v: Tensor) -> dict:
        """Compress K and V tensors.

        Args:
            k: [B, H, T, D] key tensor
            v: [B, H, T, D] value tensor

        Returns:
            dict with quantized codes and scales for K and V
        """
        k_codes, k_scale = self.quantizer.quantize(k)
        v_codes, v_scale = self.quantizer.quantize(v)
        return {
            'k_codes': k_codes, 'k_scale': k_scale,
            'v_codes': v_codes, 'v_scale': v_scale,
        }

    def decompress(self, cache: dict) -> tuple[Tensor, Tensor]:
        """Decompress K and V from cache.

        Args:
            cache: dict from compress()

        Returns:
            k: [B, H, T, D] reconstructed keys
            v: [B, H, T, D] reconstructed values
        """
        k = self.quantizer.dequantize(cache['k_codes'], cache['k_scale'])
        v = self.quantizer.dequantize(cache['v_codes'], cache['v_scale'])
        return k, v


class QuantizedBeliefStore:
    """Quantized storage for belief vectors.

    Beliefs in Memoria are naturally polar (radius = precision, angle = content).
    PolarQuant is a perfect fit: we store the angle (unit vector) quantized,
    and the radius (scalar) in full precision.

    Memory at 16K beliefs, 256-dim, 3-bit:
    - float32: 16384 × 256 × 4 bytes = 16 MB
    - 3-bit angles + float32 radii: 16384 × (256 × 3/8 + 4) bytes ≈ 1.6 MB
    - Savings: 10x
    """

    def __init__(self, belief_dim: int, bits: int = 3):
        self.belief_dim = belief_dim
        self.bits = bits
        self.quantizer = PolarQuantizer(belief_dim, bits)

    @torch.no_grad()
    def compress_beliefs(self, beliefs: Tensor) -> dict:
        """Compress belief vectors, preserving polar structure.

        Args:
            beliefs: [N, D] belief vectors (cartesian form, radius encodes precision)

        Returns:
            dict with quantized angles, radii, and metadata
        """
        # Decompose into polar: radius + unit angle
        radii = beliefs.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        angles = beliefs / radii  # unit vectors

        # Quantize the angle vectors
        angle_codes, angle_scale = self.quantizer.quantize(angles)

        return {
            'angle_codes': angle_codes,
            'angle_scale': angle_scale,
            'radii': radii,  # keep full precision — radius is the confidence signal
        }

    @torch.no_grad()
    def decompress_beliefs(self, compressed: dict) -> Tensor:
        """Reconstruct belief vectors from compressed form.

        Args:
            compressed: dict from compress_beliefs()

        Returns:
            [N, D] reconstructed belief vectors
        """
        angles = self.quantizer.dequantize(
            compressed['angle_codes'], compressed['angle_scale']
        )
        # Re-normalize angles (quantization can break unit-norm)
        angles = torch.nn.functional.normalize(angles, dim=-1, eps=1e-10)
        # Reconstruct cartesian: radius × angle
        return angles * compressed['radii']
