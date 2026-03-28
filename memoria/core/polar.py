"""Polar representation for beliefs.

Beliefs are stored in polar form where:
- Radius = precision (confidence). Large radius = high confidence, dominates message passing.
- Angle = content (what the belief is about). Unit vector in representation space.

This comes from TurboQuant's insight: PolarQuant decomposes vectors into radius + angle,
where angle patterns are concentrated and compress well. In our case, precision weighting
is free from the geometry — dot products between beliefs are naturally precision-weighted
because large-radius vectors dominate.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

EPSILON = 1e-10


def to_polar(cartesian: Tensor) -> tuple[Tensor, Tensor]:
    """Convert cartesian vectors to (radius, angle) polar form.

    Args:
        cartesian: [..., D] tensor of belief vectors

    Returns:
        radius: [...] tensor of magnitudes (precision)
        angle: [..., D] tensor of unit vectors (content direction)
    """
    radius = cartesian.norm(dim=-1, keepdim=False)
    angle = F.normalize(cartesian, dim=-1, eps=EPSILON)
    return radius, angle


def to_cartesian(radius: Tensor, angle: Tensor) -> Tensor:
    """Convert (radius, angle) polar form back to cartesian.

    Args:
        radius: [...] tensor of magnitudes
        angle: [..., D] tensor of unit vectors

    Returns:
        cartesian: [..., D] tensor
    """
    return radius.unsqueeze(-1) * angle


def angular_distance(a: Tensor, b: Tensor) -> Tensor:
    """Compute angular distance between unit vectors.

    Returns value in [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite.

    Args:
        a: [..., D] unit vectors
        b: [..., D] unit vectors

    Returns:
        distance: [...] tensor
    """
    cos_sim = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return 1.0 - cos_sim


def angular_similarity(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between unit vectors.

    Returns value in [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite.

    Args:
        a: [..., D] unit vectors
        b: [..., D] unit vectors

    Returns:
        similarity: [...] tensor
    """
    return (a * b).sum(dim=-1).clamp(-1.0, 1.0)


def precision_weighted_average(
    angles: Tensor, radii: Tensor, dim: int = 0
) -> tuple[Tensor, Tensor]:
    """Compute precision-weighted average of beliefs in polar form.

    Higher-radius (more precise) beliefs contribute more to the average.
    Used for consolidation (merging beliefs) and message fusion.

    Args:
        angles: [N, D] unit vectors
        radii: [N] magnitudes

    Returns:
        avg_angle: [D] unit vector (weighted direction)
        combined_radius: scalar (combined precision = sqrt(sum of squared radii))
    """
    weights = radii.unsqueeze(-1)  # [N, 1]
    weighted_sum = (weights * angles).sum(dim=dim)  # [D]
    avg_angle = F.normalize(weighted_sum.unsqueeze(0), dim=-1, eps=EPSILON).squeeze(0)
    combined_radius = radii.square().sum(dim=dim).sqrt()  # sqrt(Σr²)
    return avg_angle, combined_radius


def belief_is_active(radius: Tensor, threshold: float = EPSILON) -> Tensor:
    """Check which beliefs are active (allocated, non-empty).

    A belief with radius ≈ 0 is considered empty/inactive.

    Args:
        radius: [...] tensor of belief radii

    Returns:
        mask: [...] boolean tensor
    """
    return radius > threshold
