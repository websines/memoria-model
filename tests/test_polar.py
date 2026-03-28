"""Tests for polar representation utilities."""

import torch
import pytest
from memoria.core.polar import (
    to_polar, to_cartesian, angular_distance, angular_similarity,
    precision_weighted_average, belief_is_active, EPSILON,
)


def test_roundtrip():
    """Cartesian → polar → cartesian should be identity."""
    v = torch.randn(5, 256)
    r, a = to_polar(v)
    v_reconstructed = to_cartesian(r, a)
    assert torch.allclose(v, v_reconstructed, atol=1e-5)


def test_radius_is_norm():
    """Radius should equal vector norm."""
    v = torch.randn(10, 128)
    r, _ = to_polar(v)
    expected = v.norm(dim=-1)
    assert torch.allclose(r, expected, atol=1e-6)


def test_angle_is_unit():
    """Angles should be unit vectors."""
    v = torch.randn(10, 128)
    _, a = to_polar(v)
    norms = a.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_angular_distance_identical():
    """Identical vectors have zero angular distance."""
    a = torch.randn(5, 64)
    a = torch.nn.functional.normalize(a, dim=-1)
    dist = angular_distance(a, a)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-6)


def test_angular_distance_opposite():
    """Opposite vectors have distance 2."""
    a = torch.randn(5, 64)
    a = torch.nn.functional.normalize(a, dim=-1)
    dist = angular_distance(a, -a)
    assert torch.allclose(dist, 2.0 * torch.ones_like(dist), atol=1e-5)


def test_angular_similarity_range():
    """Cosine similarity should be in [-1, 1]."""
    a = torch.randn(100, 64)
    b = torch.randn(100, 64)
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    sim = angular_similarity(a, b)
    assert (sim >= -1.0 - 1e-6).all()
    assert (sim <= 1.0 + 1e-6).all()


def test_precision_weighted_average_single():
    """Weighted average of a single belief is itself."""
    angle = torch.nn.functional.normalize(torch.randn(1, 128), dim=-1)
    radius = torch.tensor([3.0])
    avg_a, avg_r = precision_weighted_average(angle, radius, dim=0)
    assert torch.allclose(avg_a, angle.squeeze(0), atol=1e-5)
    assert torch.allclose(avg_r, radius[0], atol=1e-5)


def test_precision_weighted_average_dominance():
    """High-precision belief should dominate the average."""
    a1 = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    a2 = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1)
    angles = torch.cat([a1, a2], dim=0)

    # Belief 1 has 100x the precision
    radii = torch.tensor([100.0, 1.0])
    avg_a, avg_r = precision_weighted_average(angles, radii, dim=0)

    # Average should be much closer to belief 1
    sim_to_1 = angular_similarity(avg_a.unsqueeze(0), a1).item()
    sim_to_2 = angular_similarity(avg_a.unsqueeze(0), a2).item()
    assert sim_to_1 > sim_to_2


def test_precision_weighted_average_combined_radius():
    """Combined radius should be sqrt(sum of squared radii)."""
    radii = torch.tensor([3.0, 4.0])
    angles = torch.nn.functional.normalize(torch.randn(2, 64), dim=-1)
    _, combined_r = precision_weighted_average(angles, radii, dim=0)
    expected = torch.sqrt(torch.tensor(3.0**2 + 4.0**2))
    assert torch.allclose(combined_r, expected, atol=1e-5)


def test_belief_is_active():
    """Active beliefs have radius > epsilon."""
    radii = torch.tensor([0.0, 0.001, 1.0, 5.0, 1e-11])
    mask = belief_is_active(radii)
    assert mask.tolist() == [False, True, True, True, False]
