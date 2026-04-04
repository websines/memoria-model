"""Tests for E2: Self-Verification Pass."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.self_verification import (
    verify_belief_consistency,
    supersession_scan,
    run_self_verification,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def test_verify_no_neighbors(state):
    """Verification with no causal neighbors returns empty stats."""
    slot = state.allocate_belief(torch.randn(64))
    stats = verify_belief_consistency(
        state, slot,
        divergence_threshold=torch.tensor(0.3),
        precision_decay=torch.tensor(0.2),
    )
    assert stats['neighbors_checked'] == 0
    assert stats['inconsistencies_found'] == 0


def test_verify_consistent_neighbors(state):
    """Consistent causal neighbors pass verification."""
    # Create two similar beliefs with a causal edge
    base = torch.randn(64)
    base = base / base.norm()
    slot_a = state.allocate_belief(base * 5.0)
    slot_b = state.allocate_belief((base + torch.randn(64) * 0.1) * 3.0)

    # Add causal edge a → b
    edge_slot = state.allocate_edge(slot_a, slot_b, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge_slot] = 5  # mark as causal

    stats = verify_belief_consistency(
        state, slot_a,
        divergence_threshold=torch.tensor(0.3),  # low threshold = lenient
        precision_decay=torch.tensor(0.2),
    )
    assert stats['neighbors_checked'] == 1
    # Consistent neighbors should not produce inconsistency
    assert stats['inconsistencies_found'] == 0


def test_verify_inconsistent_neighbors(state):
    """Inconsistent causal neighbors are flagged."""
    # Create two orthogonal beliefs with a causal edge
    a = torch.zeros(64)
    a[0] = 5.0
    b = torch.zeros(64)
    b[32] = 3.0
    slot_a = state.allocate_belief(a)
    slot_b = state.allocate_belief(b)

    edge_slot = state.allocate_edge(slot_a, slot_b, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge_slot] = 5

    stats = verify_belief_consistency(
        state, slot_a,
        divergence_threshold=torch.tensor(0.5),  # high threshold = strict
        precision_decay=torch.tensor(0.2),
    )
    assert stats['neighbors_checked'] == 1
    assert stats['inconsistencies_found'] == 1


def test_verify_weakens_weaker_belief(state):
    """Inconsistency weakens the lower-precision belief."""
    a = torch.zeros(64)
    a[0] = 10.0  # high precision source
    b = torch.zeros(64)
    b[32] = 2.0  # low precision target
    slot_a = state.allocate_belief(a)
    slot_b = state.allocate_belief(b)

    edge_slot = state.allocate_edge(slot_a, slot_b, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge_slot] = 5

    radius_before = state.beliefs.data[slot_b].norm().item()

    verify_belief_consistency(
        state, slot_a,
        divergence_threshold=torch.tensor(0.5),
        precision_decay=torch.tensor(0.3),
    )

    radius_after = state.beliefs.data[slot_b].norm().item()
    assert radius_after < radius_before


def test_verify_boosts_variance_on_inconsistency(state):
    """Inconsistency boosts MESU variance of weakened belief."""
    a = torch.zeros(64)
    a[0] = 10.0
    b = torch.zeros(64)
    b[32] = 2.0
    slot_a = state.allocate_belief(a)
    slot_b = state.allocate_belief(b)

    edge_slot = state.allocate_edge(slot_a, slot_b, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge_slot] = 5

    var_before = state.belief_precision_var[slot_b].item()

    verify_belief_consistency(
        state, slot_a,
        divergence_threshold=torch.tensor(0.5),
        precision_decay=torch.tensor(0.3),
    )

    var_after = state.belief_precision_var[slot_b].item()
    assert var_after > var_before


def test_supersession_empty(state):
    """Supersession scan handles empty state."""
    stats = supersession_scan(state, supersession_sim=torch.tensor(0.85))
    assert stats['supersessions_found'] == 0


def test_supersession_detects(state):
    """Supersession detected when newer belief is stronger."""
    base = torch.randn(64)
    base = base / base.norm()
    # Old belief: low precision, created early
    slot_old = state.allocate_belief(base * 2.0, step=10)
    # New belief: high precision, created late
    slot_new = state.allocate_belief((base + torch.randn(64) * 0.01) * 8.0, step=100)

    stats = supersession_scan(state, supersession_sim=torch.tensor(0.9))
    assert stats['supersessions_found'] >= 1

    # Old belief should be weakened
    old_radius = state.beliefs.data[slot_old].norm().item()
    assert old_radius < 2.0  # was 2.0, should be reduced


def test_supersession_preserves_immutable(state):
    """Immutable beliefs are not superseded."""
    base = torch.randn(64)
    base = base / base.norm()
    slot_old = state.allocate_belief(base * 2.0, step=10)
    state.immutable_beliefs[slot_old] = True
    state.allocate_belief((base + torch.randn(64) * 0.01) * 8.0, step=100)

    radius_before = state.beliefs.data[slot_old].norm().item()
    supersession_scan(state, supersession_sim=torch.tensor(0.9))
    radius_after = state.beliefs.data[slot_old].norm().item()

    assert abs(radius_before - radius_after) < 1e-6


def test_run_self_verification_empty(state):
    """Full self-verification handles empty state."""
    stats = run_self_verification(state)
    assert stats['beliefs_verified'] == 0


def test_run_self_verification_with_beliefs(state):
    """Full self-verification runs with active beliefs."""
    for i in range(10):
        state.allocate_belief(torch.randn(64) * (1.0 + i * 0.5))

    # Add some causal edges
    state.allocate_edge(0, 1, torch.randn(64), weight=0.5)
    state.edge_causal_obs[0] = 3

    stats = run_self_verification(state)
    assert stats['beliefs_verified'] > 0


def test_metaparams_e2(state):
    """MetaParams has E2 parameters."""
    assert 0 < state.meta_params.verification_divergence_threshold.item() < 1
    assert 0 < state.meta_params.verification_precision_decay.item() < 1
    assert 0 < state.meta_params.supersession_similarity.item() < 1
