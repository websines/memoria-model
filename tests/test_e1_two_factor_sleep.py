"""Tests for E1: Two-Factor Sleep Consolidation."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.two_factor_sleep import (
    homeostatic_scaling,
    conflict_scan,
    run_two_factor_sleep,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def test_homeostatic_empty_state(state):
    """Homeostatic scaling handles empty state."""
    stats = homeostatic_scaling(
        state,
        target=state.meta_params.homeostatic_target,
        rate=state.meta_params.homeostatic_rate,
    )
    assert stats['actual_total'] == 0.0
    assert stats['scale_applied'] == 1.0


def test_homeostatic_deflation(state):
    """Homeostatic scaling deflates when total precision exceeds target."""
    # Allocate beliefs with high radius (total >> target)
    for _ in range(20):
        state.allocate_belief(torch.randn(64) * 10.0)  # radius ~10 each

    total_before = state.beliefs.data[state.get_active_mask()].norm(dim=-1).sum().item()
    target = torch.tensor(50.0)  # much lower than actual ~200
    rate = torch.tensor(0.5)

    stats = homeostatic_scaling(state, target=target, rate=rate)
    total_after = state.beliefs.data[state.get_active_mask()].norm(dim=-1).sum().item()

    assert stats['scale_applied'] < 1.0
    assert total_after < total_before


def test_homeostatic_inflation(state):
    """Homeostatic scaling inflates when total precision is below target."""
    for _ in range(20):
        state.allocate_belief(torch.randn(64) * 0.1)  # very low radius

    total_before = state.beliefs.data[state.get_active_mask()].norm(dim=-1).sum().item()
    target = torch.tensor(100.0)  # much higher than actual ~2
    rate = torch.tensor(0.5)

    stats = homeostatic_scaling(state, target=target, rate=rate)
    total_after = state.beliefs.data[state.get_active_mask()].norm(dim=-1).sum().item()

    assert stats['scale_applied'] > 1.0
    assert total_after > total_before


def test_homeostatic_preserves_immutable(state):
    """Immutable beliefs are not scaled."""
    slot = state.allocate_belief(torch.randn(64) * 5.0)
    state.immutable_beliefs[slot] = True
    radius_before = state.beliefs.data[slot].norm().item()

    homeostatic_scaling(state, target=torch.tensor(0.1), rate=torch.tensor(0.9))
    radius_after = state.beliefs.data[slot].norm().item()

    assert abs(radius_before - radius_after) < 1e-6


def test_homeostatic_scale_clamped(state):
    """Scale factor is clamped to [0.5, 2.0]."""
    state.allocate_belief(torch.randn(64) * 0.001)  # tiny radius

    stats = homeostatic_scaling(
        state,
        target=torch.tensor(10000.0),  # enormous target
        rate=torch.tensor(1.0),
    )
    assert stats['scale_applied'] <= 2.0


def test_conflict_scan_empty(state):
    """Conflict scan handles empty state."""
    stats = conflict_scan(state, conflict_threshold=torch.tensor(0.85))
    assert stats['conflicts_found'] == 0


def test_conflict_scan_no_conflicts(state):
    """Orthogonal beliefs produce no conflicts."""
    # Allocate orthogonal-ish beliefs
    for i in range(5):
        v = torch.zeros(64)
        v[i * 10] = 1.0
        state.allocate_belief(v)

    stats = conflict_scan(state, conflict_threshold=torch.tensor(0.85))
    assert stats['conflicts_found'] == 0


def test_conflict_scan_detects_duplicates(state):
    """Near-duplicate beliefs are detected as conflicts."""
    base = torch.randn(64)
    base = base / base.norm()
    # Two near-identical beliefs
    state.allocate_belief(base * 5.0)
    state.allocate_belief((base + torch.randn(64) * 0.01) * 3.0)

    stats = conflict_scan(state, conflict_threshold=torch.tensor(0.9))
    assert stats['conflicts_found'] >= 1
    assert stats['beliefs_weakened'] >= 1


def test_conflict_scan_weakens_lower_precision(state):
    """The lower-precision belief is weakened in a conflict."""
    base = torch.randn(64)
    base = base / base.norm()
    slot_strong = state.allocate_belief(base * 10.0)  # high precision
    slot_weak = state.allocate_belief(base * 2.0)     # low precision

    weak_radius_before = state.beliefs.data[slot_weak].norm().item()

    conflict_scan(state, conflict_threshold=torch.tensor(0.9))
    weak_radius_after = state.beliefs.data[slot_weak].norm().item()

    # Weak belief should have been reduced
    assert weak_radius_after < weak_radius_before


def test_conflict_scan_preserves_immutable(state):
    """Immutable beliefs are not weakened by conflict scan."""
    base = torch.randn(64)
    base = base / base.norm()
    slot = state.allocate_belief(base * 2.0)
    state.immutable_beliefs[slot] = True
    state.allocate_belief(base * 10.0)

    radius_before = state.beliefs.data[slot].norm().item()
    conflict_scan(state, conflict_threshold=torch.tensor(0.9))
    radius_after = state.beliefs.data[slot].norm().item()

    assert abs(radius_before - radius_after) < 1e-6


def test_run_two_factor_sleep_empty(state):
    """Full two-factor sleep handles empty state."""
    stats = run_two_factor_sleep(state, current_step=100)
    assert 'homeostatic' in stats
    assert 'conflicts' in stats
    assert 'replay_candidates' in stats


def test_run_two_factor_sleep_with_beliefs(state):
    """Full two-factor sleep runs with active beliefs."""
    for i in range(10):
        state.allocate_belief(torch.randn(64) * (0.5 + i * 0.1))

    stats = run_two_factor_sleep(state, current_step=100)
    assert stats['homeostatic']['actual_total'] > 0
    assert stats['replay_candidates']['recent_count'] + stats['replay_candidates']['old_count'] == 10


def test_metaparams_e1(state):
    """MetaParams has E1 parameters."""
    assert state.meta_params.homeostatic_target.item() > 0
    assert 0 < state.meta_params.homeostatic_rate.item() < 1
    assert 0 < state.meta_params.sleep_conflict_threshold.item() < 1
