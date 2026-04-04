"""Tests for E3: Empirical Precision Recalibration."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.precision_recalibration import (
    record_confirmation,
    record_contradiction,
    compute_empirical_precision,
    recalibrate_beliefs,
    run_precision_recalibration,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def test_record_confirmation(state):
    """Recording confirmations increments count."""
    slot = state.allocate_belief(torch.randn(64))
    assert state.belief_confirmed_count[slot].item() == 0

    record_confirmation(state, slot)
    assert state.belief_confirmed_count[slot].item() == 1

    record_confirmation(state, slot)
    assert state.belief_confirmed_count[slot].item() == 2


def test_record_contradiction(state):
    """Recording contradictions increments count."""
    slot = state.allocate_belief(torch.randn(64))
    assert state.belief_contradicted_count[slot].item() == 0

    record_contradiction(state, slot)
    assert state.belief_contradicted_count[slot].item() == 1


def test_empirical_precision_no_data():
    """No observations returns 0.5 (neutral)."""
    confirmed = torch.zeros(5, dtype=torch.long)
    contradicted = torch.zeros(5, dtype=torch.long)
    emp = compute_empirical_precision(confirmed, contradicted)
    assert torch.allclose(emp, torch.full((5,), 0.5))


def test_empirical_precision_all_confirmed():
    """All confirmed returns 1.0."""
    confirmed = torch.tensor([10, 20, 5], dtype=torch.long)
    contradicted = torch.zeros(3, dtype=torch.long)
    emp = compute_empirical_precision(confirmed, contradicted)
    assert torch.allclose(emp, torch.ones(3))


def test_empirical_precision_all_contradicted():
    """All contradicted returns 0.0."""
    confirmed = torch.zeros(3, dtype=torch.long)
    contradicted = torch.tensor([10, 20, 5], dtype=torch.long)
    emp = compute_empirical_precision(confirmed, contradicted)
    assert torch.allclose(emp, torch.zeros(3))


def test_empirical_precision_mixed():
    """Mixed observations returns correct ratio."""
    confirmed = torch.tensor([3], dtype=torch.long)
    contradicted = torch.tensor([7], dtype=torch.long)
    emp = compute_empirical_precision(confirmed, contradicted)
    assert abs(emp[0].item() - 0.3) < 1e-6


def test_recalibrate_empty_state(state):
    """Recalibration handles empty state."""
    stats = recalibrate_beliefs(
        state,
        rate=torch.tensor(0.1),
        min_samples=torch.tensor(5.0),
    )
    assert stats['beliefs_checked'] == 0


def test_recalibrate_below_min_samples(state):
    """Beliefs with too few observations are not recalibrated."""
    slot = state.allocate_belief(torch.randn(64) * 5.0)
    record_confirmation(state, slot)  # only 1 observation

    stats = recalibrate_beliefs(
        state,
        rate=torch.tensor(0.1),
        min_samples=torch.tensor(5.0),  # need 5
    )
    assert stats['beliefs_checked'] == 0


def test_recalibrate_overconfident(state):
    """Overconfident beliefs get their precision reduced."""
    slot = state.allocate_belief(torch.randn(64) * 5.0)  # high radius
    # Build empirical record: only 30% confirmed → overconfident
    for _ in range(3):
        record_confirmation(state, slot)
    for _ in range(7):
        record_contradiction(state, slot)

    # Need running_stats to have reference precision
    state.running_stats.mean_precision.fill_(1.0)

    radius_before = state.beliefs.data[slot].norm().item()

    stats = recalibrate_beliefs(
        state,
        rate=torch.tensor(0.5),
        min_samples=torch.tensor(5.0),
    )

    radius_after = state.beliefs.data[slot].norm().item()
    assert stats['beliefs_recalibrated'] >= 1
    assert radius_after < radius_before


def test_recalibrate_accurate_belief_unchanged(state):
    """Accurately calibrated beliefs are not changed."""
    slot = state.allocate_belief(torch.randn(64) * 0.5)  # low radius
    # Perfect confirmation record
    for _ in range(10):
        record_confirmation(state, slot)

    state.running_stats.mean_precision.fill_(1.0)
    radius_before = state.beliefs.data[slot].norm().item()

    recalibrate_beliefs(
        state,
        rate=torch.tensor(0.5),
        min_samples=torch.tensor(5.0),
    )

    radius_after = state.beliefs.data[slot].norm().item()
    # Low radius (0.5) with perfect empirical (1.0) → not overconfident → no change
    assert abs(radius_before - radius_after) < 1e-6


def test_recalibrate_boosts_variance(state):
    """Recalibrated beliefs get MESU variance boost."""
    slot = state.allocate_belief(torch.randn(64) * 5.0)
    for _ in range(3):
        record_confirmation(state, slot)
    for _ in range(7):
        record_contradiction(state, slot)

    state.running_stats.mean_precision.fill_(1.0)
    var_before = state.belief_precision_var[slot].item()

    recalibrate_beliefs(
        state,
        rate=torch.tensor(0.5),
        min_samples=torch.tensor(5.0),
    )

    var_after = state.belief_precision_var[slot].item()
    assert var_after > var_before


def test_recalibrate_preserves_immutable(state):
    """Immutable beliefs are not recalibrated."""
    slot = state.allocate_belief(torch.randn(64) * 5.0)
    state.immutable_beliefs[slot] = True
    for _ in range(7):
        record_contradiction(state, slot)
    for _ in range(3):
        record_confirmation(state, slot)

    state.running_stats.mean_precision.fill_(1.0)
    radius_before = state.beliefs.data[slot].norm().item()

    recalibrate_beliefs(
        state,
        rate=torch.tensor(0.5),
        min_samples=torch.tensor(5.0),
    )

    radius_after = state.beliefs.data[slot].norm().item()
    assert abs(radius_before - radius_after) < 1e-6


def test_run_precision_recalibration(state):
    """Full recalibration run works."""
    for i in range(5):
        slot = state.allocate_belief(torch.randn(64) * 3.0)
        for _ in range(10):
            record_confirmation(state, slot)

    stats = run_precision_recalibration(state)
    assert 'beliefs_checked' in stats
    assert 'beliefs_recalibrated' in stats


def test_e3_buffers_on_state(state):
    """E3 buffers exist on CognitiveState."""
    assert hasattr(state, 'belief_confirmed_count')
    assert hasattr(state, 'belief_contradicted_count')
    assert state.belief_confirmed_count.shape == (state.config.max_beliefs,)


def test_e3_serialization_roundtrip(state):
    """E3 buffers survive serialization."""
    slot = state.allocate_belief(torch.randn(64))
    state.belief_confirmed_count[slot] = 42
    state.belief_contradicted_count[slot] = 7

    checkpoint = state.state_dict_cognitive(compress=False)
    state2 = CognitiveState(state.config)
    state2.load_state_cognitive(checkpoint)

    assert state2.belief_confirmed_count[slot].item() == 42
    assert state2.belief_contradicted_count[slot].item() == 7


def test_metaparams_e3(state):
    """MetaParams has E3 parameters."""
    assert 0 < state.meta_params.recalibration_rate.item() < 1
    assert state.meta_params.recalibration_min_samples.item() > 0
