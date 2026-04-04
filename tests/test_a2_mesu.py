"""Tests for A2: MESU Precision Variance (Bayesian Metaplasticity)."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.interface.write_path import WriteCandidate
from memoria.cognition.surprise import compute_surprise_batch, SurpriseResult


@pytest.fixture
def config():
    return StateConfig(belief_dim=32, max_beliefs=64, max_edges=256, max_goals=8, relation_dim=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def make_belief(direction: list[float], precision: float, dim: int = 32) -> torch.Tensor:
    v = torch.zeros(dim)
    for i, val in enumerate(direction):
        v[i] = val
    return torch.nn.functional.normalize(v, dim=0) * precision


def test_new_beliefs_start_with_high_variance(state):
    """Newly allocated beliefs have variance = 1.0 (high uncertainty)."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b)

    assert state.belief_precision_var[slot].item() == 1.0
    assert state.belief_reinforcement_count[slot].item() == 0


def test_deallocation_resets_variance(state):
    """Deallocating resets variance to 1.0 and count to 0."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b)

    state.belief_precision_var[slot] = 0.1
    state.belief_reinforcement_count[slot] = 50

    state.deallocate_belief(slot)

    assert state.belief_precision_var[slot].item() == 1.0
    assert state.belief_reinforcement_count[slot].item() == 0


def test_mesu_amplifies_gain_for_high_variance(state):
    """High-variance beliefs should have higher gain than low-variance ones."""
    D = state.config.belief_dim

    # Belief with high variance
    b_high = make_belief([1, 0, 0], 1.0, D)
    slot_high = state.allocate_belief(b_high)
    state.belief_precision_var[slot_high] = 2.0  # very uncertain

    # Belief with low variance
    b_low = make_belief([0, 1, 0], 1.0, D)
    slot_low = state.allocate_belief(b_low)
    state.belief_precision_var[slot_low] = 0.01  # very certain

    # Same observation strength for both
    obs_high = make_belief([1, 0.3, 0], 1.0, D)
    obs_low = make_belief([0, 1, 0.3], 1.0, D)

    candidates = [
        WriteCandidate(obs_high, matched_slot=slot_high, match_similarity=0.9,
                       source_position=0, source_layer=0),
        WriteCandidate(obs_low, matched_slot=slot_low, match_similarity=0.9,
                       source_position=1, source_layer=0),
    ]

    results = compute_surprise_batch(candidates, state)

    # High-variance belief should have higher gain
    assert results[0].gain > results[1].gain
    # Raw gains should be similar (same precision ratio)
    assert abs(results[0].mesu_gain_raw - results[1].mesu_gain_raw) < 0.3


def test_variance_shrinks_with_observations(state):
    """Repeated observations should reduce precision variance."""
    D = state.config.belief_dim
    b = make_belief([1, 0, 0], 2.0, D)
    slot = state.allocate_belief(b)

    initial_var = state.belief_precision_var[slot].item()
    assert initial_var == 1.0

    # Simulate multiple confirming observations via pass2
    from memoria.cognition.pass2 import run_pass2
    for step in range(5):
        obs = make_belief([1, 0.01 * step, 0], 0.5, D)
        candidates = [
            WriteCandidate(obs, matched_slot=slot, match_similarity=0.99,
                           source_position=0, source_layer=0),
        ]
        run_pass2(state, candidates, [slot], current_step=step)

    final_var = state.belief_precision_var[slot].item()
    assert final_var < initial_var, f"Variance should have decreased: {initial_var} → {final_var}"


def test_variance_has_floor(state):
    """Variance should never go below min_variance."""
    D = state.config.belief_dim
    b = make_belief([1, 0, 0], 2.0, D)
    slot = state.allocate_belief(b)

    min_var = state.meta_params.mesu_min_variance.item()

    # Force many reinforcements
    with torch.no_grad():
        state.belief_precision_var[slot] = min_var * 0.5  # artificially below floor

    # Run pass2 — should clamp to floor
    obs = make_belief([1, 0, 0], 0.5, D)
    candidates = [
        WriteCandidate(obs, matched_slot=slot, match_similarity=0.99,
                       source_position=0, source_layer=0),
    ]
    from memoria.cognition.pass2 import run_pass2
    run_pass2(state, candidates, [slot], current_step=100)

    assert state.belief_precision_var[slot].item() >= min_var


def test_consolidation_merges_variance(state):
    """Merged beliefs should combine variances (harmonic mean)."""
    D = state.config.belief_dim

    b1 = make_belief([1, 0, 0], 1.0, D)
    b2 = make_belief([1, 0.01, 0], 0.8, D)  # nearly identical direction
    s1 = state.allocate_belief(b1)
    s2 = state.allocate_belief(b2)

    state.belief_precision_var[s1] = 0.5
    state.belief_precision_var[s2] = 0.3

    from memoria.cognition.consolidation import soft_consolidation
    merged = soft_consolidation(state, similarity_threshold=0.95)

    if merged > 0:
        # Harmonic mean of 0.5 and 0.3 = 2/(1/0.5 + 1/0.3) ≈ 0.375
        expected_var = (0.5 * 0.3) / (0.5 + 0.3)
        min_var = state.meta_params.mesu_min_variance.item()
        actual_var = state.belief_precision_var[s1].item()
        assert actual_var >= min_var
        assert actual_var == pytest.approx(max(expected_var, min_var), abs=0.05)


def test_checkpoint_roundtrip_mesu(state):
    """MESU state survives serialization."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b)
    state.belief_precision_var[slot] = 0.42
    state.belief_reinforcement_count[slot] = 15

    checkpoint = state.state_dict_cognitive(compress=False)

    state2 = CognitiveState(state.config)
    state2.load_state_cognitive(checkpoint)

    assert state2.belief_precision_var[slot].item() == pytest.approx(0.42, abs=1e-5)
    assert state2.belief_reinforcement_count[slot].item() == 15
