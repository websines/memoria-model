"""Tests for the Internal Autoresearch Loop."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.autoresearch import (
    HypothesisGenerator, HypothesisTracker, run_autoresearch_step,
)
from memoria.cognition.telos_module import STATUS_ACTIVE


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


def setup_state_with_goals(state):
    """Add beliefs and goals to a state for testing."""
    # Add some beliefs
    for i in range(5):
        state.allocate_belief(
            make_belief([float(i + 1), 1, 0], 2.0, state.config.belief_dim)
        )

    # Add an active goal
    with torch.no_grad():
        state.goal_embeddings.data[0] = make_belief([1, 0, 0], 1.0, state.config.belief_dim)
        state.goal_status_logits[0] = torch.zeros(6)
        state.goal_status_logits[0, STATUS_ACTIVE] = 5.0


def test_hypothesis_generator_shapes(state):
    """HypothesisGenerator produces correct tensor shapes."""
    gen = HypothesisGenerator(state.config.belief_dim)

    goals = torch.randn(3, state.config.belief_dim)
    progress = torch.tensor([0.2, 0.5, 0.8])
    beliefs = state.beliefs.data.clone()

    # Add some active beliefs
    for i in range(5):
        state.allocate_belief(make_belief([float(i), 1, 0], 2.0, state.config.belief_dim))
    active_mask = state.get_active_mask()

    hypotheses, precisions, indices = gen(goals, progress, beliefs, active_mask, beta=0.5)

    # Some may be gated out, but shapes should be consistent
    assert hypotheses.shape[1] == state.config.belief_dim
    assert len(precisions) == hypotheses.shape[0]
    assert len(indices) == hypotheses.shape[0]


def test_hypothesis_generator_empty_goals(state):
    """Generator returns empty tensors with no goals."""
    gen = HypothesisGenerator(state.config.belief_dim)
    state.allocate_belief(make_belief([1, 0, 0], 2.0, state.config.belief_dim))

    goals = torch.zeros(0, state.config.belief_dim)
    progress = torch.zeros(0)
    active_mask = state.get_active_mask()

    hypotheses, precisions, indices = gen(goals, progress, state.beliefs.data, active_mask, 0.5)
    assert hypotheses.shape[0] == 0


def test_hypothesis_tracker_records(state):
    """Tracker records hypothesis outcomes correctly."""
    tracker = HypothesisTracker(max_goals=8)

    tracker._ensure_belief_buffer(64, torch.device('cpu'))

    # Record hypothesis from goal 0
    tracker.record_hypothesis(belief_slot=5, goal_idx=0)
    assert tracker.hypothesis_count[0].item() == 1
    assert tracker.belief_source_goal[5].item() == 0

    # Record success
    tracker.record_outcome(belief_slot=5, promoted=True)
    assert tracker.hypothesis_promoted[0].item() == 1
    assert tracker.goal_success_ema[0].item() > 0.5  # boosted

    # Record another from goal 0, this time failure
    tracker.record_hypothesis(belief_slot=10, goal_idx=0)
    tracker.record_outcome(belief_slot=10, promoted=False)
    assert tracker.hypothesis_evicted[0].item() == 1


def test_hypothesis_tracker_priority_boost():
    """Goals with high success rate get positive boost."""
    tracker = HypothesisTracker(max_goals=8)
    tracker._ensure_belief_buffer(64, torch.device('cpu'))

    # Goal 0: always succeeds
    for i in range(5):
        tracker.record_hypothesis(i, goal_idx=0)
        tracker.record_outcome(i, promoted=True)

    # Goal 1: always fails
    for i in range(5, 10):
        tracker.record_hypothesis(i, goal_idx=1)
        tracker.record_outcome(i, promoted=False)

    boosts = tracker.goal_priority_boost(torch.tensor([0, 1]))
    assert boosts[0].item() > 0  # successful goal → positive boost
    assert boosts[1].item() < 0  # failing goal → negative boost


def test_run_autoresearch_step_generates_hypotheses(state):
    """Full autoresearch step generates provisional beliefs from goals."""
    setup_state_with_goals(state)

    # Force the generate gate open for testing
    with torch.no_grad():
        state.hypothesis_gen.generate_gate[-1].bias.fill_(5.0)

    n_before = state.num_active_beliefs()
    stats = run_autoresearch_step(
        state, state.hypothesis_gen, state.hypothesis_tracker,
        current_step=100, current_fe=5.0,
    )

    if stats['hypotheses_generated'] > 0:
        n_after = state.num_active_beliefs()
        assert n_after > n_before
        # New beliefs should be provisional
        active_mask = state.get_active_mask()
        n_prov = state.belief_provisional[active_mask].sum().item()
        assert n_prov >= stats['hypotheses_generated']


def test_autoresearch_no_goals(state):
    """Autoresearch does nothing without active goals."""
    state.allocate_belief(make_belief([1, 0, 0], 2.0, state.config.belief_dim))

    stats = run_autoresearch_step(
        state, state.hypothesis_gen, state.hypothesis_tracker,
        current_step=100, current_fe=5.0,
    )
    assert stats['hypotheses_generated'] == 0


def test_full_autoresearch_cycle(state):
    """Full cycle: generate hypothesis → evaluate → track outcome."""
    setup_state_with_goals(state)

    # Force gate open
    with torch.no_grad():
        state.hypothesis_gen.generate_gate[-1].bias.fill_(5.0)

    # Step 1: Generate hypotheses
    stats = run_autoresearch_step(
        state, state.hypothesis_gen, state.hypothesis_tracker,
        current_step=0, current_fe=10.0,
    )
    n_generated = stats['hypotheses_generated']

    if n_generated > 0:
        # Verify tracking
        total_tracked = state.hypothesis_tracker.hypothesis_count.sum().item()
        assert total_tracked == n_generated

        # Step 2: Simulate FE improvement and evaluate
        from memoria.cognition.provisional import evaluate_provisional_beliefs

        def callback(idx, promoted):
            state.hypothesis_tracker.record_outcome(idx, promoted)

        prov_stats = evaluate_provisional_beliefs(
            state, current_step=200, current_fe=5.0,  # FE decreased
            outcome_callback=callback,
        )

        # Should have outcomes
        total_outcomes = (
            state.hypothesis_tracker.hypothesis_promoted.sum().item()
            + state.hypothesis_tracker.hypothesis_evicted.sum().item()
        )
        assert total_outcomes == n_generated


def test_autoresearch_skips_low_success_goals(state):
    """Goals with consistently failing hypotheses get skipped."""
    setup_state_with_goals(state)

    tracker = state.hypothesis_tracker
    tracker._ensure_belief_buffer(64, torch.device('cpu'))

    # Record many failures for goal 0
    for i in range(20, 30):
        tracker.record_hypothesis(i, goal_idx=0)
        tracker.record_outcome(i, promoted=False)

    # Success EMA should be very low
    assert tracker.goal_success_ema[0].item() < 0.2

    # Autoresearch should skip this goal (unless untested)
    stats = run_autoresearch_step(
        state, state.hypothesis_gen, tracker,
        current_step=100, current_fe=5.0,
    )
    # Goal 0 should have been gated out by low success rate
    assert stats['hypotheses_gated_out'] >= 1


def test_checkpoint_roundtrip_autoresearch(state):
    """Autoresearch state survives serialization."""
    state.hypothesis_tracker._ensure_belief_buffer(64, torch.device('cpu'))
    state.hypothesis_tracker.record_hypothesis(5, goal_idx=0)
    state.hypothesis_tracker.record_outcome(5, promoted=True)

    checkpoint = state.state_dict_cognitive(compress=False)

    state2 = CognitiveState(state.config)
    state2.load_state_cognitive(checkpoint)

    assert state2.hypothesis_tracker.hypothesis_promoted[0].item() == 1
    assert state2.hypothesis_tracker.goal_success_ema[0].item() > 0.5
