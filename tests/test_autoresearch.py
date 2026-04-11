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
    tracker = HypothesisTracker(max_goals=8, belief_dim=32, failed_buffer_depth=8)

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
    tracker = HypothesisTracker(max_goals=8, belief_dim=32, failed_buffer_depth=8)
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

        # Simulate post-allocation reads on all provisional beliefs so the
        # search/eval split criterion (Meta-Harness #3) does not evict them
        # for being untested. These reads model retrieval during forward
        # passes after the hypothesis was allocated.
        active_mask = state.get_active_mask()
        prov_idx = (state.belief_provisional & active_mask).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if prov_idx.numel() > 0:
            for step in (1, 2, 3):
                state.touch_beliefs(prov_idx, step=step)

        # Step 2: Simulate FE improvement and evaluate
        from memoria.cognition.provisional import evaluate_provisional_beliefs

        def callback(idx, outcome_code, metadata):
            # New 3-arg signature — outcome_code is PROMOTED (0) or EVICT_*.
            from memoria.cognition.provisional import PROMOTED
            promoted = outcome_code == PROMOTED
            state.hypothesis_tracker.record_outcome(
                idx,
                promoted,
                reason_code=outcome_code,
                fe_delta=metadata.get('fe_delta', 0.0),
            )

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


# ── #2: Failed-hypothesis log ───────────────────────────────────────────
# Meta-Harness (arXiv:2603.28052) validates that conditioning a proposer
# on raw failure history beats scalar summaries. These tests cover the
# per-goal failed ring buffer, the failure summary projection, and the
# behavior-preservation guarantee (zero-init of the new HypothesisGenerator
# columns means a freshly-built generator ignores the new signal).


def test_tracker_failed_buffer_push_on_eviction(state):
    """Evicting a hypothesis with a failure angle writes to the ring buffer."""
    tracker = state.hypothesis_tracker
    tracker._ensure_belief_buffer(64, torch.device('cpu'))

    # Record + evict three hypotheses for goal 0 with distinct angles.
    for i, slot in enumerate([3, 4, 5]):
        tracker.record_hypothesis(slot, goal_idx=0)
        angle = make_belief([float(i + 1), 0, 0], 1.0, state.config.belief_dim)
        tracker.record_outcome(
            slot,
            promoted=False,
            reason_code=1,
            fe_delta=0.5 + i * 0.1,
            failed_angle=angle,
        )

    # Goal 0 should have 3 failures logged.
    assert tracker.failed_count[0].item() == 3
    assert tracker.failed_write_idx[0].item() == 3
    # Goal 1 should have none.
    assert tracker.failed_count[1].item() == 0
    # Reasons and FE deltas should be recorded.
    assert tracker.failed_reasons[0, 0].item() == 1
    assert tracker.failed_fe_deltas[0, 0].item() == pytest.approx(0.5, abs=1e-5)


def test_tracker_failed_buffer_ring_wraps(state):
    """Ring buffer overwrites the oldest entry when saturated."""
    tracker = state.hypothesis_tracker
    tracker._ensure_belief_buffer(128, torch.device('cpu'))

    depth = tracker.failed_buffer_depth
    # Push `depth + 2` failures — two entries must wrap around.
    for i in range(depth + 2):
        tracker.record_hypothesis(i + 10, goal_idx=3)
        angle = torch.zeros(state.config.belief_dim)
        angle[0] = float(i + 1)
        tracker.record_outcome(
            i + 10,
            promoted=False,
            reason_code=1,
            fe_delta=0.0,
            failed_angle=angle,
        )

    # Count caps at depth.
    assert tracker.failed_count[3].item() == depth
    # Write index wraps back to 2 (after writing depth+2 entries starting at 0).
    assert tracker.failed_write_idx[3].item() == 2


def test_tracker_failure_summary_zero_for_empty_goal(state):
    """A goal with no failures yields a zero summary and zero count."""
    tracker = state.hypothesis_tracker
    summary, count = tracker.get_failure_summary(torch.tensor([0, 1]))
    assert summary.shape == (2, state.config.belief_dim)
    assert torch.all(summary == 0)
    assert torch.all(count == 0)


def test_tracker_failure_summary_means_over_stored_angles(state):
    """Failure summary is the mean of the stored failed angles per goal."""
    tracker = state.hypothesis_tracker
    tracker._ensure_belief_buffer(32, torch.device('cpu'))

    angles = [
        torch.tensor([2.0, 0.0] + [0.0] * (state.config.belief_dim - 2)),
        torch.tensor([0.0, 2.0] + [0.0] * (state.config.belief_dim - 2)),
    ]
    for i, a in enumerate(angles):
        tracker.record_hypothesis(i, goal_idx=2)
        tracker.record_outcome(
            i,
            promoted=False,
            reason_code=1,
            fe_delta=0.0,
            failed_angle=a,
        )

    summary, count = tracker.get_failure_summary(torch.tensor([2]))
    # Mean of the two angles.
    assert summary[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert summary[0, 1].item() == pytest.approx(1.0, abs=1e-5)
    assert count[0].item() == pytest.approx(2.0 / tracker.failed_buffer_depth, abs=1e-5)


def test_hypothesis_generator_failure_conditioning_zero_init(state):
    """A fresh generator's output is unchanged when failure conditioning is passed.

    Zero-init of the failure-conditioning columns means that at t=0 the
    network's output must be identical whether we pass zero failure summary
    (default) or a non-zero summary. This preserves backward compat of the
    autoresearch loop's early-training behavior.
    """
    gen = HypothesisGenerator(state.config.belief_dim)
    # Freeze all parameters: we want to compare forward outputs at init
    # weights, no optimizer in the loop.
    for p in gen.parameters():
        p.requires_grad_(False)

    # Force gate open so `generate_mask` is all-True and output is deterministic.
    with torch.no_grad():
        gen.generate_gate[-1].bias.fill_(5.0)

    goals = torch.randn(3, state.config.belief_dim)
    progress = torch.tensor([0.1, 0.5, 0.9])
    # Populate some active beliefs so belief_summary has signal.
    for i in range(3):
        state.allocate_belief(
            make_belief([float(i + 1), 1, 0], 2.0, state.config.belief_dim)
        )
    active_mask = state.get_active_mask()

    # No failure conditioning.
    h_none, p_none, _ = gen(
        goals, progress, state.beliefs.data, active_mask, beta=0.5
    )

    # Explicit non-zero failure conditioning.
    failure_summary = torch.randn(3, state.config.belief_dim)
    failure_count = torch.tensor([0.5, 0.75, 1.0])
    h_cond, p_cond, _ = gen(
        goals, progress, state.beliefs.data, active_mask, beta=0.5,
        failure_summary=failure_summary,
        failure_count=failure_count,
    )

    # Zero-init of the failure columns means outputs must match at init.
    assert torch.allclose(h_none, h_cond, atol=1e-6)
    assert torch.allclose(p_none, p_cond, atol=1e-6)


def test_hypothesis_generator_failure_conditioning_used_after_training(state):
    """After injecting non-zero weights on the failure columns, outputs change."""
    gen = HypothesisGenerator(state.config.belief_dim)
    for p in gen.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        gen.generate_gate[-1].bias.fill_(5.0)
        # Simulate training having learned to use the failure columns:
        # give the hypothesis_net first linear non-zero weights on the
        # failure_summary slice (indices 2D..3D).
        D = state.config.belief_dim
        gen.hypothesis_net[0].weight[:, 2 * D : 3 * D].normal_(0.0, 0.5)

    goals = torch.randn(2, D)
    progress = torch.tensor([0.3, 0.7])
    for i in range(3):
        state.allocate_belief(
            make_belief([float(i + 1), 1, 0], 2.0, D)
        )
    active_mask = state.get_active_mask()

    h_none, _, _ = gen(goals, progress, state.beliefs.data, active_mask, beta=0.5)
    failure_summary = torch.randn(2, D) * 2.0
    h_cond, _, _ = gen(
        goals, progress, state.beliefs.data, active_mask, beta=0.5,
        failure_summary=failure_summary,
    )

    # Non-zero failure columns means outputs now differ.
    assert not torch.allclose(h_none, h_cond, atol=1e-4)
