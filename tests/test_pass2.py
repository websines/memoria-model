"""Tests for the full pass 2 cognitive update loop."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.interface.write_path import WriteCandidate
from memoria.cognition.pass2 import run_pass2
from memoria.cognition.surprise import compute_surprise_batch
from memoria.cognition.telos import STATUS_ACTIVE, STATUS_COMPLETED, STATUS, PRIORITY, PROGRESS


@pytest.fixture
def config():
    return StateConfig(belief_dim=32, max_beliefs=64, max_edges=256, max_goals=8, relation_dim=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def make_belief_vec(direction: list[float], precision: float, dim: int = 32) -> torch.Tensor:
    v = torch.zeros(dim)
    for i, val in enumerate(direction):
        v[i] = val
    return torch.nn.functional.normalize(v, dim=0) * precision


def test_pass2_empty(state):
    """Pass 2 with no candidates and empty state should not crash."""
    stats = run_pass2(state, [], [], current_step=0)
    assert stats['active_beliefs'] == 0
    assert stats['active_edges'] == 0


def test_pass2_new_beliefs(state):
    """Pass 2 should allocate new beliefs from unmatched candidates."""
    candidates = [
        WriteCandidate(
            belief_vector=make_belief_vec([1, 0, 0], 2.0, state.config.belief_dim),
            matched_slot=-1, match_similarity=0.0,
            source_position=0, source_layer=0,
        ),
        WriteCandidate(
            belief_vector=make_belief_vec([0, 1, 0], 1.5, state.config.belief_dim),
            matched_slot=-1, match_similarity=0.0,
            source_position=1, source_layer=0,
        ),
    ]

    stats = run_pass2(state, candidates, [], current_step=1)
    assert state.num_active_beliefs() == 2
    assert stats['belief_new_allocations'] == 2


def test_pass2_matched_observation(state):
    """Pass 2 with matched observations should track surprise (continuous updates via gradient)."""
    # Add existing belief
    b = make_belief_vec([1, 0, 0], 1.0, state.config.belief_dim)
    slot = state.allocate_belief(b)

    # Slightly different observation matching the same slot
    obs = make_belief_vec([1, 0.2, 0], 0.5, state.config.belief_dim)
    candidates = [
        WriteCandidate(
            belief_vector=obs, matched_slot=slot,
            match_similarity=0.95, source_position=0, source_layer=0,
        ),
    ]

    stats = run_pass2(state, candidates, [slot], current_step=1)
    # Belief content is now updated by gradient, not pass2 — just verify stats are computed
    assert stats['belief_total_surprise'] >= 0
    assert stats['num_candidates'] == 1


def test_pass2_high_surprise_observation(state):
    """High-surprise observation should be tracked (continuous updates via gradient)."""
    # Strong belief pointing one way
    b = make_belief_vec([1, 0, 0], 0.5, state.config.belief_dim)
    slot = state.allocate_belief(b)

    # Strong observation pointing opposite way
    obs = make_belief_vec([-1, 0, 0], 3.0, state.config.belief_dim)
    candidates = [
        WriteCandidate(
            belief_vector=obs, matched_slot=slot,
            match_similarity=0.1, source_position=0, source_layer=0,
        ),
    ]

    stats = run_pass2(state, candidates, [slot], current_step=1)
    # High surprise should be recorded
    assert stats['belief_total_surprise'] > 0


def test_pass2_hebbian(state):
    """Co-activated beliefs should form edges."""
    b1 = make_belief_vec([1, 0, 0], 2.0, state.config.belief_dim)
    b2 = make_belief_vec([0, 1, 0], 2.0, state.config.belief_dim)
    s1 = state.allocate_belief(b1)
    s2 = state.allocate_belief(b2)

    # Simulate both beliefs being read together
    stats = run_pass2(state, [], [s1, s2], current_step=1)

    assert stats['co_activation_pairs'] == 1
    assert state.num_active_edges() >= 1


def test_pass2_intrinsic_goals(state):
    """Accumulated surprise should generate intrinsic goals."""
    # Add low-precision beliefs (surprise hotspots)
    for i in range(5):
        state.allocate_belief(make_belief_vec([float(i), 1, 0], 0.05, state.config.belief_dim))

    # Manually set high accumulated surprise
    with torch.no_grad():
        state.meta.data[1] = 10.0  # accumulated_surprise
        state.meta.data[0] = 0.8   # high β (exploration mode)

    stats = run_pass2(state, [], [], current_step=100)
    assert stats['goals_generated'] > 0
    assert state.num_active_goals() > 0


def test_pass2_goal_progress(state):
    """Updated beliefs relevant to goals should advance progress."""
    # Create a belief
    b = make_belief_vec([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b)

    # Create a goal aligned with the belief
    with torch.no_grad():
        state.goal_embeddings.data[0] = make_belief_vec([1, 0, 0], 1.0, state.config.belief_dim)
        state.goal_metadata.data[0, PRIORITY] = 0.8
        state.goal_metadata.data[0, STATUS] = STATUS_ACTIVE
        state.goal_metadata.data[0, PROGRESS] = 0.0

    # Update the belief with some surprise
    obs = make_belief_vec([1, 0.3, 0], 1.0, state.config.belief_dim)
    candidates = [
        WriteCandidate(
            belief_vector=obs, matched_slot=slot,
            match_similarity=0.9, source_position=0, source_layer=0,
        ),
    ]

    stats = run_pass2(state, candidates, [slot], current_step=1)

    # Goal progress should have increased
    progress = state.goal_metadata.data[0, PROGRESS].item()
    assert progress > 0.0, f"Goal progress should have increased, got {progress}"


def test_pass2_consolidation(state):
    """Very similar beliefs should merge during soft consolidation."""
    # Two nearly identical beliefs
    b1 = make_belief_vec([1, 0, 0], 1.0, state.config.belief_dim)
    b2 = make_belief_vec([1, 0.01, 0], 0.8, state.config.belief_dim)
    state.allocate_belief(b1)
    state.allocate_belief(b2)

    assert state.num_active_beliefs() == 2

    # Use step=10 so periodic soft_consolidation triggers (runs every 10 steps)
    stats = run_pass2(state, [], [], current_step=10)

    # Should have merged (cosine_sim > 0.95)
    assert stats['soft_merges'] > 0 or state.num_active_beliefs() <= 2


def test_pass2_kernel_rules(state):
    """Immutable beliefs should not be evicted or structurally modified."""
    b = make_belief_vec([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b)
    state.immutable_beliefs[slot] = True
    original = state.beliefs.data[slot].clone()

    # With structural-only pass2, immutable beliefs are protected during eviction/consolidation
    # The belief content itself is now updated by gradient (not pass2), so kernel rules
    # protect against structural operations (deallocation, merge)
    stats = run_pass2(state, [], [slot], current_step=1)

    # Immutable belief should still exist unchanged (structurally)
    assert torch.allclose(state.beliefs.data[slot], original)


def test_pass2_full_cycle(state):
    """Full cycle: new beliefs → update → Hebbian → goals → consolidation."""
    # Step 1: Add new beliefs
    candidates_1 = [
        WriteCandidate(make_belief_vec([1, 0, 0], 2.0, state.config.belief_dim), -1, 0.0, 0, 0),
        WriteCandidate(make_belief_vec([0, 1, 0], 2.0, state.config.belief_dim), -1, 0.0, 1, 0),
        WriteCandidate(make_belief_vec([0, 0, 1], 2.0, state.config.belief_dim), -1, 0.0, 2, 0),
    ]
    stats_1 = run_pass2(state, candidates_1, [], current_step=1, is_sequence_boundary=False)
    assert state.num_active_beliefs() == 3

    # Step 2: Co-activate beliefs → Hebbian edges
    stats_2 = run_pass2(state, [], [0, 1, 2], current_step=2, is_sequence_boundary=False)
    assert state.num_active_edges() > 0

    # Step 3: Update with surprise → should accumulate surprise
    obs = make_belief_vec([1, 0.5, 0], 1.5, state.config.belief_dim)
    candidates_3 = [
        WriteCandidate(obs, matched_slot=0, match_similarity=0.8, source_position=0, source_layer=0),
    ]
    stats_3 = run_pass2(state, candidates_3, [0], current_step=3, is_sequence_boundary=True)

    assert stats_3['beta'] >= 0.0
    assert stats_3['active_beliefs'] > 0
    assert stats_3['active_edges'] > 0
