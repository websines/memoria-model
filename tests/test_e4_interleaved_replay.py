"""Tests for E4: Interleaved Replay."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.interleaved_replay import (
    select_replay_set,
    replay_message_passing,
    run_interleaved_replay,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def test_select_replay_empty(state):
    """Replay selection handles empty state."""
    recent, old = select_replay_set(
        state, replay_ratio=torch.tensor(0.3),
    )
    assert len(recent) == 0
    assert len(old) == 0


def test_select_replay_too_few_beliefs(state):
    """Replay needs at least 4 beliefs."""
    state.allocate_belief(torch.randn(64))
    state.allocate_belief(torch.randn(64))

    recent, old = select_replay_set(
        state, replay_ratio=torch.tensor(0.3),
    )
    assert len(recent) == 0
    assert len(old) == 0


def test_select_replay_separates_groups(state):
    """Recent and old groups are non-overlapping."""
    for i in range(20):
        slot = state.allocate_belief(torch.randn(64) * (1.0 + i * 0.2), step=i * 10)
        # Make some beliefs high-variance (recent candidates)
        if i >= 15:
            state.belief_precision_var[slot] = 5.0

    recent, old = select_replay_set(
        state, replay_ratio=torch.tensor(0.3), max_replay_size=10,
    )

    assert len(recent) > 0
    assert len(old) > 0

    # No overlap
    recent_set = set(recent.tolist())
    old_set = set(old.tolist())
    assert len(recent_set & old_set) == 0


def test_select_replay_respects_ratio(state):
    """Replay ratio controls old/recent balance."""
    for i in range(20):
        state.allocate_belief(torch.randn(64), step=i * 10)

    recent, old = select_replay_set(
        state, replay_ratio=torch.tensor(0.5), max_replay_size=10,
    )

    # With 0.5 ratio, old should be ~half of total
    total = len(recent) + len(old)
    if total > 0:
        assert len(old) >= total * 0.3  # allow some tolerance


def test_replay_message_passing_empty(state):
    """Message passing with empty groups returns no contradictions."""
    empty = torch.zeros(0, dtype=torch.long)
    stats = replay_message_passing(
        state, empty, empty,
        contradiction_threshold=torch.tensor(0.5),
    )
    assert stats['contradictions_found'] == 0


def test_replay_message_passing_no_edges(state):
    """Without causal edges, no contradictions from unrelated beliefs."""
    # Allocate beliefs in very different directions
    for i in range(10):
        v = torch.zeros(64)
        v[i * 6] = 3.0
        state.allocate_belief(v, step=i * 10)

    recent = torch.tensor([0, 1, 2], dtype=torch.long)
    old = torch.tensor([7, 8, 9], dtype=torch.long)

    stats = replay_message_passing(
        state, recent, old,
        contradiction_threshold=torch.tensor(0.5),
    )
    # No edges → no contradictions (unrelated pairs skipped)
    assert stats['contradictions_found'] == 0


def test_replay_detects_contradiction(state):
    """Cross-temporal contradiction detected via causal edge."""
    # Old belief in one direction
    a = torch.zeros(64)
    a[0] = 5.0
    slot_old = state.allocate_belief(a, step=10)

    # Recent belief in opposite direction
    b = torch.zeros(64)
    b[32] = 3.0
    slot_recent = state.allocate_belief(b, step=100)

    # Causal edge between them
    edge = state.allocate_edge(slot_old, slot_recent, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge] = 5

    recent_idx = torch.tensor([slot_recent], dtype=torch.long)
    old_idx = torch.tensor([slot_old], dtype=torch.long)

    stats = replay_message_passing(
        state, recent_idx, old_idx,
        contradiction_threshold=torch.tensor(0.5),
    )
    assert stats['contradictions_found'] >= 1
    assert stats['beliefs_weakened'] >= 1


def test_replay_weakens_less_supported(state):
    """The less empirically supported belief is weakened."""
    a = torch.zeros(64)
    a[0] = 5.0
    slot_a = state.allocate_belief(a, step=10)
    state.belief_confirmed_count[slot_a] = 10

    b = torch.zeros(64)
    b[32] = 3.0
    slot_b = state.allocate_belief(b, step=100)
    state.belief_contradicted_count[slot_b] = 10

    edge = state.allocate_edge(slot_a, slot_b, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge] = 5

    radius_b_before = state.beliefs.data[slot_b].norm().item()

    replay_message_passing(
        state,
        torch.tensor([slot_b], dtype=torch.long),
        torch.tensor([slot_a], dtype=torch.long),
        contradiction_threshold=torch.tensor(0.5),
    )

    radius_b_after = state.beliefs.data[slot_b].norm().item()
    assert radius_b_after < radius_b_before


def test_replay_preserves_immutable(state):
    """Immutable beliefs are not weakened by replay."""
    a = torch.zeros(64)
    a[0] = 5.0
    slot_a = state.allocate_belief(a, step=10)
    state.immutable_beliefs[slot_a] = True

    b = torch.zeros(64)
    b[32] = 3.0
    slot_b = state.allocate_belief(b, step=100)

    edge = state.allocate_edge(slot_a, slot_b, torch.randn(64), weight=0.8)
    state.edge_causal_obs[edge] = 5

    radius_before = state.beliefs.data[slot_a].norm().item()

    replay_message_passing(
        state,
        torch.tensor([slot_b], dtype=torch.long),
        torch.tensor([slot_a], dtype=torch.long),
        contradiction_threshold=torch.tensor(0.5),
    )

    radius_after = state.beliefs.data[slot_a].norm().item()
    assert abs(radius_before - radius_after) < 1e-6


def test_run_interleaved_replay_empty(state):
    """Full replay handles empty state."""
    stats = run_interleaved_replay(state)
    assert stats['recent_selected'] == 0
    assert stats['old_selected'] == 0


def test_run_interleaved_replay_with_beliefs(state):
    """Full replay runs with active beliefs."""
    for i in range(20):
        slot = state.allocate_belief(torch.randn(64) * (1.0 + i * 0.3), step=i * 10)
        if i >= 15:
            state.belief_precision_var[slot] = 5.0

    stats = run_interleaved_replay(state)
    assert stats['recent_selected'] > 0
    assert stats['old_selected'] > 0


def test_metaparams_e4(state):
    """MetaParams has E4 parameters."""
    assert 0 < state.meta_params.replay_ratio.item() < 1
    assert 0 < state.meta_params.replay_contradiction_threshold.item() < 1
