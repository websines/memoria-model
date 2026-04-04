"""Tests for A3: Causal Cascade Revision."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.cascade_revision import cascade_revision


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


def test_cascade_no_edges(state):
    """Cascade with no edges does nothing."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b)

    stats = cascade_revision(state, [slot])

    assert stats['beliefs_decayed'] == 0
    assert state.beliefs.data[slot].norm().item() == pytest.approx(2.0, abs=0.01)


def test_cascade_single_hop(state):
    """Revision cascades one hop through a causal edge."""
    D = state.config.belief_dim

    # Source belief (will be revised)
    b_src = make_belief([1, 0, 0], 2.0, D)
    s_src = state.allocate_belief(b_src)

    # Downstream belief (should decay)
    b_tgt = make_belief([0, 1, 0], 3.0, D)
    s_tgt = state.allocate_belief(b_tgt)

    # Causal edge from src → tgt
    relation = torch.randn(state.config.relation_dim)
    eidx = state.allocate_edge(s_src, s_tgt, relation, weight=0.5)
    state.edge_causal_obs[eidx] = 1.0  # mark as causal

    initial_radius = state.beliefs.data[s_tgt].norm().item()
    initial_var = state.belief_precision_var[s_tgt].item()

    stats = cascade_revision(state, [s_src])

    assert stats['beliefs_decayed'] == 1
    # Downstream belief should have lower radius
    final_radius = state.beliefs.data[s_tgt].norm().item()
    assert final_radius < initial_radius
    # Downstream belief should have higher variance
    final_var = state.belief_precision_var[s_tgt].item()
    assert final_var > initial_var


def test_cascade_multi_hop(state):
    """Revision cascades through multiple hops: A → B → C."""
    D = state.config.belief_dim

    b_a = make_belief([1, 0, 0], 2.0, D)
    b_b = make_belief([0, 1, 0], 3.0, D)
    b_c = make_belief([0, 0, 1], 4.0, D)
    s_a = state.allocate_belief(b_a)
    s_b = state.allocate_belief(b_b)
    s_c = state.allocate_belief(b_c)

    # A → B
    e1 = state.allocate_edge(s_a, s_b, torch.randn(state.config.relation_dim), weight=0.5)
    state.edge_causal_obs[e1] = 1.0
    # B → C
    e2 = state.allocate_edge(s_b, s_c, torch.randn(state.config.relation_dim), weight=0.5)
    state.edge_causal_obs[e2] = 1.0

    r_b_before = state.beliefs.data[s_b].norm().item()
    r_c_before = state.beliefs.data[s_c].norm().item()

    stats = cascade_revision(state, [s_a])

    r_b_after = state.beliefs.data[s_b].norm().item()
    r_c_after = state.beliefs.data[s_c].norm().item()

    assert stats['beliefs_decayed'] == 2
    assert stats['max_depth_reached'] == 2

    # Both B and C should have decayed
    assert r_b_after < r_b_before
    assert r_c_after < r_c_before

    # C should decay LESS than B (further from source)
    b_decay = r_b_before - r_b_after
    c_decay = r_c_before - r_c_after
    assert c_decay < b_decay


def test_cascade_respects_immutable(state):
    """Immutable beliefs are not affected by cascade."""
    D = state.config.belief_dim

    b_src = make_belief([1, 0, 0], 2.0, D)
    b_tgt = make_belief([0, 1, 0], 3.0, D)
    s_src = state.allocate_belief(b_src)
    s_tgt = state.allocate_belief(b_tgt)

    state.immutable_beliefs[s_tgt] = True

    e = state.allocate_edge(s_src, s_tgt, torch.randn(state.config.relation_dim), weight=0.5)
    state.edge_causal_obs[e] = 1.0

    r_before = state.beliefs.data[s_tgt].norm().item()
    cascade_revision(state, [s_src])
    r_after = state.beliefs.data[s_tgt].norm().item()

    assert r_after == pytest.approx(r_before, abs=1e-6)


def test_cascade_no_cycles(state):
    """Cascade terminates with cycles in the graph (visited set prevents revisit)."""
    D = state.config.belief_dim

    b_a = make_belief([1, 0, 0], 2.0, D)
    b_b = make_belief([0, 1, 0], 3.0, D)
    s_a = state.allocate_belief(b_a)
    s_b = state.allocate_belief(b_b)

    # Cycle: A → B and B → A
    e1 = state.allocate_edge(s_a, s_b, torch.randn(state.config.relation_dim), weight=0.5)
    e2 = state.allocate_edge(s_b, s_a, torch.randn(state.config.relation_dim), weight=0.5)
    state.edge_causal_obs[e1] = 1.0
    state.edge_causal_obs[e2] = 1.0

    # Should not infinite loop
    stats = cascade_revision(state, [s_a])
    assert stats['beliefs_decayed'] >= 1


def test_cascade_increases_variance(state):
    """Downstream beliefs should have increased precision variance after cascade."""
    D = state.config.belief_dim

    b_src = make_belief([1, 0, 0], 2.0, D)
    b_tgt = make_belief([0, 1, 0], 3.0, D)
    s_src = state.allocate_belief(b_src)
    s_tgt = state.allocate_belief(b_tgt)

    state.belief_precision_var[s_tgt] = 0.1  # low initial variance

    e = state.allocate_edge(s_src, s_tgt, torch.randn(state.config.relation_dim), weight=0.5)
    state.edge_causal_obs[e] = 1.0

    cascade_revision(state, [s_src])

    # Variance should have increased
    assert state.belief_precision_var[s_tgt].item() > 0.1


def test_cascade_empty_state(state):
    """Cascade on empty state doesn't crash."""
    stats = cascade_revision(state, [])
    assert stats['beliefs_decayed'] == 0

    stats = cascade_revision(state, [0, 1, 2])
    assert stats['beliefs_decayed'] == 0
