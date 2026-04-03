"""Tests for CognitiveState."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig


@pytest.fixture
def state():
    config = StateConfig(belief_dim=64, max_beliefs=128, max_edges=512, max_goals=8)
    return CognitiveState(config)


def test_init_empty(state):
    """Fresh state has no active beliefs, edges, or goals."""
    assert state.num_active_beliefs() == 0
    assert state.num_active_edges() == 0
    assert state.num_active_goals() == 0
    assert state.beta == 1.0  # max exploration


def test_allocate_belief(state):
    """Allocating a belief increases active count."""
    belief = torch.randn(64) * 0.5  # radius ≈ 0.5 * sqrt(64) ≈ 4
    slot = state.allocate_belief(belief)
    assert slot >= 0
    assert state.num_active_beliefs() == 1


def test_allocate_multiple_beliefs(state):
    """Can allocate many beliefs."""
    for i in range(10):
        belief = torch.randn(64)
        slot = state.allocate_belief(belief)
        assert slot >= 0
    assert state.num_active_beliefs() == 10


def test_deallocate_belief(state):
    """Deallocating sets radius to 0."""
    belief = torch.randn(64)
    slot = state.allocate_belief(belief)
    assert state.num_active_beliefs() == 1

    state.deallocate_belief(slot)
    assert state.num_active_beliefs() == 0


def test_immutable_belief_cannot_deallocate(state):
    """Kernel rule: immutable beliefs resist deallocation."""
    belief = torch.randn(64)
    slot = state.allocate_belief(belief)
    state.immutable_beliefs[slot] = True

    state.deallocate_belief(slot)
    assert state.num_active_beliefs() == 1  # still there


def test_allocate_returns_minus_one_when_full(state):
    """Allocation fails gracefully when full."""
    for i in range(state.config.max_beliefs):
        state.allocate_belief(torch.randn(64))
    assert state.num_active_beliefs() == state.config.max_beliefs

    slot = state.allocate_belief(torch.randn(64))
    assert slot == -1


def test_get_active_beliefs(state):
    """Active beliefs returns only non-empty slots."""
    state.allocate_belief(torch.randn(64))
    state.allocate_belief(torch.randn(64))
    state.allocate_belief(torch.randn(64))

    indices, beliefs = state.get_active_beliefs()
    assert len(indices) == 3
    assert beliefs.shape == (3, 64)


def test_belief_radii(state):
    """Radii reflect belief vector magnitudes."""
    v = torch.randn(64)
    v = v / v.norm() * 2.5  # set radius to exactly 2.5
    slot = state.allocate_belief(v)

    radii = state.get_belief_radii()
    assert abs(radii[slot].item() - 2.5) < 1e-5


def test_allocate_edge(state):
    """Can allocate edges between beliefs."""
    s1 = state.allocate_belief(torch.randn(64))
    s2 = state.allocate_belief(torch.randn(64))

    relation = torch.randn(state.config.relation_dim)
    edge = state.allocate_edge(s1, s2, relation, weight=0.5)
    assert edge >= 0
    assert state.num_active_edges() == 1


def test_get_active_edges(state):
    """Active edges returns correct data."""
    s1 = state.allocate_belief(torch.randn(64))
    s2 = state.allocate_belief(torch.randn(64))
    relation = torch.randn(state.config.relation_dim)
    state.allocate_edge(s1, s2, relation, weight=0.8)

    src, tgt, relations, weights = state.get_active_edges()
    assert len(src) == 1
    assert src[0].item() == s1
    assert tgt[0].item() == s2
    assert abs(weights[0].item() - 0.8) < 1e-6


def test_deallocate_edge(state):
    """Deallocating an edge frees the slot."""
    s1 = state.allocate_belief(torch.randn(64))
    s2 = state.allocate_belief(torch.randn(64))
    edge = state.allocate_edge(s1, s2, torch.randn(state.config.relation_dim))

    state.deallocate_edge(edge)
    assert state.num_active_edges() == 0


def test_checkpoint_roundtrip(state):
    """Cognitive state survives serialization."""
    # Add some content
    state.allocate_belief(torch.randn(64) * 3.0)
    state.allocate_belief(torch.randn(64) * 1.5)
    s1, s2 = 0, 1
    state.allocate_edge(s1, s2, torch.randn(state.config.relation_dim), weight=0.7)

    # Checkpoint
    checkpoint = state.state_dict_cognitive()

    # Create fresh state and restore
    config = StateConfig(belief_dim=64, max_beliefs=128, max_edges=512, max_goals=8)
    state2 = CognitiveState(config)
    state2.load_state_cognitive(checkpoint)

    assert state2.num_active_beliefs() == 2
    assert state2.num_active_edges() == 1
    # Compressed checkpoints use 3-bit quantization: ~97% cosine similarity, not exact
    cos_sim = torch.nn.functional.cosine_similarity(
        state.beliefs.data[state.get_active_mask()],
        state2.beliefs.data[state2.get_active_mask()],
        dim=-1,
    )
    assert cos_sim.mean() > 0.95, f"Belief roundtrip quality too low: {cos_sim.mean():.4f}"


def test_summary(state):
    """Summary returns readable string."""
    s = state.summary()
    assert "0/128 beliefs" in s
    assert "β=1.000" in s
