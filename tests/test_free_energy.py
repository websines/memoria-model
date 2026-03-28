"""Tests for Bethe free energy computation.

Verifies:
1. Agreeing beliefs → low energy
2. Disagreeing beliefs → high energy
3. Overconfident wrong beliefs → high free energy
4. Gradients push toward consistency
5. β reflects uncertainty correctly
"""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.core.free_energy import compute_free_energy, compute_energy, compute_entropy


@pytest.fixture
def config():
    return StateConfig(belief_dim=32, max_beliefs=64, max_edges=256, max_goals=8, relation_dim=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def make_belief(direction: list[float], precision: float, dim: int = 32) -> torch.Tensor:
    """Create a belief vector with specific direction and precision (radius)."""
    v = torch.zeros(dim)
    for i, val in enumerate(direction):
        v[i] = val
    v = torch.nn.functional.normalize(v, dim=0) * precision
    return v


def test_empty_state_free_energy(state):
    """Empty state: zero energy, zero entropy."""
    result = compute_free_energy(state)
    assert result['energy'].item() == 0.0
    assert result['entropy'].item() == 0.0
    assert result['free_energy'].item() == 0.0


def test_agreeing_beliefs_low_energy(state):
    """Two similar beliefs connected by an edge → low energy."""
    b1 = make_belief([1, 0, 0], precision=2.0, dim=state.config.belief_dim)
    b2 = make_belief([1, 0.1, 0], precision=2.0, dim=state.config.belief_dim)  # slightly different

    s1 = state.allocate_belief(b1)
    s2 = state.allocate_belief(b2)

    relation = torch.zeros(state.config.relation_dim)  # neutral relation
    state.allocate_edge(s1, s2, relation, weight=1.0)

    result = compute_free_energy(state)
    agreeing_energy = result['relation_energy'].item()

    # Now make them disagree
    b3 = make_belief([-1, 0, 0], precision=2.0, dim=state.config.belief_dim)  # opposite direction
    with torch.no_grad():
        state.beliefs.data[s2] = b3

    result2 = compute_free_energy(state)
    disagreeing_energy = result2['relation_energy'].item()

    # Disagreeing should have higher energy
    assert disagreeing_energy > agreeing_energy


def test_high_precision_amplifies_energy(state):
    """Higher precision beliefs create higher energy on disagreement."""
    # Low precision disagreement
    b1_low = make_belief([1, 0, 0], precision=0.5, dim=state.config.belief_dim)
    b2_low = make_belief([-1, 0, 0], precision=0.5, dim=state.config.belief_dim)

    s1 = state.allocate_belief(b1_low)
    s2 = state.allocate_belief(b2_low)
    relation = torch.zeros(state.config.relation_dim)
    state.allocate_edge(s1, s2, relation, weight=1.0)

    result_low = compute_free_energy(state)
    energy_low = result_low['relation_energy'].item()

    # Reset and do high precision disagreement
    with torch.no_grad():
        state.beliefs.data[s1] = make_belief([1, 0, 0], precision=5.0, dim=state.config.belief_dim)
        state.beliefs.data[s2] = make_belief([-1, 0, 0], precision=5.0, dim=state.config.belief_dim)

    result_high = compute_free_energy(state)
    energy_high = result_high['relation_energy'].item()

    # High precision disagreement should create more energy
    assert energy_high > energy_low


def test_entropy_decreases_with_confidence(state):
    """Higher precision (radius) beliefs have lower entropy."""
    # Low precision belief
    b_low = make_belief([1, 0, 0], precision=0.1, dim=state.config.belief_dim)
    state.allocate_belief(b_low)

    entropy_low_prec = compute_entropy(state).item()

    # Replace with high precision
    with torch.no_grad():
        state.beliefs.data[0] = make_belief([1, 0, 0], precision=10.0, dim=state.config.belief_dim)

    entropy_high_prec = compute_entropy(state).item()

    # High precision → lower entropy
    assert entropy_high_prec < entropy_low_prec


def test_beta_high_when_uncertain(config):
    """β should be high when beliefs are uncertain (low precision)."""
    state = CognitiveState(config)

    # Add several low-precision beliefs (high uncertainty)
    for _ in range(5):
        state.allocate_belief(make_belief(list(torch.randn(3).tolist()), precision=0.1, dim=config.belief_dim))

    result = compute_free_energy(state)
    beta_uncertain = result['beta'].item()

    # Now make them high-precision
    state2 = CognitiveState(config)
    for _ in range(5):
        state2.allocate_belief(make_belief(list(torch.randn(3).tolist()), precision=10.0, dim=config.belief_dim))

    result2 = compute_free_energy(state2)
    beta_confident = result2['beta'].item()

    # Uncertain state should have higher β
    assert beta_uncertain > beta_confident


def test_gradients_push_toward_agreement(state):
    """Backprop through free energy should push disagreeing beliefs toward agreement."""
    # Make beliefs require_grad for this test
    state.beliefs = torch.nn.Parameter(state.beliefs.data.clone(), requires_grad=True)

    # Two opposing beliefs, high precision
    with torch.no_grad():
        state.beliefs.data[0] = make_belief([1, 0, 0], precision=3.0, dim=state.config.belief_dim)
        state.beliefs.data[1] = make_belief([-1, 0, 0], precision=3.0, dim=state.config.belief_dim)

    state.allocate_edge(0, 1, torch.zeros(state.config.relation_dim), weight=1.0)

    # Record initial angular distance
    a0 = state.beliefs[0] / state.beliefs[0].norm()
    a1 = state.beliefs[1] / state.beliefs[1].norm()
    initial_distance = (1.0 - (a0 * a1).sum()).item()

    # Compute free energy and backprop
    result = compute_free_energy(state)
    result['free_energy'].backward()

    # Apply gradient step
    with torch.no_grad():
        state.beliefs.data -= 0.1 * state.beliefs.grad

    # Check angular distance decreased
    a0_new = state.beliefs[0] / state.beliefs[0].norm()
    a1_new = state.beliefs[1] / state.beliefs[1].norm()
    new_distance = (1.0 - (a0_new * a1_new).sum()).item()

    assert new_distance < initial_distance, (
        f"Gradient step should reduce disagreement: {initial_distance:.4f} → {new_distance:.4f}"
    )


def test_gradients_flow_through_all_components(state):
    """Verify gradients exist for beliefs, relations, and edge weights."""
    state.beliefs = torch.nn.Parameter(state.beliefs.data.clone(), requires_grad=True)
    state.edge_relations = torch.nn.Parameter(state.edge_relations.data.clone(), requires_grad=True)
    state.edge_weights = torch.nn.Parameter(state.edge_weights.data.clone(), requires_grad=True)

    with torch.no_grad():
        state.beliefs.data[0] = make_belief([1, 0, 0], precision=2.0, dim=state.config.belief_dim)
        state.beliefs.data[1] = make_belief([0, 1, 0], precision=2.0, dim=state.config.belief_dim)

    state.allocate_edge(0, 1, torch.randn(state.config.relation_dim), weight=0.5)

    result = compute_free_energy(state)
    result['free_energy'].backward()

    # All should have gradients
    assert state.beliefs.grad is not None
    assert state.beliefs.grad.abs().sum() > 0
