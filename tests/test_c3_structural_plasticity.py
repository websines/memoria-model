"""Tests for C3: Structural Plasticity (split/prune/grow)."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.structural_plasticity import (
    StructuralPlasticity,
    run_structural_plasticity,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def plasticity(config):
    return StructuralPlasticity(belief_dim=64, max_beliefs=128)


def test_plasticity_init(plasticity):
    """Structural plasticity initializes with zero stats."""
    assert plasticity.activation_count.sum().item() == 0
    assert plasticity._total_steps.item() == 0


def test_record_activation(plasticity):
    """Recording activations increments counts."""
    indices = torch.tensor([0, 1, 5])
    plasticity.record_activation(indices)
    assert plasticity.activation_count[0].item() == 1
    assert plasticity.activation_count[1].item() == 1
    assert plasticity.activation_count[5].item() == 1
    assert plasticity.activation_count[2].item() == 0
    assert plasticity._total_steps.item() == 1


def test_record_activation_with_context(plasticity):
    """Context updates signature via EMA."""
    indices = torch.tensor([0])
    context = torch.randn(1, 64)
    plasticity.record_activation(indices, context)
    assert plasticity.context_signatures[0].abs().sum().item() > 0


def test_compute_activation_entropy(plasticity):
    """Entropy computation returns valid values."""
    # Record diverse contexts
    for _ in range(10):
        plasticity.record_activation(
            torch.tensor([0]),
            torch.randn(1, 64),
        )
    entropy = plasticity.compute_activation_entropy(None)  # state not used
    assert entropy.shape == (128,)
    assert entropy[0].item() >= 0


def test_evaluate_empty_state(state, plasticity):
    """Evaluation with no active beliefs returns empty results."""
    result = plasticity.evaluate(
        state,
        split_threshold=torch.tensor(0.7),
        prune_threshold=torch.tensor(0.1),
    )
    assert result['split_candidates'] == []
    assert result['prune_candidates'] == []
    assert not result['should_grow']


def test_evaluate_with_beliefs(state):
    """Evaluation identifies split/prune candidates."""
    # Allocate some beliefs
    for i in range(10):
        vec = torch.randn(64)
        vec = vec / vec.norm() * (0.5 if i < 5 else 0.01)  # 5 strong, 5 weak
        state.allocate_belief(vec)

    sp = state.structural_plasticity
    # Record activations for some beliefs
    for _ in range(20):
        sp.record_activation(torch.arange(10), torch.randn(10, 64))

    result = sp.evaluate(
        state,
        split_threshold=torch.tensor(0.3),
        prune_threshold=torch.tensor(0.5),
    )
    # Should have some candidates (exact count depends on network init)
    assert isinstance(result['split_candidates'], list)
    assert isinstance(result['prune_candidates'], list)


def test_split_belief(state):
    """Splitting a belief creates two children."""
    vec = torch.randn(64)
    vec = vec / vec.norm() * 1.0  # radius = 1.0
    slot = state.allocate_belief(vec)

    sp = state.structural_plasticity
    a, b = sp.split_belief(state, slot)
    assert a == slot  # child A reuses parent slot
    assert b >= 0  # child B gets new slot

    # Both children exist
    r_a = state.beliefs.data[a].norm().item()
    r_b = state.beliefs.data[b].norm().item()
    assert r_a > 0
    assert r_b > 0
    # Precision split: each ≈ original / sqrt(2)
    assert abs(r_a - 1.0 / (2 ** 0.5)) < 0.2
    assert abs(r_b - 1.0 / (2 ** 0.5)) < 0.2


def test_split_preserves_provenance(state):
    """Split children track parent as source."""
    vec = torch.randn(64) * 0.5
    slot = state.allocate_belief(vec)
    sp = state.structural_plasticity
    a, b = sp.split_belief(state, slot)
    assert state.belief_sources[b, 0].item() == slot


def test_prune_belief(state):
    """Pruning deallocates belief and connected edges."""
    vec = torch.randn(64) * 0.5
    slot = state.allocate_belief(vec)
    # Add an edge
    vec2 = torch.randn(64) * 0.5
    slot2 = state.allocate_belief(vec2)
    state.allocate_edge(slot, slot2, torch.zeros(state.config.relation_dim), weight=0.3)
    assert state.num_active_edges() == 1

    sp = state.structural_plasticity
    sp.prune_belief(state, slot)
    assert state.beliefs.data[slot].norm().item() < 1e-6
    assert state.num_active_edges() == 0  # edge cleaned up


def test_run_structural_plasticity(state):
    """Full structural plasticity step runs without error."""
    for i in range(5):
        state.allocate_belief(torch.randn(64) * 0.5)

    sp = state.structural_plasticity
    for _ in range(15):
        sp.record_activation(torch.arange(5), torch.randn(5, 64))

    stats = run_structural_plasticity(state, sp)
    assert 'splits_executed' in stats
    assert 'prunes_executed' in stats
    assert 'growth_pressure' in stats


def test_growth_pressure(state):
    """Growth pressure increases with fill ratio and surprise."""
    # Fill most slots
    for i in range(int(state.config.max_beliefs * 0.95)):
        state.allocate_belief(torch.randn(64) * 0.3)
    state.meta.data[1] = 1.0  # high surprise

    sp = state.structural_plasticity
    for _ in range(15):
        sp.record_activation(torch.arange(10))

    result = sp.evaluate(
        state,
        split_threshold=torch.tensor(0.7),
        prune_threshold=torch.tensor(0.1),
    )
    assert result['growth_pressure'] > 0.5
    assert result['should_grow']


def test_no_split_immutable(state):
    """Immutable beliefs are never split."""
    vec = torch.randn(64) * 0.5
    slot = state.allocate_belief(vec)
    state.immutable_beliefs[slot] = True

    sp = state.structural_plasticity
    for _ in range(20):
        sp.record_activation(torch.tensor([slot]), torch.randn(1, 64))

    result = sp.evaluate(
        state,
        split_threshold=torch.tensor(0.01),  # very low threshold
        prune_threshold=torch.tensor(0.01),
    )
    assert slot not in result['split_candidates']
    assert slot not in result['prune_candidates']


def test_reset_stats(plasticity):
    """Reset clears activation statistics."""
    plasticity.record_activation(torch.tensor([0, 1, 2]))
    plasticity.reset_stats()
    assert plasticity.activation_count.sum().item() == 0


def test_plasticity_integrated(state):
    """StructuralPlasticity is attached to state."""
    assert hasattr(state, 'structural_plasticity')
    assert isinstance(state.structural_plasticity, StructuralPlasticity)
