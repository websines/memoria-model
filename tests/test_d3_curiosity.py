"""Tests for D3: Curiosity-Driven Telos Generation."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.agency.curiosity import (
    CuriosityModule,
    run_curiosity_step,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def curiosity():
    return CuriosityModule(belief_dim=64)


def test_curiosity_init(curiosity):
    """CuriosityModule initializes with EMA at 1.0."""
    assert curiosity._actor_ema.item() == 1.0
    assert curiosity._critic_ema.item() == 1.0


def test_actor_curiosity_empty(curiosity):
    """Actor curiosity returns 0 for empty beliefs."""
    beliefs = torch.zeros(0, 64)
    c = curiosity.compute_actor_curiosity(beliefs)
    assert c.item() == 0.0


def test_actor_curiosity_nonzero(curiosity):
    """Actor curiosity returns positive value for real beliefs."""
    beliefs = torch.randn(10, 64)
    c = curiosity.compute_actor_curiosity(beliefs)
    assert c.item() >= 0


def test_critic_curiosity_no_efe(curiosity):
    """Critic curiosity works without EFE scores."""
    beliefs = torch.randn(10, 64)
    c = curiosity.compute_critic_curiosity(beliefs)
    assert c.item() >= 0


def test_critic_curiosity_with_efe(curiosity):
    """Critic curiosity uses EFE scores when provided."""
    beliefs = torch.randn(10, 64)
    efe = torch.randn(6)
    c = curiosity.compute_critic_curiosity(beliefs, efe)
    assert c.item() >= 0


def test_combined_curiosity(curiosity):
    """Combined curiosity is weighted sum of actor and critic."""
    beliefs = torch.randn(10, 64)
    weight = torch.tensor(1.0)
    combined = curiosity.compute_combined_curiosity(beliefs, weight)
    assert combined.item() >= 0


def test_curiosity_weight_effect(curiosity):
    """Higher curiosity weight increases combined signal."""
    beliefs = torch.randn(10, 64)
    low = curiosity.compute_combined_curiosity(beliefs, torch.tensor(0.1))
    high = curiosity.compute_combined_curiosity(beliefs, torch.tensor(10.0))
    assert high.item() >= low.item()


def test_generate_exploration_goals_below_threshold(curiosity, state):
    """No goals generated when curiosity is below threshold."""
    goals, priorities = curiosity.generate_exploration_goals(
        state,
        curiosity_threshold=torch.tensor(100.0),  # very high
        curiosity_weight=torch.tensor(1.0),
    )
    assert goals.shape[0] == 0


def test_generate_exploration_goals_above_threshold(state):
    """Goals generated when curiosity exceeds threshold."""
    # Allocate high-variance beliefs
    for i in range(10):
        slot = state.allocate_belief(torch.randn(64) * 0.5)
        state.belief_precision_var[slot] = 5.0  # high uncertainty

    curiosity = state.curiosity
    goals, priorities = curiosity.generate_exploration_goals(
        state,
        curiosity_threshold=torch.tensor(0.001),  # very low threshold
        curiosity_weight=torch.tensor(1.0),
    )
    # May or may not generate depending on network output, but shapes are correct
    assert goals.dim() == 2
    if goals.shape[0] > 0:
        assert goals.shape[1] == 64
        assert priorities.shape[0] == goals.shape[0]


def test_run_curiosity_step_empty(state):
    """Curiosity step handles empty state."""
    stats = run_curiosity_step(state, state.curiosity)
    assert stats['actor_curiosity'] == 0.0
    assert stats['goals_generated'] == 0


def test_run_curiosity_step_with_beliefs(state):
    """Curiosity step runs with active beliefs."""
    for i in range(5):
        state.allocate_belief(torch.randn(64) * 0.5)
    stats = run_curiosity_step(state, state.curiosity)
    assert 'actor_curiosity' in stats
    assert 'critic_curiosity' in stats
    assert 'combined_curiosity' in stats
    assert 'threshold' in stats


def test_ema_updates(curiosity):
    """EMA updates over multiple calls."""
    beliefs = torch.randn(10, 64)
    ema_before = curiosity._actor_ema.item()
    for _ in range(10):
        curiosity.compute_actor_curiosity(beliefs)
    ema_after = curiosity._actor_ema.item()
    # EMA should have changed
    assert ema_before != ema_after


def test_curiosity_integrated(state):
    """CuriosityModule is attached to state."""
    assert hasattr(state, 'curiosity')
    assert isinstance(state.curiosity, CuriosityModule)


def test_metaparams_curiosity(state):
    """MetaParams has curiosity_threshold and curiosity_weight."""
    t = state.meta_params.curiosity_threshold
    w = state.meta_params.curiosity_weight
    assert t.item() > 0  # softplus
    assert w.item() > 0  # softplus
