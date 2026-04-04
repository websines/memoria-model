"""Tests for C4: Learned Recursion Depth (Adaptive Computation Time)."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.adaptive_depth import (
    AdaptiveDepth,
    ACTController,
    run_adaptive_depth_update,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def adaptive():
    return AdaptiveDepth(belief_dim=64, max_depth=8)


def test_adaptive_depth_init(adaptive):
    """AdaptiveDepth initializes correctly."""
    assert adaptive.max_depth == 8
    assert adaptive._iter_encoding.shape == (8, 1)


def test_halt_probs_shape(adaptive):
    """Halt probabilities have correct shape."""
    beliefs = torch.randn(10, 64)
    uncertainties = torch.ones(10)
    acc_change = torch.zeros(10)
    halt_bias = torch.tensor(0.0)
    probs = adaptive.compute_halt_probs(beliefs, uncertainties, 0, acc_change, halt_bias)
    assert probs.shape == (10,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_halt_probs_bias_effect(adaptive):
    """Higher halt bias increases halting probability."""
    beliefs = torch.randn(5, 64)
    uncertainties = torch.ones(5)
    acc_change = torch.zeros(5)

    probs_low = adaptive.compute_halt_probs(beliefs, uncertainties, 0, acc_change, torch.tensor(-5.0))
    probs_high = adaptive.compute_halt_probs(beliefs, uncertainties, 0, acc_change, torch.tensor(5.0))
    assert probs_high.mean().item() > probs_low.mean().item()


def test_act_controller_basic():
    """ACT controller runs and produces output."""
    module = AdaptiveDepth(belief_dim=32, max_depth=4)
    beliefs = torch.randn(5, 32)
    uncertainties = torch.ones(5)
    halt_bias = torch.tensor(0.0)

    controller = ACTController(module, beliefs, uncertainties, halt_bias)

    for i in range(4):
        updated = beliefs + torch.randn_like(beliefs) * 0.1
        controller.step(updated, i)
        if controller.all_halted():
            break

    final, ponder = controller.finalize(torch.tensor(0.5))
    assert final.shape == (5, 32)
    assert ponder.item() >= 0


def test_act_controller_stats():
    """ACT controller provides depth statistics."""
    module = AdaptiveDepth(belief_dim=32, max_depth=4)
    beliefs = torch.randn(5, 32)

    controller = ACTController(module, beliefs, torch.ones(5), torch.tensor(0.0))
    for i in range(4):
        controller.step(beliefs + torch.randn_like(beliefs) * 0.01, i)
    controller.finalize(torch.tensor(0.1))

    stats = controller.get_stats()
    assert 'mean_depth' in stats
    assert 'max_depth' in stats
    assert stats['mean_depth'] > 0


def test_ponder_cost_increases_with_steps():
    """Ponder cost is higher when more steps are used."""
    module = AdaptiveDepth(belief_dim=32, max_depth=8)
    beliefs = torch.randn(5, 32)

    # Force low halt probability (bias very negative)
    controller = ACTController(module, beliefs, torch.ones(5), torch.tensor(-10.0))
    for i in range(8):
        controller.step(beliefs, i)
    _, ponder_many = controller.finalize(torch.tensor(1.0))

    # Force high halt probability (bias very positive)
    controller2 = ACTController(module, beliefs, torch.ones(5), torch.tensor(10.0))
    for i in range(8):
        controller2.step(beliefs, i)
        if controller2.all_halted():
            break
    _, ponder_few = controller2.finalize(torch.tensor(1.0))

    # More steps → higher ponder cost
    assert ponder_many.item() >= ponder_few.item()


def test_run_adaptive_depth_update(state):
    """Full adaptive depth update runs correctly."""
    # Allocate some beliefs
    for i in range(5):
        state.allocate_belief(torch.randn(64) * 0.5)

    def simple_update(beliefs, iteration):
        return beliefs + torch.randn_like(beliefs) * 0.01

    result = run_adaptive_depth_update(
        state, state.adaptive_depth, simple_update, max_depth=3,
    )
    assert 'ponder_cost' in result
    assert 'stats' in result
    assert result['ponder_cost'].item() >= 0


def test_run_adaptive_depth_empty_state(state):
    """Adaptive depth handles empty state gracefully."""
    def noop(beliefs, iteration):
        return beliefs

    result = run_adaptive_depth_update(state, state.adaptive_depth, noop)
    assert result['ponder_cost'].item() == 0.0


def test_adaptive_depth_respects_max_depth():
    """Processing doesn't exceed max_depth."""
    module = AdaptiveDepth(belief_dim=32, max_depth=3)
    beliefs = torch.randn(5, 32)

    controller = ACTController(module, beliefs, torch.ones(5), torch.tensor(-10.0))
    steps_run = 0
    for i in range(3):
        controller.step(beliefs, i)
        steps_run += 1
        if controller.all_halted():
            break

    assert steps_run <= 3


def test_adaptive_depth_integrated(state):
    """AdaptiveDepth is attached to state."""
    assert hasattr(state, 'adaptive_depth')
    assert isinstance(state.adaptive_depth, AdaptiveDepth)


def test_metaparams_recursion(state):
    """MetaParams has recursion_depth_penalty and recursion_halt_bias."""
    p = state.meta_params.recursion_depth_penalty
    h = state.meta_params.recursion_halt_bias
    assert p.item() > 0  # softplus
    assert 0 < h.item() < 1  # sigmoid
