"""Tests for C2: Meta-Learned Belief Update Function."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.learned_update import (
    LearnedUpdateFunction,
    apply_learned_update,
    get_edge_context,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def update_fn():
    return LearnedUpdateFunction(belief_dim=64)


def test_learned_update_init(update_fn):
    """Learned update initializes with near-zero delta and ~1.0 precision scale."""
    belief = torch.randn(64)
    obs = torch.randn(64)
    prec = torch.tensor(1.0)
    err = torch.tensor(0.5)
    ctx = torch.zeros(64)
    delta, scale, merge = update_fn(belief, obs, prec, err, ctx)
    assert delta.shape == (1, 64)
    assert scale.shape == (1,)
    assert merge.shape == (1,)
    # Delta should be near zero at init
    assert delta.abs().mean().item() < 0.1
    # Scale should be near 1.0
    assert abs(scale.item() - 1.0) < 0.5
    # Merge should be low
    assert merge.item() < 0.3


def test_learned_update_batched(update_fn):
    """Batched input produces correct shapes."""
    N = 10
    belief = torch.randn(N, 64)
    obs = torch.randn(N, 64)
    prec = torch.randn(N).abs()
    err = torch.randn(N).abs()
    ctx = torch.zeros(N, 64)
    delta, scale, merge = update_fn(belief, obs, prec, err, ctx)
    assert delta.shape == (N, 64)
    assert scale.shape == (N,)
    assert merge.shape == (N,)


def test_precision_scale_positive(update_fn):
    """Precision scale is always positive (softplus)."""
    for _ in range(5):
        delta, scale, _ = update_fn(
            torch.randn(64), torch.randn(64),
            torch.tensor(0.1), torch.tensor(2.0), torch.zeros(64),
        )
        assert scale.item() > 0


def test_merge_signal_bounded(update_fn):
    """Merge signal is bounded [0, 1] (sigmoid)."""
    for _ in range(5):
        _, _, merge = update_fn(
            torch.randn(64), torch.randn(64),
            torch.tensor(1.0), torch.tensor(0.5), torch.zeros(64),
        )
        assert 0.0 <= merge.item() <= 1.0


def test_apply_learned_update_gated():
    """Gated blend respects gate value."""
    fn = LearnedUpdateFunction(belief_dim=32)
    N = 5
    beliefs = torch.randn(N, 32)
    obs = torch.randn(N, 32)
    prec = torch.ones(N)
    err = torch.ones(N) * 0.5
    ctx = torch.zeros(N, 32)
    handcoded_delta = torch.randn(N, 32)
    handcoded_scale = torch.ones(N)

    # Gate = 0: fully hand-coded
    final_d, final_s, _ = apply_learned_update(
        beliefs, obs, prec, err, ctx, fn,
        gate=torch.tensor(0.0),
        handcoded_deltas=handcoded_delta,
        handcoded_precision_scales=handcoded_scale,
    )
    assert torch.allclose(final_d, handcoded_delta, atol=1e-5)

    # Gate = 1: fully learned
    final_d1, _, _ = apply_learned_update(
        beliefs, obs, prec, err, ctx, fn,
        gate=torch.tensor(1.0),
        handcoded_deltas=handcoded_delta,
        handcoded_precision_scales=handcoded_scale,
    )
    # Should differ from handcoded
    assert not torch.allclose(final_d1, handcoded_delta, atol=0.01)


def test_get_edge_context_no_edges(state):
    """Edge context returns zeros when no edges exist."""
    indices = torch.tensor([0, 1, 2])
    ctx = get_edge_context(state, indices)
    assert ctx.shape == (3, 64)
    assert ctx.abs().sum().item() == 0.0


def test_get_edge_context_with_edges(state):
    """Edge context returns neighbor means when edges exist."""
    # Allocate two beliefs and an edge
    b0 = torch.randn(64) * 0.5
    b1 = torch.randn(64) * 0.5
    with torch.no_grad():
        state.beliefs.data[0] = b0
        state.beliefs.data[1] = b1
    state.allocate_edge(0, 1, torch.zeros(state.config.relation_dim), weight=0.5)

    indices = torch.tensor([0])
    ctx = get_edge_context(state, indices)
    assert ctx.shape == (1, 64)
    # Context for belief 0 should be close to belief 1 (its neighbor)
    assert ctx[0].norm().item() > 0


def test_learned_update_integrated_in_state(state):
    """LearnedUpdateFunction is attached to state."""
    assert hasattr(state, 'learned_update')
    assert isinstance(state.learned_update, LearnedUpdateFunction)


def test_update_fn_gate_metaparam(state):
    """MetaParams has update_fn_gate."""
    gate = state.meta_params.update_fn_gate
    assert 0 < gate.item() < 1
