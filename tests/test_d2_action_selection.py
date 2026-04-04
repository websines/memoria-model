"""Tests for D2: EFE Action Selection."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.agency.action_selection import (
    ActionSelector,
    ACTION_TYPES,
    N_ACTIONS,
    select_action,
)
from memoria.agency.daemon import ActionType


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def selector():
    return ActionSelector(belief_dim=64)


def test_selector_init(selector):
    """ActionSelector initializes correctly."""
    assert selector.n_actions == N_ACTIONS
    assert selector.action_embeddings.shape == (N_ACTIONS, 64)


def test_extract_state_features(state):
    """Feature extraction produces correct-size vector."""
    selector = state.action_selector
    features = selector.extract_state_features(state)
    assert features.shape == (64 + 8,)


def test_score_actions(state):
    """Action scoring produces correct shapes."""
    selector = state.action_selector
    scores = selector.score_actions(
        state,
        state.meta_params.efe_epistemic_weight,
        state.meta_params.efe_risk_weight,
        state.meta_params.action_risk_aversion,
    )
    assert scores['efe_scores'].shape == (N_ACTIONS,)
    assert scores['pragmatic'].shape == (N_ACTIONS,)
    assert scores['epistemic'].shape == (N_ACTIONS,)
    assert scores['risk'].shape == (N_ACTIONS,)


def test_risk_non_negative(state):
    """Risk scores are non-negative (softplus)."""
    selector = state.action_selector
    scores = selector.score_actions(
        state,
        state.meta_params.efe_epistemic_weight,
        state.meta_params.efe_risk_weight,
        state.meta_params.action_risk_aversion,
    )
    assert (scores['risk'] >= 0).all()


def test_select_action_returns_valid(state):
    """Action selection returns valid action type and info."""
    selector = state.action_selector
    temperature = state.meta_params.action_temperature
    risk = state.meta_params.action_risk_aversion

    idx, info = selector.select_action(state, temperature, risk)
    assert 0 <= idx < N_ACTIONS
    assert info['action_type'] in ACTION_TYPES
    assert 'efe_score' in info
    assert 'probs' in info
    assert info['probs'].sum().item() == pytest.approx(1.0, abs=0.01)


def test_select_action_convenience(state):
    """Convenience function works."""
    action_type, info = select_action(state)
    assert isinstance(action_type, ActionType)
    assert 'efe_score' in info


def test_temperature_affects_selection(state):
    """Lower temperature → more deterministic selection."""
    selector = state.action_selector
    risk = state.meta_params.action_risk_aversion

    # High temperature → more uniform
    _, info_hot = selector.select_action(state, torch.tensor(10.0), risk)
    probs_hot = info_hot['probs']

    # Low temperature → more peaked
    _, info_cold = selector.select_action(state, torch.tensor(0.01), risk)
    probs_cold = info_cold['probs']

    # Entropy of cold should be lower (more peaked)
    entropy_hot = -(probs_hot * (probs_hot + 1e-8).log()).sum().item()
    entropy_cold = -(probs_cold * (probs_cold + 1e-8).log()).sum().item()
    assert entropy_cold <= entropy_hot + 0.1  # allow small numerical tolerance


def test_action_types_complete():
    """All action types are represented."""
    assert ActionType.RESPOND in ACTION_TYPES
    assert ActionType.TOOL_CALL in ACTION_TYPES
    assert ActionType.SEARCH in ACTION_TYPES
    assert ActionType.WAIT in ACTION_TYPES
    assert ActionType.EXPLORE in ACTION_TYPES
    assert ActionType.CONSOLIDATE in ACTION_TYPES


def test_selector_with_beliefs(state):
    """Selection works with active beliefs."""
    for i in range(5):
        state.allocate_belief(torch.randn(64) * 0.5)
    action_type, info = select_action(state)
    assert isinstance(action_type, ActionType)


def test_action_selector_integrated(state):
    """ActionSelector is attached to state."""
    assert hasattr(state, 'action_selector')
    assert isinstance(state.action_selector, ActionSelector)


def test_metaparams_action(state):
    """MetaParams has action_temperature and action_risk_aversion."""
    t = state.meta_params.action_temperature
    r = state.meta_params.action_risk_aversion
    assert t.item() > 0  # softplus
    assert r.item() > 0  # softplus
