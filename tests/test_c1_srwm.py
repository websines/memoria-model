"""Tests for C1: Self-Referential Weight Matrix (SRWM)."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.srwm import SRWM


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def srwm():
    return SRWM(state_dim=16, n_meta_params=10, rank=8)


def test_srwm_init(srwm):
    """SRWM initializes with zero fast weights."""
    assert srwm.W_fast.abs().sum().item() == 0.0
    assert srwm._update_count.item() == 0


def test_srwm_query_returns_ones_at_init(srwm):
    """Before any updates, query returns modulations near 1.0."""
    features = torch.randn(16)
    modulation = srwm.query(features)
    assert modulation.shape == (10,)
    # With zero W_fast and zero output bias, tanh(0)=0, so modulation=1+0=1
    assert torch.allclose(modulation, torch.ones(10), atol=0.01)


def test_srwm_update_changes_W_fast(srwm):
    """Hebbian update modifies the fast-weight matrix."""
    features = torch.randn(16)
    lr = torch.tensor(0.1)
    decay = torch.tensor(0.05)
    srwm.update(features, lr, decay)
    assert srwm.W_fast.abs().sum().item() > 0
    assert srwm._update_count.item() == 1


def test_srwm_query_after_update(srwm):
    """After update with trained output proj, query returns non-trivial modulations."""
    features = torch.randn(16)
    lr = torch.tensor(0.5)
    decay = torch.tensor(0.01)
    # Initialize output_proj with small random weights (simulates training)
    torch.nn.init.xavier_normal_(srwm.output_proj.weight, gain=0.5)
    # Multiple updates to build up fast weights
    for _ in range(10):
        srwm.update(features, lr, decay)
    modulation = srwm.query(features)
    # With non-zero output weights and built-up W_fast, should differ from exactly 1.0
    # Modulations are intentionally small (tanh of small values), but non-zero
    assert not torch.allclose(modulation, torch.ones(10), atol=1e-4)


def test_srwm_modulation_bounded(srwm):
    """Modulations are bounded to (0, 2) by tanh."""
    features = torch.randn(16) * 10  # large features
    lr = torch.tensor(0.9)
    decay = torch.tensor(0.01)
    for _ in range(50):
        srwm.update(features, lr, decay)
    modulation = srwm.query(features)
    assert (modulation > 0).all()
    assert (modulation < 2).all()


def test_srwm_spectral_norm_clamp(srwm):
    """Spectral norm of W_fast is clamped to 1.0."""
    features = torch.ones(16)
    lr = torch.tensor(1.0)
    decay = torch.tensor(0.0)
    for _ in range(100):
        srwm.update(features, lr, decay)
    sv = torch.linalg.svdvals(srwm.W_fast)
    assert sv[0].item() <= 1.0 + 1e-5


def test_srwm_decay_reduces_W_fast(srwm):
    """Decay reduces fast-weight magnitude."""
    features = torch.randn(16)
    lr = torch.tensor(0.5)
    srwm.update(features, lr, torch.tensor(0.0))
    norm_before = srwm.W_fast.norm().item()
    # Now apply pure decay
    srwm.update(torch.zeros(16), torch.tensor(0.0), torch.tensor(0.5))
    norm_after = srwm.W_fast.norm().item()
    assert norm_after < norm_before


def test_srwm_reset(srwm):
    """Reset clears fast weights."""
    features = torch.randn(16)
    srwm.update(features, torch.tensor(0.5), torch.tensor(0.01))
    srwm.reset_fast_weights()
    assert srwm.W_fast.abs().sum().item() == 0.0
    assert srwm._update_count.item() == 0


def test_srwm_extract_state_features(state):
    """State feature extraction produces correct-size vector."""
    features = state.srwm.extract_state_features(state)
    assert features.shape == (min(state.config.belief_dim, 64),)


def test_srwm_get_dynamic_params(state):
    """Dynamic params returns modulated values."""
    base_params = {'lr': torch.tensor(0.1), 'decay': torch.tensor(0.5)}
    dynamic = state.srwm.get_dynamic_params(state, base_params)
    assert 'lr' in dynamic
    assert 'decay' in dynamic
    # At init, modulations are ~1.0, so dynamic ≈ base
    assert abs(dynamic['lr'].item() - 0.1) < 0.02


def test_srwm_summary(srwm):
    """Summary is readable."""
    s = srwm.summary()
    assert 'SRWM' in s
    assert 'rank=8' in s


def test_srwm_integrated_in_state(state):
    """SRWM is properly attached to CognitiveState."""
    assert hasattr(state, 'srwm')
    assert isinstance(state.srwm, SRWM)


def test_srwm_serialization_roundtrip(state):
    """SRWM survives serialization."""
    # Do some updates
    features = state.srwm.extract_state_features(state)
    state.srwm.update(features, torch.tensor(0.3), torch.tensor(0.05))
    w_before = state.srwm.W_fast.clone()

    checkpoint = state.state_dict_cognitive(compress=False)
    state2 = CognitiveState(state.config)
    state2.load_state_cognitive(checkpoint)

    assert torch.allclose(state2.srwm.W_fast, w_before, atol=1e-6)
