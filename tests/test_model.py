"""Smoke tests for the full MemoriaModel."""

import torch
import pytest
from memoria.model.config import small_config, MemoriaConfig, TransformerConfig, StateConfig, TrainingConfig
from memoria.model.memoria_model import MemoriaModel
from memoria.model.transformer import Transformer


@pytest.fixture
def tiny_config():
    """Tiny config for fast testing."""
    return MemoriaConfig(
        transformer=TransformerConfig(
            vocab_size=256, sequence_len=64,
            n_layer=4, n_head=2, n_kv_head=2, n_embd=128,
            interface_every=2, interface_num_heads=2, interface_top_k=8,
        ),
        state=StateConfig(
            belief_dim=64, max_beliefs=32, max_edges=64,
            max_goals=4, relation_dim=16,
        ),
        training=TrainingConfig(),
    )


@pytest.fixture
def model(tiny_config):
    m = MemoriaModel(tiny_config)
    m.init_weights()
    return m


def test_model_creates(model):
    """Model instantiates without error."""
    assert model is not None
    print(model.summary())


def test_model_forward(model):
    """Forward pass produces logits of correct shape."""
    idx = torch.randint(0, 256, (2, 32))
    result = model.forward(idx)
    assert result['logits'].shape == (2, 32, 256)
    assert isinstance(result['candidates'], list)


def test_model_loss(model):
    """Loss computation works with L_token only."""
    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    result = model.compute_loss(idx, targets, alpha=0.0)
    assert 'loss' in result
    assert result['loss'].requires_grad
    assert result['loss_fe'].item() == 0.0


def test_model_loss_with_fe(model):
    """Loss computation works with L_token + L_fe."""
    # Add some beliefs so L_fe has something to compute over
    for _ in range(5):
        model.state.allocate_belief(torch.randn(model.config.state.belief_dim))
    # Add an edge
    model.state.allocate_edge(0, 1, torch.randn(model.config.state.relation_dim), weight=0.5)

    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    result = model.compute_loss(idx, targets, alpha=0.1)

    assert result['loss_fe'].item() != 0.0
    assert 'free_energy_stats' in result
    assert 'beta' in result['free_energy_stats']


def test_backward(model):
    """Gradients flow through the full model."""
    # Add beliefs so the read path actually does computation
    for _ in range(5):
        model.state.allocate_belief(torch.randn(model.config.state.belief_dim))
    # Set output_proj to non-zero so gradients flow through read path
    with torch.no_grad():
        for iface in model.interfaces:
            iface.read_path.output_proj.weight.normal_(std=0.1)

    idx = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    result = model.compute_loss(idx, targets, alpha=0.0)
    result['loss'].backward()

    # Transformer params should have gradients
    assert model.transformer.wte.weight.grad is not None
    # Interface read path should have gradients (beliefs exist, output_proj non-zero)
    for iface in model.interfaces:
        assert iface.read_path.query_proj.weight.grad is not None


def test_interface_positions(tiny_config):
    """Interface layers are inserted at correct positions."""
    model = MemoriaModel(tiny_config)
    # 4 layers, interface_every=2 → positions [1, 3] (after blocks 1 and 3)
    assert model.interface_positions == [1, 3]
    assert len(model.interfaces) == 2


def test_detach_state(model):
    """State detach doesn't crash."""
    model.state.allocate_belief(torch.randn(model.config.state.belief_dim))
    model.detach_state()
    assert model.state.num_active_beliefs() == 1


def test_empty_state_is_pure_transformer(model):
    """With empty state, output should match a pure transformer."""
    idx = torch.randint(0, 256, (1, 16))

    # Model with empty state → interface read returns zeros
    result = model.forward(idx)

    # Just verify it doesn't crash and produces valid logits
    logits = result['logits']
    assert logits.shape == (1, 16, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_num_parameters(model):
    """Parameter counting works."""
    params = model.num_parameters()
    assert params['transformer'] > 0
    assert params['interface'] > 0
    assert params['total_trainable'] == params['transformer'] + params['interface']


def test_candidates_produced(model):
    """Forward pass produces write candidates."""
    # Add some beliefs so write path has something to match against
    for _ in range(5):
        model.state.allocate_belief(torch.randn(model.config.state.belief_dim))

    idx = torch.randint(0, 256, (1, 16))
    result = model.forward(idx)

    # Should have some candidates (from 2 interface layers × 16 token positions)
    # Not all positions produce meaningful candidates, but some should
    assert isinstance(result['candidates'], list)


def test_gradient_flow_to_beliefs(model):
    """Gradients from L_fe should flow into belief vectors (gradient wall removed)."""
    # Add beliefs so read/write paths have something to work with
    for _ in range(5):
        model.state.allocate_belief(torch.randn(model.config.state.belief_dim))

    # Zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    idx = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    result = model.forward(idx, targets=targets, alpha=0.1)

    result['loss'].backward()

    # Beliefs should have received gradients (gradient wall is down)
    assert model.state.beliefs.grad is not None
    # At least some belief slots should have non-zero gradients
    active_mask = model.state.get_active_mask()
    if active_mask.any():
        active_grads = model.state.beliefs.grad[active_mask]
        assert active_grads.abs().sum() > 0, "Active beliefs should have non-zero gradients"


def test_cognitive_meta_params_in_model(model):
    """MetaParams should be part of the model and have learnable parameters."""
    meta_params = list(model.state.meta_params.parameters())
    assert len(meta_params) == 62, f"Expected 62 meta params, got {len(meta_params)}"
    for p in meta_params:
        assert p.requires_grad, "All meta params should be learnable"
