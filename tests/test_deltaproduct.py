"""Tests for DeltaProduct and Log-Linear DeltaProduct layers.

Run on a CUDA machine with flash-linear-attention installed:
    pip install flash-linear-attention
    pytest tests/test_deltaproduct.py -v

On Mac/CPU (no FLA), only the Fenwick tree and config tests will run.
FLA-dependent tests are skipped automatically.
"""

import math
import torch
import pytest
from memoria.model.config import (
    MemoriaConfig, TransformerConfig, StateConfig, TrainingConfig, small_config,
)
from memoria.model.fenwick_state import FenwickStateTree, _lssb, MAX_NUM_LEVELS


# ── Check if FLA/hattention are available ──

_fla_available = False
try:
    from fla.layers import GatedDeltaProduct
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    _fla_available = True
except ImportError:
    pass

_hattention_available = False
try:
    from hattention.modeling_h_gated_deltanet import HGatedDeltaNet
    _hattention_available = True
except ImportError:
    pass

requires_fla = pytest.mark.skipif(not _fla_available, reason="flash-linear-attention not installed")
requires_hattention = pytest.mark.skipif(not _hattention_available, reason="hattention not installed")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ═══════════════════════════════════════════════════════════════════════════
# Fenwick tree tests (run anywhere, no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestLSSB:
    """Test least significant set bit computation."""

    def test_powers_of_two(self):
        assert _lssb(1) == 0
        assert _lssb(2) == 1
        assert _lssb(4) == 2
        assert _lssb(8) == 3
        assert _lssb(16) == 4

    def test_odd_numbers(self):
        assert _lssb(3) == 0
        assert _lssb(5) == 0
        assert _lssb(7) == 0

    def test_mixed(self):
        assert _lssb(6) == 1   # 110 → bit 1
        assert _lssb(12) == 2  # 1100 → bit 2
        assert _lssb(10) == 1  # 1010 → bit 1

    def test_zero(self):
        assert _lssb(0) == 0


class TestFenwickStateTree:
    """Test Fenwick tree state management."""

    @pytest.fixture
    def tree(self):
        return FenwickStateTree(
            num_heads=2, head_k_dim=8, head_v_dim=16,
            device=torch.device('cpu'), dtype=torch.float32, batch_size=1,
        )

    def test_init_zeros(self, tree):
        assert tree.states.shape == (MAX_NUM_LEVELS, 1, 2, 8, 16)
        assert tree.states.abs().sum() == 0
        assert not tree.active.any()

    def test_query_empty_returns_zeros(self, tree):
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)
        init = tree.query(lw)
        assert init.shape == (1, 2, 8, 16)
        assert init.abs().sum() == 0

    def test_single_update(self, tree):
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)
        state = torch.randn(1, 2, 8, 16)
        tree.update(0, state, lw)
        assert tree.active[0]
        assert tree._chunk_count == 1

    def test_query_after_update(self, tree):
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)
        state = torch.randn(1, 2, 8, 16)
        tree.update(0, state, lw)
        init = tree.query(lw)
        # Should return something nonzero now
        assert init.abs().sum() > 0

    def test_multiple_updates_fenwick_merge(self, tree):
        """After 2 updates, Fenwick should merge level 0 into level 1."""
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)

        s0 = torch.randn(1, 2, 8, 16)
        tree.update(0, s0, lw)
        assert tree.active[0]

        s1 = torch.randn(1, 2, 8, 16)
        tree.update(1, s1, lw)
        # After chunk 1: lssb(2)=1, so levels 0..0 merge into level 1
        assert tree.active[0]  # level 0 always gets latest
        assert tree.active[1]  # level 1 should be active (merged)

    def test_four_updates_hierarchy(self, tree):
        """After 4 updates, should have state at level 2."""
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)
        for ci in range(4):
            state = torch.randn(1, 2, 8, 16)
            tree.update(ci, state, lw)

        # After chunk 3: lssb(4)=2, so levels 0..1 merge into level 2
        assert tree.active[0]
        assert tree.active[2]
        assert tree._chunk_count == 4

    def test_reset(self, tree):
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)
        tree.update(0, torch.randn(1, 2, 8, 16), lw)
        tree.reset()
        assert tree.states.abs().sum() == 0
        assert not tree.active.any()
        assert tree._chunk_count == 0

    def test_level_scales_modulate_query(self, tree):
        """Different level scales should produce different query results."""
        lw = torch.ones(1, 2, MAX_NUM_LEVELS)

        # Need enough updates so that levels 0 and 2 have DIFFERENT states
        for ci in range(4):
            tree.update(ci, torch.randn(1, 2, 8, 16), lw)

        # After 4 chunks: level 0 has chunk 3's state, level 2 has merged 0+1
        # Query with weight only on level 0
        lw_level0 = torch.zeros(1, 2, MAX_NUM_LEVELS)
        lw_level0[:, :, 0] = 10.0
        init_0 = tree.query(lw_level0)

        # Query with weight only on level 2
        lw_level2 = torch.zeros(1, 2, MAX_NUM_LEVELS)
        lw_level2[:, :, 2] = 10.0
        init_2 = tree.query(lw_level2)

        # Level 0 and level 2 hold different states, so queries should differ
        assert not torch.allclose(init_0, init_2)

    def test_batch_support(self):
        tree = FenwickStateTree(
            num_heads=2, head_k_dim=8, head_v_dim=16,
            device=torch.device('cpu'), dtype=torch.float32, batch_size=4,
        )
        lw = torch.ones(4, 2, MAX_NUM_LEVELS)
        state = torch.randn(4, 2, 8, 16)
        tree.update(0, state, lw)
        init = tree.query(lw)
        assert init.shape == (4, 2, 8, 16)


# ═══════════════════════════════════════════════════════════════════════════
# Config tests (run anywhere)
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_default_pattern_is_hhhhl(self):
        cfg = TransformerConfig()
        assert cfg.window_pattern == "HHHHL"

    def test_small_config_pattern(self):
        cfg = small_config()
        assert cfg.transformer.window_pattern == "HHHHL"

    def test_deltaproduct_defaults(self):
        cfg = TransformerConfig()
        assert cfg.deltaproduct_n_householder == 3
        assert cfg.deltaproduct_allow_neg_eigval is True
        assert cfg.deltaproduct_head_dim == 128
        assert cfg.deltaproduct_expand_v == 2
        assert cfg.loglinear_chunk_size == 64

    def test_legacy_mamba_fields_still_exist(self):
        """Legacy Mamba fields kept for backward compat."""
        cfg = TransformerConfig()
        assert hasattr(cfg, 'mamba_d_state')
        assert hasattr(cfg, 'mamba_d_conv')
        assert hasattr(cfg, 'mamba_expand')


# ═══════════════════════════════════════════════════════════════════════════
# Block dispatch tests (run anywhere — tests fallback behavior)
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockDispatch:
    def test_h_layer_fallback_without_fla(self):
        """H layers should fall back gracefully without FLA installed."""
        from memoria.model.transformer import Block
        config = TransformerConfig(
            window_pattern='HHHHL', n_layer=5,
            n_head=6, n_kv_head=6, n_embd=768,
        )
        block = Block(config, layer_idx=0)
        # Should be SOME attention type, not crash
        assert block.attn is not None

    def test_d_layer_fallback_without_fla(self):
        from memoria.model.transformer import Block
        config = TransformerConfig(
            window_pattern='DDDML', n_layer=5,
            n_head=6, n_kv_head=6, n_embd=768,
        )
        block = Block(config, layer_idx=0)
        assert block.attn is not None

    def test_l_layer_uses_mla(self):
        from memoria.model.transformer import Block
        config = TransformerConfig(
            window_pattern='HHHHL', n_layer=5,
            n_head=6, n_kv_head=6, n_embd=768,
            mla_latent_dim=192,
        )
        block = Block(config, layer_idx=4)  # L layer
        from memoria.model.transformer import MLACausalSelfAttention
        assert isinstance(block.attn, MLACausalSelfAttention)


# ═══════════════════════════════════════════════════════════════════════════
# DeltaProduct layer tests (require FLA + CUDA)
# ═══════════════════════════════════════════════════════════════════════════

@requires_fla
@requires_cuda
class TestDeltaProductBlock:
    @pytest.fixture
    def config(self):
        return TransformerConfig(
            vocab_size=256, sequence_len=128,
            n_layer=5, n_head=4, n_kv_head=4, n_embd=128,
            window_pattern='DDDML',
            deltaproduct_head_dim=32,
            deltaproduct_n_householder=3,
            deltaproduct_expand_v=2,
        )

    def test_creates(self, config):
        from memoria.model.deltaproduct_layers import DeltaProductBlock
        block = DeltaProductBlock(config, layer_idx=0)
        assert block is not None

    def test_forward_shape(self, config):
        from memoria.model.deltaproduct_layers import DeltaProductBlock
        block = DeltaProductBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda')
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        assert out.shape == (2, 64, 128)

    def test_gradients_flow(self, config):
        from memoria.model.deltaproduct_layers import DeltaProductBlock
        block = DeltaProductBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda', requires_grad=True)
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


@requires_fla
@requires_cuda
class TestLogLinearDeltaProductBlock:
    @pytest.fixture
    def config(self):
        return TransformerConfig(
            vocab_size=256, sequence_len=128,
            n_layer=5, n_head=4, n_kv_head=4, n_embd=128,
            window_pattern='HHHHL',
            deltaproduct_head_dim=32,
            deltaproduct_n_householder=3,
            deltaproduct_expand_v=2,
            loglinear_chunk_size=32,  # small chunks for testing
        )

    def test_creates(self, config):
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0)
        assert block is not None
        assert block.num_householder == 3

    def test_has_level_projections(self, config):
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0)
        assert hasattr(block, 'l_proj')
        assert hasattr(block, 'L')
        assert block.L.shape == (4, MAX_NUM_LEVELS)

    def test_forward_shape(self, config):
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda')
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        assert out.shape == (2, 64, 128)

    def test_forward_different_from_zeros(self, config):
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda')
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        assert out.abs().sum() > 0

    def test_gradients_flow(self, config):
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda', requires_grad=True)
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_level_params_get_gradients(self, config):
        """Level scale params (L, l_proj) should receive gradients."""
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda')
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        loss = out.sum()
        loss.backward()
        assert block.L.grad is not None
        assert block.l_proj.weight.grad is not None

    def test_handles_non_chunk_aligned_length(self, config):
        """Sequence length not a multiple of chunk_size should still work."""
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        block = LogLinearDeltaProductBlock(config, layer_idx=0).cuda()
        # 50 tokens, chunk_size=32 → needs padding to 64
        x = torch.randn(2, 50, 128, device='cuda')
        cos = sin = torch.zeros(1, 50, 1, 32, device='cuda')
        out = block(x, cos, sin)
        assert out.shape == (2, 50, 128)


@requires_hattention
@requires_cuda
class TestLogLinearGDNBlock:
    @pytest.fixture
    def config(self):
        return TransformerConfig(
            vocab_size=256, sequence_len=128,
            n_layer=5, n_head=4, n_kv_head=4, n_embd=128,
            window_pattern='DDDEL',
            deltaproduct_head_dim=32,
            deltaproduct_expand_v=2,
        )

    def test_creates(self, config):
        from memoria.model.deltaproduct_layers import LogLinearGDNBlock
        block = LogLinearGDNBlock(config, layer_idx=0)
        assert block is not None

    def test_forward_shape(self, config):
        from memoria.model.deltaproduct_layers import LogLinearGDNBlock
        block = LogLinearGDNBlock(config, layer_idx=0).cuda()
        x = torch.randn(2, 64, 128, device='cuda')
        cos = sin = torch.zeros(1, 64, 1, 32, device='cuda')
        out = block(x, cos, sin)
        assert out.shape == (2, 64, 128)


# ═══════════════════════════════════════════════════════════════════════════
# Full model integration test (requires FLA + CUDA)
# ═══════════════════════════════════════════════════════════════════════════

@requires_fla
@requires_cuda
class TestFullModelIntegration:
    """Test that the full MemoriaModel works with DeltaProduct layers."""

    @pytest.fixture
    def config(self):
        return MemoriaConfig(
            transformer=TransformerConfig(
                vocab_size=256, sequence_len=128,
                n_layer=5, n_head=4, n_kv_head=4, n_embd=128,
                window_pattern='HHHHL',
                deltaproduct_head_dim=32,
                deltaproduct_n_householder=3,
                deltaproduct_expand_v=2,
                loglinear_chunk_size=32,
                mla_latent_dim=32,
                mla_rope_dim=16,
                interface_every=5, interface_num_heads=2, interface_top_k=8,
            ),
            state=StateConfig(
                belief_dim=64, max_beliefs=32, max_edges=64,
                max_goals=4, relation_dim=16,
            ),
            training=TrainingConfig(),
        )

    def test_model_creates_with_deltaproduct(self, config):
        from memoria.model.memoria_model import MemoriaModel
        model = MemoriaModel(config)
        model.init_weights()
        # Check that H layers got LogLinearDeltaProductBlock
        from memoria.model.deltaproduct_layers import LogLinearDeltaProductBlock
        h_layers = [b for b in model.transformer.blocks
                     if isinstance(b.attn, LogLinearDeltaProductBlock)]
        assert len(h_layers) == 4  # 4 H layers in HHHHL pattern with 5 total

    def test_model_forward(self, config):
        from memoria.model.memoria_model import MemoriaModel
        model = MemoriaModel(config).cuda()
        model.init_weights()
        tokens = torch.randint(0, 256, (2, 64), device='cuda')
        targets = torch.randint(0, 256, (2, 64), device='cuda')
        result = model(tokens, targets, alpha=0.0)
        assert 'loss' in result
        assert result['loss'].isfinite()

    def test_model_backward(self, config):
        from memoria.model.memoria_model import MemoriaModel
        model = MemoriaModel(config).cuda()
        model.init_weights()
        tokens = torch.randint(0, 256, (2, 64), device='cuda')
        targets = torch.randint(0, 256, (2, 64), device='cuda')
        result = model(tokens, targets, alpha=0.0)
        result['loss'].backward()
        # Check that DeltaProduct params got gradients
        for block in model.transformer.blocks:
            if hasattr(block.attn, 'q_proj'):
                assert block.attn.q_proj.weight.grad is not None
                break
