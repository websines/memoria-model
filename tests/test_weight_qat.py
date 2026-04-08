"""Tests for weight QAT (RotorQuant + CAGE)."""

import torch
import torch.nn as nn
import pytest

from memoria.core.quantize import (
    WeightQuantLinear, apply_weight_qat, cage_step, get_cage_lambda,
    _make_quantizer, ste_quantize, PolarQuantizer,
)
from memoria.model.config import (
    TransformerConfig, TrainingConfig, MemoriaConfig, small_config,
)


# ── WeightQuantLinear ──

class TestWeightQuantLinear:
    def test_creates(self):
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        assert wql.in_features == 128
        assert wql.out_features == 256
        assert wql.bits == 4

    def test_forward_train_injects_noise(self):
        """In training mode, output should differ from plain linear due to quantization noise."""
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        wql.train()
        x = torch.randn(2, 10, 128)
        out_quant = wql(x)
        out_plain = linear(x)
        # Should be close but not identical (quantization noise)
        assert out_quant.shape == out_plain.shape
        assert not torch.allclose(out_quant, out_plain, atol=1e-7)

    def test_forward_eval_matches_linear(self):
        """In eval mode, output should exactly match plain linear."""
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        wql.eval()
        x = torch.randn(2, 10, 128)
        out_quant = wql(x)
        out_plain = linear(x)
        assert torch.allclose(out_quant, out_plain, atol=1e-7)

    def test_gradients_flow(self):
        """STE should allow gradients to flow through."""
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        wql.train()
        x = torch.randn(2, 10, 128, requires_grad=True)
        out = wql(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert linear.weight.grad is not None

    def test_weight_property(self):
        """Weight property should expose the underlying linear weight."""
        linear = nn.Linear(64, 128, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        assert wql.weight is linear.weight

    def test_with_bias(self):
        """Should work with biased linear layers."""
        linear = nn.Linear(128, 256, bias=True)
        wql = WeightQuantLinear(linear, bits=4)
        wql.train()
        x = torch.randn(2, 10, 128)
        out = wql(x)
        assert out.shape == (2, 10, 256)

    def test_bf16_input(self):
        """Should work with bfloat16 input (H-block sends bf16 through projections)."""
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        wql.train()
        x = torch.randn(2, 10, 128, dtype=torch.bfloat16)
        out = wql(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 10, 256)

    def test_3bit_vs_4bit_noise(self):
        """3-bit should inject more noise than 4-bit."""
        linear = nn.Linear(128, 256, bias=False)
        wql3 = WeightQuantLinear(nn.Linear(128, 256, bias=False), bits=3)
        wql4 = WeightQuantLinear(nn.Linear(128, 256, bias=False), bits=4)
        # Copy same weights
        wql3.linear.weight.data.copy_(linear.weight.data)
        wql4.linear.weight.data.copy_(linear.weight.data)
        wql3.train()
        wql4.train()
        x = torch.randn(2, 10, 128)
        out_plain = linear(x)
        err3 = (wql3(x) - out_plain).abs().mean().item()
        err4 = (wql4(x) - out_plain).abs().mean().item()
        assert err3 > err4, "3-bit should have more quantization error than 4-bit"


# ── apply_weight_qat ──

class TestApplyWeightQAT:
    @pytest.fixture
    def model(self):
        """Create a small MemoriaModel for testing."""
        config = small_config()
        config.transformer.n_layer = 5
        config.transformer.vocab_size = 256
        config.transformer.sequence_len = 64
        config.transformer.n_embd = 128
        config.transformer.n_head = 4
        config.transformer.n_kv_head = 4
        config.transformer.deltaproduct_head_dim = 32
        config.transformer.mla_latent_dim = 32
        config.transformer.mla_rope_dim = 16
        config.transformer.engram_table_size = 100
        config.state.belief_dim = 64
        config.state.max_beliefs = 128
        config.state.max_edges = 256
        config.state.max_goals = 16
        config.state.relation_dim = 32
        from memoria.model.memoria_model import MemoriaModel
        model = MemoriaModel(config)
        model.init_weights()
        return model

    def test_patches_layers(self, model):
        """Should patch nn.Linear layers in transformer blocks."""
        patched = apply_weight_qat(model, bits=4, mlp_bits=3)
        assert len(patched) > 0

    def test_mlp_gets_different_bits(self, model):
        """MLP layers should get mlp_bits, others get bits."""
        patched = apply_weight_qat(model, bits=4, mlp_bits=3)
        mlp_entries = [p for p in patched if '3-bit' in p]
        attn_entries = [p for p in patched if '4-bit' in p]
        assert len(mlp_entries) > 0, "Some MLP layers should be 3-bit"
        assert len(attn_entries) > 0, "Some attention layers should be 4-bit"

    def test_skips_embeddings(self, model):
        """Should NOT patch embeddings or lm_head."""
        apply_weight_qat(model, bits=4)
        # Check wte and lm_head are still plain
        assert isinstance(model.transformer.wte, nn.Embedding)
        assert isinstance(model.transformer.lm_head, nn.Linear)

    def test_skips_interface_layers(self, model):
        """Should NOT patch state interface projections."""
        apply_weight_qat(model, bits=4)
        for interface in model.interfaces:
            for name, module in interface.named_modules():
                assert not isinstance(module, WeightQuantLinear), \
                    f"Interface module {name} should not be quantized"

    def test_forward_still_works(self, model):
        """Model should still produce valid output after patching."""
        apply_weight_qat(model, bits=4, mlp_bits=3)
        model.train()
        tokens = torch.randint(0, 256, (1, 32))
        targets = torch.randint(0, 256, (1, 32))
        result = model(tokens, targets, alpha=0.0)
        assert 'loss' in result
        assert not torch.isnan(result['loss'])

    def test_disabled_when_bits_zero(self, model):
        """bits=0 should patch nothing."""
        patched = apply_weight_qat(model, bits=0)
        assert len(patched) == 0


# ── CAGE ──

class TestCAGE:
    def test_cage_step_nudges_weights(self):
        """CAGE step should move weights closer to quantization grid."""
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        model = nn.Sequential(wql)

        # Compute initial quantization error
        w = linear.weight.data
        q = wql.quantizer
        if isinstance(q, PolarQuantizer):
            codes, scale = q.quantize(w)
            w_hat = q.dequantize(codes, scale)
        else:
            w_hat, _ = q(w)
        err_before = (w - w_hat).abs().mean().item()

        # Apply CAGE step
        cage_step(model, lr=0.01, cage_lambda=10.0)

        # Recompute error
        w = linear.weight.data
        if isinstance(q, PolarQuantizer):
            codes, scale = q.quantize(w)
            w_hat = q.dequantize(codes, scale)
        else:
            w_hat, _ = q(w)
        err_after = (w - w_hat).abs().mean().item()

        assert err_after < err_before, "CAGE should reduce quantization error"

    def test_cage_step_zero_lambda_noop(self):
        """CAGE with lambda=0 should not change weights."""
        linear = nn.Linear(128, 256, bias=False)
        wql = WeightQuantLinear(linear, bits=4)
        model = nn.Sequential(wql)
        w_before = linear.weight.data.clone()
        cage_step(model, lr=0.01, cage_lambda=0.0)
        assert torch.equal(linear.weight.data, w_before)


class TestCAGESchedule:
    @pytest.fixture
    def config(self):
        c = small_config()
        c.training.phase1_steps = 100
        c.training.phase2_steps = 200
        c.training.cage_lambda_base = 10.0
        c.training.cage_silence_ratio = 0.0  # phase-aligned
        return c

    def test_phase1_silent(self, config):
        assert get_cage_lambda(0, config) == 0.0
        assert get_cage_lambda(50, config) == 0.0
        assert get_cage_lambda(99, config) == 0.0

    def test_phase2_ramps(self, config):
        lam_start = get_cage_lambda(100, config)
        lam_mid = get_cage_lambda(200, config)
        lam_end = get_cage_lambda(299, config)
        assert lam_start == 0.0
        assert 0 < lam_mid < 10.0
        assert lam_mid == pytest.approx(5.0, abs=0.1)
        assert lam_end > lam_mid

    def test_phase3_full(self, config):
        assert get_cage_lambda(300, config) == 10.0
        assert get_cage_lambda(1000, config) == 10.0

    def test_manual_silence_ratio(self, config):
        config.training.cage_silence_ratio = 0.8
        config.training._total_steps = 1000
        assert get_cage_lambda(0, config) == 0.0
        assert get_cage_lambda(799, config) == 0.0
        lam = get_cage_lambda(900, config)
        assert 0 < lam < 10.0
        assert get_cage_lambda(1000, config) == 10.0
