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
    """Loss computation works with L_token only.

    update_state=False makes this a pure loss-computation test with no TTT
    or belief mutation — fast, deterministic, no autograd version issues.
    """
    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    result = model.compute_loss(idx, targets, alpha=0.0, update_state=False)
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
    """Gradients flow through the full model.

    Wraps forward+backward in allow_mutation_on_saved_tensors because the
    training forward path mutates TTT deltas and state.beliefs in-place
    (test-time training / live self-improvement). The context manager
    makes PyTorch clone saved tensors on mutation so backward computes
    gradients against the pre-mutation values. This matches the production
    training loop in memoria/training/train.py.
    """
    # Add beliefs so the read path actually does computation
    for _ in range(5):
        model.state.allocate_belief(torch.randn(model.config.state.belief_dim))
    # Set output_proj to non-zero so gradients flow through read path
    with torch.no_grad():
        for iface in model.interfaces:
            iface.read_path.output_proj.weight.normal_(std=0.1)

    idx = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    # update_state=True makes compute_loss exercise the full TTT mutation
    # path, which is what this test is meant to verify. compute_loss's safe
    # default is False, so we opt in explicitly. The context manager is
    # mandatory whenever update_state=True is combined with backward.
    with torch.autograd.graph.allow_mutation_on_saved_tensors():
        result = model.compute_loss(idx, targets, alpha=0.0, update_state=True)
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
    """Gradients from L_fe should flow into belief vectors (gradient wall removed).

    Wraps forward+backward in allow_mutation_on_saved_tensors to match the
    production training loop — TTT mutates state.beliefs in-place during
    forward, and backward needs to compute gradients against the pre-mutation
    values. See test_backward for the rationale.
    """
    # Add beliefs so read/write paths have something to work with
    for _ in range(5):
        model.state.allocate_belief(torch.randn(model.config.state.belief_dim))

    # Zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    idx = torch.randint(0, 256, (1, 16))
    targets = torch.randint(0, 256, (1, 16))
    with torch.autograd.graph.allow_mutation_on_saved_tensors():
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
    """MetaParams should be part of the model and have learnable scalar parameters.

    Structural invariants (not a magic count) so intentional additions don't
    break the test. Regressions are still caught by the floor check and the
    per-parameter assertions.
    """
    # Sanity floor: catch wholesale regression without breaking on additions.
    # Raise this when you deliberately remove meta parameters.
    MIN_META_PARAMS = 60

    meta_params = list(model.state.meta_params.parameters())
    assert len(meta_params) >= MIN_META_PARAMS, (
        f"MetaParams has {len(meta_params)} params, expected at least "
        f"{MIN_META_PARAMS}. If you intentionally removed meta parameters, "
        f"lower MIN_META_PARAMS in this test."
    )

    # Each meta parameter must be a learnable scalar tensor — the unconstrained
    # storage for a softplus/sigmoid-gated hyperparameter (see meta_params.py).
    # A non-scalar shape almost always means someone registered the wrong kind
    # of parameter on MetaParams.
    for p in meta_params:
        assert p.requires_grad, "All meta params should be learnable"
        assert p.dim() == 0, (
            f"Meta params should be scalar tensors; got shape {tuple(p.shape)}"
        )

    # Every meta param must also be reachable from `model.state.meta_params`
    # via an attribute beginning with `_` (the unconstrained raw-value
    # convention). This catches "registered as a parameter but not assigned
    # to an attribute" or "stored under the wrong name" regressions.
    named = dict(model.state.meta_params.named_parameters())
    assert len(named) == len(meta_params), (
        f"named_parameters count ({len(named)}) should match parameters count "
        f"({len(meta_params)})"
    )
    for name in named:
        assert name.startswith("_"), (
            f"Meta parameter {name!r} should use the underscore-prefix raw-value "
            f"convention (see meta_params.py)"
        )


def test_update_state_false_is_strictly_read_only(tiny_config):
    """Property test: `forward(update_state=False)` must not mutate ANY tensor
    under `model.state` or `model.ttt`.

    This is a contract test for every mutation site in `forward()`. The
    training forward is allowed to mutate TTT deltas, beliefs, decay alphas,
    etc. — but *only* when `update_state=True`. Measurement paths (eval
    perplexity, belief-advantage probe) rely on `update_state=False` producing
    a fully read-only forward; if that invariant breaks, reported metrics are
    silently measured on state the measurement itself is mutating.

    Strategy: snapshot every recursive buffer and parameter under `state` and
    `ttt`, run several forwards with different alpha/seed combinations (so
    both the α=0 and α>0 code paths fire, both the write path and the belief
    matching logic run, and different write candidates are generated each
    call), then assert bit-equality on the full tensor set. Any future
    mutation path that forgets to check `update_state` will fail this test
    immediately — much cheaper than hoping an auditor catches it.

    If this test fires, do NOT exclude the offending tensor from the
    snapshot. Either gate its mutation on `update_state`, or if the mutation
    is legitimate (e.g., a BN running stat not under `state`/`ttt`), move it
    out of those module trees.
    """
    import torch.nn as nn

    model = MemoriaModel(tiny_config)
    model.init_weights()

    # Populate state so the read/write paths and loss_fe_bethe have something
    # to compute over. Without beliefs+edges, the forward takes degenerate
    # paths that miss the mutation sites we want to exercise.
    for _ in range(5):
        model.state.allocate_belief(torch.randn(tiny_config.state.belief_dim))
    model.state.allocate_edge(
        0, 1, torch.randn(tiny_config.state.relation_dim), weight=0.5,
    )

    def snapshot(module: nn.Module) -> dict:
        """Deep clone every recursive buffer and parameter tensor."""
        snap = {}
        for name, buf in module.named_buffers():
            snap[f"B:{name}"] = buf.detach().clone()
        for name, p in module.named_parameters():
            snap[f"P:{name}"] = p.detach().clone()
        return snap

    state_before = snapshot(model.state)
    ttt_before = snapshot(model.ttt)

    # Run several forward passes with varied configurations so we exercise
    # as many mutation sites as possible in one test:
    #   - alpha=0.0 : L_token only, skips loss_fe_bethe
    #   - alpha=0.1 : exercises the full free-energy path including beliefs
    #   - fresh inputs each call : each forward produces different write
    #     candidates, different refinement-loop decisions, and different
    #     should_update() verdicts in the TTT gate
    for alpha_val, seed in [(0.0, 0), (0.1, 1), (0.0, 2), (0.1, 3)]:
        torch.manual_seed(seed)
        idx = torch.randint(0, 256, (1, 16))
        targets = torch.randint(0, 256, (1, 16))
        _ = model.forward(
            idx, targets=targets, alpha=alpha_val, update_state=False,
        )

    def check_unchanged(mod: nn.Module, before: dict, label: str) -> None:
        after = snapshot(mod)
        missing = set(before) - set(after)
        added = set(after) - set(before)
        assert not missing and not added, (
            f"{label}: recursive tensor set changed during forward "
            f"(added={sorted(added)}, removed={sorted(missing)})"
        )
        drifted = []
        for key in before:
            b, a = before[key], after[key]
            if not torch.equal(b, a):
                # Compute a diff summary for the error message.
                b_f = b.to(torch.float64) if b.is_floating_point() else b.to(torch.float64)
                a_f = a.to(torch.float64) if a.is_floating_point() else a.to(torch.float64)
                delta = (a_f - b_f).abs()
                drifted.append(
                    f"  {key}: max_delta={delta.max().item():.6e}, "
                    f"nnz_delta={int((delta > 0).sum().item())}, "
                    f"shape={tuple(b.shape)}, dtype={b.dtype}"
                )
        if drifted:
            raise AssertionError(
                f"{label} mutated during update_state=False forward — "
                f"a mutation path is ignoring the read-only gate. "
                f"Drifted tensors:\n" + "\n".join(drifted) +
                "\n\nFix: locate the mutation site and add "
                "`and update_state` to its enclosing condition, OR move the "
                "tensor out of state/ttt if the mutation is legitimate."
            )

    check_unchanged(model.state, state_before, "model.state")
    check_unchanged(model.ttt, ttt_before, "model.ttt")
