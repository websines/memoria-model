"""Optimizer setup: Muon for matrix params, AdamW for everything else.

Muon uses Newton-Schulz orthogonalization on gradient updates for matrix params,
achieving ~35% faster convergence than AdamW on transformer blocks.

Reference: github.com/karpathy/autoresearch (train.py)
Reference: kellerjordan.github.io/posts/muon/ (Muon blog post)
"""

import torch
import torch.nn as nn
from torch import Tensor
from ..model.config import MemoriaConfig


# ── Muon Optimizer ──

@torch.no_grad()
def _newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    """Newton-Schulz iteration for approximate matrix orthogonalization.

    Computes the orthogonal polar factor of G in O(steps) matrix multiplies.
    Coefficients from the Muon paper (degree-5 polynomial).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)

    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer: momentum + Newton-Schulz orthogonalization.

    For 2D+ parameters (attention, MLP weights), the gradient update is
    orthogonalized via Newton-Schulz iteration, making updates explore
    the loss landscape more efficiently.

    For 1D parameters (biases, norms), falls back to standard SGD with momentum.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    update = g + momentum * buf
                else:
                    update = buf.clone()

                # Newton-Schulz orthogonalization for 2D params
                if update.ndim >= 2:
                    original_shape = update.shape
                    update_2d = update.view(update.shape[0], -1) if update.ndim > 2 else update
                    # Capture flattened dims before orthogonalization so the
                    # aspect-ratio scale below uses (H, W*K*...) for ndim>2
                    # tensors — not original_shape[0]/original_shape[-1], which
                    # would drop all middle dims from the width.
                    flat_rows, flat_cols = update_2d.shape
                    update_2d = _newton_schulz(update_2d, ns_steps)
                    update = update_2d.view(original_shape)
                    # Muon paper §3.2 — RMS-norm scaling by flattened aspect ratio
                    update *= max(1, flat_rows / flat_cols) ** 0.5

                p.add_(update, alpha=-lr)


def setup_optimizer(model: nn.Module, config: MemoriaConfig) -> torch.optim.Optimizer:
    """Set up optimizer with per-group learning rates.

    Scratch mode (MemoriaModel):
    - 2D matrix params in transformer blocks (attention, MLP, DeltaProduct) → Muon
    - 1D params in transformer blocks (biases, norms) → AdamW
    - Embedding / unembedding → AdamW
    - Scalar params (lambdas) → AdamW with higher LR
    - State interface params → AdamW
    - Cognitive state params → NOT optimized (updated by pass 2)

    Pretrained mode (PretrainedMemoriaModel):
    - Backbone → frozen (not in optimizer at all)
    - State interface params + read gates → AdamW
    - Cognitive state params → NOT optimized (updated by pass 2)

    Args:
        model: MemoriaModel or PretrainedMemoriaModel instance
        config: full configuration

    Returns:
        optimizer
    """
    tc = config.training
    betas = tc.adam_betas

    if config.backbone == "pretrained":
        return _setup_pretrained_optimizer(model, config)

    # ── Scratch mode ──
    param_groups = []

    # 1. Transformer embedding
    embedding_params = list(model.transformer.wte.parameters())
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': tc.embedding_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 2. Transformer LM head (unembedding)
    lm_head_params = list(model.transformer.lm_head.parameters())
    if lm_head_params:
        param_groups.append({
            'params': lm_head_params,
            'lr': tc.unembedding_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 3. Transformer scalar params (resid_lambdas, x0_lambdas)
    scalar_params = [model.transformer.resid_lambdas, model.transformer.x0_lambdas]
    param_groups.append({
        'params': scalar_params,
        'lr': tc.scalar_lr,
        'betas': betas,
        'eps': 1e-8,
        'weight_decay': 0.0,
    })

    # 4. Transformer matrix params (attention, MLP) → Muon for 2D, AdamW for 1D
    matrix_params_2d = []
    matrix_params_1d = []
    for block in model.transformer.blocks:
        for p in block.parameters():
            if p.ndim >= 2:
                matrix_params_2d.append(p)
            else:
                matrix_params_1d.append(p)

    # 1D params (biases, norms) still use AdamW
    if matrix_params_1d:
        param_groups.append({
            'params': matrix_params_1d,
            'lr': tc.scalar_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 5. State interface params
    interface_params = list(model.interfaces.parameters())
    if interface_params:
        param_groups.append({
            'params': interface_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 6. Cognitive state continuous params (beliefs, edge_weights, edge_relations)
    # NO weight decay: beliefs are memory entries, not network weights.
    # Their lifecycle (creation, consolidation, eviction, sleep-cycle
    # homeostatic scaling) is managed entirely by pass 2. Weight decay
    # uniformly shrinks radii every step, preventing beliefs from ever
    # building precision — the opposite of what a memory system needs.
    cognitive_params = [p for p in [model.state.beliefs, model.state.edge_weights, model.state.edge_relations] if p.requires_grad]
    if cognitive_params:
        param_groups.append({
            'params': cognitive_params,
            'lr': tc.belief_lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 7. Cognitive meta-parameters (learned thresholds replacing magic numbers)
    cognitive_meta_params = list(model.state.meta_params.parameters())
    if cognitive_meta_params:
        param_groups.append({
            'params': cognitive_meta_params,
            'lr': tc.cognitive_meta_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 8. Telos module (learned goal generation, progress, transitions)
    telos_params = [p for p in model.state.telos.parameters() if p.requires_grad]
    if telos_params:
        param_groups.append({
            'params': telos_params,
            'lr': tc.interface_lr,  # same LR as interface layers
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 9. Goal embeddings (differentiable, slow LR like beliefs)
    if model.state.goal_embeddings.requires_grad:
        param_groups.append({
            'params': [model.state.goal_embeddings],
            'lr': tc.belief_lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': tc.weight_decay * 0.5,
        })

    # 10. In-Place TTT module (fast-weight deltas, lr modulator)
    if hasattr(model, 'ttt'):
        ttt_params = [p for p in model.ttt.parameters() if p.requires_grad]
        if ttt_params:
            param_groups.append({
                'params': ttt_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 11. Edge proposal network (learned edge creation)
    edge_proposal_params = [p for p in model.state.edge_proposal.parameters() if p.requires_grad]
    if edge_proposal_params:
        param_groups.append({
            'params': edge_proposal_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 12. Edge directions (CoED, learned per-edge direction angles)
    if model.state.edge_direction.requires_grad:
        param_groups.append({
            'params': [model.state.edge_direction],
            'lr': tc.belief_lr * 10,  # faster than beliefs (structural, not content)
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0,  # no decay — directions should persist
        })

    # 13. Cognitive controller (SEAL-style learned pass2 policy)
    controller_params = [p for p in model.state.controller.parameters() if p.requires_grad]
    if controller_params:
        param_groups.append({
            'params': controller_params,
            'lr': tc.interface_lr * 0.1,  # slow — controller should be stable
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 14. SleepGate (learned sleep cycle consolidation)
    sleep_params = [p for p in model.state.sleep_gate.parameters() if p.requires_grad]
    if sleep_params:
        param_groups.append({
            'params': sleep_params,
            'lr': tc.interface_lr * 0.1,  # slow — consolidation should be stable
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 15. Message passing (relation transform + DEQ parameters)
    mp_params = [p for p in model.state.message_passing.parameters() if p.requires_grad]
    if mp_params:
        param_groups.append({
            'params': mp_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 16. Kendall/Gal uncertainty weighting parameters (log_sigma per loss group)
    if hasattr(model, 'log_sigma'):
        sigma_params = list(model.log_sigma.parameters())
        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': tc.scalar_lr if hasattr(tc, 'scalar_lr') else 0.5,
                'betas': (0.8, 0.95),
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 17. Hypothesis generator (autoresearch loop)
    if hasattr(model.state, 'hypothesis_gen'):
        hyp_params = [p for p in model.state.hypothesis_gen.parameters() if p.requires_grad]
        if hyp_params:
            param_groups.append({
                'params': hyp_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 17b. Strategy bank + selector + evolver (LSR refinement loop strategies).
    # strategy_bank is an nn.Parameter used in forward (perturbation injection).
    # strategy_selector/evolver have learned weights for selection/generation.
    # Without optimizer registration, their .grad accumulates unboundedly
    # because zero_grad only touches params it owns — a latent NaN source.
    if model.state._strategy_bank_initialized:
        strat_params = [model.state.strategy_bank]
        if model.state.strategy_selector is not None:
            strat_params.extend(
                p for p in model.state.strategy_selector.parameters() if p.requires_grad
            )
        if model.state.strategy_evolver is not None:
            strat_params.extend(
                p for p in model.state.strategy_evolver.parameters() if p.requires_grad
            )
        if strat_params:
            param_groups.append({
                'params': strat_params,
                'lr': tc.interface_lr * 0.1,  # slow — strategies evolve via pass2, not gradient
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 18a. BLT byte encoder (byte embedding, N-gram conv, local DeltaProduct, pooling)
    if hasattr(model, 'blt_enabled') and model.blt_enabled:
        enc_params = [p for p in model.byte_encoder.parameters() if p.requires_grad]
        if enc_params:
            param_groups.append({
                'params': enc_params,
                'lr': tc.interface_lr,  # same LR as interface layers
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 18b. BLT byte decoder (down projection, local DeltaProduct, byte heads)
    if hasattr(model, 'blt_enabled') and model.blt_enabled:
        dec_params = [p for p in model.byte_decoder.parameters() if p.requires_grad]
        if dec_params:
            param_groups.append({
                'params': dec_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 19. DFlash draft head (block diffusion speculative decoding)
    # (was group 18 before BLT groups were added)
    # Includes: draft layers, mask_embed, pos_embed, feature_proj, out_norm,
    # and per-layer KV injection projections (k_inject, v_inject).
    # KV injection projections are RotorQuant 3-bit eligible (marked with _qat_bits).
    if hasattr(model, 'dflash_enabled') and model.dflash_enabled:
        dflash_params = [p for p in model.dflash_head.parameters() if p.requires_grad]
        if dflash_params:
            param_groups.append({
                'params': dflash_params,
                'lr': tc.interface_lr,  # same LR as interface layers
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 19. Goal routers (PARL-style per-head goal assignment in read paths)
    # NOTE: GoalRouter params are already included in group 5 (interface params)
    # since model.interfaces.parameters() traverses all submodules including
    # read_path.goal_router. No separate group needed — they train at interface_lr.

    # 20. Refinement router (MoR-style per-position adaptive refinement)
    if hasattr(model, 'refinement_router'):
        router_params = [p for p in model.refinement_router.parameters() if p.requires_grad]
        if router_params:
            param_groups.append({
                'params': router_params,
                'lr': tc.interface_lr,  # same LR as interface layers
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 21. Working memory prefix (Mamba-inspired learnable scratchpad)
    if hasattr(model, 'working_memory') and model.working_memory_size > 0:
        param_groups.append({
            'params': [model.working_memory],
            'lr': tc.scalar_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 22. Refinement probe + gate (halt decision + lifeline/loop encoding)
    if hasattr(model, 'refinement_probe'):
        probe_params = [p for p in model.refinement_probe.parameters() if p.requires_grad]
        if probe_params:
            param_groups.append({
                'params': probe_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })
    if hasattr(model, 'refinement_gate'):
        param_groups.append({
            'params': [model.refinement_gate],
            'lr': tc.scalar_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # 23. Engram cache (N-gram hash tables, value projection, gate norms)
    if hasattr(model, 'engram_cache'):
        engram_params = [p for p in model.engram_cache.parameters() if p.requires_grad]
        if engram_params:
            param_groups.append({
                'params': engram_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # 24-33. Cognitive subsystems (C1–D4). These modules have standard nn.Linear
    # parameters that need SGD updates — without a group, their .grad is zeroed
    # each step by the orphan-grad cleanup in train.py and they never learn.
    # LR selection follows the existing convention:
    #   - fast-adaptation nets (match forward path latency) → interface_lr
    #   - slow meta/structural controllers (shouldn't oscillate) → interface_lr * 0.1
    #   - content embedding banks → belief_lr (same as goal_embeddings)
    _add_cognitive_subsystem_groups(param_groups, model, tc, betas)

    # AdamW for non-matrix params
    adamw_optimizer = torch.optim.AdamW(param_groups) if param_groups else None

    # Muon for 2D matrix params (attention, MLP weights)
    muon_optimizer = Muon(matrix_params_2d, lr=tc.matrix_lr) if matrix_params_2d else None

    # Combine into a single interface
    optimizer = _CombinedOptimizer(adamw_optimizer, muon_optimizer)

    # Store initial LRs for scheduling
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']

    _assert_optimizer_covers_model(optimizer, model)

    return optimizer


def _add_cognitive_subsystem_groups(param_groups: list, model: nn.Module, tc, betas) -> None:
    """Register the 10 cognitive subsystems (C1–D4) with the optimizer.

    Each is guarded by hasattr so models instantiated without a subsystem
    (future config variants) still produce a valid optimizer.
    """
    # Fast-adaptation networks — operate inside the forward path, should track
    # model LR so their updates arrive on the same timescale as the backbone.
    fast_subsystems = [
        ('srwm',             'state.srwm'),
        ('adaptive_depth',   'state.adaptive_depth'),
        ('daemon',           'state.daemon'),
        ('curiosity',        'state.curiosity'),
        ('action_selector',  'state.action_selector'),
        ('skill_composer',   'state.skill_composer'),
        ('skill_detector',   'state.skill_detector'),
    ]
    for attr, _label in fast_subsystems:
        if hasattr(model.state, attr):
            mod = getattr(model.state, attr)
            ps = [p for p in mod.parameters() if p.requires_grad]
            if ps:
                param_groups.append({
                    'params': ps,
                    'lr': tc.interface_lr,
                    'betas': betas,
                    'eps': 1e-8,
                    'weight_decay': 0.0,
                })

    # Slow meta / structural controllers — decide consolidation and learned
    # update rules. Oscillation here is destabilizing (they modulate the
    # training signal for every other subsystem), so 10× slower than fast nets
    # — same convention used for controller and sleep_gate.
    slow_subsystems = [
        ('learned_update',          'state.learned_update'),
        ('structural_plasticity',   'state.structural_plasticity'),
    ]
    for attr, _label in slow_subsystems:
        if hasattr(model.state, attr):
            mod = getattr(model.state, attr)
            ps = [p for p in mod.parameters() if p.requires_grad]
            if ps:
                param_groups.append({
                    'params': ps,
                    'lr': tc.interface_lr * 0.1,
                    'betas': betas,
                    'eps': 1e-8,
                    'weight_decay': 0.0,
                })

    # Skill embedding bank — a learnable content store, same pattern as
    # goal_embeddings: belief_lr + half weight decay.
    if hasattr(model.state, 'skill_bank'):
        sb = model.state.skill_bank
        sb_params = [p for p in sb.parameters() if p.requires_grad]
        if sb_params:
            param_groups.append({
                'params': sb_params,
                'lr': tc.belief_lr,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': tc.weight_decay * 0.5,
            })


def _assert_optimizer_covers_model(optimizer, model: nn.Module) -> None:
    """Verify every trainable parameter is in exactly one optimizer group.

    Three classes of silent training bugs this catches:

    1. Duplicate registration (HARD ERROR) — a param in two groups gets a
       double update per step with inconsistent LR/beta. Never intentional.

    2. Missing registration (WARNING) — a trainable param that no group
       owns. orphan-grad cleanup zeros its .grad every step so it never
       moves, but gradients still accumulate in the forward/backward path.
       Usually either:
         - the module needs to be added to setup_optimizer(), or
         - the module is pass2-only updated and should have requires_grad=False.
       Downgraded to a warning because the correct fix is a modeling
       decision, not something this function can safely automate.

    Note: a THIRD silent bug class — "param in optimizer but never reachable
    under autograd because its forward runs under no_grad" — is NOT detected
    here (it requires a live training step). The companion function
    `assert_gradient_reachability()` detects it at startup by running a
    synthetic forward under autograd and flagging any optimizer param whose
    .grad stays None. Call that separately, after prepare() + warmup.
    """
    import sys
    seen: dict[int, str] = {}
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    for g in optimizer.param_groups:
        for p in g['params']:
            pid = id(p)
            if pid in seen:
                raise RuntimeError(
                    f"Optimizer: parameter {name_by_id.get(pid, '<unknown>')} "
                    f"(shape {tuple(p.shape)}) registered in multiple groups — "
                    f"would be double-updated per step."
                )
            seen[pid] = name_by_id.get(pid, '<unknown>')
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in seen]
    if missing:
        # Cluster by top-level module so the report is scannable.
        from collections import defaultdict
        by_subsystem: dict[str, list[str]] = defaultdict(list)
        for n in missing:
            parts = n.split('.')
            key = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
            by_subsystem[key].append(n)
        print(
            f"\n[optimizer] WARNING: {len(missing)} trainable param(s) not in "
            f"any optimizer group — their .grad will be zeroed each step but "
            f"they will never be updated. Subsystems:",
            file=sys.stderr,
        )
        for k in sorted(by_subsystem):
            print(f"  - {k}: {len(by_subsystem[k])} params", file=sys.stderr)
        print(
            "  Fix: add to setup_optimizer() (train via SGD) OR set "
            "requires_grad=False (pass2-only updates).\n",
            file=sys.stderr,
        )


def assert_gradient_reachability(
    optimizer,
    model: nn.Module,
    post_backward: bool = True,
) -> dict:
    """Runtime check: every optimizer param saw a non-None .grad.

    Call this AFTER the first real training backward pass. Any param that
    was registered in a group but whose .grad is still None is unreachable
    under autograd (usually because its forward call site is wrapped in
    torch.no_grad()). Such params exist as dead weight — AdamW skips them,
    they stay at their init values forever, and the subsystem they belong
    to runs as a deterministic function of random init.

    This is the third silent-bug class that `_assert_optimizer_covers_model`
    does NOT catch because it requires a live step to detect.

    Args:
        optimizer: the configured optimizer (_CombinedOptimizer or AdamW)
        model: the top-level model
        post_backward: if True, assumes a backward just completed so .grad
            has been set for reachable params. Call right before the
            corresponding zero_grad().

    Returns:
        dict with keys:
          'reachable':  list of (name, shape) for params with non-None .grad
          'unreachable': list of (name, shape) for params with None .grad
          'unreachable_by_subsystem': {top_level_module: count}
    """
    import sys
    from collections import defaultdict

    name_by_id = {id(p): n for n, p in model.named_parameters()}
    reachable: list[tuple[str, tuple]] = []
    unreachable: list[tuple[str, tuple]] = []
    for g in optimizer.param_groups:
        for p in g['params']:
            pid = id(p)
            name = name_by_id.get(pid, '<unknown>')
            shape = tuple(p.shape)
            if p.grad is None:
                unreachable.append((name, shape))
            else:
                reachable.append((name, shape))

    by_subsystem: dict[str, int] = defaultdict(int)
    for name, _ in unreachable:
        parts = name.split('.')
        key = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        by_subsystem[key] += 1

    if unreachable and post_backward:
        print(
            f"\n[optimizer] WARNING: {len(unreachable)} optimizer param(s) "
            f"have None grad after backward — forward call site likely runs "
            f"under torch.no_grad(). These params will never update:",
            file=sys.stderr,
        )
        for k in sorted(by_subsystem):
            print(f"  - {k}: {by_subsystem[k]} params", file=sys.stderr)
        print(
            "  Fix options: (a) remove no_grad() if gradient path was intended, "
            "(b) add a dedicated compute_loss() + backward for that module "
            "(see CognitiveController pattern), or (c) set requires_grad=False "
            "and remove the group to honestly document the module as frozen.\n",
            file=sys.stderr,
        )

    return {
        'reachable': reachable,
        'unreachable': unreachable,
        'unreachable_by_subsystem': dict(by_subsystem),
    }


class _CombinedOptimizer:
    """Wraps AdamW + Muon into a single optimizer interface.

    LR scheduling works via param_groups (both optimizers' groups are exposed).
    """

    def __init__(self, adamw: torch.optim.Optimizer | None, muon: Muon | None):
        self.adamw = adamw
        self.muon = muon
        self._optimizers = [o for o in [adamw, muon] if o is not None]

    @property
    def param_groups(self) -> list:
        groups = []
        for o in self._optimizers:
            groups.extend(o.param_groups)
        return groups

    def step(self):
        for o in self._optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = False):
        for o in self._optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            'optimizers': [o.state_dict() for o in self._optimizers],
        }

    def load_state_dict(self, state_dict: dict):
        saved = state_dict['optimizers']
        if len(saved) != len(self._optimizers):
            raise ValueError(
                f"Optimizer count mismatch: checkpoint has {len(saved)}, "
                f"current model has {len(self._optimizers)}. "
                f"Cannot resume across different optimizer configurations."
            )
        for o, sd in zip(self._optimizers, saved):
            o.load_state_dict(sd)


def _setup_pretrained_optimizer(model: nn.Module, config: MemoriaConfig) -> torch.optim.Optimizer:
    """Optimizer for pretrained mode: interface layers + cognitive state + telos."""
    tc = config.training
    betas = tc.adam_betas

    param_groups = [
        {
            'params': list(model.interfaces.parameters()),
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        },
        {
            'params': [model.read_gate],
            'lr': tc.interface_lr * 10,  # gates learn faster
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        },
    ]

    # Cognitive state (beliefs, edges, relations) -- no weight decay (see group 6 comment)
    cognitive_params = [p for p in [model.state.beliefs, model.state.edge_weights, model.state.edge_relations] if p.requires_grad]
    if cognitive_params:
        param_groups.append({
            'params': cognitive_params,
            'lr': tc.belief_lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Cognitive meta-parameters
    cognitive_meta_params = list(model.state.meta_params.parameters())
    if cognitive_meta_params:
        param_groups.append({
            'params': cognitive_meta_params,
            'lr': tc.cognitive_meta_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Telos module (surprise predictor, goal generator, transition net, progress head)
    telos_params = [p for p in model.state.telos.parameters() if p.requires_grad]
    if telos_params:
        param_groups.append({
            'params': telos_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Goal embeddings
    if model.state.goal_embeddings.requires_grad:
        param_groups.append({
            'params': [model.state.goal_embeddings],
            'lr': tc.belief_lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': tc.weight_decay * 0.5,
        })

    # Edge proposal network
    edge_proposal_params = [p for p in model.state.edge_proposal.parameters() if p.requires_grad]
    if edge_proposal_params:
        param_groups.append({
            'params': edge_proposal_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Edge directions
    if model.state.edge_direction.requires_grad:
        param_groups.append({
            'params': [model.state.edge_direction],
            'lr': tc.belief_lr * 10,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Cognitive controller
    controller_params = [p for p in model.state.controller.parameters() if p.requires_grad]
    if controller_params:
        param_groups.append({
            'params': controller_params,
            'lr': tc.interface_lr * 0.1,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # In-Place TTT module (step-size modulators + meta-learned init)
    if hasattr(model, 'ttt'):
        ttt_params = [p for p in model.ttt.parameters() if p.requires_grad]
        if ttt_params:
            param_groups.append({
                'params': ttt_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # SleepGate (learned sleep cycle)
    sleep_params = [p for p in model.state.sleep_gate.parameters() if p.requires_grad]
    if sleep_params:
        param_groups.append({
            'params': sleep_params,
            'lr': tc.interface_lr * 0.1,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Message passing (relation transform + DEQ parameters)
    mp_params = [p for p in model.state.message_passing.parameters() if p.requires_grad]
    if mp_params:
        param_groups.append({
            'params': mp_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Kendall/Gal uncertainty weighting parameters (log_sigma per loss group)
    if hasattr(model, 'log_sigma'):
        sigma_params = list(model.log_sigma.parameters())
        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': tc.scalar_lr if hasattr(tc, 'scalar_lr') else 0.5,
                'betas': (0.8, 0.95),
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # Hypothesis generator (autoresearch loop)
    if hasattr(model.state, 'hypothesis_gen'):
        hyp_params = [p for p in model.state.hypothesis_gen.parameters() if p.requires_grad]
        if hyp_params:
            param_groups.append({
                'params': hyp_params,
                'lr': tc.interface_lr,
                'betas': betas,
                'eps': 1e-8,
                'weight_decay': 0.0,
            })

    # Cognitive subsystems (C1–D4) — see scratch-mode comment.
    _add_cognitive_subsystem_groups(param_groups, model, tc, betas)

    optimizer = torch.optim.AdamW(param_groups)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']

    _assert_optimizer_covers_model(optimizer, model)

    return optimizer
