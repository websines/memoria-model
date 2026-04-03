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
                    update_2d = _newton_schulz(update_2d, ns_steps)
                    update = update_2d.view(original_shape)
                    # Scale: larger matrices get proportionally larger updates
                    update *= max(1, update.shape[0] / update.shape[-1]) ** 0.5

                p.add_(update, alpha=-lr)


def setup_optimizer(model: nn.Module, config: MemoriaConfig) -> torch.optim.Optimizer:
    """Set up optimizer with per-group learning rates.

    Scratch mode (MemoriaModel):
    - Matrix params in transformer blocks → AdamW (TODO: Muon)
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
    # Slow LR + weight decay replaces: radius clamp, edge death thresholds, sequence boundary decay
    cognitive_params = [p for p in [model.state.beliefs, model.state.edge_weights, model.state.edge_relations] if p.requires_grad]
    if cognitive_params:
        param_groups.append({
            'params': cognitive_params,
            'lr': tc.belief_lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': tc.weight_decay,
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

    # AdamW for non-matrix params
    adamw_optimizer = torch.optim.AdamW(param_groups) if param_groups else None

    # Muon for 2D matrix params (attention, MLP weights)
    muon_optimizer = Muon(matrix_params_2d, lr=tc.matrix_lr) if matrix_params_2d else None

    # Combine into a single interface
    optimizer = _CombinedOptimizer(adamw_optimizer, muon_optimizer)

    # Store initial LRs for scheduling
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']

    return optimizer


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
        for o, sd in zip(self._optimizers, state_dict['optimizers']):
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

    # Cognitive state (beliefs, edges, relations)
    cognitive_params = [p for p in [model.state.beliefs, model.state.edge_weights, model.state.edge_relations] if p.requires_grad]
    if cognitive_params:
        param_groups.append({
            'params': cognitive_params,
            'lr': tc.belief_lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': tc.weight_decay,
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

    # In-Place TTT module (step-size modulators)
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

    optimizer = torch.optim.AdamW(param_groups)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']

    return optimizer
