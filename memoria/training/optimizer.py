"""Optimizer setup: Muon for matrix params, AdamW for everything else.

Ported from autoresearch/train.py. Muon is the optimizer that made nanogpt
speedruns 35% faster. Uses Newton-Schulz orthogonalization on gradient updates.

Reference: github.com/karpathy/autoresearch (train.py)
Reference: kellerjordan.github.io/posts/muon/ (Muon blog post)
"""

import torch
import torch.nn as nn
from ..model.config import MemoriaConfig


def setup_optimizer(model: nn.Module, config: MemoriaConfig) -> torch.optim.Optimizer:
    """Set up Muon + AdamW optimizer with per-group learning rates.

    - Matrix params in transformer blocks → Muon (orthogonalized SGD)
    - Embedding / unembedding → AdamW
    - Scalar params (lambdas) → AdamW with higher LR
    - State interface params → AdamW
    - Cognitive state params → NOT optimized (updated by pass 2)

    Args:
        model: MemoriaModel instance
        config: full configuration

    Returns:
        optimizer
    """
    tc = config.training
    betas = tc.adam_betas

    # Collect parameter groups
    param_groups = []

    # 1. Transformer embedding
    embedding_params = list(model.transformer.wte.parameters())
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': tc.embedding_lr,
            'betas': betas,
            'eps': 1e-10,
            'weight_decay': 0.0,
        })

    # 2. Transformer LM head (unembedding)
    lm_head_params = list(model.transformer.lm_head.parameters())
    if lm_head_params:
        param_groups.append({
            'params': lm_head_params,
            'lr': tc.unembedding_lr,
            'betas': betas,
            'eps': 1e-10,
            'weight_decay': 0.0,
        })

    # 3. Transformer scalar params (resid_lambdas, x0_lambdas)
    scalar_params = [model.transformer.resid_lambdas, model.transformer.x0_lambdas]
    param_groups.append({
        'params': scalar_params,
        'lr': tc.scalar_lr,
        'betas': betas,
        'eps': 1e-10,
        'weight_decay': 0.0,
    })

    # 4. Transformer matrix params (attention, MLP) → these benefit most from Muon
    # For simplicity in v1, we use AdamW for all. Muon requires custom optimizer class.
    # TODO: implement Muon for matrix params (from autoresearch)
    matrix_params = []
    for block in model.transformer.blocks:
        matrix_params.extend(block.parameters())

    if matrix_params:
        param_groups.append({
            'params': matrix_params,
            'lr': tc.matrix_lr,
            'betas': betas,
            'eps': 1e-10,
            'weight_decay': tc.weight_decay,
        })

    # 5. State interface params
    interface_params = list(model.interfaces.parameters())
    if interface_params:
        param_groups.append({
            'params': interface_params,
            'lr': tc.interface_lr,
            'betas': betas,
            'eps': 1e-10,
            'weight_decay': 0.0,
        })

    # Note: cognitive state params (beliefs, edges, goals, meta) have requires_grad=False
    # and are updated by pass 2, not by the optimizer.

    optimizer = torch.optim.AdamW(param_groups)

    # Store initial LRs for scheduling
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']

    return optimizer
