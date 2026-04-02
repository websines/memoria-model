"""Shared loss utilities and differentiable free energy computation.

Contains:
- chunked_cross_entropy: memory-efficient cross-entropy for large vocabularies
- compute_differentiable_free_energy: proxy free energy over forward-pass tensors,
  providing actual gradient signal to interface layers

The original compute_free_energy() in free_energy.py operates on detached state
tensors (requires_grad=False) and produces zero gradients for all trainable params.
This module provides the differentiable alternative used as the L_fe training loss.

compute_free_energy() is still used for beta computation and logging in Pass 2.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def chunked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -1,
) -> Tensor:
    """Cross-entropy computed in chunks to avoid materializing [B*T, vocab] in memory.

    Standard for large-vocab models (151K vocab × 8K tokens = 9+ GiB in float32).
    Chunks along the sequence dimension to keep peak memory bounded.
    """
    BT, V = logits.shape
    if BT <= chunk_size:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    total_loss = 0.0
    n_tokens = 0
    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        chunk_logits = logits[start:end]
        chunk_targets = targets[start:end]
        valid = (chunk_targets != ignore_index).sum().item()
        if valid > 0:
            total_loss = total_loss + F.cross_entropy(
                chunk_logits, chunk_targets, ignore_index=ignore_index, reduction='sum'
            )
            n_tokens += valid

    return total_loss / max(n_tokens, 1)


def compute_differentiable_free_energy(
    attn_weights: list[Tensor],
    retrieved: list[Tensor],
    observations: list[Tensor],
    belief_dim: int,
    fe_lambda: float | None = None,
) -> Tensor:
    """Compute free energy proxy over forward-pass tensors (fully differentiable).

    F_proxy = E_consistency - lambda * H_attention

    Where:
    - E_consistency: disagreement between retrieved beliefs and observations.
      Trains read path to attend to beliefs consistent with current input,
      and write path to project compatible observations.
    - H_attention: entropy of attention distribution over beliefs.
      Prevents overconfident retrieval; encourages appropriate uncertainty.

    Gradients flow to:
    - query_proj (via attention weights in retrieved)
    - obs_proj (via observation vectors)
    - write_gate / precision_head (via observation scaling)
    - goal_gate, log_temperature (via attention scores)

    Args:
        attn_weights: per-layer [B, T, num_heads, N_active] attention tensors
        retrieved: per-layer [B, T, num_heads, D] retrieved belief representations
        observations: per-layer [B, T, D] observation projections from write path
        belief_dim: D, for reshaping

    Returns:
        Scalar free energy proxy (differentiable w.r.t. interface parameters)
    """
    if not attn_weights or not observations:
        device = attn_weights[0].device if attn_weights else torch.device('cpu')
        return torch.tensor(0.0, device=device, requires_grad=True)

    device = attn_weights[0].device
    energy_terms = []
    entropy_terms = []

    for attn, ret, obs in zip(attn_weights, retrieved, observations):
        # Skip layers with no active beliefs (attn has N_active=0)
        if attn.shape[-1] == 0:
            continue

        # --- Energy: disagreement between retrieved and observed ---
        # retrieved: [B, T, num_heads, D], observations: [B, T, D]
        ret_avg = ret.mean(dim=2)  # [B, T, D] — average over heads
        cos_sim = F.cosine_similarity(ret_avg, obs, dim=-1)  # [B, T]
        energy_terms.append((1.0 - cos_sim).mean())

        # --- Entropy: attention distribution entropy ---
        # attn: [B, T, num_heads, N_active]
        log_attn = torch.log(attn + 1e-8)
        h = -(attn * log_attn).sum(dim=-1)  # [B, T, num_heads]
        entropy_terms.append(h.mean())

    if not energy_terms:
        return torch.tensor(0.0, device=device, requires_grad=True)

    energy = torch.stack(energy_terms).mean()
    entropy = torch.stack(entropy_terms).mean()

    # F = E - lambda * H  (lambda=0.1 prevents entropy from dominating)
    lam = fe_lambda if fe_lambda is not None else 0.1
    return energy - lam * entropy
