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


def fused_chunked_cross_entropy(
    hidden: Tensor,
    targets: Tensor,
    head_weight: Tensor,
    chunk_size: int = 2048,
    ignore_index: int = -1,
) -> Tensor:
    """Fused LM head + cross-entropy in chunks — never materializes full logits.

    At 128K context with 151K vocab, logits = [128K, 151K] × 2 bytes = 38 GB.
    This function computes head(hidden) → logits → loss in chunks of chunk_size
    positions, keeping peak memory at chunk_size × vocab_size × 2 bytes.

    At chunk_size=2048: peak = 2048 × 151K × 2 = 620 MB (vs 38 GB).

    Gradients flow correctly through both the hidden states and head_weight
    because each chunk's logits are created from the original hidden tensor
    (not detached). PyTorch accumulates gradients across chunks automatically.

    Args:
        hidden: [B*T, D] hidden states (BEFORE LM head projection)
        targets: [B*T] target token IDs
        head_weight: [vocab_size, D] LM head weight matrix
        chunk_size: positions to process at once (controls peak memory)
        ignore_index: target value to ignore in loss

    Returns:
        Scalar cross-entropy loss (differentiable w.r.t. hidden and head_weight)
    """
    BT, D = hidden.shape

    # Short sequences: just compute directly
    if BT <= chunk_size:
        logits = F.linear(hidden, head_weight)
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    total_loss = 0.0
    n_tokens = 0
    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        # Compute logits for this chunk only — peak memory = chunk_size × vocab
        chunk_logits = F.linear(hidden[start:end], head_weight)
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
    huber_delta: float | Tensor | None = None,
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
        # Huber loss: quadratic for small errors, linear for outliers.
        # Prevents spurious matches from dominating belief update gradients.
        # Reference: MIRAS/YAAD (arXiv:2504.13173); Huber (1964)
        ret_avg = ret.mean(dim=2)  # [B, T, D] — average over heads
        cos_sim = F.cosine_similarity(ret_avg, obs, dim=-1)  # [B, T]
        disagreement = 1.0 - cos_sim  # [B, T], range [0, 2]
        if huber_delta is not None:
            energy_terms.append(F.huber_loss(
                disagreement, torch.zeros_like(disagreement),
                reduction='mean', delta=huber_delta.detach().item() if isinstance(huber_delta, Tensor) else huber_delta,
            ))
        else:
            energy_terms.append(disagreement.mean())

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
