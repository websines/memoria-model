"""Meta-learning: β computation and SPSA self-tuning of cognitive parameters.

β = H / (|E| + H + ε) — computed from actual state, not a hyperparameter.

SPSA (Simultaneous Perturbation Stochastic Approximation) tunes cognitive
parameters by minimizing free energy:
- Perturb parameters slightly
- Measure free energy before/after
- Gradient-free update: adjust in direction that reduced free energy
- Only needs 2 evaluations per step regardless of parameter count

Ported from: prototype-research/src/dynamics/meta_learning.rs
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.free_energy import compute_free_energy
from ..core.polar import EPSILON


def compute_beta(state: CognitiveState, temperature: float = 5.0) -> float:
    """Compute β from current state (already done inside compute_free_energy).

    This is a convenience wrapper that returns just β.
    β is also written to state.meta[0] as a side effect.

    Args:
        state: cognitive state
        temperature: for energy computation

    Returns:
        β ∈ [0, 1]
    """
    result = compute_free_energy(state, temperature)
    return result['beta'].item()


def spsa_step(
    state: CognitiveState,
    temperature: float = 5.0,
    perturbation_scale: float = 0.01,
    step_size: float = 0.001,
):
    """One step of SPSA to tune meta parameters.

    Tunes: reconsolidation_threshold (meta[4]), match_threshold (meta[5]),
    and any other SPSA-tunable parameters in meta[6:].

    Algorithm:
    1. Perturb parameters by ±Δ (Bernoulli random)
    2. Compute free energy at +Δ and -Δ
    3. Estimate gradient: g = (F(+Δ) - F(-Δ)) / (2Δ)
    4. Update: param -= step_size × g

    Args:
        state: cognitive state
        temperature: for free energy computation
        perturbation_scale: size of random perturbations
        step_size: learning rate for parameter updates
    """
    # Indices of tunable meta parameters (skip β[0] and accumulated_surprise[1])
    tunable_start = 4  # reconsolidation_threshold, match_threshold, ...
    tunable_end = min(8, state.config.meta_dim)  # first few are tunable

    if tunable_end <= tunable_start:
        return

    device = state.meta.device

    with torch.no_grad():
        original_values = state.meta.data[tunable_start:tunable_end].clone()

        # Random perturbation direction (Bernoulli ±1) — on correct device
        n = tunable_end - tunable_start
        delta = torch.where(
            torch.rand(n, device=device) > 0.5,
            torch.ones(n, device=device),
            -torch.ones(n, device=device),
        ) * perturbation_scale

        # Evaluate at +Δ
        state.meta.data[tunable_start:tunable_end] = original_values + delta
        fe_plus = compute_free_energy(state, temperature)['free_energy'].item()

        # Evaluate at -Δ
        state.meta.data[tunable_start:tunable_end] = original_values - delta
        fe_minus = compute_free_energy(state, temperature)['free_energy'].item()

        # SPSA gradient estimate
        gradient = (fe_plus - fe_minus) / (2.0 * delta + EPSILON)

        # Update parameters (minimize free energy)
        new_values = original_values - step_size * gradient

        # Clamp to valid ranges
        new_values[0] = new_values[0].clamp(0.1, 0.9)   # reconsolidation_threshold
        new_values[1] = new_values[1].clamp(0.3, 0.95)   # match_threshold
        if n > 2:
            new_values[2] = new_values[2].clamp(0.2, 0.9)   # goal_dedup_threshold

        state.meta.data[tunable_start:tunable_end] = new_values


def apply_sequence_boundary_decay(state: CognitiveState, decay_factor: float = 0.95):
    """Apply exponential decay to belief precision at sequence boundaries.

    Vectorized: applies decay to all active non-immutable beliefs in one op.
    Beliefs not reinforced fade over ~20 sequences (0.95^20 ≈ 0.36).

    Args:
        state: cognitive state
        decay_factor: multiplicative decay per sequence boundary
    """
    with torch.no_grad():
        active = state.get_active_mask()
        mutable = active & ~state.immutable_beliefs

        if mutable.any():
            # Scale all mutable belief vectors by decay_factor (shrinks radius, preserves angle)
            state.beliefs.data[mutable] *= decay_factor

            # Zero out beliefs that decayed below epsilon
            decayed_radii = state.beliefs.data[mutable].norm(dim=-1)
            dead = decayed_radii < EPSILON
            if dead.any():
                mutable_indices = mutable.nonzero(as_tuple=False).squeeze(-1)
                state.beliefs.data[mutable_indices[dead]] = 0.0

        # Increment consolidation timer
        state.meta.data[2] += 1.0
