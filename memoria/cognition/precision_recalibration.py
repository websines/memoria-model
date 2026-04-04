"""E3: Empirical Precision Recalibration — track prediction accuracy per belief,
decay stored precision toward empirical precision when overconfident.

Each belief tracks how many times its predictions were confirmed vs contradicted.
The empirical precision = confirmed / (confirmed + contradicted) gives a
frequency-based estimate of belief quality. When stored radius >> empirical
precision, the radius is decayed toward the empirical value.

This prevents the #1 failure mode: a belief that *feels* confident (high radius)
but is *actually* wrong (low empirical accuracy).

Reference: Epistemic Uncertainty Collapse (arXiv:2409.02628)
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import belief_is_active, EPSILON


def record_confirmation(state: CognitiveState, belief_idx: int) -> None:
    """Record that a belief's prediction was confirmed.

    Called when a belief's causal prediction matches the observed outcome.
    """
    state.belief_confirmed_count[belief_idx] += 1


def record_contradiction(state: CognitiveState, belief_idx: int) -> None:
    """Record that a belief's prediction was contradicted.

    Called when a belief's causal prediction diverges from the observed outcome.
    """
    state.belief_contradicted_count[belief_idx] += 1


def compute_empirical_precision(
    confirmed: Tensor,
    contradicted: Tensor,
) -> Tensor:
    """Compute empirical precision from confirmed/contradicted counts.

    empirical = confirmed / (confirmed + contradicted)
    Returns values in [0, 1]. Returns 0.5 when both counts are zero (no data).

    Args:
        confirmed: [N] confirmed counts
        contradicted: [N] contradicted counts

    Returns:
        [N] empirical precision values in [0, 1]
    """
    total = confirmed.float() + contradicted.float()
    # Default to 0.5 (neutral) when no data available
    empirical = torch.where(
        total > 0,
        confirmed.float() / total.clamp(min=1),
        torch.full_like(total, 0.5),
    )
    return empirical


def recalibrate_beliefs(
    state: CognitiveState,
    rate: Tensor,
    min_samples: Tensor,
) -> dict:
    """Recalibrate overconfident beliefs toward their empirical precision.

    For each active belief with enough observations:
    1. Compute empirical precision from confirmed/contradicted counts
    2. Compare against stored radius (normalized to [0, 1])
    3. If stored > empirical (overconfident): decay radius toward empirical
    4. Also boost MESU variance for recalibrated beliefs (make them more plastic)

    The recalibration formula:
        new_radius = radius * (1 - rate * max(0, stored_precision - empirical))

    Args:
        state: cognitive state (modified in-place)
        rate: correction rate per cycle (from MetaParams)
        min_samples: minimum total observations before recalibrating

    Returns:
        dict with {beliefs_checked, beliefs_recalibrated, total_radius_reduction}
    """
    stats = {
        'beliefs_checked': 0,
        'beliefs_recalibrated': 0,
        'total_radius_reduction': 0.0,
        'mean_empirical_precision': 0.0,
    }

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return stats

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    n_active = len(active_idx)

    confirmed = state.belief_confirmed_count[active_idx]
    contradicted = state.belief_contradicted_count[active_idx]
    total_obs = confirmed + contradicted

    # Only recalibrate beliefs with enough observations
    min_s = int(min_samples.item())
    has_enough = total_obs >= max(min_s, 1)

    if not has_enough.any():
        return stats

    eligible_mask = has_enough
    eligible_local = eligible_mask.nonzero(as_tuple=False).squeeze(-1)

    empirical = compute_empirical_precision(
        confirmed[eligible_local],
        contradicted[eligible_local],
    )
    stats['mean_empirical_precision'] = empirical.mean().item()

    rate_val = rate.item()

    with torch.no_grad():
        for i, local_idx in enumerate(eligible_local.tolist()):
            global_idx = active_idx[local_idx].item()
            if state.immutable_beliefs[global_idx]:
                continue

            stats['beliefs_checked'] += 1

            current_radius = state.beliefs.data[global_idx].norm().item()
            if current_radius < EPSILON:
                continue

            # Normalize stored radius to [0, 1] scale for comparison
            # Use mean_precision from running_stats as reference scale
            ref_precision = state.running_stats.mean_precision.item()
            if ref_precision < EPSILON:
                ref_precision = 1.0
            stored_precision = min(current_radius / ref_precision, 1.0)
            emp_val = empirical[i].item()

            # Only correct overconfident beliefs (stored > empirical)
            overconfidence = max(0.0, stored_precision - emp_val)
            if overconfidence > EPSILON:
                # Decay radius proportional to overconfidence gap
                decay = 1.0 - rate_val * overconfidence
                decay = max(decay, 0.5)  # floor: never reduce more than half

                old_radius = current_radius
                state.beliefs.data[global_idx] *= decay
                stats['beliefs_recalibrated'] += 1
                stats['total_radius_reduction'] += old_radius - state.beliefs.data[global_idx].norm().item()

                # Boost MESU variance for recalibrated beliefs
                # (they need to be more plastic to re-learn)
                state.belief_precision_var[global_idx] += overconfidence

    return stats


def run_precision_recalibration(
    state: CognitiveState,
) -> dict:
    """Run the full empirical precision recalibration pass.

    All thresholds from MetaParams (no hardcoded magic numbers).

    Args:
        state: cognitive state (modified in-place)

    Returns:
        dict with statistics
    """
    return recalibrate_beliefs(
        state,
        rate=state.meta_params.recalibration_rate,
        min_samples=state.meta_params.recalibration_min_samples,
    )
