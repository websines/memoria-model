"""Tentative belief evaluation: the internal autoresearch loop.

Provisional beliefs are hypotheses that participate normally in forward passes
but don't build reinforcement (access_count stays frozen). After a learned
evaluation window, they're tested: did they help?

Promotion criteria (both must hold):
  1. Global free energy decreased since allocation (the state got better)
  2. Belief precision held above a learned retention fraction of its initial radius

If neither holds, the belief is evicted — freeing the slot for a better hypothesis.

Reference: BrainCL (arXiv:2504.14727) — wake/sleep staging buffer
Reference: Synaptic tagging (Nature Comms 2021) — provisional encoding
"""

import torch
from ..core.state import CognitiveState


def evaluate_provisional_beliefs(
    state: CognitiveState,
    current_step: int,
    current_fe: float,
    outcome_callback: 'callable | None' = None,
) -> dict:
    """Evaluate all provisional beliefs that have passed their evaluation window.

    Called from pass2 every step. Only evaluates beliefs whose provisional
    period has elapsed (current_step - alloc_step >= eval_window).

    All thresholds are derived from state.meta_params — no magic numbers.

    Args:
        state: cognitive state (modified in-place)
        current_step: current training step
        current_fe: current global free energy (from last computed Bethe FE)
        outcome_callback: optional function(belief_idx, promoted: bool) called
                          for each evaluated belief. Used by HypothesisTracker
                          to record which goals produce successful hypotheses.

    Returns:
        dict with statistics: promoted, evicted, still_provisional counts
    """
    stats = {'promoted': 0, 'evicted': 0, 'still_provisional': 0}

    prov_mask = state.belief_provisional & state.get_active_mask()
    if not prov_mask.any():
        return stats

    prov_indices = prov_mask.nonzero(as_tuple=False).squeeze(-1)

    # Learned parameters (no hardcoded values)
    eval_window = state.meta_params.provisional_eval_window.item()
    fe_threshold = state.meta_params.provisional_fe_threshold.item()
    precision_retention = state.meta_params.provisional_precision_retention.item()

    with torch.no_grad():
        for idx in prov_indices.tolist():
            alloc_step = state.belief_provisional_step[idx].item()
            elapsed = current_step - alloc_step

            if elapsed < eval_window:
                stats['still_provisional'] += 1
                continue

            # Criterion 1: Did free energy improve?
            alloc_fe = state.belief_provisional_fe[idx].item()
            # FE must have decreased by at least fe_threshold fraction
            fe_improved = current_fe < alloc_fe * (1.0 - fe_threshold)

            # Criterion 2: Did belief precision hold?
            alloc_radius = state.belief_provisional_radius[idx].item()
            current_radius = state.beliefs.data[idx].norm().item()
            precision_held = current_radius >= alloc_radius * precision_retention

            if fe_improved and precision_held:
                # Promote: clear provisional flag, belief becomes committed
                state.belief_provisional[idx] = False
                state.belief_provisional_step[idx] = 0.0
                state.belief_provisional_fe[idx] = 0.0
                state.belief_provisional_radius[idx] = 0.0
                stats['promoted'] += 1
                if outcome_callback:
                    outcome_callback(idx, True)
            else:
                # Evict: hypothesis failed
                if outcome_callback:
                    outcome_callback(idx, False)
                state.deallocate_belief(idx)
                stats['evicted'] += 1

    return stats
