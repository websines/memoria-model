"""Tentative belief evaluation: the internal autoresearch loop.

Provisional beliefs are hypotheses that participate normally in forward passes
but don't build reinforcement (access_count stays frozen). After a learned
evaluation window, they're tested: did they help?

Promotion criteria (all must hold):
  1. Global free energy decreased since allocation (the state got better)
  2. Belief precision held above a learned retention fraction of its initial radius
  3. The belief was actually retrieved at least `provisional_min_reads` times
     strictly *after* the allocation step — the search/eval split from
     Meta-Harness (arXiv:2603.28052). A hypothesis that was never read during
     its evaluation window was never tested on data that did not spawn it,
     and any FE improvement over that window is unattributable.

If any criterion fails, the belief is evicted. Eviction reasons are surfaced
to the outcome callback so downstream trackers (HypothesisTracker) can log
which failure mode fired, which feeds back into future hypothesis generation.

Reference: Meta-Harness (arXiv:2603.28052) — search/eval split, richer outcome signal
Reference: BrainCL (arXiv:2504.14727) — wake/sleep staging buffer
Reference: Synaptic tagging (Nature Comms 2021) — provisional encoding
"""

import torch
from ..core.state import CognitiveState


# Eviction reason codes (surfaced to outcome callback).
# Keep stable — HypothesisTracker stores these as categorical features.
EVICT_FE_NOT_IMPROVED = 1   # global FE did not decrease enough
EVICT_PRECISION_COLLAPSED = 2   # belief radius shrank below retention
EVICT_NEVER_READ = 3   # no post-allocation reads — untested
PROMOTED = 0


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
        outcome_callback: optional function(belief_idx, outcome_code, metadata)
                          called for each evaluated belief. `outcome_code` is
                          PROMOTED or one of EVICT_*. `metadata` is a dict with
                          `fe_delta`, `precision_ratio`, `post_alloc_reads`.
                          Used by HypothesisTracker to record which goals
                          produce successful hypotheses and why failures
                          happened.

    Returns:
        dict with statistics: promoted, evicted, still_provisional counts,
        plus per-reason eviction counters.
    """
    stats = {
        'promoted': 0,
        'evicted': 0,
        'still_provisional': 0,
        'evicted_fe_not_improved': 0,
        'evicted_precision_collapsed': 0,
        'evicted_never_read': 0,
    }

    prov_mask = state.belief_provisional & state.get_active_mask()
    if not prov_mask.any():
        return stats

    prov_indices = prov_mask.nonzero(as_tuple=False).squeeze(-1)

    # Learned parameters (no hardcoded values)
    eval_window = state.meta_params.provisional_eval_window.item()
    fe_threshold = state.meta_params.provisional_fe_threshold.item()
    precision_retention = state.meta_params.provisional_precision_retention.item()
    min_reads = state.meta_params.provisional_min_reads.item()

    with torch.no_grad():
        for idx in prov_indices.tolist():
            alloc_step = state.belief_provisional_step[idx].item()
            elapsed = current_step - alloc_step

            if elapsed < eval_window:
                stats['still_provisional'] += 1
                continue

            # Criterion 1: Did free energy improve?
            alloc_fe = state.belief_provisional_fe[idx].item()
            fe_improved = current_fe < alloc_fe * (1.0 - fe_threshold)

            # Criterion 2: Did belief precision hold?
            alloc_radius = state.belief_provisional_radius[idx].item()
            current_radius = state.beliefs.data[idx].norm().item()
            precision_ratio = (
                current_radius / alloc_radius if alloc_radius > 0 else 0.0
            )
            precision_held = precision_ratio >= precision_retention

            # Criterion 3: Was the belief actually read on held-out data?
            # Search/eval split (Meta-Harness, arXiv:2603.28052): a belief that
            # never got retrieved post-allocation was never tested on data
            # that didn't shape it. We cannot attribute FE improvement to it.
            post_alloc_reads = state.belief_provisional_reads[idx].item()
            was_tested = post_alloc_reads >= min_reads

            fe_delta = current_fe - alloc_fe
            metadata = {
                'fe_delta': fe_delta,
                'precision_ratio': precision_ratio,
                'post_alloc_reads': post_alloc_reads,
            }

            if fe_improved and precision_held and was_tested:
                # Promote: clear provisional flag, belief becomes committed.
                state.belief_provisional[idx] = False
                state.belief_provisional_step[idx] = 0.0
                state.belief_provisional_fe[idx] = 0.0
                state.belief_provisional_radius[idx] = 0.0
                state.belief_provisional_reads[idx] = 0.0
                stats['promoted'] += 1
                if outcome_callback:
                    outcome_callback(idx, PROMOTED, metadata)
            else:
                # Evict: attribute the failure to the most specific reason.
                # Precedence: untested > FE stagnation > precision collapse.
                # Untested dominates because the other tests are uninformative
                # in that regime. Precision collapse is reported only when
                # both other criteria passed.
                if not was_tested:
                    reason = EVICT_NEVER_READ
                    stats['evicted_never_read'] += 1
                elif not fe_improved:
                    reason = EVICT_FE_NOT_IMPROVED
                    stats['evicted_fe_not_improved'] += 1
                else:
                    reason = EVICT_PRECISION_COLLAPSED
                    stats['evicted_precision_collapsed'] += 1
                if outcome_callback:
                    outcome_callback(idx, reason, metadata)
                state.deallocate_belief(idx)
                stats['evicted'] += 1

    return stats
