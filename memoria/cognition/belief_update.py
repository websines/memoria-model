"""Belief update: precision-weighted revision of existing beliefs.

Implements the Kalman-like update from Memoria's belief_update.rs:
- gain = obs_precision / (belief_precision + obs_precision)
- If gain < reconsolidation_threshold: incremental update
- If gain >= threshold: full reconsolidation (belief rewritten)

In polar form:
- Angle (content) shifts toward observation proportional to gain
- Radius (precision) adjusts based on update consistency

Ported from: prototype-research/src/aif/belief_update.rs
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import EPSILON
from .surprise import SurpriseResult


def apply_belief_updates(
    surprise_results: list[SurpriseResult],
    state: CognitiveState,
) -> dict:
    """Apply all belief updates from surprise results.

    For each result:
    - If reconsolidate: fully rewrite the belief to the observation
    - If incremental: shift belief angle toward observation by gain, adjust radius
    - If new: allocate a new belief slot
    - Kernel rules respected: immutable beliefs skip updates

    Args:
        surprise_results: computed surprise for each write candidate
        state: cognitive state to update

    Returns:
        dict with update statistics
    """
    stats = {
        'incremental_updates': 0,
        'reconsolidations': 0,
        'new_allocations': 0,
        'evictions': 0,
        'blocked_by_kernel': 0,
        'total_surprise': 0.0,
    }

    with torch.no_grad():
        for sr in surprise_results:
            stats['total_surprise'] += sr.surprise

            if sr.is_new:
                # Allocate new belief
                slot = state.allocate_belief(sr.observation)
                if slot == -1:
                    # Full — evict lowest-precision non-immutable belief
                    slot = _evict_weakest(state)
                    if slot >= 0:
                        state.beliefs.data[slot] = sr.observation
                        stats['evictions'] += 1
                    # If still -1, we can't allocate (all immutable). Skip.
                else:
                    stats['new_allocations'] += 1

            elif sr.should_reconsolidate:
                # Full reconsolidation: rewrite belief to observation
                if state.immutable_beliefs[sr.slot]:
                    stats['blocked_by_kernel'] += 1
                    continue

                state.beliefs.data[sr.slot] = sr.observation
                stats['reconsolidations'] += 1

            else:
                # Incremental update: shift angle by gain, adjust radius
                if state.immutable_beliefs[sr.slot]:
                    stats['blocked_by_kernel'] += 1
                    continue

                existing = state.beliefs.data[sr.slot]
                existing_radius = existing.norm().clamp(min=EPSILON)
                existing_angle = existing / existing_radius

                obs = sr.observation
                obs_radius = obs.norm().clamp(min=EPSILON)
                obs_angle = obs / obs_radius

                gain = sr.gain

                # Angle update: interpolate toward observation
                new_angle = (1.0 - gain) * existing_angle + gain * obs_angle
                new_angle = torch.nn.functional.normalize(new_angle, dim=0, eps=EPSILON)

                # Radius update: consistent updates increase precision,
                # contradictory updates decrease it
                if sr.surprise < 0.5:
                    # Low surprise (agreement) → precision increases slightly
                    new_radius = existing_radius + 0.1 * obs_radius * (1.0 - sr.surprise)
                else:
                    # High surprise (disagreement) → precision decreases
                    new_radius = existing_radius * (1.0 - gain * 0.5)

                new_radius = new_radius.clamp(min=EPSILON, max=100.0)

                state.beliefs.data[sr.slot] = new_angle * new_radius
                stats['incremental_updates'] += 1

        # Update accumulated surprise in meta region
        state.meta.data[1] += stats['total_surprise']

    return stats


def _evict_weakest(state: CognitiveState) -> int:
    """Find and evict the lowest-priority belief.

    Eviction score = radius × recency_proxy × (1 - is_immutable)
    We don't track recency yet, so just use radius (precision).

    Returns:
        slot index of evicted belief, or -1 if all immutable
    """
    radii = state.get_belief_radii()
    active = state.get_active_mask()

    # Score: low radius = evict first. Immutable = never evict.
    scores = radii.clone()
    scores[~active] = float('inf')  # don't evict empty slots
    scores[state.immutable_beliefs] = float('inf')  # don't evict immutable

    if (scores == float('inf')).all():
        return -1

    slot = scores.argmin().item()
    state.deallocate_belief(slot)
    return slot
