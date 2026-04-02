"""Belief allocation: assign new observations to belief slots.

Beliefs are now differentiable — content updates happen via optimizer.step().
This module handles only structural allocation:
- New observations get a fresh slot (or evict the weakest if full)
- Existing beliefs are left untouched (the optimizer updates them)

Ported from: prototype-research/src/aif/belief_update.rs
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import EPSILON
from .surprise import SurpriseResult


def allocate_new_beliefs(
    surprise_results: list[SurpriseResult],
    state: CognitiveState,
) -> dict:
    """Allocate slots for new observations; collect stats for existing beliefs.

    For existing beliefs (slot >= 0): collect surprise stats only — the
    optimizer handles content updates.
    For new observations (is_new=True): allocate a belief slot, evicting the
    weakest belief if the pool is full.

    Args:
        surprise_results: computed surprise for each write candidate
        state: cognitive state to update

    Returns:
        dict with allocation statistics
    """
    stats = {
        'new_allocations': 0,
        'evictions': 0,
        'total_surprise': 0.0,
    }

    if not surprise_results:
        return stats

    # Classify results
    new_obs = []
    total_surprise = 0.0

    for sr in surprise_results:
        total_surprise += sr.surprise
        if sr.is_new:
            new_obs.append(sr.observation)

    stats['total_surprise'] = total_surprise

    with torch.no_grad():
        # ── New belief allocations ──
        for obs in new_obs:
            slot = state.allocate_belief(obs)
            if slot == -1:
                slot = _evict_weakest(state)
                if slot >= 0:
                    state.beliefs.data[slot] = obs
                    stats['evictions'] += 1
            else:
                stats['new_allocations'] += 1

        # Update accumulated surprise in meta region
        state.meta.data[1] += total_surprise

    return stats


def _evict_weakest(state: CognitiveState) -> int:
    """Find and evict the lowest-priority belief.

    Eviction score = radius × recency_factor.
    Stale beliefs (not accessed recently) are evicted first.

    Returns:
        slot index of evicted belief, or -1 if all immutable
    """
    radii = state.get_belief_radii()
    active = state.get_active_mask()

    recency = state.belief_access_count.clamp(min=0)
    recency_weight = state.running_stats.eviction_recency_weight
    recency_factor = 1.0 + recency_weight * recency

    scores = radii * recency_factor
    scores[~active] = float('inf')
    scores[state.immutable_beliefs] = float('inf')

    if (scores == float('inf')).all():
        return -1

    slot = scores.argmin().item()
    state.deallocate_belief(slot)
    return slot
