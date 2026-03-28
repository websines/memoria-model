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

    Vectorized: groups updates by type (new/reconsolidate/incremental) and
    applies them as batched tensor operations.

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

    if not surprise_results:
        return stats

    device = state.beliefs.device

    # Classify all results into groups
    new_obs = []
    recon_slots = []
    recon_obs = []
    incr_slots = []
    incr_obs = []
    incr_gains = []
    incr_surprises = []

    total_surprise = 0.0
    for sr in surprise_results:
        total_surprise += sr.surprise
        if sr.is_new:
            new_obs.append(sr.observation)
        elif sr.should_reconsolidate:
            recon_slots.append(sr.slot)
            recon_obs.append(sr.observation)
        else:
            incr_slots.append(sr.slot)
            incr_obs.append(sr.observation)
            incr_gains.append(sr.gain)
            incr_surprises.append(sr.surprise)

    stats['total_surprise'] = total_surprise

    with torch.no_grad():
        # ── Batch reconsolidations ──
        if recon_slots:
            slots_t = torch.tensor(recon_slots, dtype=torch.long, device=device)
            obs_t = torch.stack(recon_obs).to(device)

            # Filter out immutable beliefs
            immutable = state.immutable_beliefs[slots_t]
            mutable = ~immutable
            n_blocked = immutable.sum().item()
            stats['blocked_by_kernel'] += n_blocked

            if mutable.any():
                m_slots = slots_t[mutable]
                m_obs = obs_t[mutable]
                state.beliefs.data[m_slots] = m_obs
                stats['reconsolidations'] = mutable.sum().item()

        # ── Batch incremental updates ──
        if incr_slots:
            slots_t = torch.tensor(incr_slots, dtype=torch.long, device=device)
            obs_t = torch.stack(incr_obs).to(device)
            gains_t = torch.tensor(incr_gains, device=device)
            surprises_t = torch.tensor(incr_surprises, device=device)

            # Filter out immutable
            immutable = state.immutable_beliefs[slots_t]
            mutable = ~immutable
            stats['blocked_by_kernel'] += immutable.sum().item()

            if mutable.any():
                m_slots = slots_t[mutable]
                m_obs = obs_t[mutable]
                m_gains = gains_t[mutable]
                m_surprises = surprises_t[mutable]

                existing = state.beliefs.data[m_slots]  # [K, D]
                existing_radii = existing.norm(dim=-1).clamp(min=EPSILON)  # [K]
                existing_angles = existing / existing_radii.unsqueeze(-1)  # [K, D]

                obs_radii = m_obs.norm(dim=-1).clamp(min=EPSILON)
                obs_angles = m_obs / obs_radii.unsqueeze(-1)

                # Angle update: interpolate toward observation
                g = m_gains.unsqueeze(-1)  # [K, 1]
                new_angles = (1.0 - g) * existing_angles + g * obs_angles
                new_angles = torch.nn.functional.normalize(new_angles, dim=-1, eps=EPSILON)

                # Radius update
                low_surprise = m_surprises < 0.5
                headroom = (1.0 - existing_radii / 10.0).clamp(min=0.0)
                # Agreement: precision increases (diminishing returns)
                r_agree = existing_radii + 0.1 * obs_radii * (1.0 - m_surprises) * headroom
                # Disagreement: precision decreases
                r_disagree = existing_radii * (1.0 - m_gains * 0.5)

                new_radii = torch.where(low_surprise, r_agree, r_disagree)
                new_radii = new_radii.clamp(min=EPSILON, max=100.0)

                state.beliefs.data[m_slots] = new_angles * new_radii.unsqueeze(-1)
                stats['incremental_updates'] = mutable.sum().item()

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
    recency_factor = 1.0 + 0.1 * recency

    scores = radii * recency_factor
    scores[~active] = float('inf')
    scores[state.immutable_beliefs] = float('inf')

    if (scores == float('inf')).all():
        return -1

    slot = scores.argmin().item()
    state.deallocate_belief(slot)
    return slot
