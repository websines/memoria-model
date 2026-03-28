"""Consolidation: merge similar beliefs to free capacity and build abstractions.

Two modes:
1. Soft merge (continuous, differentiable): very similar beliefs blend together
2. Hard cleanup (periodic, non-differentiable): cluster and merge, treated like data augmentation

Merged belief radius = sqrt(r_A² + r_B²) — combined evidence increases precision.

Ported from: prototype-research/src/dynamics/compression.rs
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, precision_weighted_average, belief_is_active, EPSILON


def soft_consolidation(
    state: CognitiveState,
    similarity_threshold: float = 0.95,
) -> int:
    """Merge very similar active beliefs (differentiable-friendly).

    For each pair of active beliefs with cosine_sim > threshold:
    - Merge into the higher-precision one using precision_weighted_average
    - Zero the lower-precision one (free the slot)

    Args:
        state: cognitive state
        similarity_threshold: how similar beliefs must be to merge

    Returns:
        Number of beliefs merged
    """
    active_mask = state.get_active_mask()
    if active_mask.sum() < 2:
        return 0

    active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_beliefs = state.beliefs.data[active_indices]
    n = len(active_indices)

    radii = active_beliefs.norm(dim=-1)
    angles = active_beliefs / radii.unsqueeze(-1).clamp(min=EPSILON)

    # Pairwise cosine similarity
    sim_matrix = angles @ angles.T  # [N, N]

    merged = 0
    merged_set = set()

    with torch.no_grad():
        for i in range(n):
            if i in merged_set:
                continue
            idx_i = active_indices[i].item()
            if state.immutable_beliefs[idx_i]:
                continue

            for j in range(i + 1, n):
                if j in merged_set:
                    continue
                idx_j = active_indices[j].item()
                if state.immutable_beliefs[idx_j]:
                    continue

                if sim_matrix[i, j] > similarity_threshold:
                    # Merge j into i (keep i, free j)
                    pair_angles = torch.stack([angles[i], angles[j]])
                    pair_radii = torch.stack([radii[i], radii[j]])
                    new_angle, new_radius = precision_weighted_average(pair_angles, pair_radii, dim=0)

                    state.beliefs.data[idx_i] = new_angle * new_radius
                    state.beliefs.data[idx_j].zero_()  # free slot

                    # Redirect edges from j to i
                    _redirect_edges(state, from_idx=idx_j, to_idx=idx_i)

                    merged_set.add(j)
                    merged += 1

                    # Update local tracking
                    radii[i] = new_radius
                    angles[i] = new_angle
                    break  # only merge one partner per belief per pass

    return merged


def periodic_hard_cleanup(
    state: CognitiveState,
    low_precision_threshold: float = 0.1,
) -> int:
    """Periodic hard consolidation: remove very low-precision beliefs.

    Non-differentiable. Run every N sequences. Frees slots for new knowledge.

    Args:
        state: cognitive state
        low_precision_threshold: beliefs with radius below this are candidates for removal

    Returns:
        Number of beliefs removed
    """
    active_mask = state.get_active_mask()
    if not active_mask.any():
        return 0

    radii = state.get_belief_radii()
    removed = 0

    with torch.no_grad():
        for i in range(state.config.max_beliefs):
            if not active_mask[i]:
                continue
            if state.immutable_beliefs[i]:
                continue
            if radii[i].item() < low_precision_threshold:
                state.deallocate_belief(i)
                removed += 1

    return removed


def _redirect_edges(state: CognitiveState, from_idx: int, to_idx: int):
    """Redirect all edges pointing to from_idx to point to to_idx instead.

    If an edge would create a duplicate (to_idx already connected to the same
    neighbor), keep the stronger edge and drop the weaker one.
    """
    if not state.edge_active.any():
        return

    active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)

    for eidx in active_edges.tolist():
        src = state.edge_src[eidx].item()
        tgt = state.edge_tgt[eidx].item()

        if src == from_idx:
            state.edge_src[eidx] = to_idx
        elif tgt == from_idx:
            state.edge_tgt[eidx] = to_idx

    # Deduplicate edges that now connect the same pair
    _deduplicate_edges(state)


def _deduplicate_edges(state: CognitiveState):
    """Remove duplicate edges (same src-tgt pair), keeping the stronger one."""
    if not state.edge_active.any():
        return

    active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    seen = {}  # (min_idx, max_idx) → edge_slot with highest weight

    for eidx in active_edges.tolist():
        src = state.edge_src[eidx].item()
        tgt = state.edge_tgt[eidx].item()
        key = (min(src, tgt), max(src, tgt))
        weight = state.edge_weights.data[eidx].item()

        if key in seen:
            prev_eidx, prev_weight = seen[key]
            if weight > prev_weight:
                state.deallocate_edge(prev_eidx)
                seen[key] = (eidx, weight)
            else:
                state.deallocate_edge(eidx)
        else:
            seen[key] = (eidx, weight)
