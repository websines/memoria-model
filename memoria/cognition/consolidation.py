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

    Vectorized: computes full similarity matrix and extracts merge pairs with
    torch.nonzero instead of nested Python loops.

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

    # Mask: upper triangle only, above threshold, neither immutable
    immutable_local = state.immutable_beliefs[active_indices]
    # Zero out pairs involving immutable beliefs
    immutable_row = immutable_local.unsqueeze(1).expand_as(sim_matrix)
    immutable_col = immutable_local.unsqueeze(0).expand_as(sim_matrix)
    eligible = ~(immutable_row | immutable_col)

    # Upper triangle + threshold + eligible
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=sim_matrix.device), diagonal=1)
    merge_mask = upper & (sim_matrix > similarity_threshold) & eligible

    if not merge_mask.any():
        return 0

    # Get all merge pairs at once
    pairs = merge_mask.nonzero(as_tuple=False)  # [P, 2] local indices

    merged = 0
    merged_set = set()

    with torch.no_grad():
        for p in range(pairs.shape[0]):
            i = pairs[p, 0].item()
            j = pairs[p, 1].item()

            if i in merged_set or j in merged_set:
                continue

            idx_i = active_indices[i].item()
            idx_j = active_indices[j].item()

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

    return merged


def periodic_hard_cleanup(
    state: CognitiveState,
    low_precision_threshold: float = 0.1,
) -> int:
    """Periodic hard consolidation: remove very low-precision beliefs.

    Vectorized: uses tensor masks instead of per-belief Python loop.

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
    access_counts = state.belief_access_count

    # Vectorized: find all removable beliefs at once
    removable = (
        active_mask
        & ~state.immutable_beliefs
        & (radii < low_precision_threshold)
        & (access_counts < 3)
    )

    if not removable.any():
        return 0

    remove_indices = removable.nonzero(as_tuple=False).squeeze(-1)
    removed = len(remove_indices)

    with torch.no_grad():
        for idx in remove_indices.tolist():
            state.deallocate_belief(idx)

    return removed


def _redirect_edges(state: CognitiveState, from_idx: int, to_idx: int):
    """Redirect all edges pointing to from_idx to point to to_idx instead."""
    if not state.edge_active.any():
        return

    active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    srcs = state.edge_src[active_edges]
    tgts = state.edge_tgt[active_edges]

    # Vectorized redirect
    src_match = srcs == from_idx
    tgt_match = tgts == from_idx
    if src_match.any():
        state.edge_src[active_edges[src_match]] = to_idx
    if tgt_match.any():
        state.edge_tgt[active_edges[tgt_match]] = to_idx

    # Deduplicate
    _deduplicate_edges(state)


def _deduplicate_edges(state: CognitiveState):
    """Remove duplicate edges (same src-tgt pair), keeping the stronger one."""
    if not state.edge_active.any():
        return

    active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    if len(active_edges) <= 1:
        return

    srcs = state.edge_src[active_edges]
    tgts = state.edge_tgt[active_edges]
    weights = state.edge_weights.data[active_edges]

    # Canonical key: (min, max) for undirected dedup
    mins = torch.min(srcs, tgts)
    maxs = torch.max(srcs, tgts)

    seen: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(len(active_edges)):
        key = (mins[i].item(), maxs[i].item())
        eidx = active_edges[i].item()
        w = weights[i].item()

        if key in seen:
            prev_eidx, prev_w = seen[key]
            if w > prev_w:
                state.deallocate_edge(prev_eidx)
                seen[key] = (eidx, w)
            else:
                state.deallocate_edge(eidx)
        else:
            seen[key] = (eidx, w)
