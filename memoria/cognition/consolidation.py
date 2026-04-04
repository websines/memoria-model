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
    max_beliefs_to_check: int = 512,
) -> int:
    """Merge very similar active beliefs (differentiable-friendly).

    Samples a subset of active beliefs if count exceeds max_beliefs_to_check,
    capping the similarity matrix to max_beliefs_to_check^2 instead of N^2.

    Args:
        state: cognitive state
        similarity_threshold: how similar beliefs must be to merge
        max_beliefs_to_check: cap on beliefs to compare (limits O(N^2) cost)

    Returns:
        Number of beliefs merged
    """
    active_mask = state.get_active_mask()
    n_active = active_mask.sum().item()
    if n_active < 2:
        return 0

    active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)

    # Sample subset if too many active beliefs
    if n_active > max_beliefs_to_check:
        perm = torch.randperm(n_active, device=active_indices.device)[:max_beliefs_to_check]
        active_indices = active_indices[perm]

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

    # Level-aware thresholds: higher-level beliefs need higher similarity to merge.
    # Penalty per level = remaining gap to 1.0, divided equally across 3 levels.
    # E.g., threshold=0.95 → gap=0.05 → per_level=0.05/3 ≈ 0.017
    # This is derived, not hardcoded: tighter threshold at higher levels.
    # Level 3 (core beliefs) never merge — they are immutable abstractions.
    if hasattr(state, 'belief_level'):
        gap_to_max = 1.0 - similarity_threshold
        per_level_penalty = gap_to_max / 3.0  # 3 non-core levels
        levels = state.belief_level[active_indices]  # [N]
        pair_candidates = merge_mask.nonzero(as_tuple=False)  # [P, 2]
        for pair_idx in range(pair_candidates.shape[0]):
            i = pair_candidates[pair_idx, 0].item()
            j = pair_candidates[pair_idx, 1].item()
            max_level = max(levels[i].item(), levels[j].item())
            if max_level >= 3:
                merge_mask[i, j] = False  # core beliefs are immutable abstractions
                continue
            level_penalty = max_level * per_level_penalty
            if sim_matrix[i, j] < similarity_threshold + level_penalty:
                merge_mask[i, j] = False

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

            # A2: MESU — merge precision variances.
            # Combined variance = harmonic mean (two independent estimates reduce uncertainty).
            var_i = state.belief_precision_var[idx_i].item()
            var_j = state.belief_precision_var[idx_j].item()
            combined_var = (var_i * var_j) / max(var_i + var_j, 1e-10)
            min_var = state.meta_params.mesu_min_variance.item()
            state.belief_precision_var[idx_i] = max(combined_var, min_var)
            # Combined reinforcement count
            state.belief_reinforcement_count[idx_i] = max(
                state.belief_reinforcement_count[idx_i].item(),
                state.belief_reinforcement_count[idx_j].item(),
            )

            # SDFT-inspired abstraction metadata: merged belief is promoted one level.
            if hasattr(state, 'belief_level'):
                new_level = min(
                    max(state.belief_level[idx_i].item(), state.belief_level[idx_j].item()) + 1,
                    3,
                )
                state.belief_level[idx_i] = new_level
            if hasattr(state, 'belief_sources'):
                state.belief_sources[idx_i] = torch.tensor(
                    [-1, -1, -1, -1], dtype=torch.long, device=state.beliefs.device
                )
                state.belief_sources[idx_i, 0] = idx_i  # self (as continuation)
                state.belief_sources[idx_i, 1] = idx_j  # merged partner
            if hasattr(state, 'belief_source_type'):
                state.belief_source_type[idx_i] = 1  # merge

            # Merged belief inherits the higher access count of its two sources.
            state.belief_access_count[idx_i] = max(
                state.belief_access_count[idx_i].item(),
                state.belief_access_count[idx_j].item(),
            )

            # Redirect edges from j to i (no dedup per-merge; dedup once after all merges)
            _redirect_edges(state, from_idx=idx_j, to_idx=idx_i)

            merged_set.add(j)
            merged += 1

            # Update local tracking
            radii[i] = new_radius
            angles[i] = new_angle

    # Deduplicate edges once after all merges complete (not per-merge)
    if merged > 0:
        _deduplicate_edges(state)

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

    # Promote beliefs that have accumulated enough evidence for their next level.
    if hasattr(state, 'belief_level'):
        _promote_eligible_beliefs(state)

    return removed


def _promote_eligible_beliefs(state: CognitiveState) -> None:
    """Promote beliefs whose radius and access count justify a higher abstraction level.

    Thresholds tighten at each level so that only the most stable, frequently-used
    beliefs advance to level 3 (which is thereafter protected from merging).

    Level 3 beliefs are never promoted further (they are permanent abstractions).
    """
    active = state.get_active_mask()
    if not active.any():
        return

    active_idx = active.nonzero(as_tuple=False).squeeze(-1)

    # Thresholds derived from running_stats.promotion_thresholds() — same
    # method used by state.promote_belief(). No hardcoded constants.
    with torch.no_grad():
        for idx in active_idx.tolist():
            current_level = state.belief_level[idx].item()
            if current_level >= 3:
                continue  # already at maximum abstraction level

            radius = state.beliefs.data[idx].norm().item()
            access = state.belief_access_count[idx].item()

            min_r, min_a = state.running_stats.promotion_thresholds(current_level)
            if radius >= min_r and access >= min_a:
                state.belief_level[idx] = current_level + 1
                if hasattr(state, 'belief_source_type'):
                    state.belief_source_type[idx] = 2  # promotion


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

    # Dedup removed from per-merge redirect; now called once after all merges


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

    # Directed key: (src, tgt) preserves causal edge direction
    seen: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(len(active_edges)):
        key = (srcs[i].item(), tgts[i].item())
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
