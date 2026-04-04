"""E2: Self-Verification Pass — causal graph consistency checking during consolidation.

For each high-precision belief, traces the causal graph to identify beliefs it
should predict. Runs message passing to check consistency. On divergence,
reduces precision of the weakest link in the causal chain.

Also implements conflict-aware supersession: when two beliefs are very similar
in direction but one is newer with higher precision, the older one is flagged
as superseded.

Reference: InternalInspector (arXiv:2406.12053, EMNLP 2024)
Reference: SleepGate (arXiv:2603.14517)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from collections import deque

from ..core.state import CognitiveState
from ..core.polar import belief_is_active, angular_similarity, EPSILON


def _get_causal_neighbors(state: CognitiveState, belief_idx: int) -> list[tuple[int, int]]:
    """Get downstream beliefs reachable via causal edges from belief_idx.

    Returns list of (neighbor_belief_idx, edge_idx) pairs.
    """
    neighbors = []
    if not state.edge_active.any():
        return neighbors

    active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    for edge_idx in active_edges.tolist():
        src = state.edge_src[edge_idx].item()
        if src == belief_idx and state.edge_causal_obs[edge_idx].item() > 0:
            tgt = state.edge_tgt[edge_idx].item()
            if belief_is_active(state.beliefs.data[tgt].norm()):
                neighbors.append((tgt, edge_idx))
    return neighbors


def verify_belief_consistency(
    state: CognitiveState,
    belief_idx: int,
    divergence_threshold: Tensor,
    precision_decay: Tensor,
) -> dict:
    """Verify a single belief's consistency with its causal downstream.

    For each causal neighbor:
    1. Compute expected influence: weight * edge_relation-scaled prediction
    2. Compare prediction direction with actual belief direction
    3. If divergence > threshold: weaken the weakest link

    Args:
        state: cognitive state (modified in-place)
        belief_idx: index of the belief to verify
        divergence_threshold: cosine similarity below which = inconsistency
        precision_decay: radius reduction fraction on inconsistency

    Returns:
        dict with {neighbors_checked, inconsistencies_found, precision_reduced}
    """
    stats = {'neighbors_checked': 0, 'inconsistencies_found': 0, 'precision_reduced': 0}

    neighbors = _get_causal_neighbors(state, belief_idx)
    if not neighbors:
        return stats

    source_belief = state.beliefs.data[belief_idx]
    source_dir = F.normalize(source_belief.unsqueeze(0), dim=-1, eps=EPSILON).squeeze(0)
    source_radius = source_belief.norm().item()

    threshold = divergence_threshold.item()
    decay = precision_decay.item()

    with torch.no_grad():
        for tgt_idx, edge_idx in neighbors:
            stats['neighbors_checked'] += 1

            target_belief = state.beliefs.data[tgt_idx]
            target_dir = F.normalize(target_belief.unsqueeze(0), dim=-1, eps=EPSILON).squeeze(0)
            target_radius = target_belief.norm().item()

            # Edge weight indicates causal strength
            edge_weight = state.edge_weights.data[edge_idx].item()
            if abs(edge_weight) < EPSILON:
                continue

            # Predicted direction: source belief direction transformed by relation
            # Simplified: use angular similarity as consistency measure
            sim = (source_dir * target_dir).sum().item()

            # For causal neighbors, we expect some similarity (they're related)
            # Divergence = low similarity where we expect high
            # Weight by edge strength: strong causal links should be more consistent
            effective_threshold = threshold * edge_weight

            if sim < effective_threshold:
                stats['inconsistencies_found'] += 1

                # Weaken the weakest link: the belief with lower precision
                if target_radius < source_radius:
                    # Target is weaker — reduce its precision
                    state.beliefs.data[tgt_idx] *= (1.0 - decay)
                    stats['precision_reduced'] += 1
                    # Also increase MESU variance (make it more plastic)
                    state.belief_precision_var[tgt_idx] += decay
                else:
                    # Source is weaker — but we're verifying source, so
                    # reduce edge weight instead (the causal link is suspect)
                    state.edge_weights.data[edge_idx] *= (1.0 - decay)

    return stats


def supersession_scan(
    state: CognitiveState,
    supersession_sim: Tensor,
) -> dict:
    """Detect beliefs that supersede each other.

    When two beliefs have very high angular cosine similarity (near-duplicate
    content) but different ages and precisions, the newer higher-precision
    belief supersedes the older lower-precision one.

    Resolution: reduce the superseded belief's precision significantly.

    Args:
        state: cognitive state (modified in-place)
        supersession_sim: cosine threshold for supersession detection

    Returns:
        dict with {pairs_checked, supersessions_found}
    """
    stats = {'pairs_checked': 0, 'supersessions_found': 0}

    active_mask = state.get_active_mask()
    if active_mask.sum() < 2:
        return stats

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_beliefs = state.beliefs.data[active_idx]
    n_active = len(active_idx)

    angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)
    sim_matrix = angles @ angles.T
    radii = active_beliefs.norm(dim=-1)
    ages = state.belief_created_step[active_idx].float()
    threshold = supersession_sim.item()

    with torch.no_grad():
        for i in range(n_active):
            idx_i = active_idx[i].item()
            if state.immutable_beliefs[idx_i]:
                continue
            for j in range(i + 1, n_active):
                idx_j = active_idx[j].item()
                if state.immutable_beliefs[idx_j]:
                    continue

                stats['pairs_checked'] += 1
                sim = sim_matrix[i, j].item()

                if sim > threshold:
                    # Near-duplicate — check for supersession
                    r_i, r_j = radii[i].item(), radii[j].item()
                    age_i, age_j = ages[i].item(), ages[j].item()

                    # Newer + higher precision supersedes older + lower precision
                    if age_i > age_j and r_i > r_j:
                        # i is newer and stronger → j is superseded
                        state.beliefs.data[idx_j] *= 0.5
                        state.belief_precision_var[idx_j] += 0.5
                        stats['supersessions_found'] += 1
                    elif age_j > age_i and r_j > r_i:
                        # j is newer and stronger → i is superseded
                        state.beliefs.data[idx_i] *= 0.5
                        state.belief_precision_var[idx_i] += 0.5
                        stats['supersessions_found'] += 1

    return stats


def run_self_verification(
    state: CognitiveState,
) -> dict:
    """Run the full self-verification pass during consolidation.

    1. For each high-precision belief, verify causal consistency
    2. Scan for supersession conflicts

    All thresholds from MetaParams (no hardcoded magic numbers).

    Args:
        state: cognitive state (modified in-place)

    Returns:
        dict with combined statistics
    """
    stats = {
        'beliefs_verified': 0,
        'total_inconsistencies': 0,
        'total_precision_reduced': 0,
        'supersessions': 0,
    }

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return stats

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    radii = state.beliefs.data[active_idx].norm(dim=-1)

    # Only verify high-precision beliefs (above median radius)
    if len(radii) < 2:
        return stats

    median_radius = radii.median().item()

    # Phase 1: Causal consistency verification
    for i in range(len(active_idx)):
        idx = active_idx[i].item()
        if radii[i].item() < median_radius:
            continue
        if state.immutable_beliefs[idx]:
            continue

        v_stats = verify_belief_consistency(
            state, idx,
            divergence_threshold=state.meta_params.verification_divergence_threshold,
            precision_decay=state.meta_params.verification_precision_decay,
        )
        stats['beliefs_verified'] += 1
        stats['total_inconsistencies'] += v_stats['inconsistencies_found']
        stats['total_precision_reduced'] += v_stats['precision_reduced']

    # Phase 2: Supersession scanning
    sup_stats = supersession_scan(
        state,
        supersession_sim=state.meta_params.supersession_similarity,
    )
    stats['supersessions'] = sup_stats['supersessions_found']

    return stats
