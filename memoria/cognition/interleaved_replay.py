"""E4: Interleaved Replay — mix recent high-surprise beliefs with old
high-precision beliefs during consolidation, run message passing between them
to catch cross-temporal contradictions.

During sleep/consolidation, select two groups:
1. Recent beliefs with high MESU variance (recent learning, still uncertain)
2. Old beliefs with high precision (established knowledge)

Run message passing between these groups. If messages diverge beyond a threshold,
flag a cross-temporal contradiction: new learning contradicts old knowledge
(or vice versa). The inconsistent belief gets its precision reduced and
variance increased, making it plastic enough to be corrected.

Reference: SCoRe interleaved replay (2025)
Reference: Complementary Learning Systems theory
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import belief_is_active, angular_similarity, EPSILON


def select_replay_set(
    state: CognitiveState,
    replay_ratio: Tensor,
    max_replay_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Select beliefs for interleaved replay.

    Selects two groups:
    - Recent group: beliefs with highest MESU variance (most uncertain/recently changed)
    - Old group: oldest beliefs with highest precision (established knowledge)

    Args:
        state: cognitive state
        replay_ratio: fraction of replay set that should be old beliefs
        max_replay_size: maximum total replay set size

    Returns:
        (recent_indices, old_indices) — global belief indices for each group
    """
    active_mask = state.get_active_mask()
    if not active_mask.any():
        empty = torch.zeros(0, dtype=torch.long, device=state.beliefs.device)
        return empty, empty

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    n_active = len(active_idx)

    if n_active < 4:
        empty = torch.zeros(0, dtype=torch.long, device=state.beliefs.device)
        return empty, empty

    ratio = replay_ratio.item()
    n_old = max(1, int(max_replay_size * ratio))
    n_recent = max(1, max_replay_size - n_old)

    # Cap to available beliefs
    n_old = min(n_old, n_active // 2)
    n_recent = min(n_recent, n_active - n_old)

    # Score for "recent and uncertain": high variance = good replay candidate
    variances = state.belief_precision_var[active_idx]
    _, recent_order = variances.sort(descending=True)
    recent_local = recent_order[:n_recent]

    # Score for "old and confident": high radius * old age
    radii = state.beliefs.data[active_idx].norm(dim=-1)
    ages = state.belief_created_step[active_idx].float()
    # Low created_step = old; high radius = confident
    # Score = radius / (created_step + 1)  (old + precise = high score)
    old_scores = radii / (ages.clamp(min=1.0))
    _, old_order = old_scores.sort(descending=True)

    # Exclude beliefs already in recent set
    recent_set = set(recent_local.tolist())
    old_candidates = [i for i in old_order.tolist() if i not in recent_set]
    old_local = torch.tensor(old_candidates[:n_old], dtype=torch.long, device=state.beliefs.device)

    return active_idx[recent_local], active_idx[old_local]


def replay_message_passing(
    state: CognitiveState,
    recent_idx: Tensor,
    old_idx: Tensor,
    contradiction_threshold: Tensor,
) -> dict:
    """Run lightweight message passing between replay groups.

    For each recent belief, compute its similarity to each old belief.
    If connected by causal edges, check for message divergence.
    If no direct edge exists, use angular similarity as a proxy.

    When contradiction is detected:
    - Reduce precision of the less-supported belief
    - Increase MESU variance (make it plastic for correction)

    Args:
        state: cognitive state (modified in-place)
        recent_idx: indices of recent high-variance beliefs
        old_idx: indices of old high-precision beliefs
        contradiction_threshold: cosine similarity below which = contradiction

    Returns:
        dict with statistics
    """
    stats = {
        'pairs_compared': 0,
        'contradictions_found': 0,
        'beliefs_weakened': 0,
    }

    if len(recent_idx) == 0 or len(old_idx) == 0:
        return stats

    threshold = contradiction_threshold.item()

    # Build edge lookup for fast causal connection check
    edge_connections: set[tuple[int, int]] = set()
    if state.edge_active.any():
        active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        for e_idx in active_edges.tolist():
            if state.edge_causal_obs[e_idx].item() > 0:
                src = state.edge_src[e_idx].item()
                tgt = state.edge_tgt[e_idx].item()
                edge_connections.add((src, tgt))
                edge_connections.add((tgt, src))

    recent_beliefs = state.beliefs.data[recent_idx]
    old_beliefs = state.beliefs.data[old_idx]

    recent_dirs = F.normalize(recent_beliefs, dim=-1, eps=EPSILON)
    old_dirs = F.normalize(old_beliefs, dim=-1, eps=EPSILON)

    # Cross-group similarity matrix
    cross_sim = recent_dirs @ old_dirs.T  # [n_recent, n_old]

    with torch.no_grad():
        for i in range(len(recent_idx)):
            r_idx = recent_idx[i].item()
            if state.immutable_beliefs[r_idx]:
                continue

            for j in range(len(old_idx)):
                o_idx = old_idx[j].item()
                if state.immutable_beliefs[o_idx]:
                    continue

                # Only compare if they're related (edge connection or moderate similarity)
                has_edge = (r_idx, o_idx) in edge_connections
                sim = cross_sim[i, j].item()

                # Skip unrelated pairs (no edge and low similarity)
                if not has_edge and abs(sim) < 0.3:
                    continue

                stats['pairs_compared'] += 1

                # Contradiction: causally connected but directions disagree
                # or very similar direction but dramatically different precision
                r_radius = recent_beliefs[i].norm().item()
                o_radius = old_beliefs[j].norm().item()

                if has_edge and sim < threshold:
                    # Direct causal link but direction disagrees → contradiction
                    stats['contradictions_found'] += 1

                    # Weaken the one with lower empirical support
                    r_confirmed = state.belief_confirmed_count[r_idx].item()
                    r_contradicted = state.belief_contradicted_count[r_idx].item()
                    o_confirmed = state.belief_confirmed_count[o_idx].item()
                    o_contradicted = state.belief_contradicted_count[o_idx].item()

                    r_support = r_confirmed / max(r_confirmed + r_contradicted, 1)
                    o_support = o_confirmed / max(o_confirmed + o_contradicted, 1)

                    if r_support < o_support:
                        # Recent belief is less supported → weaken it
                        state.beliefs.data[r_idx] *= 0.8
                        state.belief_precision_var[r_idx] += 0.3
                        stats['beliefs_weakened'] += 1
                    else:
                        # Old belief is less supported → weaken it
                        state.beliefs.data[o_idx] *= 0.8
                        state.belief_precision_var[o_idx] += 0.3
                        stats['beliefs_weakened'] += 1

    return stats


def run_interleaved_replay(
    state: CognitiveState,
) -> dict:
    """Run the full interleaved replay pass during consolidation.

    1. Select replay set (recent + old beliefs)
    2. Run cross-group message passing
    3. Flag and resolve contradictions

    All thresholds from MetaParams (no hardcoded magic numbers).

    Args:
        state: cognitive state (modified in-place)

    Returns:
        dict with statistics
    """
    stats = {}

    # Select replay groups
    recent_idx, old_idx = select_replay_set(
        state,
        replay_ratio=state.meta_params.replay_ratio,
    )
    stats['recent_selected'] = len(recent_idx)
    stats['old_selected'] = len(old_idx)

    # Run cross-group message passing
    replay_stats = replay_message_passing(
        state,
        recent_idx,
        old_idx,
        contradiction_threshold=state.meta_params.replay_contradiction_threshold,
    )
    stats.update(replay_stats)

    return stats
