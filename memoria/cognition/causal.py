"""Causal reasoning: d-separation and interventional queries on the relation graph.

d-separation: determine conditional independence from graph structure.
Intervention (do-calculus): clamp a belief, zero incoming edges, propagate.

These are graph algorithms implemented as tensor operations on the relation region.

Ported from: prototype-research/src/causal/bayes_ball.rs, do_operator.rs
Reference: dfgo (github.com/brentyi/dfgo) — differentiable factor graph ops in PyTorch
"""

import torch
from torch import Tensor
from collections import deque

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, EPSILON


def build_adjacency(state: CognitiveState) -> dict[int, list[tuple[int, float]]]:
    """Build adjacency list from active edges.

    Returns:
        Dict mapping belief_idx → list of (neighbor_idx, edge_weight)
        Directed: only src → tgt entries.
    """
    adj: dict[int, list[tuple[int, float]]] = {}

    if not state.edge_active.any():
        return adj

    active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    for eidx in active_idx.tolist():
        src = state.edge_src[eidx].item()
        tgt = state.edge_tgt[eidx].item()
        w = state.edge_weights.data[eidx].item()
        if src not in adj:
            adj[src] = []
        adj[src].append((tgt, w))

    return adj


def _build_edge_index(state: CognitiveState) -> dict[tuple[int, int], int]:
    """Build (src, tgt) → edge_slot hash map for O(1) edge lookups.

    Only includes active edges.
    """
    index = {}
    if not state.edge_active.any():
        return index
    active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    srcs = state.edge_src[active_idx]
    tgts = state.edge_tgt[active_idx]
    for i, eidx in enumerate(active_idx.tolist()):
        index[(srcs[i].item(), tgts[i].item())] = eidx
    return index


def d_separated(
    state: CognitiveState,
    source: int,
    target: int,
    conditioned_on: set[int] | None = None,
) -> bool:
    """Test if source and target are d-separated given conditioned_on.

    Uses a simplified reachability check on the directed graph:
    - If there's a directed path from source to target not blocked by
      conditioned nodes, they are NOT d-separated.

    This is a simplified version. Full Bayes-Ball handles colliders (explaining away).
    For v1, we use directional reachability with blocking.

    Args:
        source: belief index
        target: belief index
        conditioned_on: set of belief indices that are observed/conditioned

    Returns:
        True if d-separated (conditionally independent)
    """
    if conditioned_on is None:
        conditioned_on = set()

    adj = build_adjacency(state)

    # BFS from source, blocked by conditioned nodes
    visited = set()
    queue = deque([source])
    visited.add(source)

    while queue:
        node = queue.popleft()

        if node == target:
            return False  # path exists → NOT d-separated

        if node in conditioned_on and node != source:
            continue  # blocked

        for neighbor, weight in adj.get(node, []):
            if neighbor not in visited and weight > 0.01:
                visited.add(neighbor)
                queue.append(neighbor)

    return True  # no unblocked path → d-separated


def intervene(
    state: CognitiveState,
    target_belief: int,
    intervention_value: Tensor,
    propagation_steps: int = 3,
) -> dict[int, Tensor]:
    """Perform an intervention (do-calculus): clamp a belief and propagate.

    do(X = v):
    1. Set belief X to intervention_value (clamp)
    2. Zero all incoming edges to X (break causal influence on X)
    3. Propagate outward from X through the causal graph
    4. Return the resulting belief values at downstream nodes

    This does NOT modify the actual state — it creates a copy for simulation.

    Args:
        state: cognitive state (not modified)
        target_belief: index of belief to intervene on
        intervention_value: [D] tensor to set the belief to
        propagation_steps: how many hops to propagate

    Returns:
        Dict mapping belief_idx → predicted belief tensor after intervention
    """
    # Copy beliefs for simulation (don't modify actual state)
    sim_beliefs = state.beliefs.data.clone()
    sim_beliefs[target_belief] = intervention_value

    adj = build_adjacency(state)

    # Build reverse adjacency (incoming edges)
    reverse_adj: dict[int, list[int]] = {}
    for src, neighbors in adj.items():
        for tgt, _ in neighbors:
            if tgt not in reverse_adj:
                reverse_adj[tgt] = []
            reverse_adj[tgt].append(src)

    # Zero incoming influence on the target (do-calculus: cut incoming edges)
    # In simulation, we just don't propagate TO the target
    clamped = {target_belief}

    # BFS propagation from target
    results = {target_belief: intervention_value}
    frontier = [target_belief]

    for step in range(propagation_steps):
        next_frontier = []
        for node in frontier:
            for neighbor, weight in adj.get(node, []):
                if neighbor in clamped:
                    continue  # don't update clamped nodes

                # Simple propagation: neighbor shifts toward node's value
                # weighted by edge weight and node's precision
                node_belief = sim_beliefs[node]
                node_radius = node_belief.norm().clamp(min=EPSILON)
                neighbor_belief = sim_beliefs[neighbor]
                neighbor_radius = neighbor_belief.norm().clamp(min=EPSILON)

                # Influence proportional to weight × source precision
                influence = weight * node_radius / (node_radius + neighbor_radius + EPSILON)

                node_angle = node_belief / node_radius
                neighbor_angle = neighbor_belief / max(neighbor_radius.item(), EPSILON)

                # Shift neighbor toward node (scaled by influence)
                new_angle = (1 - influence) * neighbor_angle + influence * node_angle
                new_angle = torch.nn.functional.normalize(new_angle, dim=0, eps=EPSILON)
                new_belief = new_angle * neighbor_radius  # keep neighbor's precision

                sim_beliefs[neighbor] = new_belief
                results[neighbor] = new_belief
                next_frontier.append(neighbor)

        frontier = next_frontier
        if not frontier:
            break

    return results


# ── Causal Edge Learning from Experience ──
# Learns directed edges from temporal surprise patterns:
# If belief A is updated with surprise at step t, and belief B is updated
# with surprise at step t+1, the signal A→B accumulates via Bayesian averaging.
#
# Ported from: prototype-research/src/causal/notears.rs (Bayesian edge accumulation)
# Distinct from Hebbian: Hebbian is undirected co-activation, causal is directed temporal precedence.


def causal_edge_learning(
    state: CognitiveState,
    updated_indices: list[int],
    surprise_values: list[float],
    min_signal: float | None = None,
    decay_rate: float | None = None,
) -> dict:
    """Learn directed causal edges from temporal surprise patterns.

    Vectorized: builds edge index hash for O(1) lookups, caps the cross-product
    of (prev × curr) pairs, and batches edge decay.

    Args:
        state: cognitive state (modified in-place)
        updated_indices: belief indices updated THIS step (from surprise computation)
        surprise_values: corresponding surprise values
        min_signal: minimum geometric-mean surprise to create/strengthen an edge
            (defaults to state.meta_params.causal_min_signal)
        decay_rate: per-step decay for unreinforced causal edges (slower than Hebbian)
            (defaults to state.meta_params.causal_decay_rate)

    Returns:
        dict with statistics
    """
    if min_signal is None:
        min_signal = state.meta_params.causal_min_signal.item()
    if decay_rate is None:
        decay_rate = state.meta_params.causal_decay_rate.item()
    stats = {'edges_created': 0, 'edges_strengthened': 0, 'edges_decayed': 0}

    with torch.no_grad():
        prev_surprise = state.belief_prev_surprise  # [max_beliefs]
        prev_updated = (prev_surprise > EPSILON).nonzero(as_tuple=False).squeeze(-1)

        if len(prev_updated) > 0 and len(updated_indices) > 0:
            # Build edge index once for O(1) lookups
            edge_index = _build_edge_index(state)

            # Build current update map — filter by EPSILON
            curr_map = {idx: s for idx, s in zip(updated_indices, surprise_values) if s > EPSILON}

            if not curr_map:
                # No meaningful current updates; skip to decay + store
                pass
            else:
                reinforced = set()

                # Pre-fetch belief angles once (used for relation vectors)
                belief_angles = state.get_belief_angles()

                K = state.config.relation_dim
                D = min(K, state.config.belief_dim)

                for prev_idx in prev_updated.tolist():
                    prev_s = prev_surprise[prev_idx].item()

                    for curr_idx, curr_s in curr_map.items():
                        if prev_idx == curr_idx:
                            continue

                        signal = (prev_s * curr_s) ** 0.5
                        if signal < min_signal:
                            continue

                        # O(1) edge lookup via hash map
                        edge_idx = edge_index.get((prev_idx, curr_idx), -1)

                        if edge_idx >= 0:
                            # Bayesian posterior update
                            obs = state.edge_causal_obs[edge_idx].item()
                            obs = max(obs, 1.0)
                            old_w = state.edge_weights.data[edge_idx].item()
                            new_w = (old_w * obs + signal) / (obs + 1.0)

                            state.edge_weights.data[edge_idx] = min(new_w, 1.0)
                            state.edge_causal_obs[edge_idx] = obs + 1.0

                            # Update relation vector
                            cause_angle = belief_angles[prev_idx]
                            effect_angle = belief_angles[curr_idx]
                            new_rel = torch.zeros(K, device=state.beliefs.device)
                            new_rel[:D] = (effect_angle[:D] - cause_angle[:D]) * state.meta_params.causal_relation_scale.item()
                            alpha = 1.0 / (obs + 1.0)
                            state.edge_relations.data[edge_idx] = (
                                (1.0 - alpha) * state.edge_relations.data[edge_idx] + alpha * new_rel
                            )

                            reinforced.add(edge_idx)
                            stats['edges_strengthened'] += 1
                        else:
                            # Create new directed causal edge
                            cause_angle = belief_angles[prev_idx]
                            effect_angle = belief_angles[curr_idx]
                            relation = torch.zeros(K, device=state.beliefs.device)
                            relation[:D] = (effect_angle[:D] - cause_angle[:D]) * state.meta_params.causal_relation_scale.item()

                            new_idx = state.allocate_edge(
                                prev_idx, curr_idx, relation, weight=signal * state.meta_params.causal_initial_weight_scale.item(),
                            )
                            if new_idx >= 0:
                                state.edge_causal_obs[new_idx] = 1.0
                                reinforced.add(new_idx)
                                edge_index[(prev_idx, curr_idx)] = new_idx
                                stats['edges_created'] += 1

                # Batch decay unreinforced causal edges
                if state.edge_active.any():
                    active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
                    causal_mask = state.edge_causal_obs[active_idx] > 0
                    immutable_mask = state.immutable_edges[active_idx]
                    decay_mask = causal_mask & ~immutable_mask

                    if decay_mask.any():
                        decay_edges = active_idx[decay_mask]
                        # Filter out reinforced edges
                        reinforced_t = torch.tensor(list(reinforced), dtype=torch.long, device=decay_edges.device) if reinforced else torch.tensor([], dtype=torch.long, device=decay_edges.device)
                        if len(reinforced_t) > 0:
                            # Create mask for non-reinforced
                            is_reinforced = (decay_edges.unsqueeze(-1) == reinforced_t.unsqueeze(0)).any(dim=-1)
                            decay_edges = decay_edges[~is_reinforced]

                        if len(decay_edges) > 0:
                            weights = state.edge_weights.data[decay_edges]
                            weights *= (1.0 - decay_rate)
                            state.edge_weights.data[decay_edges] = weights
                            # Deallocate edges that decayed below threshold
                            dead = weights < 0.01
                            if dead.any():
                                for eidx in decay_edges[dead].tolist():
                                    state.deallocate_edge(eidx)
                            stats['edges_decayed'] = len(decay_edges)

        # Store current step's surprises for next step
        state.belief_prev_surprise.zero_()
        for idx, s in zip(updated_indices, surprise_values):
            state.belief_prev_surprise[idx] = s

    return stats
