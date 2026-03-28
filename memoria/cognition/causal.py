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
