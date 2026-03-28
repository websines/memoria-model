"""Hebbian association: co-activated beliefs strengthen connecting edges.

"Neurons that fire together wire together."

After each forward pass, beliefs that were both read (co-activated) during
the same pass strengthen their connecting edge. Edges not activated decay.

Update rule (saturating Hebb):
    w_new = w_old + η(1 - w_old)    # strengthening, saturates at 1.0
    w_new = w_old × (1 - decay)      # decay for inactive edges

Ported from: prototype-research/src/pipeline/hebbian.rs
Reference: Fast Weights (Ba & Hinton, 2016) — outer-product Hebbian updates
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, EPSILON


def hebbian_update(
    state: CognitiveState,
    co_activated_pairs: list[tuple[int, int]],
    learning_rate: float = 0.05,
    decay_rate: float = 0.01,
):
    """Update edge weights based on co-activation.

    Args:
        state: cognitive state
        co_activated_pairs: list of (belief_idx_a, belief_idx_b) pairs that
            were co-activated (both read in the same forward pass)
        learning_rate: η in the Hebb rule
        decay_rate: how much inactive edges decay per step
    """
    with torch.no_grad():
        if not state.edge_active.any():
            # No edges → create edges for co-activated pairs
            for a, b in co_activated_pairs:
                _ensure_edge(state, a, b)
            return

        # Get active edges
        active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) == 0:
            return

        # Build set of co-activated pairs for fast lookup
        co_active_set = set()
        for a, b in co_activated_pairs:
            co_active_set.add((min(a, b), max(a, b)))  # canonical order

        # Update each active edge
        for idx in active_idx.tolist():
            if state.immutable_edges[idx]:
                continue

            src = state.edge_src[idx].item()
            tgt = state.edge_tgt[idx].item()
            pair = (min(src, tgt), max(src, tgt))

            w = state.edge_weights.data[idx].item()

            if pair in co_active_set:
                # Strengthening: saturating Hebb rule
                w_new = w + learning_rate * (1.0 - w)
                state.edge_weights.data[idx] = min(w_new, 1.0)
                co_active_set.discard(pair)  # handled
            else:
                # Decay: not co-activated this step
                w_new = w * (1.0 - decay_rate)
                if w_new < 0.01:
                    # Edge too weak → deallocate
                    state.deallocate_edge(idx)
                else:
                    state.edge_weights.data[idx] = w_new

        # Create new edges for co-activated pairs that don't have edges yet
        for a, b in co_active_set:
            _ensure_edge(state, a, b)


def _ensure_edge(state: CognitiveState, a: int, b: int):
    """Create an edge between two beliefs if none exists."""
    # Check if edge already exists
    if state.edge_active.any():
        active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        for idx in active_idx.tolist():
            src = state.edge_src[idx].item()
            tgt = state.edge_tgt[idx].item()
            if (src == a and tgt == b) or (src == b and tgt == a):
                return  # already exists

    # Create with initial low weight
    relation = torch.zeros(state.config.relation_dim)
    state.allocate_edge(a, b, relation, weight=0.1)


def extract_co_activations(
    state: CognitiveState,
    read_indices: list[int],
) -> list[tuple[int, int]]:
    """Extract co-activation pairs from beliefs read in the same pass.

    All pairs of beliefs that were retrieved together are co-activated.

    Args:
        state: cognitive state
        read_indices: indices of beliefs that were read during this forward pass

    Returns:
        list of (idx_a, idx_b) pairs
    """
    pairs = []
    indices = sorted(set(read_indices))
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pairs.append((indices[i], indices[j]))
    return pairs
