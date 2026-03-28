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

    Vectorized: builds edge index hash for O(1) lookups, batches
    strengthening and decay operations.

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
                _create_edge(state, a, b)
            return

        active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) == 0:
            return

        # Build edge index for O(1) lookups
        edge_index: dict[tuple[int, int], int] = {}
        srcs = state.edge_src[active_idx]
        tgts = state.edge_tgt[active_idx]
        for i, eidx in enumerate(active_idx.tolist()):
            key = (min(srcs[i].item(), tgts[i].item()), max(srcs[i].item(), tgts[i].item()))
            edge_index[key] = eidx

        # Build set of co-activated pairs (canonical order)
        co_active_set = set()
        for a, b in co_activated_pairs:
            co_active_set.add((min(a, b), max(a, b)))

        # Separate active edges into strengthened vs decayed
        strengthen_edges = []
        decay_edges = []

        for eidx_local in range(len(active_idx)):
            eidx = active_idx[eidx_local].item()
            if state.immutable_edges[eidx]:
                continue
            src = srcs[eidx_local].item()
            tgt = tgts[eidx_local].item()
            pair = (min(src, tgt), max(src, tgt))

            if pair in co_active_set:
                strengthen_edges.append(eidx)
                co_active_set.discard(pair)
            else:
                # Skip causal edges — they have their own decay
                if state.edge_causal_obs[eidx].item() > 0:
                    continue
                decay_edges.append(eidx)

        # Batch strengthen
        if strengthen_edges:
            s_idx = torch.tensor(strengthen_edges, dtype=torch.long, device=state.beliefs.device)
            w = state.edge_weights.data[s_idx]
            w_new = (w + learning_rate * (1.0 - w)).clamp(max=1.0)
            state.edge_weights.data[s_idx] = w_new

        # Batch decay
        if decay_edges:
            d_idx = torch.tensor(decay_edges, dtype=torch.long, device=state.beliefs.device)
            w = state.edge_weights.data[d_idx]
            w_new = w * (1.0 - decay_rate)
            state.edge_weights.data[d_idx] = w_new
            # Deallocate edges too weak
            dead = w_new < 0.01
            if dead.any():
                for eidx in d_idx[dead].tolist():
                    state.deallocate_edge(eidx)

        # Create new edges for remaining co-activated pairs that don't have edges
        for a, b in co_active_set:
            _create_edge(state, a, b)


def _create_edge(state: CognitiveState, a: int, b: int):
    """Create an edge between two beliefs (no existence check — caller ensures uniqueness)."""
    relation = torch.zeros(state.config.relation_dim, device=state.beliefs.device)
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
