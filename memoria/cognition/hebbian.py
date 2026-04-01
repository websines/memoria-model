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

    Uses tensor operations instead of Python dicts for edge matching:
    builds canonical (min, max) edge keys as tensors and matches via
    broadcasting, avoiding per-edge Python iteration.

    Args:
        state: cognitive state
        co_activated_pairs: list of (belief_idx_a, belief_idx_b) pairs that
            were co-activated (both read in the same forward pass)
        learning_rate: η in the Hebb rule
        decay_rate: how much inactive edges decay per step
    """
    with torch.no_grad():
        if not co_activated_pairs:
            return

        device = state.beliefs.device

        if not state.edge_active.any():
            for a, b in co_activated_pairs:
                _create_edge(state, a, b)
            return

        active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) == 0:
            return

        srcs = state.edge_src[active_idx]
        tgts = state.edge_tgt[active_idx]

        # Canonical edge keys: (min, max) as tensors
        edge_mins = torch.min(srcs, tgts)  # [E]
        edge_maxs = torch.max(srcs, tgts)  # [E]

        # Co-activated pairs as tensors
        pairs_t = torch.tensor(co_activated_pairs, dtype=torch.long, device=device)
        pair_mins = pairs_t.min(dim=1).values  # [P]
        pair_maxs = pairs_t.max(dim=1).values  # [P]

        # Match: [E, P] boolean — which edges match which pairs
        match_matrix = (
            (edge_mins.unsqueeze(1) == pair_mins.unsqueeze(0)) &
            (edge_maxs.unsqueeze(1) == pair_maxs.unsqueeze(0))
        )

        edge_matched = match_matrix.any(dim=1)   # [E] — edges with a co-activated pair
        pair_matched = match_matrix.any(dim=0)    # [P] — pairs with existing edges

        immutable = state.immutable_edges[active_idx]
        causal = state.edge_causal_obs[active_idx] > 0

        # Strengthen matched, mutable edges
        strengthen_mask = edge_matched & ~immutable
        if strengthen_mask.any():
            s_idx = active_idx[strengthen_mask]
            w = state.edge_weights.data[s_idx]
            state.edge_weights.data[s_idx] = (w + learning_rate * (1.0 - w)).clamp(max=1.0)

        # Decay non-matched, non-causal, mutable edges
        decay_mask = ~edge_matched & ~immutable & ~causal
        if decay_mask.any():
            d_idx = active_idx[decay_mask]
            w = state.edge_weights.data[d_idx]
            w_new = w * (1.0 - decay_rate)
            state.edge_weights.data[d_idx] = w_new
            dead = w_new < 0.01
            if dead.any():
                for eidx in d_idx[dead].tolist():
                    state.deallocate_edge(eidx)

        # Create edges for unmatched pairs
        if not pair_matched.all():
            new_pairs = pairs_t[~pair_matched]
            for i in range(len(new_pairs)):
                _create_edge(state, new_pairs[i, 0].item(), new_pairs[i, 1].item())


def _create_edge(state: CognitiveState, a: int, b: int):
    """Create an edge between two beliefs (no existence check — caller ensures uniqueness)."""
    relation = torch.zeros(state.config.relation_dim, device=state.beliefs.device)
    state.allocate_edge(a, b, relation, weight=0.1)


def extract_co_activations(
    state: CognitiveState,
    read_indices: list[int],
    max_pairs: int = 256,
) -> list[tuple[int, int]]:
    """Extract co-activation pairs from beliefs read in the same pass.

    Caps the number of indices to prevent O(N^2) explosion. With 23 indices,
    we get 253 pairs (just under 256 cap). With 80+ indices uncapped,
    we'd get 3000+ pairs.

    Args:
        state: cognitive state
        read_indices: indices of beliefs that were read during this forward pass
        max_pairs: maximum number of pairs to return

    Returns:
        list of (idx_a, idx_b) pairs
    """
    import random
    indices = sorted(set(read_indices))

    # Cap indices to prevent quadratic blowup
    # N choose 2 <= max_pairs → N <= ~sqrt(2 * max_pairs)
    max_indices = int((2 * max_pairs) ** 0.5) + 1
    if len(indices) > max_indices:
        indices = random.sample(indices, max_indices)
        indices.sort()

    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pairs.append((indices[i], indices[j]))
    return pairs[:max_pairs]
