"""Hebbian association: create, reinforce, and decay edges between beliefs.

"Neurons that fire together wire together" — and edges that don't fire, fade.

Three mechanisms, matching Ba-Hinton Fast Weights (2016) A_t = λ·A_{t-1} +
η·h_t·h_tᵀ applied to the edge graph instead of a dense matrix:

1. **Decay** (λ): every active edge's weight is multiplied by
   (1 − hebbian_decay) each step. Unused edges exponentially approach zero.
2. **Reinforcement** (η): when a pair of beliefs co-activates AND an edge
   already connects them, its weight is incremented by hebbian_lr.
3. **Creation**: co-activated pairs without an existing edge get one
   allocated at initial_edge_weight (structural plasticity).

Decayed edges that cross the EPSILON threshold are collected by
pass2's structural zero-weight sweep (pass2.py:207), closing the
lifecycle. No separate pruning path is needed.

Both rates (hebbian_decay, hebbian_lr) and the initial weight
(initial_edge_weight) are learned MetaParams — no constants.

References:
- Ba, Hinton, Mnih, Leibo, Ionescu (NeurIPS 2016), "Using Fast Weights
  to Attend to the Recent Past". Canonical decay+reinforce formulation
  for associative memory.
- Prior Rust port: prototype-research/src/pipeline/hebbian.rs
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, EPSILON


def decay_edge_weights(state: CognitiveState) -> dict:
    """Apply Ba-Hinton fast-weights decay to all active edges.

    edge_weights_t = (1 − hebbian_decay) · edge_weights_{t-1}

    Call once per step, before the structural zero-weight sweep in pass2.
    Unused edges (never reinforced) geometrically approach zero and are
    collected by the EPSILON-threshold cleanup, closing the lifecycle
    without explicit pruning logic.

    hebbian_decay is a learned sigmoid MetaParam (default ≈ 0.01 per step;
    half-life ≈ 70 steps). Runs under no_grad — this is a fast-weights
    update, not part of the slow-weight autograd path.

    Returns:
        dict with n_decayed (count of edges touched) and mean_weight_after.
    """
    stats = {'n_decayed': 0, 'mean_weight_after': 0.0}
    with torch.no_grad():
        if not state.edge_active.any():
            return stats
        active_mask = state.edge_active
        decay_rate = state.meta_params.hebbian_decay   # sigmoid output, scalar tensor
        retention = 1.0 - decay_rate
        state.edge_weights.data[active_mask] = (
            state.edge_weights.data[active_mask] * retention
        )
        stats['n_decayed'] = int(active_mask.sum().item())
        stats['mean_weight_after'] = float(
            state.edge_weights.data[active_mask].abs().mean().item()
        )
    return stats


def reinforce_existing_edges(
    state: CognitiveState,
    co_activated_pairs: list[tuple[int, int]],
) -> int:
    """Increment weight of edges that connect a co-activated pair.

    This is the η·h·hᵀ half of Ba-Hinton Fast Weights applied to the edge
    graph. Runs unconditionally — unlike creation (gated by edge_fill < 0.9
    in pass2), reinforcement is most critical when the graph is saturated:
    it's what separates used edges (weight grows) from dormant ones
    (weight decays to zero).

    Args:
        state: cognitive state
        co_activated_pairs: list of (belief_idx_a, belief_idx_b) pairs

    Returns:
        Number of edges whose weight was incremented.
    """
    if not co_activated_pairs:
        return 0
    with torch.no_grad():
        if not state.edge_active.any():
            return 0
        device = state.beliefs.device
        active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) == 0:
            return 0

        srcs = state.edge_src[active_idx]
        tgts = state.edge_tgt[active_idx]
        edge_mins = torch.min(srcs, tgts)
        edge_maxs = torch.max(srcs, tgts)

        pairs_t = torch.tensor(co_activated_pairs, dtype=torch.long, device=device)
        pair_mins = pairs_t.min(dim=1).values
        pair_maxs = pairs_t.max(dim=1).values

        match_matrix = (
            (edge_mins.unsqueeze(1) == pair_mins.unsqueeze(0)) &
            (edge_maxs.unsqueeze(1) == pair_maxs.unsqueeze(0))
        )
        edge_reinforced = match_matrix.any(dim=1)
        if not edge_reinforced.any():
            return 0

        lr = state.meta_params.hebbian_lr
        edges_to_reinforce = active_idx[edge_reinforced]
        state.edge_weights.data[edges_to_reinforce] = (
            state.edge_weights.data[edges_to_reinforce] + lr
        )
        return int(edge_reinforced.sum().item())


def hebbian_update(
    state: CognitiveState,
    co_activated_pairs: list[tuple[int, int]],
) -> dict:
    """Create edges for new co-activation pairs AND reinforce existing ones.

    For each co-activated pair (a, b):
      - if no edge connects (a, b): allocate one at initial_edge_weight.
      - if one already exists: edge_weights[eidx] += hebbian_lr (bounded
        at the same max-weight the homeostatic system targets).

    This is the η·h·hᵀ half of Ba-Hinton fast weights; the λ·A part lives
    in decay_edge_weights() and is called once per step by pass2.

    Uses tensor operations instead of Python dicts for edge matching:
    builds canonical (min, max) edge keys as tensors and matches via
    broadcasting, avoiding per-edge Python iteration.

    Args:
        state: cognitive state
        co_activated_pairs: list of (belief_idx_a, belief_idx_b) pairs that
            were co-activated (both read in the same forward pass)

    Returns:
        dict with n_reinforced (existing edges whose weight was incremented)
        and n_created (new edges allocated).
    """
    out = {'n_reinforced': 0, 'n_created': 0}
    with torch.no_grad():
        if not co_activated_pairs:
            return out

        device = state.beliefs.device

        if not state.edge_active.any():
            for a, b in co_activated_pairs:
                if _create_edge(state, a, b):
                    out['n_created'] += 1
            return out

        active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) == 0:
            return out

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

        pair_matched = match_matrix.any(dim=0)     # [P] — pairs with existing edges
        edge_reinforced = match_matrix.any(dim=1)  # [E] — edges that match some pair

        # Reinforce existing edges. Each matched edge's weight gets += hebbian_lr.
        # If multiple pairs map to the same edge (shouldn't happen for unique pairs
        # from extract_co_activations but we handle it), we still only reinforce
        # once — the pair→edge mapping is many-to-one at most via match_matrix.
        if edge_reinforced.any():
            lr = state.meta_params.hebbian_lr  # scalar tensor
            edges_to_reinforce = active_idx[edge_reinforced]
            state.edge_weights.data[edges_to_reinforce] = (
                state.edge_weights.data[edges_to_reinforce] + lr
            )
            out['n_reinforced'] = int(edge_reinforced.sum().item())

        # Create edges for unmatched pairs
        if not pair_matched.all():
            new_pairs = pairs_t[~pair_matched]
            for i in range(len(new_pairs)):
                if _create_edge(state, new_pairs[i, 0].item(), new_pairs[i, 1].item()):
                    out['n_created'] += 1

        return out


def _create_edge(state: CognitiveState, a: int, b: int) -> bool:
    """Create an edge between two beliefs (no existence check — caller ensures uniqueness).

    Returns True if an edge was actually allocated, False if allocation failed
    (e.g. edge slots full). The caller uses this for bookkeeping.
    """
    relation = torch.zeros(state.config.relation_dim, device=state.beliefs.device)
    eidx = state.allocate_edge(a, b, relation, weight=state.meta_params.initial_edge_weight.item())
    return eidx >= 0


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
