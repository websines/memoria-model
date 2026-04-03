"""Cross-run cognitive seed: save and load transferable cognitive knowledge.

Saves meta-parameters, Telos weights, running statistics, and high-confidence
core beliefs that survived training. On load, matches beliefs by content
similarity (cosine clustering), NOT by slot index.

Design principle: belief slots are mutable storage locations, not stable
semantic coordinates. EWC on raw slots is invalid. Content matching is required.

Reference: SDFT (arXiv:2601.19897) — self-distillation for continual learning
Reference: CALM (ICLR 2024) — composition via cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path

from ..core.state import CognitiveState
from ..core.polar import belief_is_active, EPSILON


def save_cognitive_seed(state: CognitiveState, path: str | Path, min_confidence: float = 0.5):
    """Save transferable cognitive knowledge from a trained state.

    Extracts:
    1. Meta-parameters (learned thresholds — most transferable)
    2. Telos module weights (learned goal system)
    3. Running statistics (accumulated signal distributions)
    4. Core beliefs: high-confidence beliefs that survived consolidation
    5. Core edges: edges between core beliefs with high weight
    6. Edge proposal network weights (if present)

    Args:
        state: trained cognitive state
        path: output file path
        min_confidence: minimum belief radius to include (filters noise)
    """
    path = Path(path)

    # Extract core beliefs (high confidence, well-connected)
    radii = state.get_belief_radii()
    active = belief_is_active(radii)
    confident = active & (radii > min_confidence)

    core_indices = confident.nonzero(as_tuple=False).squeeze(-1)
    core_beliefs = state.beliefs.data[core_indices].clone()
    core_radii = radii[core_indices].clone()
    core_access_counts = state.belief_access_count[core_indices].clone()

    # Extract edges between core beliefs
    core_set = set(core_indices.tolist())
    core_edges = []
    if state.edge_active.any():
        active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        for eidx in active_edges.tolist():
            src = state.edge_src[eidx].item()
            tgt = state.edge_tgt[eidx].item()
            if src in core_set and tgt in core_set:
                core_edges.append({
                    'src_belief': state.beliefs.data[src].clone(),
                    'tgt_belief': state.beliefs.data[tgt].clone(),
                    'relation': state.edge_relations.data[eidx].clone(),
                    'weight': state.edge_weights.data[eidx].item(),
                    'causal_obs': state.edge_causal_obs[eidx].item(),
                })

    # Edge proposal network (if it exists)
    edge_proposal_state = None
    if hasattr(state, 'edge_proposal'):
        edge_proposal_state = state.edge_proposal.state_dict()

    seed = {
        'version': 1,
        'meta_params': state.meta_params.state_dict(),
        'telos': state.telos.state_dict(),
        'running_stats': {k: v.clone() for k, v in state.running_stats._buffers.items()},
        'core_beliefs': core_beliefs,
        'core_radii': core_radii,
        'core_access_counts': core_access_counts,
        'core_edges': core_edges,
        'edge_proposal': edge_proposal_state,
        'n_core_beliefs': len(core_indices),
        'n_core_edges': len(core_edges),
        'belief_dim': state.config.belief_dim,
    }

    torch.save(seed, path)
    print(f"Cognitive seed saved: {path}")
    print(f"  {len(core_indices)} core beliefs (min radius={min_confidence})")
    print(f"  {len(core_edges)} core edges")


def load_cognitive_seed(
    state: CognitiveState,
    path: str | Path,
    transfer_beliefs: bool = True,
    transfer_edges: bool = True,
    match_threshold: float = 0.8,
):
    """Load cognitive seed into a fresh state, matching beliefs by content.

    Content matching: for each seed belief, find the closest existing belief
    (by cosine similarity). If sim > match_threshold AND the existing slot
    is empty or lower confidence, transfer the seed belief.

    New beliefs that don't match anything get allocated to empty slots.

    Args:
        state: target cognitive state (modified in-place)
        path: seed file path
        transfer_beliefs: whether to transfer core beliefs
        transfer_edges: whether to transfer core edges
        match_threshold: cosine similarity threshold for matching
    """
    path = Path(path)
    seed = torch.load(path, map_location=state.beliefs.device, weights_only=True)

    print(f"Loading cognitive seed: {path}")

    # 1. Always transfer meta-parameters (most stable across runs)
    if 'meta_params' in seed:
        state.meta_params.load_state_dict(seed['meta_params'])
        print("  Loaded meta_params")

    # 2. Always transfer Telos weights
    if 'telos' in seed:
        state.telos.load_state_dict(seed['telos'])
        print("  Loaded telos module")

    # 3. Always transfer running statistics
    if 'running_stats' in seed:
        for k, v in seed['running_stats'].items():
            if hasattr(state.running_stats, k):
                getattr(state.running_stats, k).copy_(v)
        print("  Loaded running_stats")

    # 4. Transfer edge proposal network
    if seed.get('edge_proposal') is not None and hasattr(state, 'edge_proposal'):
        state.edge_proposal.load_state_dict(seed['edge_proposal'])
        print("  Loaded edge_proposal")

    # 5. Content-matched belief transfer
    if transfer_beliefs and seed['n_core_beliefs'] > 0:
        n_transferred = _transfer_beliefs_by_content(
            state, seed['core_beliefs'], seed['core_radii'],
            seed.get('core_access_counts'), match_threshold,
        )
        print(f"  Transferred {n_transferred}/{seed['n_core_beliefs']} beliefs (match_threshold={match_threshold})")

    # 6. Content-matched edge transfer
    if transfer_edges and seed['n_core_edges'] > 0 and transfer_beliefs:
        n_edges = _transfer_edges_by_content(
            state, seed['core_edges'], match_threshold,
        )
        print(f"  Transferred {n_edges}/{seed['n_core_edges']} edges")


def _transfer_beliefs_by_content(
    state: CognitiveState,
    seed_beliefs: Tensor,
    seed_radii: Tensor,
    seed_access_counts: Tensor | None,
    match_threshold: float,
) -> int:
    """Transfer beliefs by content similarity matching.

    For each seed belief:
    1. Compute cosine similarity with all active beliefs in target
    2. If best match > threshold AND seed has higher confidence → update slot
    3. If no match → allocate to empty slot

    Returns number of beliefs transferred.
    """
    n_transferred = 0
    device = state.beliefs.device

    with torch.no_grad():
        seed_beliefs = seed_beliefs.to(device)
        seed_radii = seed_radii.to(device)

        # Normalize seed beliefs for cosine sim
        seed_angles = F.normalize(seed_beliefs, dim=-1, eps=EPSILON)

        for i in range(len(seed_beliefs)):
            seed_b = seed_beliefs[i]
            seed_r = seed_radii[i].item()
            seed_angle = seed_angles[i]

            # Check existing active beliefs
            active_mask = state.get_active_mask()

            if active_mask.any():
                active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
                active_beliefs = state.beliefs.data[active_indices]
                active_angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)
                active_radii = active_beliefs.norm(dim=-1)

                # Cosine similarity with all active beliefs
                sims = (seed_angle.unsqueeze(0) @ active_angles.T).squeeze(0)
                best_sim, best_local = sims.max(dim=0)

                if best_sim.item() > match_threshold:
                    # Match found — only transfer if seed is more confident
                    target_slot = active_indices[best_local].item()
                    target_radius = active_radii[best_local].item()

                    if seed_r > target_radius and not state.immutable_beliefs[target_slot]:
                        state.beliefs.data[target_slot] = seed_b
                        if seed_access_counts is not None:
                            state.belief_access_count[target_slot] = seed_access_counts[i].item()
                        n_transferred += 1
                    continue

            # No match — allocate to empty slot
            slot = state.allocate_belief(seed_b)
            if slot >= 0:
                if seed_access_counts is not None:
                    state.belief_access_count[slot] = seed_access_counts[i].item()
                n_transferred += 1

    return n_transferred


def _transfer_edges_by_content(
    state: CognitiveState,
    seed_edges: list[dict],
    match_threshold: float,
) -> int:
    """Transfer edges by matching endpoint beliefs by content.

    For each seed edge:
    1. Find the belief slot most similar to seed's src belief
    2. Find the belief slot most similar to seed's tgt belief
    3. If both match above threshold, create the edge
    """
    n_transferred = 0
    device = state.beliefs.device

    with torch.no_grad():
        active_mask = state.get_active_mask()
        if not active_mask.any():
            return 0

        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        active_beliefs = state.beliefs.data[active_indices]
        active_angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)

        for edge_data in seed_edges:
            src_belief = edge_data['src_belief'].to(device)
            tgt_belief = edge_data['tgt_belief'].to(device)

            src_angle = F.normalize(src_belief.unsqueeze(0), dim=-1, eps=EPSILON)
            tgt_angle = F.normalize(tgt_belief.unsqueeze(0), dim=-1, eps=EPSILON)

            # Match src
            src_sims = (src_angle @ active_angles.T).squeeze(0)
            src_best_sim, src_best_local = src_sims.max(dim=0)

            # Match tgt
            tgt_sims = (tgt_angle @ active_angles.T).squeeze(0)
            tgt_best_sim, tgt_best_local = tgt_sims.max(dim=0)

            if src_best_sim.item() > match_threshold and tgt_best_sim.item() > match_threshold:
                src_slot = active_indices[src_best_local].item()
                tgt_slot = active_indices[tgt_best_local].item()

                if src_slot != tgt_slot:
                    relation = edge_data['relation'].to(device)
                    weight = edge_data['weight']
                    eidx = state.allocate_edge(src_slot, tgt_slot, relation, weight)
                    if eidx >= 0:
                        state.edge_causal_obs[eidx] = edge_data.get('causal_obs', 0.0)
                        n_transferred += 1

    return n_transferred
