"""Pass 2: structural cleanup after gradient-based state updates.

The gradient (from L_token + L_fe_bethe + L_fe_proxy + L_utility + L_surprise)
handles ALL continuous updates to beliefs, edges, and relations via optimizer.step().

Pass 2 handles only discrete structural operations that gradients cannot:
1. Allocate new belief slots for unmatched write candidates
2. Create new edges for co-activated belief pairs (topology, not weights)
3. Structural cleanup: zero-norm beliefs → free slot, zero-weight edges → inactive
4. Periodic consolidation (merge near-identical beliefs, prune dead ones)
5. Differentiable goal generation via TelosModule (runs with gradients when called from forward)
6. Beta computation and running statistics update

This runs ONCE per step after optimizer.step() and detach_state().
The cognitive state is modified in-place via .data access (no gradients).
"""

import torch
from ..core.state import CognitiveState
from ..core.polar import EPSILON
from ..interface.write_path import WriteCandidate
from .surprise import compute_surprise_batch
from .belief_update import allocate_new_beliefs
from .hebbian import hebbian_update, extract_co_activations
from .causal import causal_edge_learning
from .consolidation import soft_consolidation, periodic_hard_cleanup
from .meta_learning import compute_beta


def run_pass2(
    state: CognitiveState,
    candidates: list[WriteCandidate],
    read_belief_indices: list[int],
    current_step: int,
    is_sequence_boundary: bool = True,
    total_steps: int = 1,  # must be provided by caller (training loop knows max_steps)
    spsa_controller=None,  # deprecated, kept for call-site compat
) -> dict:
    """Structural cleanup pass after gradient-based updates.

    All continuous updates (belief content, precision, edge weights, relations)
    are handled by optimizer.step() through L_fe_bethe. This function handles
    only discrete structural operations.

    Args:
        state: cognitive state to update (modified in-place via .data)
        candidates: write candidates from state interface layers
        read_belief_indices: indices of beliefs that were read during pass 1
        current_step: current training/inference step
        is_sequence_boundary: whether we're at a sequence boundary
        spsa_controller: deprecated, ignored

    Returns:
        dict with statistics
    """
    stats = {'step': current_step}

    # ── 1. Structural cleanup: zero-norm beliefs and edges ──
    with torch.no_grad():
        # Beliefs driven to zero by weight decay → free the slot
        radii = state.beliefs.data.norm(dim=-1)
        dead_beliefs = (radii > 0) & (radii < EPSILON) & ~state.immutable_beliefs
        if dead_beliefs.any():
            dead_idx = dead_beliefs.nonzero(as_tuple=False).squeeze(-1)
            for idx in dead_idx.tolist():
                state.deallocate_belief(idx)
        stats['beliefs_cleaned'] = dead_beliefs.sum().item()

        # Edges driven to zero by weight decay → deactivate
        dead_edges = state.edge_active & (state.edge_weights.data.abs() < EPSILON)
        if dead_edges.any():
            dead_idx = dead_edges.nonzero(as_tuple=False).squeeze(-1)
            for idx in dead_idx.tolist():
                state.deallocate_edge(idx)
        stats['edges_cleaned'] = dead_edges.sum().item()

    # ── 2. New belief allocation (discrete: pick a slot) ──
    max_candidates = 1024
    if len(candidates) > max_candidates:
        import random
        candidates = random.sample(candidates, max_candidates)

    surprise_results = compute_surprise_batch(candidates, state)
    stats['num_candidates'] = len(candidates)
    stats['total_surprise'] = sum(sr.surprise for sr in surprise_results)

    update_stats = allocate_new_beliefs(surprise_results, state)
    stats.update({f'belief_{k}': v for k, v in update_stats.items()})

    # Collect indices for edge creation
    updated_indices = []
    surprise_values = []
    for sr in surprise_results:
        if sr.slot >= 0:
            updated_indices.append(sr.slot)
            surprise_values.append(sr.surprise)

    # ── 3. Edge topology creation (discrete: create new edges, no weight updates) ──
    causal_stats = causal_edge_learning(state, updated_indices, surprise_values)
    stats['causal_edges_created'] = causal_stats['edges_created']

    co_activations = extract_co_activations(state, read_belief_indices)
    hebbian_update(state, co_activations)
    stats['co_activation_pairs'] = len(co_activations)

    # ── 4. Periodic consolidation (structural: merge similar beliefs) ──
    if current_step % state.running_stats.soft_consolidation_interval == 0:
        merged = soft_consolidation(
            state,
            similarity_threshold=state.running_stats.merge_similarity_threshold,
        )
    else:
        merged = 0
    stats['soft_merges'] = merged

    consolidation_interval = state.running_stats.hard_consolidation_interval
    consolidation_timer = state.meta.data[2].item()
    if consolidation_timer >= consolidation_interval:
        removed = periodic_hard_cleanup(
            state,
            low_precision_threshold=state.running_stats.hard_cleanup_precision_threshold,
        )
        stats['hard_cleanup_removed'] = removed
        state.meta.data[2] = 0.0
    else:
        stats['hard_cleanup_removed'] = 0

    # ── 5. Differentiable goal generation (via TelosModule) ──
    # Gated by cooldown from running_stats to prevent slot flooding.
    cooldown = state.running_stats.goal_cooldown_steps
    with torch.no_grad():
        active_mask = state.get_active_mask()
        if active_mask.any() and current_step % max(cooldown, 1) == 0:
            beta = state.meta.data[0].item()
            goal_embeds, goal_surprise = state.telos.generate_goals(
                state.beliefs.data, active_mask, beta, max_new=3,
            )
            # Allocate goal slots
            n_allocated = 0
            for i in range(goal_embeds.shape[0]):
                # Find empty goal slot (status_logits argmax == 0 means empty)
                status_probs = torch.softmax(state.goal_status_logits, dim=-1)
                empty_mask = status_probs[:, 0] > 0.5  # mostly empty
                if empty_mask.any():
                    slot = empty_mask.nonzero(as_tuple=False)[0].item()
                    state.goal_embeddings.data[slot] = goal_embeds[i]
                    # Set status to proposed (index 1)
                    state.goal_status_logits[slot] = torch.zeros(6, device=state.beliefs.device)
                    state.goal_status_logits[slot, 1] = 5.0  # strong prior on proposed
                    state.goal_metadata.data[slot, 6] = float(current_step)  # created_step
                    n_allocated += 1
            stats['goals_generated'] = n_allocated
        else:
            stats['goals_generated'] = 0

    # ── 6. Beta computation + running stats ──
    beta = compute_beta(state, state.meta_params.fe_temperature.item())
    stats['beta'] = beta
    stats['active_goals'] = state.num_active_goals()

    # Increment consolidation timer
    if is_sequence_boundary:
        state.meta.data[2] += 1.0

    # Update running statistics
    mean_surprise = stats.get('total_surprise', 0) / max(stats.get('num_candidates', 1), 1)
    state.running_stats.update(state, {
        'mean_surprise': mean_surprise,
        'current_step': current_step,
    })

    # Anneal Telos Gumbel-Softmax temperature
    state.telos.anneal_temperature(current_step, total_steps)

    stats['active_beliefs'] = state.num_active_beliefs()
    stats['active_edges'] = state.num_active_edges()

    return stats
