"""Pass 2 orchestrator: runs all cognitive updates after the forward pass.

Order:
1. Surprise computation
2. Belief updates (precision-weighted revision)
3. Causal edge learning (temporal surprise precedence → directed edges)
4. Hebbian edge strengthening (co-activation → undirected edges)
5. Goal progress tracking
6. Intrinsic goal generation (from surprise)
7. Stall detection + deadline enforcement
8. Meta update (β computation, periodic SPSA)
9. Consolidation (periodic)

This runs ONCE per sequence (training) or per response (inference).
The cognitive state is modified in-place.
"""

from ..core.state import CognitiveState
from ..interface.write_path import WriteCandidate
from .surprise import compute_surprise_batch
from .belief_update import allocate_new_beliefs
from .hebbian import hebbian_update, extract_co_activations
from .causal import causal_edge_learning
from .telos import (
    generate_intrinsic_goals, update_goal_progress,
    detect_stalls, enforce_deadlines,
)
from .consolidation import soft_consolidation, periodic_hard_cleanup
from .meta_learning import compute_beta, SPSAController, apply_sequence_boundary_decay


MAX_CANDIDATES = 1024  # safety valve: subsample if too many candidates


def run_pass2(
    state: CognitiveState,
    candidates: list[WriteCandidate],
    read_belief_indices: list[int],
    current_step: int,
    is_sequence_boundary: bool = True,
    consolidation_interval: int = 50,
    max_candidates: int = MAX_CANDIDATES,
    spsa_controller: SPSAController | None = None,
) -> dict:
    """Run the full pass 2 cognitive update loop.

    Args:
        state: cognitive state to update (modified in-place)
        candidates: write candidates from state interface layers
        read_belief_indices: indices of beliefs that were read during pass 1
        current_step: current training/inference step
        is_sequence_boundary: whether we're at a sequence boundary (apply decay)
        consolidation_interval: run consolidation every N steps
        max_candidates: subsample candidates if exceeding this count
        spsa_controller: multi-step SPSA controller (created by training loop, None to skip)

    Returns:
        dict with statistics from all sub-operations
    """
    stats = {
        'step': current_step,
    }

    # Subsample candidates if too many (safety valve)
    if len(candidates) > max_candidates:
        import random
        candidates = random.sample(candidates, max_candidates)

    # 1. Surprise computation
    surprise_results = compute_surprise_batch(candidates, state)
    stats['num_candidates'] = len(candidates)
    stats['num_surprise_results'] = len(surprise_results)
    stats['total_surprise'] = sum(sr.surprise for sr in surprise_results)

    # 2. Belief updates
    update_stats = allocate_new_beliefs(surprise_results, state)
    stats.update({f'belief_{k}': v for k, v in update_stats.items()})

    # Collect indices and surprises of updated beliefs (for goal progress + Hebbian)
    # For existing slots: direct from surprise results
    # For new allocations: batch cosine similarity search
    updated_indices = []
    surprise_values = []
    new_observations = []
    new_surprises = []

    for sr in surprise_results:
        if sr.slot >= 0:
            updated_indices.append(sr.slot)
            surprise_values.append(sr.surprise)
        elif sr.is_new:
            new_observations.append(sr.observation)
            new_surprises.append(sr.surprise)

    # Batch-find slots for newly allocated beliefs
    n_new = min(len(new_observations), update_stats['new_allocations'])
    if n_new > 0:
        import torch
        active_mask = state.get_active_mask()
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(active_indices) > 0:
            active_beliefs = state.beliefs.data[active_indices]  # [A, D]
            new_obs_t = torch.stack(new_observations[:n_new])    # [K, D]
            # Batch cosine similarity: [K, A]
            sims = torch.nn.functional.cosine_similarity(
                new_obs_t.unsqueeze(1),       # [K, 1, D]
                active_beliefs.unsqueeze(0),  # [1, A, D]
                dim=-1,
            )
            best_local = sims.argmax(dim=-1)  # [K]
            best_slots = active_indices[best_local]  # [K]
            for i in range(n_new):
                updated_indices.append(best_slots[i].item())
                surprise_values.append(new_surprises[i])

    # 3. Causal edge learning (temporal surprise precedence → directed edges)
    causal_stats = causal_edge_learning(state, updated_indices, surprise_values)
    stats['causal_edges_created'] = causal_stats['edges_created']
    stats['causal_edges_strengthened'] = causal_stats['edges_strengthened']

    # 4. Hebbian edge strengthening (co-activation → undirected edges)
    co_activations = extract_co_activations(state, read_belief_indices)
    hebbian_update(state, co_activations)
    stats['co_activation_pairs'] = len(co_activations)

    # 5. Goal progress
    update_goal_progress(state, updated_indices, surprise_values)
    stats['active_goals'] = state.num_active_goals()

    # 6. Meta update: β + multi-step SPSA
    beta = compute_beta(state, state.meta_params.fe_temperature.item())
    stats['beta'] = beta

    # SPSA removed — its tunable params are now nn.Parameters in MetaParams,
    # trained by backprop through L_fe. Nevergrad available as fallback if
    # future discrete parameters need gradient-free optimization.
    if spsa_controller is not None:
        did_update = spsa_controller.step(state, current_step, state.meta_params.fe_temperature.item())
        stats['spsa_step'] = did_update
        stats['spsa_phase'] = spsa_controller.phase
    else:
        stats['spsa_step'] = False
        stats['spsa_phase'] = 'disabled'

    # 7. Consolidation (periodic, not every step)
    if current_step % 10 == 0:
        merged = soft_consolidation(
            state,
            similarity_threshold=state.running_stats.merge_similarity_threshold,
        )
    else:
        merged = 0
    stats['soft_merges'] = merged

    consolidation_timer = state.meta.data[2].item()
    if consolidation_timer >= consolidation_interval:
        removed = periodic_hard_cleanup(
            state,
            low_precision_threshold=state.running_stats.hard_cleanup_precision_threshold,
        )
        stats['hard_cleanup_removed'] = removed
        state.meta.data[2] = 0.0  # reset timer
    else:
        stats['hard_cleanup_removed'] = 0

    # 8. Intrinsic goal generation (AFTER consolidation so goals reference valid beliefs)
    new_goals = generate_intrinsic_goals(state, current_step)
    stats['goals_generated'] = new_goals

    # 9. Stall detection + deadline enforcement
    detect_stalls(state, current_step)
    enforce_deadlines(state, current_step)

    # Sequence boundary: decay belief precision (every 10 steps, not every step)
    if is_sequence_boundary and current_step % 10 == 0:
        apply_sequence_boundary_decay(state)

    # Update running statistics
    mean_surprise = stats.get('total_surprise', 0) / max(stats.get('num_candidates', 1), 1)
    state.running_stats.update(state, {
        'mean_surprise': mean_surprise,
        'current_step': current_step,
    })

    # Final stats
    stats['active_beliefs'] = state.num_active_beliefs()
    stats['active_edges'] = state.num_active_edges()

    return stats
