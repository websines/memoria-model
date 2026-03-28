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
from .belief_update import apply_belief_updates
from .hebbian import hebbian_update, extract_co_activations
from .causal import causal_edge_learning
from .telos import (
    generate_intrinsic_goals, update_goal_progress,
    detect_stalls, enforce_deadlines,
)
from .consolidation import soft_consolidation, periodic_hard_cleanup
from .meta_learning import compute_beta, spsa_step, apply_sequence_boundary_decay


MAX_CANDIDATES = 1024  # safety valve: subsample if too many candidates


def run_pass2(
    state: CognitiveState,
    candidates: list[WriteCandidate],
    read_belief_indices: list[int],
    current_step: int,
    is_sequence_boundary: bool = True,
    consolidation_interval: int = 50,
    spsa_interval: int = 100,
    temperature: float = 5.0,
    max_candidates: int = MAX_CANDIDATES,
) -> dict:
    """Run the full pass 2 cognitive update loop.

    Args:
        state: cognitive state to update (modified in-place)
        candidates: write candidates from state interface layers
        read_belief_indices: indices of beliefs that were read during pass 1
        current_step: current training/inference step
        is_sequence_boundary: whether we're at a sequence boundary (apply decay)
        consolidation_interval: run consolidation every N steps
        spsa_interval: run SPSA every N steps
        temperature: for free energy computation
        max_candidates: subsample candidates if exceeding this count

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
    update_stats = apply_belief_updates(surprise_results, state)
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

    # 6. Intrinsic goal generation
    new_goals = generate_intrinsic_goals(state, current_step)
    stats['goals_generated'] = new_goals

    # 7. Stall detection + deadline enforcement
    detect_stalls(state, current_step)
    enforce_deadlines(state, current_step)

    # 8. Meta update: β + periodic SPSA
    beta = compute_beta(state, temperature)
    stats['beta'] = beta

    if current_step > 0 and current_step % spsa_interval == 0:
        spsa_step(state, temperature)
        stats['spsa_step'] = True
    else:
        stats['spsa_step'] = False

    # 9. Consolidation
    merged = soft_consolidation(state)
    stats['soft_merges'] = merged

    consolidation_timer = state.meta.data[2].item()
    if consolidation_timer >= consolidation_interval:
        removed = periodic_hard_cleanup(state)
        stats['hard_cleanup_removed'] = removed
        state.meta.data[2] = 0.0  # reset timer
    else:
        stats['hard_cleanup_removed'] = 0

    # Sequence boundary: decay belief precision
    if is_sequence_boundary:
        apply_sequence_boundary_decay(state)

    # Final stats
    stats['active_beliefs'] = state.num_active_beliefs()
    stats['active_edges'] = state.num_active_edges()

    return stats
