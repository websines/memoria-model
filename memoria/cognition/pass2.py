"""Pass 2: structural cleanup after gradient-based state updates.

The gradient (from L_token + L_fe_bethe + L_fe_proxy + L_utility + L_surprise)
handles ALL continuous updates to beliefs, edges, and relations via optimizer.step().

Pass 2 handles only discrete structural operations that gradients cannot:
1. Allocate new belief slots for unmatched write candidates
2. Create new edges via learned EdgeProposer (replaces Hebbian/causal heuristics)
3. Structural cleanup: zero-norm beliefs → free slot, zero-weight edges → inactive
4. Periodic consolidation (merge near-identical beliefs, prune dead ones)
5. Differentiable goal generation via TelosModule
6. Beta computation and running statistics update
7. CognitiveController makes structural decisions (merge/prune/connect rates)

This runs ONCE per step after optimizer.step() and detach_state().
The cognitive state is modified in-place via .data access (no gradients).
"""

import torch
import torch.nn as nn
from ..core.state import CognitiveState
from ..core.polar import EPSILON
from ..interface.write_path import WriteCandidate
from .surprise import compute_surprise_batch
from .belief_update import allocate_new_beliefs
from .hebbian import extract_co_activations
from .consolidation import soft_consolidation, periodic_hard_cleanup
from .meta_learning import compute_beta


class Pass2Probe(nn.Module):
    """Lightweight probe that decides which pass 2 operations are needed this step.

    Most steps don't need full consolidation or heavy edge creation.
    This probe takes state features → probabilities for each operation,
    allowing pass 2 to skip expensive work when not needed.

    Reference: Mamba's HaltingHead — learn WHEN to do expensive compute.

    Input features (6-dim):
        [0] fill_ratio: fraction of belief slots occupied
        [1] mean_surprise: average surprise from current candidates
        [2] candidate_load: fraction of max_candidates present
        [3] edge_fill: fraction of edge slots occupied
        [4] beta: current exploration/exploitation parameter
        [5] consolidation_timer: normalized time since last hard consolidation

    Output (4-dim, sigmoid):
        [0] P(need_allocation): should we allocate new beliefs?
        [1] P(need_edges): should we create edges?
        [2] P(need_consolidation): should we run soft consolidation?
        [3] P(need_goals): should we generate goals?
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )
        # Initialize with positive bias so everything runs by default
        # (probe learns to SKIP, not to ENABLE — safer default)
        nn.init.constant_(self.net[-2].bias, 2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute operation probabilities from state features.

        Args:
            features: [6] state feature vector

        Returns:
            [4] probabilities: allocation, edges, consolidation, goals
        """
        return self.net(features)


def run_pass2(
    state: CognitiveState,
    candidates: list[WriteCandidate],
    read_belief_indices: list[int],
    current_step: int,
    is_sequence_boundary: bool = True,
    total_steps: int = 1,
    belief_advantage: float = 0.0,
    spsa_controller=None,  # deprecated
) -> dict:
    """Structural cleanup pass after gradient-based updates.

    With adaptive depth: rule-based skip logic using running statistics
    decides which operations are needed this step. Skips expensive
    consolidation and edge creation when the state doesn't need them.

    Args:
        state: cognitive state to update (modified in-place via .data)
        candidates: write candidates from state interface layers
        read_belief_indices: indices of beliefs that were read during pass 1
        current_step: current training/inference step
        is_sequence_boundary: whether we're at a sequence boundary
        total_steps: total training steps (for temperature annealing)
        belief_advantage: current belief advantage for controller reward
        spsa_controller: deprecated, ignored

    Returns:
        dict with statistics
    """
    stats = {'step': current_step}

    # ── 0. Controller actions (learned structural decisions) ──
    actions = state.controller.get_actions(state)
    state.controller.record_reward(belief_advantage)
    stats['controller_actions'] = actions

    # ── 0b. Adaptive depth — decide which operations are needed this step ──
    # Rule-based using running statistics (not a learned probe — the training
    # signal for a probe here is too noisy to be useful). Each operation runs
    # only when the state indicates it's needed.
    n_active = state.num_active_beliefs()
    n_max = state.config.max_beliefs
    fill_ratio = n_active / max(n_max, 1)
    n_new_candidates = sum(1 for c in candidates if c.matched_slot == -1)
    edge_fill = state.num_active_edges() / max(state.config.max_edges, 1)
    consol_timer = state.meta.data[2].item()

    # Allocation: always run if there are candidates (cheap, core function)
    need_allocation = len(candidates) > 0
    # Edges: skip if edge slots are >90% full or no co-activation data
    need_edges = edge_fill < 0.9 and len(read_belief_indices) > 0
    # Consolidation: only run on the configured interval OR when near capacity
    soft_interval = state.running_stats.soft_consolidation_interval
    need_consolidation = (
        (current_step % max(soft_interval, 1) == 0) or fill_ratio > 0.9
    )
    # Goals: only when there are active beliefs and not too many active goals
    need_goals = n_active > 0 and state.num_active_goals() < state.config.max_goals * 0.8

    stats['pass2_skip'] = {
        'allocation': not need_allocation,
        'edges': not need_edges,
        'consolidation': not need_consolidation,
        'goals': not need_goals,
    }

    # ── 1. Structural cleanup: zero-norm beliefs and edges (always runs) ──
    with torch.no_grad():
        radii = state.beliefs.data.norm(dim=-1)
        dead_beliefs = (radii > 0) & (radii < EPSILON) & ~state.immutable_beliefs
        if dead_beliefs.any():
            dead_idx = dead_beliefs.nonzero(as_tuple=False).squeeze(-1)
            for idx in dead_idx.tolist():
                state.deallocate_belief(idx)
        stats['beliefs_cleaned'] = dead_beliefs.sum().item()

        dead_edges = state.edge_active & (state.edge_weights.data.abs() < EPSILON)
        if dead_edges.any():
            dead_idx = dead_edges.nonzero(as_tuple=False).squeeze(-1)
            for idx in dead_idx.tolist():
                state.deallocate_edge(idx)
        stats['edges_cleaned'] = dead_edges.sum().item()

    # ── 2. New belief allocation (discrete: pick a slot) ──
    max_candidates = state.running_stats.max_candidates
    if len(candidates) > max_candidates:
        import random
        candidates = random.sample(candidates, max_candidates)

    allocate_rate = actions.get('allocate_rate', 1.0)
    n_candidates_for_subsample = max(3, max_candidates // 10)
    if allocate_rate < 0.9 and len(candidates) > n_candidates_for_subsample:
        import random
        n_keep = max(1, int(len(candidates) * allocate_rate))
        candidates = random.sample(candidates, n_keep)

    if need_allocation:
        surprise_results = compute_surprise_batch(candidates, state)
        stats['num_candidates'] = len(candidates)
        stats['total_surprise'] = sum(sr.surprise for sr in surprise_results)

        update_stats = allocate_new_beliefs(surprise_results, state)
        stats.update({f'belief_{k}': v for k, v in update_stats.items()})
    else:
        surprise_results = compute_surprise_batch(candidates, state)
        stats['num_candidates'] = len(candidates)
        stats['total_surprise'] = sum(sr.surprise for sr in surprise_results)
        stats['belief_allocated'] = 0
        stats['belief_updated'] = 0
        stats['belief_skipped'] = len(candidates)

    # Collect indices for edge creation
    updated_indices = []
    surprise_values = []
    for sr in surprise_results:
        if sr.slot >= 0:
            updated_indices.append(sr.slot)
            surprise_values.append(sr.surprise)

    # ── 3. Learned edge proposal ──
    if need_edges:
        co_activations = extract_co_activations(state, read_belief_indices)
        causal_pairs = _extract_causal_candidates(state, updated_indices, surprise_values)
        all_candidate_pairs = co_activations + causal_pairs

        connect_rate = actions.get('connect_rate', 1.0)
        if connect_rate < 1.0 and len(all_candidate_pairs) > 0:
            import random
            n_keep = max(1, int(len(all_candidate_pairs) * connect_rate))
            all_candidate_pairs = random.sample(all_candidate_pairs, n_keep)

        with torch.no_grad():
            accepted_edges, _ = state.edge_proposal.propose_edges(
                all_candidate_pairs, state.beliefs.data,
            )
            n_created = 0
            for src, tgt, weight, theta in accepted_edges:
                relation = torch.zeros(state.config.relation_dim, device=state.beliefs.device)
                eidx = state.allocate_edge(src, tgt, relation, weight=weight)
                if eidx >= 0:
                    state.edge_direction.data[eidx] = theta
                    n_created += 1
            stats['edges_proposed'] = len(all_candidate_pairs)
            stats['edges_created'] = n_created
            state.edge_proposal.update_ada_threshold(state.num_active_edges(), len(all_candidate_pairs))
        stats['co_activation_pairs'] = len(co_activations)
    else:
        stats['edges_proposed'] = 0
        stats['edges_created'] = 0
        stats['co_activation_pairs'] = 0

    # Store surprise for next-step causal detection (always runs — cheap)
    with torch.no_grad():
        state.belief_prev_surprise.zero_()
        for idx, s in zip(updated_indices, surprise_values):
            state.belief_prev_surprise[idx] = s

    # ── 4. Periodic consolidation (structural: merge similar beliefs) ──
    merge_threshold = actions.get('merge_threshold', state.running_stats.merge_similarity_threshold)
    prune_threshold = actions.get('prune_threshold', state.running_stats.hard_cleanup_precision_threshold)

    if need_consolidation:
        merged = soft_consolidation(state, similarity_threshold=merge_threshold)
    else:
        merged = 0
    stats['soft_merges'] = merged

    consolidation_interval = state.running_stats.hard_consolidation_interval
    consolidation_timer = state.meta.data[2].item()
    if need_consolidation and consolidation_timer >= consolidation_interval:
        removed = periodic_hard_cleanup(state, low_precision_threshold=prune_threshold)
        stats['hard_cleanup_removed'] = removed
        state.meta.data[2] = 0.0
    else:
        stats['hard_cleanup_removed'] = 0

    # ── 5. Differentiable goal generation (via TelosModule) ──
    cooldown = state.running_stats.goal_cooldown_steps
    goal_rate = int(actions.get('goal_rate', 3.0))
    if need_goals:
        with torch.no_grad():
            active_mask = state.get_active_mask()
            if active_mask.any() and current_step % max(cooldown, 1) == 0 and goal_rate > 0:
                beta = state.meta.data[0].item()
                goal_embeds, goal_surprise = state.telos.generate_goals(
                    state.beliefs.data, active_mask, beta, max_new=goal_rate,
                )
                n_allocated = 0
                for i in range(goal_embeds.shape[0]):
                    best_status = state.goal_status_logits.argmax(dim=-1)
                    empty_mask = best_status == 0
                    if empty_mask.any():
                        slot = empty_mask.nonzero(as_tuple=False)[0].item()
                        state.goal_embeddings.data[slot] = goal_embeds[i]
                        init_logit = 1.0 / max(state.telos.gumbel_tau.item(), 0.1)
                        state.goal_status_logits[slot] = torch.zeros(6, device=state.beliefs.device)
                        state.goal_status_logits[slot, 1] = init_logit
                        state.goal_metadata.data[slot, 6] = float(current_step)
                        n_allocated += 1
                stats['goals_generated'] = n_allocated
            else:
                stats['goals_generated'] = 0
    else:
        stats['goals_generated'] = 0

    # ── 6. Beta computation + running stats ──
    beta = compute_beta(state, state.meta_params.fe_temperature.item())
    stats['beta'] = beta
    stats['active_goals'] = state.num_active_goals()

    if is_sequence_boundary:
        state.meta.data[2] += 1.0

    mean_surprise = stats.get('total_surprise', 0) / max(stats.get('num_candidates', 1), 1)
    state.running_stats.update(state, {
        'mean_surprise': mean_surprise,
        'current_step': current_step,
    })

    state.telos.anneal_temperature(current_step, total_steps)

    stats['active_beliefs'] = state.num_active_beliefs()
    stats['active_edges'] = state.num_active_edges()

    return stats


def _extract_causal_candidates(
    state: CognitiveState,
    updated_indices: list[int],
    surprise_values: list[float],
    min_signal: float | None = None,
) -> list[tuple[int, int]]:
    """Extract candidate pairs from temporal surprise precedence.

    Beliefs surprised at step t-1 × beliefs surprised at step t = causal candidates.
    The EdgeProposer decides which become actual edges.
    """
    if min_signal is None:
        min_signal = state.meta_params.causal_min_signal.item()

    pairs = []
    prev_surprise = state.belief_prev_surprise
    prev_updated = (prev_surprise > EPSILON).nonzero(as_tuple=False).squeeze(-1)

    if len(prev_updated) == 0 or len(updated_indices) == 0:
        return pairs

    curr_map = {idx: s for idx, s in zip(updated_indices, surprise_values) if s > EPSILON}
    if not curr_map:
        return pairs

    for prev_idx in prev_updated.tolist():
        prev_s = prev_surprise[prev_idx].item()
        for curr_idx, curr_s in curr_map.items():
            if prev_idx == curr_idx:
                continue
            signal = (prev_s * curr_s) ** 0.5
            if signal >= min_signal:
                pairs.append((prev_idx, curr_idx))

    # Cap to prevent explosion
    if len(pairs) > 256:
        import random
        pairs = random.sample(pairs, 256)

    return pairs
