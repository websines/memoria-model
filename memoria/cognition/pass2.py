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
8. Per-belief adaptive LR updates (RWKV-7)
9. Confidence propagation through source chains (MemOS)
10. Belief promotion through abstraction hierarchy (SDFT)
11. Periodic sleep cycle (SleepGate) + dream phase (NeuroDream)
12. Belief shift from message passing (confidence cascade)

This runs ONCE per step after optimizer.step() and detach_state().
The cognitive state is modified in-place via .data access (no gradients).
"""

import torch
import torch.nn as nn
from ..core.state import CognitiveState
from ..core.polar import EPSILON
from ..core.message_passing import FactorGraphMessagePassing, apply_belief_shift
from ..interface.write_path import WriteCandidate
from .surprise import compute_surprise_batch
from .belief_update import allocate_new_beliefs
from .hebbian import extract_co_activations
from .consolidation import soft_consolidation, periodic_hard_cleanup
from .meta_learning import compute_beta
from .sleep import SleepGate, run_sleep_cycle, run_dream_phase
from .provisional import evaluate_provisional_beliefs
from .cascade_revision import cascade_revision
from .autoresearch import run_autoresearch_step
from .planning import run_planning_step
from .structural_plasticity import run_structural_plasticity
from .two_factor_sleep import run_two_factor_sleep
from .self_verification import run_self_verification
from .precision_recalibration import run_precision_recalibration
from .interleaved_replay import run_interleaved_replay


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


def _make_tracker_callback(state: CognitiveState):
    """Create a callback that feeds provisional outcomes to the hypothesis tracker.

    The provisional evaluator emits (belief_idx, outcome_code, metadata) where
    outcome_code is PROMOTED (0) or EVICT_* (>0). Failed hypotheses carry the
    angle snapshot and metadata through to the tracker so future generations
    can condition on recent failures for the same goal.
    """
    from .provisional import PROMOTED
    tracker = state.hypothesis_tracker

    def callback(belief_idx: int, outcome_code: int, metadata: dict):
        promoted = outcome_code == PROMOTED
        # Capture belief angle *before* deallocation clears the slot so the
        # tracker can store it in the per-goal failed buffer.
        angle_snapshot = (
            state.beliefs.data[belief_idx].detach().clone()
            if not promoted
            else None
        )
        tracker.record_outcome(
            belief_idx,
            promoted,
            reason_code=outcome_code,
            fe_delta=metadata.get('fe_delta', 0.0),
            failed_angle=angle_snapshot,
        )

    return callback


def run_pass2(
    state: CognitiveState,
    candidates: list[WriteCandidate],
    read_belief_indices: list[int],
    current_step: int,
    is_sequence_boundary: bool = True,
    total_steps: int = 1,
    belief_advantage: float = 0.0,
    current_fe: float = 0.0,
    training_progress: float = 0.0,
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
        current_fe: current global free energy (for provisional evaluation)

    Returns:
        dict with statistics
    """
    stats = {'step': current_step}

    # ── 0. Controller actions (learned structural decisions) ──
    actions = state.controller.get_actions(state)
    dense_reward = state.controller.compute_dense_reward(
        state, belief_advantage, training_progress=training_progress,
    )
    state.controller.record_reward(dense_reward)
    stats['controller_actions'] = actions

    # ── 0b. Adaptive depth — decide which operations are needed this step ──
    # Rule-based using running statistics (not a learned probe — the training
    # signal for a probe here is too noisy to be useful). Each operation runs
    # only when the state indicates it's needed.
    n_active = state.num_active_beliefs()
    n_max = state.config.max_beliefs
    fill_ratio = n_active / max(n_max, 1)
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

    # Collect indices for edge creation + per-belief adaptive LR
    updated_indices = []
    surprise_values = []
    reconsolidated_indices = []
    for sr in surprise_results:
        if sr.slot >= 0:
            updated_indices.append(sr.slot)
            surprise_values.append(sr.surprise)
            if sr.should_reconsolidate and not sr.is_new:
                reconsolidated_indices.append(sr.slot)

    # ── 2b-i. A2: MESU — update precision variance for matched beliefs ──
    # Observations reduce uncertainty about a belief's precision.
    # High gain → large variance reduction. Windowed posterior prevents
    # variance from collapsing to zero.
    if updated_indices:
        with torch.no_grad():
            min_var = state.meta_params.mesu_min_variance.item()
            shrink_rate = state.meta_params.mesu_variance_shrink.item()
            window_size = int(state.meta_params.mesu_window_size.item())
            for sr in surprise_results:
                if sr.slot >= 0 and not sr.is_new:
                    idx = sr.slot
                    var = state.belief_precision_var[idx].item()
                    # Variance shrinks proportional to gain^2
                    new_var = var * (1.0 - sr.gain ** 2 * shrink_rate)
                    # Window: if too many reinforcements, floor is raised
                    count = state.belief_reinforcement_count[idx].item()
                    effective_floor = min_var * (1.0 + max(0, count - window_size) / window_size)
                    state.belief_precision_var[idx] = max(new_var, effective_floor)
                    state.belief_reinforcement_count[idx] += 1
        stats['mesu_variance_updates'] = sum(
            1 for sr in surprise_results if sr.slot >= 0 and not sr.is_new
        )

    # ── 2b-ii. A3: Causal cascade revision ──
    # When beliefs are reconsolidated (high surprise), propagate precision
    # decay to downstream beliefs in the causal graph.
    if reconsolidated_indices:
        cascade_stats = cascade_revision(state, reconsolidated_indices)
        stats['cascade_beliefs_decayed'] = cascade_stats['beliefs_decayed']
        stats['cascade_total_decay'] = cascade_stats['total_decay']
        stats['cascade_max_depth'] = cascade_stats['max_depth_reached']
    else:
        stats['cascade_beliefs_decayed'] = 0

    # ── 2b. Per-belief adaptive LR update (RWKV-7 + MESU variance boost) ──
    if updated_indices:
        idx_t = torch.tensor(updated_indices, device=state.beliefs.device)
        surp_t = torch.tensor(surprise_values, device=state.beliefs.device)
        state.update_belief_lr_scale(idx_t, surp_t)
        # A2: Boost LR for high-variance beliefs (uncertain → learn faster)
        with torch.no_grad():
            var_boost = (1.0 + state.belief_precision_var[idx_t]).sqrt()
            state.belief_lr_scale[idx_t] *= var_boost
        stats['beliefs_lr_updated'] = len(updated_indices)

    # ── 2c. Confidence propagation through source chains (MemOS) ──
    if updated_indices and hasattr(state, 'belief_sources'):
        with torch.no_grad():
            idx_t = torch.tensor(updated_indices, device=state.beliefs.device, dtype=torch.long)
            # Snapshot radii before pass2 belief changes for propagation
            old_radii = state.beliefs.data[idx_t].norm(dim=-1)
            state.propagate_confidence(idx_t, old_radii)
        stats['confidence_propagated'] = len(updated_indices)

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
                        state.goal_status_logits.data[slot].zero_()
                        state.goal_status_logits.data[slot, 1] = init_logit
                        state.goal_metadata.data[slot, 6] = float(current_step)
                        n_allocated += 1
                stats['goals_generated'] = n_allocated
            else:
                stats['goals_generated'] = 0
    else:
        stats['goals_generated'] = 0

    # ── 5b. Internal Autoresearch Loop: generate hypothesis beliefs from goals ──
    # After goals are generated/updated, synthesize candidate beliefs that might
    # reduce free energy in goal directions. Allocate as provisional (A1).
    # Only runs when there are active goals and room for hypotheses.
    if (need_goals and state.num_active_goals() > 0
            and hasattr(state, 'hypothesis_gen')
            and state.hypothesis_gen is not None):
        fill_ratio = state.num_active_beliefs() / max(state.config.max_beliefs, 1)
        # Don't generate hypotheses if state is near capacity
        if fill_ratio < 0.85:
            ar_stats = run_autoresearch_step(
                state, state.hypothesis_gen, state.hypothesis_tracker,
                current_step, current_fe,
            )
            stats['hypotheses_generated'] = ar_stats['hypotheses_generated']
            stats['hypotheses_gated_out'] = ar_stats['hypotheses_gated_out']
        else:
            stats['hypotheses_generated'] = 0
            stats['hypotheses_gated_out'] = 0
    else:
        stats['hypotheses_generated'] = 0
        stats['hypotheses_gated_out'] = 0

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

    # ── 6b. A1: Evaluate provisional beliefs (internal autoresearch loop) ──
    # Every step, check if any provisional beliefs have passed their evaluation
    # window. Promote winners, evict losers. Feed outcomes to tracker.
    prov_stats = evaluate_provisional_beliefs(
        state, current_step, current_fe,
        outcome_callback=_make_tracker_callback(state) if hasattr(state, 'hypothesis_tracker') else None,
    )
    stats['provisional_promoted'] = prov_stats['promoted']
    stats['provisional_evicted'] = prov_stats['evicted']
    stats['provisional_pending'] = prov_stats['still_provisional']

    # ── 7. Belief promotion (SDFT-inspired abstraction hierarchy) ──
    # Check all updated beliefs for promotion eligibility. Cheap: just compares
    # radius + access count against stats-derived thresholds.
    if hasattr(state, 'belief_level'):
        n_promoted = 0
        for idx in updated_indices:
            old_level = state.belief_level[idx].item()
            state.promote_belief(idx)
            if state.belief_level[idx].item() > old_level:
                n_promoted += 1
        stats['beliefs_promoted'] = n_promoted

    # ── 8. Periodic sleep cycle (SleepGate, arXiv:2603.14517) ──
    # Run when the consolidation timer fires AND the state has a sleep gate.
    # The sleep cycle scores each belief for strengthen/maintain/forget.
    sleep_interval = state.running_stats.hard_consolidation_interval
    if (hasattr(state, 'sleep_gate') and state.sleep_gate is not None
            and consolidation_timer >= sleep_interval):
        sleep_stats = run_sleep_cycle(state, state.sleep_gate, current_step)
        stats['sleep_strengthened'] = sleep_stats['strengthened']
        stats['sleep_forgotten'] = sleep_stats['forgotten']
        stats['sleep_deallocated'] = sleep_stats['deallocated']

    # ── 9. Periodic dream phase (NeuroDream, SSRN 5377250) ──
    # Run at sequence boundaries when there are enough beliefs and edges to
    # make internal propagation meaningful. The dream phase converges the
    # belief graph toward internal consistency without external input.
    if (is_sequence_boundary and hasattr(state, 'message_passing')
            and state.message_passing is not None
            and state.num_active_beliefs() > 10
            and state.num_active_edges() > 5):
        dream_stats = run_dream_phase(state, state.message_passing)
        stats['dream_iterations'] = dream_stats['iterations']
        stats['dream_converged'] = dream_stats['converged']

        # ── 9b. Belief shift from message passing (confidence cascade) ──
        # After dream propagation, shift beliefs toward their messages.
        if dream_stats['iterations'] > 0:
            mp_result = state.message_passing(state)
            shifted = apply_belief_shift(state, mp_result['messages'], mp_result['precisions'])
            stats['beliefs_shifted'] = len(shifted)

    # ── 9c. E1: Two-factor sleep consolidation ──
    # Runs during sleep cycle: homeostatic precision normalization +
    # conflict scanning + replay candidate identification.
    if (consolidation_timer >= sleep_interval
            and state.num_active_beliefs() > 0):
        tfs_stats = run_two_factor_sleep(state, current_step)
        stats['two_factor_sleep'] = tfs_stats

    # ── 9d. E2: Self-verification pass ──
    # Runs during sleep: causal consistency check, weakest-link precision
    # reduction, conflict-aware supersession.
    if (consolidation_timer >= sleep_interval
            and state.num_active_beliefs() > 2
            and state.num_active_edges() > 0):
        sv_stats = run_self_verification(state)
        stats['self_verification'] = sv_stats

    # ── 9e. E3: Empirical precision recalibration ──
    # Runs during sleep: decay overconfident beliefs toward empirical precision.
    if (consolidation_timer >= sleep_interval
            and state.num_active_beliefs() > 0):
        recal_stats = run_precision_recalibration(state)
        stats['precision_recalibration'] = recal_stats

    # ── 9f. E4: Interleaved replay ──
    # Runs during sleep: cross-temporal contradiction detection between
    # recent high-surprise and old high-precision beliefs.
    if (consolidation_timer >= sleep_interval
            and state.num_active_beliefs() >= 4):
        replay_stats = run_interleaved_replay(state)
        stats['interleaved_replay'] = replay_stats

    # ── 10. Planning step (B1-B4) ──
    # Run at sequence boundaries when there are active goals and beliefs.
    # Planning computes preference/epistemic priors, causal rollouts,
    # and optionally MCTS for multi-goal decisions.
    if (is_sequence_boundary and state.num_active_goals() > 0
            and state.num_active_beliefs() > 0):
        planning_stats = run_planning_step(state, current_step)
        stats['planning'] = planning_stats

    # ── 11. C1: SRWM update (self-referential fast-weight matrix) ──
    # Update the fast-weight matrix with current state features.
    # This adapts meta-parameter modulations based on cognitive state.
    if hasattr(state, 'srwm') and state.srwm is not None:
        features = state.srwm.extract_state_features(state)
        state.srwm.update(
            features,
            lr=state.meta_params.srwm_lr,
            decay=state.meta_params.srwm_decay,
        )
        stats['srwm_updated'] = True

    # ── 12. C3: Structural plasticity (split/prune/grow) ──
    # Run periodically when there's enough activation data.
    # Splits polysemantic beliefs, prunes dead ones.
    if (hasattr(state, 'structural_plasticity')
            and state.structural_plasticity is not None
            and is_sequence_boundary
            and state.structural_plasticity._total_steps.item() > 10):
        sp_stats = run_structural_plasticity(state, state.structural_plasticity)
        stats['structural_plasticity'] = sp_stats

    stats['active_beliefs'] = state.num_active_beliefs()
    stats['active_edges'] = state.num_active_edges()

    # Level distribution for logging
    if hasattr(state, 'belief_level'):
        active = state.get_active_mask()
        if active.any():
            levels = state.belief_level[active]
            stats['level_distribution'] = {
                f'L{i}': (levels == i).sum().item() for i in range(4)
            }

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
