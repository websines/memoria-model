"""Telos: goal lifecycle, intrinsic generation, decomposition, progress tracking.

The model's executive function. Goals:
- Emerge from surprise hotspots (intrinsic generation)
- Shape attention (goal-directed retrieval in read path)
- Persist across sessions
- Have full lifecycle: proposed → active → stalled → completed/failed

Ported from: prototype-research/src/api/telos*.rs
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, EPSILON


# Goal metadata indices
PRIORITY = 0
URGENCY = 1
PROGRESS = 2
STATUS = 3
DEPTH = 4
SURPRISE_ACCUM = 5
CREATED_STEP = 6
DEADLINE = 7

# Status encoding
STATUS_EMPTY = 0.0
STATUS_PROPOSED = 0.2
STATUS_ACTIVE = 0.4
STATUS_STALLED = 0.6
STATUS_COMPLETED = 0.8
STATUS_FAILED = 1.0


def generate_intrinsic_goals(
    state: CognitiveState,
    current_step: int,
    cooldown_steps: int | None = None,
) -> int:
    """Generate goals from surprise hotspots.

    When accumulated surprise exceeds β-adjusted threshold, create new goals
    from the highest-surprise belief regions.

    Ported from: prototype-research/src/dynamics/intrinsic.rs

    Args:
        state: cognitive state
        current_step: current training/inference step
        cooldown_steps: minimum steps between goal generations
            (defaults to state.running_stats.goal_cooldown_steps)

    Returns:
        Number of goals generated
    """
    if cooldown_steps is None:
        cooldown_steps = state.running_stats.goal_cooldown_steps

    beta = state.beta
    accumulated = state.accumulated_surprise

    # Adaptive threshold: higher β (uncertainty) → lower threshold (more goals)
    threshold = state.running_stats.goal_threshold_scale / (1.0 + beta)

    if accumulated < threshold:
        return 0

    # Check cooldown: don't generate if a recent goal was just created
    _, _, goal_meta = state.get_active_goals()
    if len(goal_meta) > 0:
        latest_created = goal_meta[:, CREATED_STEP].max().item()
        if current_step - latest_created < cooldown_steps:
            return 0

    # Find high-surprise beliefs (candidates for goal generation)
    active_mask = state.get_active_mask()
    if not active_mask.any():
        return 0

    active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_beliefs = state.beliefs.data[active_indices]
    active_radii = active_beliefs.norm(dim=-1)

    # Low-precision beliefs in active regions = surprise hotspots
    # (they have been updated frequently but remain uncertain)
    # Use inverse radius as "surprise proxy" for goal generation
    surprise_proxy = 1.0 / (active_radii + EPSILON)

    # How many goals to generate (scaled by β)
    max_new = max(1, int(beta * 3))

    # Top-k most surprising beliefs
    k = min(max_new, len(active_indices))
    _, topk_local = surprise_proxy.topk(k)
    topk_indices = active_indices[topk_local]

    generated = 0
    with torch.no_grad():
        for local_idx in range(k):
            belief_idx = topk_indices[local_idx].item()
            belief_vec = state.beliefs.data[belief_idx]

            # Goal embedding = the belief direction (investigate THIS)
            goal_embed = belief_vec.clone()

            # Dedup: skip if too similar to existing active goal
            if _is_duplicate_goal(state, goal_embed):
                continue

            # Allocate goal slot
            slot = _allocate_goal_slot(state)
            if slot < 0:
                break  # no slots available

            # Set goal
            state.goal_embeddings.data[slot] = goal_embed

            # Metadata
            priority = surprise_proxy[topk_local[local_idx]].item()
            priority = min(priority / (surprise_proxy.max().item() + EPSILON), 1.0)  # normalize

            state.goal_metadata.data[slot, PRIORITY] = priority
            state.goal_metadata.data[slot, URGENCY] = 0.0
            state.goal_metadata.data[slot, PROGRESS] = 0.0
            state.goal_metadata.data[slot, STATUS] = STATUS_ACTIVE
            state.goal_metadata.data[slot, DEPTH] = float(goal_embed.norm().item())  # depth from goal complexity
            state.goal_metadata.data[slot, SURPRISE_ACCUM] = 0.0
            state.goal_metadata.data[slot, CREATED_STEP] = float(current_step)
            state.goal_metadata.data[slot, DEADLINE] = 0.0  # no deadline

            generated += 1

        # Only reset accumulated surprise if we actually generated goals
        if generated > 0:
            state.meta.data[1] = 0.0

    return generated


def update_goal_progress(
    state: CognitiveState,
    updated_belief_indices: list[int],
    surprise_values: list[float],
):
    """Update progress on active goals based on belief updates.

    When beliefs relevant to a goal are updated, the goal's progress increases
    proportional to the relevance × surprise of the update.

    Args:
        state: cognitive state
        updated_belief_indices: indices of beliefs that were updated in this pass
        surprise_values: surprise values for each updated belief
    """
    goal_indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(goal_indices) == 0 or len(updated_belief_indices) == 0:
        return

    with torch.no_grad():
        for gi, gidx in enumerate(goal_indices):
            gidx = gidx.item()
            status = state.goal_metadata.data[gidx, STATUS].item()

            # Only update active goals
            if abs(status - STATUS_ACTIVE) > 0.05:
                continue

            goal_embed = state.goal_embeddings.data[gidx]
            goal_angle = F.normalize(goal_embed, dim=0, eps=EPSILON)

            for bi, belief_idx in enumerate(updated_belief_indices):
                belief = state.beliefs.data[belief_idx]
                belief_angle = F.normalize(belief, dim=0, eps=EPSILON)

                relevance = angular_similarity(
                    goal_angle.unsqueeze(0), belief_angle.unsqueeze(0)
                ).item()

                if relevance > state.meta_params.goal_relevance_threshold.item():  # only count meaningfully relevant updates
                    surprise = surprise_values[bi] if bi < len(surprise_values) else 0.0
                    progress_delta = relevance * min(surprise, 1.0) * state.meta_params.goal_progress_rate.item()
                    state.goal_metadata.data[gidx, PROGRESS] = min(
                        state.goal_metadata.data[gidx, PROGRESS].item() + progress_delta,
                        1.0,
                    )

            # Auto-complete
            if state.goal_metadata.data[gidx, PROGRESS].item() >= 1.0:
                state.goal_metadata.data[gidx, STATUS] = STATUS_COMPLETED


def detect_stalls(state: CognitiveState, current_step: int):
    """Mark active goals as stalled if no progress for too long.

    Stall threshold scales with urgency derived from running_stats.stall_threshold_base
    multiplied by [1, 2.5, 5, 10] for decreasing urgency bands.

    Ported from: prototype-research/src/queue/worker.rs (detect_stalls)
    """
    goal_indices, _, goal_meta = state.get_active_goals()
    if len(goal_indices) == 0:
        return

    with torch.no_grad():
        stall_base = state.running_stats.stall_threshold_base
        for gi, gidx in enumerate(goal_indices):
            gidx = gidx.item()
            status = state.goal_metadata.data[gidx, STATUS].item()

            if abs(status - STATUS_ACTIVE) > 0.05:
                continue

            urgency = state.goal_metadata.data[gidx, URGENCY].item()
            created = state.goal_metadata.data[gidx, CREATED_STEP].item()

            # Urgency-scaled threshold derived from running_stats.stall_threshold_base
            if urgency >= 0.8:
                threshold = int(stall_base * 1)
            elif urgency >= 0.5:
                threshold = int(stall_base * 2.5)
            elif urgency >= 0.2:
                threshold = int(stall_base * 5)
            else:
                threshold = int(stall_base * 10)

            idle_steps = current_step - created
            progress = state.goal_metadata.data[gidx, PROGRESS].item()

            if idle_steps > threshold and progress < 0.1:
                state.goal_metadata.data[gidx, STATUS] = STATUS_STALLED


def enforce_deadlines(state: CognitiveState, current_step: int):
    """Mark overdue active goals as stalled.

    Ported from: prototype-research/src/dynamics/deadline.rs
    """
    goal_indices, _, goal_meta = state.get_active_goals()
    if len(goal_indices) == 0:
        return

    with torch.no_grad():
        for gi, gidx in enumerate(goal_indices):
            gidx = gidx.item()
            status = state.goal_metadata.data[gidx, STATUS].item()

            if abs(status - STATUS_ACTIVE) > 0.05:
                continue

            deadline = state.goal_metadata.data[gidx, DEADLINE].item()
            if deadline > 0 and current_step > deadline:
                state.goal_metadata.data[gidx, STATUS] = STATUS_STALLED
                # Boost priority for overdue goals
                state.goal_metadata.data[gidx, PRIORITY] = min(
                    state.goal_metadata.data[gidx, PRIORITY].item() + 0.2,
                    1.0,
                )


# ── Helpers ──

def _is_duplicate_goal(state: CognitiveState, goal_embed: Tensor) -> bool:
    """Check if a goal is too similar to existing active goals.

    Only compares against truly active goals (proposed/active), not
    completed/stalled/failed ones. Threshold is dynamic — stored in
    meta[6] and tuned by SPSA, so the model learns its own dedup sensitivity.
    """
    goal_indices, existing_embeds, _ = state.get_active_goals()
    if len(goal_indices) == 0:
        return False

    # Dynamic threshold from learned meta-parameters
    threshold = state.meta_params.goal_dedup_threshold.item()

    goal_angle = F.normalize(goal_embed, dim=0, eps=EPSILON)
    existing_angles = F.normalize(existing_embeds, dim=-1, eps=EPSILON)

    sims = existing_angles @ goal_angle
    return (sims > threshold).any().item()


def _allocate_goal_slot(state: CognitiveState) -> int:
    """Find an empty goal slot.

    Eviction priority: empty > completed/failed > stalled (oldest first).
    Returns slot index or -1 if all occupied by active/proposed goals.
    """
    for i in range(state.config.max_goals):
        if state.immutable_goals[i]:
            continue
        status = state.goal_metadata.data[i, STATUS].item()
        if status == STATUS_EMPTY:
            return i

    # No empty slots — try to evict a completed/failed goal
    for i in range(state.config.max_goals):
        if state.immutable_goals[i]:
            continue
        status = state.goal_metadata.data[i, STATUS].item()
        if abs(status - STATUS_COMPLETED) < 0.05 or abs(status - STATUS_FAILED) < 0.05:
            state.goal_metadata.data[i].zero_()
            state.goal_embeddings.data[i].zero_()
            return i

    # Still no slots — evict the oldest stalled goal
    oldest_stalled = -1
    oldest_step = float('inf')
    for i in range(state.config.max_goals):
        if state.immutable_goals[i]:
            continue
        status = state.goal_metadata.data[i, STATUS].item()
        if abs(status - STATUS_STALLED) < 0.05:
            created = state.goal_metadata.data[i, CREATED_STEP].item()
            if created < oldest_step:
                oldest_step = created
                oldest_stalled = i

    if oldest_stalled >= 0:
        state.goal_metadata.data[oldest_stalled].zero_()
        state.goal_embeddings.data[oldest_stalled].zero_()
        return oldest_stalled

    return -1
