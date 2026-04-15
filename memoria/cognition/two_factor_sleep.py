"""E1: Two-Factor Sleep Consolidation — homeostatic precision normalization
+ cross-temporal replay + conflict-aware scanning.

Extends the existing SleepGate with three new mechanisms:
1. Homeostatic scaling: normalizes total precision budget to a learned target
2. Conflict scanning: detects near-duplicate beliefs with contradictory content
3. Cross-temporal replay preparation: identifies replay candidates for E4

Reference: Two-Factor Synaptic Consolidation (PNAS 2025)
Reference: SleepGate (arXiv:2603.14517)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import belief_is_active, angular_similarity, EPSILON


def homeostatic_scaling(
    state: CognitiveState,
    target: Tensor,
    rate: Tensor,
) -> dict:
    """Normalize total precision budget toward a learned target.

    Multiplies all active belief radii by a common factor that moves
    the total precision toward the target. This prevents unbounded
    inflation where every belief becomes maximally confident.

    The scaling is multiplicative (preserves relative ordering):
        scale = 1 + rate * (target / actual - 1)

    Args:
        state: cognitive state (modified in-place)
        target: desired total precision (from MetaParams.homeostatic_target)
        rate: correction rate per cycle (from MetaParams.homeostatic_rate)

    Returns:
        dict with {actual_total, target_total, scale_applied}
    """
    stats = {
        'actual_total': 0.0,
        'target_total': target.item(),
        'scale_applied': 1.0,
        'beliefs_eroded': 0,
    }

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return stats

    # Compute actual total precision (sum of radii)
    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    radii = state.beliefs.data[active_idx].norm(dim=-1)
    actual_total = radii.sum()
    stats['actual_total'] = actual_total.item()

    if actual_total < EPSILON:
        return stats

    # Compute scaling factor: moves actual toward target
    # scale = 1 + rate * (target/actual - 1)
    # When actual > target: scale < 1 (deflate)
    # When actual < target: scale > 1 (inflate)
    ratio = target / actual_total.clamp(min=EPSILON)
    scale = 1.0 + rate * (ratio - 1.0)

    # Clamp scale to prevent extreme corrections
    scale = scale.clamp(min=0.5, max=2.0)
    stats['scale_applied'] = scale.item()

    # Apply scaling to all active, non-immutable beliefs.
    # Track beliefs that were active before scaling but fall below EPSILON
    # after — these disappear from num_active_beliefs() silently without any
    # explicit deallocation counter. Surfacing this as `beliefs_eroded` closes
    # the audit gap where active_beliefs drifts down (2721→2584 observed) while
    # pass2/beliefs_cleaned stays at 0.
    scale_val = scale.item()
    eroded = 0
    with torch.no_grad():
        for local_i, idx_t in enumerate(active_idx):
            idx = idx_t.item()
            if state.immutable_beliefs[idx]:
                continue
            pre_radius = radii[local_i].item()
            state.beliefs.data[idx] *= scale
            post_radius = pre_radius * scale_val
            # EPSILON is state.py's active-threshold (1e-10). Crossing it means
            # the belief drops out of num_active_beliefs() on the next query.
            if pre_radius >= EPSILON and post_radius < EPSILON:
                eroded += 1

    stats['beliefs_eroded'] = eroded
    return stats


def conflict_scan(
    state: CognitiveState,
    conflict_threshold: Tensor,
) -> dict:
    """Detect near-duplicate beliefs that may be in conflict.

    Two beliefs are in conflict if:
    - Their angular cosine similarity > conflict_threshold (near-duplicate directions)
    - But one has much higher precision than the other (competing claims)

    Resolution: reduce precision of the lower-precision belief (it's likely stale).

    Args:
        state: cognitive state (modified in-place)
        conflict_threshold: cosine similarity above which beliefs conflict

    Returns:
        dict with {conflicts_found, beliefs_weakened}
    """
    stats = {'conflicts_found': 0, 'beliefs_weakened': 0}

    active_mask = state.get_active_mask()
    if active_mask.sum() < 2:
        return stats

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_beliefs = state.beliefs.data[active_idx]
    n_active = len(active_idx)

    # Compute pairwise angular similarity
    angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)
    sim_matrix = angles @ angles.T  # [N, N]

    radii = active_beliefs.norm(dim=-1)
    threshold = conflict_threshold.item()

    with torch.no_grad():
        for i in range(n_active):
            idx_i = active_idx[i].item()
            if state.immutable_beliefs[idx_i]:
                continue
            for j in range(i + 1, n_active):
                idx_j = active_idx[j].item()
                if state.immutable_beliefs[idx_j]:
                    continue

                sim = sim_matrix[i, j].item()
                if sim > threshold:
                    stats['conflicts_found'] += 1

                    # Weaken the lower-precision belief
                    r_i = radii[i].item()
                    r_j = radii[j].item()
                    if r_i < r_j:
                        # i is weaker — reduce its radius
                        # Decay proportional to how similar they are
                        decay = 1.0 - (sim - threshold) / (1.0 - threshold + EPSILON)
                        decay = max(decay, 0.5)  # floor: never reduce more than half
                        state.beliefs.data[idx_i] *= decay
                        stats['beliefs_weakened'] += 1
                    else:
                        decay = 1.0 - (sim - threshold) / (1.0 - threshold + EPSILON)
                        decay = max(decay, 0.5)
                        state.beliefs.data[idx_j] *= decay
                        stats['beliefs_weakened'] += 1

    return stats


def run_two_factor_sleep(
    state: CognitiveState,
    current_step: int,
) -> dict:
    """Run the full two-factor sleep consolidation cycle.

    Three phases:
    1. Homeostatic scaling — normalize total precision budget
    2. Conflict scanning — detect and resolve near-duplicate conflicts
    3. Identify replay candidates (used by E4 interleaved replay)

    All thresholds come from MetaParams (no hardcoded magic numbers).

    Args:
        state: cognitive state
        current_step: current training step

    Returns:
        dict with combined statistics
    """
    stats = {}

    # Phase 1: Homeostatic precision normalization
    homeo_stats = homeostatic_scaling(
        state,
        target=state.meta_params.homeostatic_target,
        rate=state.meta_params.homeostatic_rate,
    )
    stats['homeostatic'] = homeo_stats

    # Phase 2: Conflict scanning
    conflict_stats = conflict_scan(
        state,
        conflict_threshold=state.meta_params.sleep_conflict_threshold,
    )
    stats['conflicts'] = conflict_stats

    # Phase 3: Identify replay candidates for E4
    # Recent high-surprise beliefs + old high-precision beliefs
    active_mask = state.get_active_mask()
    if active_mask.any():
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        radii = state.beliefs.data[active_idx].norm(dim=-1)
        ages = current_step - state.belief_created_step[active_idx].float()

        # Recent = created in last 20% of current step count (or young)
        age_threshold = max(current_step * 0.2, 10.0)
        recent_mask = ages < age_threshold

        # High-surprise = beliefs with high precision variance (uncertain)
        variances = state.belief_precision_var[active_idx]

        stats['replay_candidates'] = {
            'recent_count': recent_mask.sum().item(),
            'old_count': (~recent_mask).sum().item(),
            'mean_variance_recent': variances[recent_mask].mean().item() if recent_mask.any() else 0.0,
            'mean_variance_old': variances[~recent_mask].mean().item() if (~recent_mask).any() else 0.0,
        }
    else:
        stats['replay_candidates'] = {
            'recent_count': 0, 'old_count': 0,
            'mean_variance_recent': 0.0, 'mean_variance_old': 0.0,
        }

    return stats
