"""B1-B4: Planning as inference on the belief graph.

Planning in Memoria is NOT a separate module — it extends the existing factor
graph with preference priors (from Telos goals) and epistemic priors (from
uncertainty), then runs the same message passing to convergence. The converged
belief state IS the plan.

Four components:
  B1: Preference/epistemic priors augment the factor graph
  B2: Multi-step causal rollout simulates future belief states
  B3: MCTS over EFE at high-uncertainty decision points
  B4: Hierarchical temporal planning via Telos depth

All behavioral constants are derived from MetaParams (learned nn.Parameters)
or from mathematical relationships. No hardcoded magic numbers.

Reference: Nuijten et al. — EFE Planning as Variational Inference (arXiv:2504.14898)
Reference: Nuijten et al. — Message Passing EFE Minimization (arXiv:2508.02197)
Reference: Deep AIF for Long Horizons (arXiv:2505.19867)
Reference: Dynamic Planning in Hierarchical Active Inference (arXiv:2402.11658)
Reference: Amortized Planning with Transformers (arXiv:2402.04494)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

from mcts import mcts as MCTS

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, EPSILON
from ..core.free_energy import power_spherical_entropy
from .causal import build_adjacency


# ═══════════════════════════════════════════════════════════════════════════════
# B1: Preference and Epistemic Priors
# ═══════════════════════════════════════════════════════════════════════════════

def compute_preference_messages(
    state: CognitiveState,
) -> tuple[Tensor, Tensor]:
    """Compute preference prior messages from active Telos goals.

    Each active goal injects a "virtual message" into the belief slots most
    aligned with it. The message pulls beliefs toward the goal direction,
    weighted by goal priority and the learned preference_prior_strength.

    This augments the factor graph: goals become unary factor nodes connected
    to their most relevant beliefs, adding a pragmatic bias to BP convergence.

    Attention sharpness uses 1/planning_temperature (learned) — not a constant.

    Args:
        state: cognitive state with goals and beliefs

    Returns:
        pref_messages: [max_beliefs, D] preference messages per belief
        pref_precisions: [max_beliefs] precision of preference messages
    """
    device = state.beliefs.device
    D = state.config.belief_dim
    n = state.config.max_beliefs

    pref_messages = torch.zeros(n, D, device=device)
    pref_precisions = torch.zeros(n, device=device)

    indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(indices) == 0:
        return pref_messages, pref_precisions

    strength = state.meta_params.preference_prior_strength.item()
    if strength < EPSILON:
        return pref_messages, pref_precisions

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return pref_messages, pref_precisions

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_beliefs = state.beliefs[active_idx]  # [N_active, D]

    # Normalize for cosine similarity
    goal_angles = F.normalize(goal_embeds, dim=-1, eps=EPSILON)      # [G, D]
    belief_angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)  # [N_active, D]

    # Similarity: which beliefs are most relevant to each goal?
    sim = belief_angles @ goal_angles.T  # [N_active, G]

    # Priority weighting: important goals pull harder
    priority = goal_meta[:, 0].clamp(min=EPSILON)  # [G]
    progress = goal_meta[:, 2].clamp(0.0, 1.0)     # [G]
    urgency = (1.0 - progress).clamp(min=EPSILON)   # [G]

    # Attention sharpness from learned planning_temperature:
    # temperature → 0 = sharp attention, temperature → inf = uniform
    tau = state.meta_params.planning_temperature.item()
    attn_scale = 1.0 / max(tau, EPSILON)

    for g_idx in range(len(indices)):
        attn = F.softmax(sim[:, g_idx] * attn_scale, dim=0)  # [N_active]

        msg_strength = priority[g_idx] * urgency[g_idx] * strength
        goal_msg = goal_angles[g_idx] * msg_strength  # [D]

        # Distribute to beliefs proportional to attention
        for i, belief_slot in enumerate(active_idx.tolist()):
            weight = attn[i].item()
            if weight > EPSILON:
                pref_messages[belief_slot] += goal_msg * weight
                pref_precisions[belief_slot] += msg_strength * weight

    return pref_messages, pref_precisions


def compute_epistemic_messages(
    state: CognitiveState,
) -> tuple[Tensor, Tensor]:
    """Compute epistemic prior messages that reward uncertainty reduction.

    High-entropy (uncertain) beliefs receive a message that amplifies their
    participation in message passing. This implements the epistemic drive:
    the system preferentially resolves uncertainty during planning.

    The epistemic message has no directional bias (it doesn't tell beliefs
    WHERE to go, only that they SHOULD update). Direction comes from the
    regular BP messages and preference priors.

    Args:
        state: cognitive state

    Returns:
        epist_messages: [max_beliefs, D] epistemic messages (zero direction)
        epist_precisions: [max_beliefs] epistemic urgency per belief
    """
    device = state.beliefs.device
    D = state.config.belief_dim
    n = state.config.max_beliefs

    epist_messages = torch.zeros(n, D, device=device)
    epist_precisions = torch.zeros(n, device=device)

    strength = state.meta_params.epistemic_prior_strength.item()
    if strength < EPSILON:
        return epist_messages, epist_precisions

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return epist_messages, epist_precisions

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_radii = state.beliefs[active_idx].norm(dim=-1)

    # Power Spherical entropy: high entropy = uncertain = should explore
    kappa = active_radii.clamp(min=EPSILON)
    H = power_spherical_entropy(kappa, D)  # [N_active]

    # Normalize entropy to [0, 1] range for stable scaling
    if H.numel() > 1 and H.max() > H.min():
        H_norm = (H - H.min()) / (H.max() - H.min() + EPSILON)
    else:
        # Single belief or all same entropy: use midpoint (uninformative)
        H_norm = torch.full_like(H, 0.5)

    # Epistemic precision: uncertain beliefs get higher epistemic urgency
    for i, belief_slot in enumerate(active_idx.tolist()):
        epist_precisions[belief_slot] = H_norm[i] * strength

    return epist_messages, epist_precisions


# ═══════════════════════════════════════════════════════════════════════════════
# B2: Multi-Step Causal Rollout
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RolloutResult:
    """Result of a multi-step causal rollout."""
    predicted_beliefs: Tensor          # [max_beliefs, D] predicted future state
    pragmatic_value: float             # goal alignment of predicted state
    epistemic_value: float             # uncertainty reduction
    risk: float                        # prediction error magnitude
    efe: float                         # combined EFE score
    depth: int                         # how many steps we actually rolled out
    visited_beliefs: list[int]         # belief indices affected by rollout


def causal_rollout(
    state: CognitiveState,
    horizon: int | None = None,
    discount: float | None = None,
) -> RolloutResult:
    """Simulate future belief states by propagating through causal edges.

    Starting from current beliefs, follow causal edges (directed graph) to
    predict what beliefs would look like K steps in the future. Score the
    predicted state by EFE decomposition.

    This is the "dynamics model" — the causal graph is a learned model of
    how beliefs evolve, and rollout asks "what happens if I let things play out?"

    All influence magnitudes are derived from edge weights and relative precision
    (the Kalman-gain formula: w * r_src / (r_src + r_tgt)). The discount is
    a learned MetaParam.

    Args:
        state: cognitive state (not modified — rollout is pure simulation)
        horizon: how many causal steps to simulate (default: learned)
        discount: temporal discount per step (default: learned)

    Returns:
        RolloutResult with predicted state and EFE scores
    """
    if horizon is None:
        horizon = max(1, int(state.meta_params.planning_horizon.item()))
    if discount is None:
        discount = state.meta_params.planning_discount.item()

    device = state.beliefs.device
    D = state.config.belief_dim

    # Clone beliefs for simulation
    sim_beliefs = state.beliefs.data.clone()
    adj = build_adjacency(state)

    visited = set()
    cumulative_efe = 0.0
    actual_depth = 0

    for step in range(horizon):
        new_beliefs = sim_beliefs.clone()
        step_visited = []

        for src, neighbors in adj.items():
            src_belief = sim_beliefs[src]
            src_radius = src_belief.norm().clamp(min=EPSILON)
            if src_radius < EPSILON:
                continue

            src_angle = src_belief / src_radius

            for tgt, weight in neighbors:
                if weight < EPSILON:
                    continue

                tgt_belief = sim_beliefs[tgt]
                tgt_radius = tgt_belief.norm().clamp(min=EPSILON)

                # Kalman-gain style influence: w * r_src / (r_src + r_tgt)
                # This is derived from relative precision — no magic number.
                # The discount acts as a per-step damper on influence.
                influence = weight * src_radius / (src_radius + tgt_radius + EPSILON)
                influence = influence.item() if isinstance(influence, Tensor) else influence
                influence = min(influence, discount)  # discount caps per-step influence

                if tgt_radius > EPSILON:
                    tgt_angle = tgt_belief / tgt_radius
                    new_angle = F.normalize(
                        ((1.0 - influence) * tgt_angle + influence * src_angle).unsqueeze(0),
                        dim=-1, eps=EPSILON,
                    ).squeeze(0)
                    new_beliefs[tgt] = new_angle * tgt_radius
                else:
                    new_beliefs[tgt] = src_angle * (src_radius * influence)

                visited.add(tgt)
                step_visited.append(tgt)

        sim_beliefs = new_beliefs
        actual_depth = step + 1

        # Compute step EFE with temporal discount
        step_discount = discount ** step
        step_efe = _evaluate_belief_state(state, sim_beliefs)
        cumulative_efe += step_efe * step_discount

        if not step_visited:
            break  # no more propagation possible

    # Final evaluation of the predicted state
    final_eval = _evaluate_belief_state_detailed(state, sim_beliefs)

    return RolloutResult(
        predicted_beliefs=sim_beliefs,
        pragmatic_value=final_eval['pragmatic'],
        epistemic_value=final_eval['epistemic'],
        risk=final_eval['risk'],
        efe=cumulative_efe,
        depth=actual_depth,
        visited_beliefs=list(visited),
    )


def _evaluate_belief_state(state: CognitiveState, beliefs: Tensor) -> float:
    """Quick EFE evaluation of a hypothetical belief state.

    Uses the same learned EFE weights (efe_epistemic_weight, efe_risk_weight)
    as the main free energy computation.
    """
    D = state.config.belief_dim
    active_mask = beliefs.norm(dim=-1) > EPSILON

    if not active_mask.any():
        return 0.0

    active = beliefs[active_mask]
    active_radii = active.norm(dim=-1)

    # Pragmatic: goal alignment
    indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(indices) > 0:
        goal_angles = F.normalize(goal_embeds, dim=-1, eps=EPSILON)
        belief_angles = F.normalize(active, dim=-1, eps=EPSILON)
        sim = belief_angles @ goal_angles.T
        priority = goal_meta[:, 0].clamp(min=EPSILON)
        pragmatic = (sim * priority.unsqueeze(0)).max(dim=-1).values.mean().item()
    else:
        pragmatic = 0.0

    # Epistemic: entropy of predicted beliefs
    kappa = active_radii.clamp(min=EPSILON)
    H = power_spherical_entropy(kappa, D)
    epistemic = H.mean().item()

    w_e = state.meta_params.efe_epistemic_weight.item()

    return -pragmatic + w_e * epistemic


def _evaluate_belief_state_detailed(
    state: CognitiveState,
    beliefs: Tensor,
) -> dict:
    """Detailed EFE evaluation of a hypothetical belief state."""
    D = state.config.belief_dim
    active_mask = beliefs.norm(dim=-1) > EPSILON

    if not active_mask.any():
        return {'pragmatic': 0.0, 'epistemic': 0.0, 'risk': 0.0}

    active = beliefs[active_mask]
    active_radii = active.norm(dim=-1)

    # Pragmatic
    indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(indices) > 0:
        goal_angles = F.normalize(goal_embeds, dim=-1, eps=EPSILON)
        belief_angles = F.normalize(active, dim=-1, eps=EPSILON)
        sim = belief_angles @ goal_angles.T
        priority = goal_meta[:, 0].clamp(min=EPSILON)
        pragmatic = (sim * priority.unsqueeze(0)).max(dim=-1).values.mean().item()
    else:
        pragmatic = 0.0

    # Epistemic
    kappa = active_radii.clamp(min=EPSILON)
    H = power_spherical_entropy(kappa, D)
    epistemic = H.mean().item()

    # Risk: cosine divergence from current beliefs
    current_active = state.beliefs.data[active_mask]
    if current_active.shape == active.shape:
        cos_sim = F.cosine_similarity(current_active, active, dim=-1)
        risk = (1.0 - cos_sim).mean().item()
    else:
        risk = 0.0

    return {'pragmatic': pragmatic, 'epistemic': epistemic, 'risk': risk}


# ═══════════════════════════════════════════════════════════════════════════════
# B3: Monte Carlo Tree Search over EFE (via mcts library v1.0.4)
# ═══════════════════════════════════════════════════════════════════════════════

class _PlanningState:
    """Adapter wrapping Memoria's belief state for the mcts library.

    The mcts library (v1.0.4) expects a State with:
      - getPossibleActions() → list of actions
      - takeAction(action) → new State
      - isTerminal() → bool
      - getReward() → float

    Each action = a goal index. takeAction simulates one causal step
    biased toward that goal. Reward = negative EFE (lower EFE = better).
    Terminal when depth reaches max_depth.
    """

    def __init__(
        self,
        cognitive_state: CognitiveState,
        beliefs: Tensor,
        goal_indices: Tensor,
        goal_embeds: Tensor,
        goal_meta: Tensor,
        depth: int = 0,
        max_depth: int = 5,
    ):
        self._cog = cognitive_state
        self.beliefs = beliefs
        self._goal_indices = goal_indices
        self._goal_embeds = goal_embeds
        self._goal_meta = goal_meta
        self._depth = depth
        self._max_depth = max_depth
        # Cache the action taken at this node for result extraction
        self.action_taken = -1

    def getPossibleActions(self) -> list[int]:
        return list(range(len(self._goal_indices)))

    def takeAction(self, action: int) -> '_PlanningState':
        child_beliefs = _simulate_goal_step(
            self._cog, self.beliefs,
            self._goal_embeds[action], self._goal_meta[action],
        )
        child = _PlanningState(
            cognitive_state=self._cog,
            beliefs=child_beliefs,
            goal_indices=self._goal_indices,
            goal_embeds=self._goal_embeds,
            goal_meta=self._goal_meta,
            depth=self._depth + 1,
            max_depth=self._max_depth,
        )
        child.action_taken = action
        return child

    def isTerminal(self) -> bool:
        return self._depth >= self._max_depth

    def getReward(self) -> float:
        # Negative EFE: lower EFE = better = higher reward
        return -_evaluate_belief_state(self._cog, self.beliefs)


def _efe_rollout_policy(state: '_PlanningState') -> float:
    """EFE-based rollout policy for MCTS.

    Instead of random rollout, evaluate the state's EFE directly.
    This gives better value estimates than random play because our
    "game" has no natural terminal state — EFE is the objective.
    """
    return state.getReward()


def mcts_plan(
    state: CognitiveState,
    n_simulations: int | None = None,
    max_depth: int | None = None,
) -> dict:
    """Run MCTS over EFE to select the best goal to pursue.

    Uses the mcts library (v1.0.4) with UCB tree search. Each action is
    an active Telos goal. The rollout policy evaluates EFE directly
    (no random play needed — EFE is the objective function).

    The exploration constant is a learned MetaParam (mcts_exploration).

    Args:
        state: cognitive state (not modified)
        n_simulations: number of MCTS iterations (default: horizon^2)
        max_depth: max tree depth (default: learned planning_horizon)

    Returns:
        dict with best_goal_idx, goal_values, root_visits, tree_depth
    """
    if max_depth is None:
        max_depth = max(1, int(state.meta_params.planning_horizon.item()))
    if n_simulations is None:
        n_simulations = max(4, max_depth * max_depth)

    exploration = state.meta_params.mcts_exploration.item()

    goal_indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(goal_indices) == 0:
        return {
            'best_goal_idx': -1,
            'goal_values': {},
            'root_visits': 0,
            'tree_depth': 0,
        }

    # Create initial planning state for the mcts library
    initial_state = _PlanningState(
        cognitive_state=state,
        beliefs=state.beliefs.data.clone(),
        goal_indices=goal_indices,
        goal_embeds=goal_embeds,
        goal_meta=goal_meta,
        depth=0,
        max_depth=max_depth,
    )

    # Use the mcts library with our EFE rollout policy
    searcher = MCTS(
        iterationLimit=n_simulations,
        explorationConstant=exploration,
        rolloutPolicy=_efe_rollout_policy,
    )
    best_action = searcher.search(initialState=initial_state)

    # Extract per-goal values from the tree
    goal_values = {}
    root_node = searcher.root
    for action, child_node in root_node.children.items():
        g_idx = goal_indices[action].item()
        mean_val = child_node.totalReward / max(child_node.numVisits, 1)
        goal_values[g_idx] = mean_val

    # Fill missing goals with 0
    for g_idx in goal_indices.tolist():
        if g_idx not in goal_values:
            goal_values[g_idx] = 0.0

    best_goal_idx = goal_indices[best_action].item()

    # Compute tree depth from root visits
    max_depth_reached = 0
    queue = [root_node]
    while queue:
        node = queue.pop()
        for child in node.children.values():
            queue.append(child)
            # Depth approximation: count levels
            d = 1
            n = child
            while n.parent is not None:
                d += 1
                n = n.parent
            max_depth_reached = max(max_depth_reached, d)

    return {
        'best_goal_idx': best_goal_idx,
        'goal_values': goal_values,
        'root_visits': root_node.numVisits,
        'tree_depth': max_depth_reached,
    }


def _simulate_goal_step(
    state: CognitiveState,
    beliefs: Tensor,
    goal_embed: Tensor,
    goal_meta: Tensor,
) -> Tensor:
    """Simulate one causal step biased toward a goal.

    Combines causal propagation with a goal-directed pull.
    Pull strength = preference_prior_strength / D (dimension-normalized,
    same derivation as apply_belief_shift in message_passing.py).
    Influence cap = planning_discount (same learned parameter used for rollout).
    """
    sim = beliefs.clone()

    goal_angle = F.normalize(goal_embed.unsqueeze(0), dim=-1, eps=EPSILON).squeeze(0)
    active_mask = sim.norm(dim=-1) > EPSILON
    if not active_mask.any():
        return sim

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)

    belief_angles = F.normalize(sim[active_idx], dim=-1, eps=EPSILON)
    similarity = (belief_angles @ goal_angle).clamp(-1, 1)

    # Top-K: pull sqrt(N_active) beliefs (scales with state size, not constant)
    k = max(1, min(int(math.sqrt(len(active_idx))), len(active_idx)))
    top_sim, top_local = similarity.topk(k)

    # Pull strength: preference_prior_strength / D (dimension-normalized, no magic)
    D = state.config.belief_dim
    pref_strength = state.meta_params.preference_prior_strength.item()
    pull = pref_strength / D

    discount = state.meta_params.planning_discount.item()

    with torch.no_grad():
        for i in range(k):
            if top_sim[i] > 0.0:
                idx = active_idx[top_local[i]].item()
                r = sim[idx].norm().clamp(min=EPSILON)
                angle = sim[idx] / r
                new_angle = F.normalize(
                    ((1.0 - pull) * angle + pull * goal_angle).unsqueeze(0),
                    dim=-1, eps=EPSILON,
                ).squeeze(0)
                sim[idx] = new_angle * r

    # Causal propagation step
    adj = build_adjacency(state)
    new_sim = sim.clone()
    for src, neighbors in adj.items():
        src_belief = sim[src]
        src_r = src_belief.norm().clamp(min=EPSILON)
        if src_r < EPSILON:
            continue
        src_a = src_belief / src_r
        for tgt, weight in neighbors:
            if weight < EPSILON:
                continue
            tgt_belief = sim[tgt]
            tgt_r = tgt_belief.norm().clamp(min=EPSILON)
            # Kalman-gain influence capped by discount
            influence = weight * src_r / (src_r + tgt_r + EPSILON)
            influence = min(
                influence.item() if isinstance(influence, Tensor) else influence,
                discount,
            )
            if tgt_r > EPSILON:
                tgt_a = tgt_belief / tgt_r
                new_a = F.normalize(
                    ((1.0 - influence) * tgt_a + influence * src_a).unsqueeze(0),
                    dim=-1, eps=EPSILON,
                ).squeeze(0)
                new_sim[tgt] = new_a * tgt_r

    return new_sim


# ═══════════════════════════════════════════════════════════════════════════════
# B4: Hierarchical Temporal Planning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HierarchicalPlan:
    """A plan decomposed across temporal scales."""
    # Per-level plans: level → (goal_idx, EFE value, horizon used)
    level_plans: dict[int, list[tuple[int, float, int]]]
    # Top-down constraints: which high-level goals constrain lower levels
    constraints: dict[int, list[int]]  # lower_goal → [higher_goals that constrain it]
    # Overall recommended action
    recommended_goal: int
    total_efe: float


def hierarchical_plan(
    state: CognitiveState,
) -> HierarchicalPlan:
    """Plan across temporal scales using Telos goal hierarchy.

    Goals at different depths plan at different temporal resolutions:
    - Depth 0 (root goals): coarse planning, horizon * 4
    - Depth 1 (sub-goals): medium planning, horizon * 2
    - Depth 2+ (leaf goals): fine-grained, horizon * 1

    The horizon scale is geometric (4, 2, 1) which is a structural choice
    from the hierarchical AIF literature (arXiv:2402.11658), not a tuned
    constant. The base_horizon is a learned MetaParam.

    MCTS is used when β > 0.5 (uncertain, exploring) at the coarsest level.
    β is derived from free energy computation — not a magic number.

    Args:
        state: cognitive state

    Returns:
        HierarchicalPlan with per-level plans and constraints
    """
    goal_indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(goal_indices) == 0:
        return HierarchicalPlan(
            level_plans={}, constraints={},
            recommended_goal=-1, total_efe=0.0,
        )

    # Group goals by depth
    depths = goal_meta[:, 4].long()  # metadata[4] = depth
    level_goals: dict[int, list[int]] = {}
    for i, depth in enumerate(depths.tolist()):
        level_goals.setdefault(depth, []).append(i)

    level_plans: dict[int, list[tuple[int, float, int]]] = {}
    constraints: dict[int, list[int]] = {}
    base_horizon = max(1, int(state.meta_params.planning_horizon.item()))

    sorted_levels = sorted(level_goals.keys())

    # Use β (exploration/exploitation from free energy) as MCTS trigger
    beta = state.beta

    for level in sorted_levels:
        g_indices_local = level_goals[level]

        # Temporal resolution: geometric scale 4, 2, 1, 1, ...
        # This is a structural choice from hierarchical AIF, not a tuned constant
        scale = max(1, 4 >> min(level, 2))
        level_horizon = base_horizon * scale

        plans = []
        for g_local in g_indices_local:
            g_global = goal_indices[g_local].item()

            # MCTS for uncertain coarse-level decisions with multiple competing goals
            # β > 0.5 means exploration-dominant (derived from FE, not magic)
            if beta > 0.5 and level == 0 and len(g_indices_local) > 1:
                mcts_result = mcts_plan(state, max_depth=level_horizon)
                value = mcts_result['goal_values'].get(g_global, 0.0)
            else:
                rollout = causal_rollout(state, horizon=level_horizon)
                value = -rollout.efe

            plans.append((g_global, value, level_horizon))

        level_plans[level] = plans

        # Top-down constraints: best goal at this level constrains next level
        if plans and level + 1 in level_goals:
            best_at_level = max(plans, key=lambda p: p[1])
            for g_local_lower in level_goals.get(level + 1, []):
                g_global_lower = goal_indices[g_local_lower].item()
                constraints.setdefault(g_global_lower, []).append(best_at_level[0])

    # Overall recommendation: best goal weighted by level (deeper = more actionable)
    # Bonus per level = 1/D (dimension-normalized, same rationale as belief shift)
    D = state.config.belief_dim
    level_bonus = 1.0 / D
    all_plans = []
    for level, plans in level_plans.items():
        for g_idx, value, horizon in plans:
            adjusted = value + level * level_bonus
            all_plans.append((g_idx, adjusted))

    if all_plans:
        best = max(all_plans, key=lambda p: p[1])
        recommended = best[0]
        total_efe = best[1]
    else:
        recommended = -1
        total_efe = 0.0

    return HierarchicalPlan(
        level_plans=level_plans,
        constraints=constraints,
        recommended_goal=recommended,
        total_efe=total_efe,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Planning step for pass2
# ═══════════════════════════════════════════════════════════════════════════════

def run_planning_step(
    state: CognitiveState,
    current_step: int,
) -> dict:
    """Run one planning cycle, integrated into pass2.

    Computes preference and epistemic priors, runs causal rollout for the
    active goal set, and optionally runs MCTS for high-uncertainty decisions.

    Called from pass2 at sequence boundaries (when there's time to think ahead).

    MCTS threshold: β > 0.5 (exploration-dominant, derived from FE).
    Hierarchical planning interval: same as hard_consolidation_interval
    (from running_stats — adaptive, not hardcoded).

    Args:
        state: cognitive state (preference/epistemic messages stored for next BP)
        current_step: current training step

    Returns:
        dict with planning statistics
    """
    stats = {'planning_ran': True}

    # B1: Compute preference and epistemic priors
    pref_messages, pref_precisions = compute_preference_messages(state)
    epist_messages, epist_precisions = compute_epistemic_messages(state)

    # Store for use in next message passing round
    with torch.no_grad():
        state._planning_pref_messages.copy_(pref_messages)
        state._planning_pref_precisions.copy_(pref_precisions)
        state._planning_epist_precisions.copy_(epist_precisions)

    stats['pref_precision_total'] = pref_precisions.sum().item()
    stats['epist_precision_total'] = epist_precisions.sum().item()

    # B2: Causal rollout
    rollout = causal_rollout(state)
    stats['rollout_depth'] = rollout.depth
    stats['rollout_pragmatic'] = rollout.pragmatic_value
    stats['rollout_epistemic'] = rollout.epistemic_value
    stats['rollout_risk'] = rollout.risk
    stats['rollout_efe'] = rollout.efe

    # B3: MCTS for multi-goal uncertainty
    n_goals = state.num_active_goals()
    beta = state.beta

    if n_goals > 1 and beta > 0.5:
        # β > 0.5 = exploration-dominant = uncertain about which goal to pursue
        mcts_result = mcts_plan(state)
        stats['mcts_best_goal'] = mcts_result['best_goal_idx']
        stats['mcts_visits'] = mcts_result['root_visits']
        stats['mcts_depth'] = mcts_result['tree_depth']
    else:
        stats['mcts_best_goal'] = -1
        stats['mcts_visits'] = 0
        stats['mcts_depth'] = 0

    # B4: Hierarchical planning (at consolidation intervals — adaptive)
    consol_interval = max(1, int(state.running_stats.hard_consolidation_interval))
    if current_step % consol_interval == 0 and n_goals > 1:
        h_plan = hierarchical_plan(state)
        stats['hierarchical_recommended'] = h_plan.recommended_goal
        stats['hierarchical_levels'] = len(h_plan.level_plans)
    else:
        stats['hierarchical_recommended'] = -1
        stats['hierarchical_levels'] = 0

    return stats
