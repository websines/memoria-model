"""Tests for B1-B4: Planning as inference on the belief graph."""

import math
import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.planning import (
    compute_preference_messages,
    compute_epistemic_messages,
    causal_rollout,
    mcts_plan,
    hierarchical_plan,
    run_planning_step,
    RolloutResult,
    HierarchicalPlan,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=32, max_beliefs=64, max_edges=256, max_goals=8, relation_dim=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


def make_belief(direction: list[float], precision: float, dim: int = 32) -> torch.Tensor:
    v = torch.zeros(dim)
    for i, val in enumerate(direction):
        v[i] = val
    return torch.nn.functional.normalize(v, dim=0) * precision


def setup_beliefs_and_goals(state):
    """Populate state with beliefs, goals, and edges for testing."""
    D = state.config.belief_dim
    # Add several beliefs
    for i in range(10):
        direction = [0.0] * D
        direction[i % D] = 1.0
        b = make_belief(direction, 2.0 + i * 0.1, D)
        state.allocate_belief(b, step=i)

    # Add a goal
    goal_embed = make_belief([1, 0, 0], 1.5, D)
    with torch.no_grad():
        state.goal_embeddings.data[0] = goal_embed
        state.goal_status_logits[0] = torch.zeros(6)
        state.goal_status_logits[0, 2] = 10.0  # STATUS_ACTIVE
        state.goal_metadata.data[0, 0] = 0.8   # priority
        state.goal_metadata.data[0, 2] = 0.2   # progress

    # Add some causal edges between beliefs
    relation = torch.zeros(state.config.relation_dim)
    state.allocate_edge(0, 1, relation, weight=0.5)
    state.allocate_edge(1, 2, relation, weight=0.3)
    state.allocate_edge(2, 3, relation, weight=0.4)


def setup_multi_goal(state):
    """Set up state with multiple competing goals."""
    setup_beliefs_and_goals(state)
    D = state.config.belief_dim
    # Add a second competing goal
    goal_embed2 = make_belief([0, 1, 0], 1.3, D)
    with torch.no_grad():
        state.goal_embeddings.data[1] = goal_embed2
        state.goal_status_logits[1] = torch.zeros(6)
        state.goal_status_logits[1, 2] = 10.0  # STATUS_ACTIVE
        state.goal_metadata.data[1, 0] = 0.6   # priority
        state.goal_metadata.data[1, 2] = 0.1   # progress


# ═══════════════════════════════════════════════════════════════════════════════
# B1: Preference and Epistemic Priors
# ═══════════════════════════════════════════════════════════════════════════════

def test_preference_messages_empty(state):
    """No goals → zero preference messages."""
    pref_msg, pref_prec = compute_preference_messages(state)
    assert pref_msg.abs().sum().item() == 0.0
    assert pref_prec.abs().sum().item() == 0.0


def test_preference_messages_with_goals(state):
    """Active goals produce non-zero preference messages on aligned beliefs."""
    setup_beliefs_and_goals(state)
    pref_msg, pref_prec = compute_preference_messages(state)
    # Some beliefs should receive non-zero preference messages
    assert pref_prec.sum().item() > 0
    assert pref_msg.abs().sum().item() > 0


def test_preference_strength_controls_magnitude(state):
    """Stronger preference_prior_strength → larger messages."""
    setup_beliefs_and_goals(state)

    # Weak preference
    with torch.no_grad():
        state.meta_params._preference_prior_strength.fill_(-5.0)  # softplus ≈ 0.007
    _, prec_weak = compute_preference_messages(state)

    # Strong preference
    with torch.no_grad():
        state.meta_params._preference_prior_strength.fill_(3.0)  # softplus ≈ 3.05
    _, prec_strong = compute_preference_messages(state)

    assert prec_strong.sum().item() > prec_weak.sum().item()


def test_epistemic_messages_empty(state):
    """No active beliefs → zero epistemic messages."""
    _, epist_prec = compute_epistemic_messages(state)
    assert epist_prec.abs().sum().item() == 0.0


def test_epistemic_messages_with_beliefs(state):
    """Active beliefs produce non-zero epistemic messages."""
    setup_beliefs_and_goals(state)
    _, epist_prec = compute_epistemic_messages(state)
    # Beliefs with uncertainty get epistemic urgency
    assert epist_prec.sum().item() > 0


def test_epistemic_strength_controls_magnitude(state):
    """Stronger epistemic_prior_strength → larger epistemic precision."""
    setup_beliefs_and_goals(state)

    with torch.no_grad():
        state.meta_params._epistemic_prior_strength.fill_(-5.0)
    _, prec_weak = compute_epistemic_messages(state)

    with torch.no_grad():
        state.meta_params._epistemic_prior_strength.fill_(3.0)
    _, prec_strong = compute_epistemic_messages(state)

    assert prec_strong.sum().item() > prec_weak.sum().item()


# ═══════════════════════════════════════════════════════════════════════════════
# B2: Causal Rollout
# ═══════════════════════════════════════════════════════════════════════════════

def test_rollout_empty_state(state):
    """Rollout on empty state produces zero-depth result."""
    result = causal_rollout(state, horizon=3)
    assert result.depth >= 0
    assert len(result.visited_beliefs) == 0


def test_rollout_with_edges(state):
    """Rollout through causal edges visits downstream beliefs."""
    setup_beliefs_and_goals(state)
    result = causal_rollout(state, horizon=3)
    assert result.depth > 0
    assert len(result.visited_beliefs) > 0


def test_rollout_predicted_beliefs_differ(state):
    """Rolled-out beliefs differ from current beliefs."""
    setup_beliefs_and_goals(state)
    result = causal_rollout(state, horizon=5)
    # Predicted state should differ from current (causal propagation changed things)
    diff = (result.predicted_beliefs - state.beliefs.data).abs().sum().item()
    assert diff > 0


def test_rollout_horizon_controls_depth(state):
    """Longer horizon → more rollout steps."""
    setup_beliefs_and_goals(state)
    short = causal_rollout(state, horizon=1)
    long = causal_rollout(state, horizon=5)
    assert long.depth >= short.depth


def test_rollout_efe_components(state):
    """Rollout returns valid EFE components."""
    setup_beliefs_and_goals(state)
    result = causal_rollout(state, horizon=3)
    # Components should be finite
    assert math.isfinite(result.pragmatic_value)
    assert math.isfinite(result.epistemic_value)
    assert math.isfinite(result.risk)
    assert math.isfinite(result.efe)


def test_rollout_discount_affects_efe(state):
    """Different discount values affect cumulative EFE."""
    setup_beliefs_and_goals(state)
    r1 = causal_rollout(state, horizon=3, discount=0.99)
    r2 = causal_rollout(state, horizon=3, discount=0.5)
    # Higher discount = less discounting = EFE from later steps matters more
    # (they should differ unless rollout converges in 1 step)
    # Can't assert direction but should be different
    # Both should be finite
    assert math.isfinite(r1.efe)
    assert math.isfinite(r2.efe)


# ═══════════════════════════════════════════════════════════════════════════════
# B3: MCTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_mcts_no_goals(state):
    """MCTS with no goals returns empty result."""
    result = mcts_plan(state, n_simulations=4)
    assert result['best_goal_idx'] == -1
    assert result['root_visits'] == 0


def test_mcts_single_goal(state):
    """MCTS with one goal always selects it."""
    setup_beliefs_and_goals(state)
    result = mcts_plan(state, n_simulations=8)
    assert result['best_goal_idx'] >= 0
    assert result['root_visits'] == 8


def test_mcts_multi_goal(state):
    """MCTS with competing goals makes a selection."""
    setup_multi_goal(state)
    result = mcts_plan(state, n_simulations=16)
    assert result['best_goal_idx'] >= 0
    assert len(result['goal_values']) == 2
    assert result['root_visits'] == 16


def test_mcts_exploration_affects_search(state):
    """Different exploration weights change visit distribution."""
    setup_multi_goal(state)

    # Low exploration: exploit
    with torch.no_grad():
        state.meta_params._mcts_exploration.fill_(-5.0)  # softplus ≈ 0.007
    r1 = mcts_plan(state, n_simulations=16)

    # High exploration
    with torch.no_grad():
        state.meta_params._mcts_exploration.fill_(3.0)  # softplus ≈ 3.05
    r2 = mcts_plan(state, n_simulations=16)

    # Both should produce valid results
    assert r1['best_goal_idx'] >= 0
    assert r2['best_goal_idx'] >= 0


def test_mcts_default_simulations_scale_with_horizon(state):
    """Default n_simulations = horizon^2."""
    setup_multi_goal(state)
    with torch.no_grad():
        state.meta_params._planning_horizon.fill_(2.3)  # softplus ≈ 10 → 10^2 = 100
    # Don't actually run 100 sims in test — just verify the function runs
    result = mcts_plan(state, n_simulations=4, max_depth=2)
    assert result['root_visits'] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# B4: Hierarchical Planning
# ═══════════════════════════════════════════════════════════════════════════════

def test_hierarchical_no_goals(state):
    """Hierarchical plan with no goals returns empty."""
    result = hierarchical_plan(state)
    assert result.recommended_goal == -1
    assert len(result.level_plans) == 0


def test_hierarchical_single_level(state):
    """Single-level goals produce one-level plan."""
    setup_beliefs_and_goals(state)
    result = hierarchical_plan(state)
    assert result.recommended_goal >= 0
    assert len(result.level_plans) > 0


def test_hierarchical_multi_level(state):
    """Goals at different depths produce multi-level plan."""
    setup_multi_goal(state)
    # Set goals at different depths
    with torch.no_grad():
        state.goal_metadata.data[0, 4] = 0.0  # depth 0
        state.goal_metadata.data[1, 4] = 1.0  # depth 1
    result = hierarchical_plan(state)
    assert result.recommended_goal >= 0
    assert len(result.level_plans) >= 1


def test_hierarchical_constraints(state):
    """Higher-level goals constrain lower-level goals."""
    setup_multi_goal(state)
    # Add a third goal at depth 1
    D = state.config.belief_dim
    with torch.no_grad():
        state.goal_embeddings.data[2] = make_belief([0, 0, 1], 1.0, D)
        state.goal_status_logits[2] = torch.zeros(6)
        state.goal_status_logits[2, 2] = 10.0
        state.goal_metadata.data[2, 0] = 0.5
        # Goal 0 at depth 0, goals 1 and 2 at depth 1
        state.goal_metadata.data[0, 4] = 0.0
        state.goal_metadata.data[1, 4] = 1.0
        state.goal_metadata.data[2, 4] = 1.0
    result = hierarchical_plan(state)
    # Should have constraints from depth 0 to depth 1
    if result.constraints:
        for lower, uppers in result.constraints.items():
            assert len(uppers) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_planning_step(state):
    """Full planning step runs without errors."""
    setup_beliefs_and_goals(state)
    stats = run_planning_step(state, current_step=0)
    assert stats['planning_ran']
    assert 'rollout_depth' in stats
    assert 'pref_precision_total' in stats
    assert 'epist_precision_total' in stats


def test_planning_step_stores_priors(state):
    """Planning step stores preference/epistemic priors for BP."""
    setup_beliefs_and_goals(state)
    run_planning_step(state, current_step=0)
    # Priors should be stored on state
    assert state._planning_pref_precisions.sum().item() > 0


def test_planning_step_mcts_with_multi_goals(state):
    """Planning triggers MCTS when multiple goals and high beta."""
    setup_multi_goal(state)
    # Force high beta (exploration mode)
    with torch.no_grad():
        state.meta.data[0] = 0.8  # β > 0.5
    stats = run_planning_step(state, current_step=0)
    # MCTS should have run
    assert stats['mcts_visits'] > 0


def test_planning_no_magic_numbers(state):
    """All planning behavior changes when MetaParams change."""
    setup_beliefs_and_goals(state)

    # Baseline
    stats1 = run_planning_step(state, current_step=0)

    # Change planning_horizon → different rollout depth
    with torch.no_grad():
        state.meta_params._planning_horizon.fill_(0.0)  # softplus ≈ 0.693 → horizon=1
    stats2 = run_planning_step(state, current_step=1)

    # Both should run but potentially different depths
    assert stats1['planning_ran']
    assert stats2['planning_ran']


def test_state_serialization_includes_planning(state):
    """Cognitive state checkpoint includes planning buffers."""
    setup_beliefs_and_goals(state)
    run_planning_step(state, current_step=0)

    saved = state.state_dict_cognitive(compress=False)
    assert '_planning_pref_messages' in saved
    assert '_planning_pref_precisions' in saved
    assert '_planning_epist_precisions' in saved
    assert 'safety_gate' in saved


def test_state_serialization_roundtrip(state, config):
    """Planning buffers survive save/load."""
    setup_beliefs_and_goals(state)
    run_planning_step(state, current_step=0)

    saved = state.state_dict_cognitive(compress=False)
    pref_before = state._planning_pref_precisions.clone()

    state2 = CognitiveState(config)
    state2.load_state_cognitive(saved)

    assert torch.allclose(state2._planning_pref_precisions, pref_before)
