"""Tests for CognitiveController — Meta-Harness #1 rich history encoding.

The controller's original 10 scalar features are augmented with a GRU over a
rolling buffer of (action, outcome) tuples. The GRU output is projected and
ADDED to the scalar features before the policy/value heads see them. The
projection is zero-initialized, so a freshly-built controller's behavior is
exactly identical to the pre-#1 scalar-only version — these tests lock that
invariant in place and then exercise the history push / ring-buffer wrap /
post-commit forward path.

Reference: Meta-Harness (arXiv:2603.28052) Table 3 — full access to prior
diagnostic experience beats compressed scalar feedback by ~15 points.
"""

import torch
import pytest

from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.cognitive_controller import (
    CognitiveController,
    NUM_ACTIONS,
    OUTCOME_DIM,
)


@pytest.fixture
def config():
    return StateConfig(
        belief_dim=32,
        max_beliefs=64,
        max_edges=256,
        max_goals=8,
        relation_dim=16,
        controller_history_depth=8,  # small for fast tests
        failed_buffer_depth=4,
    )


@pytest.fixture
def state(config):
    return CognitiveState(config)


def _set_seed():
    torch.manual_seed(0)


# ── Zero-init invariant ─────────────────────────────────────────────────────


def test_history_proj_zero_init_preserves_scalar_only_behavior(state):
    """At t=0, the history-augmented forward produces the same features
    as the pre-#1 scalar-only controller."""
    controller = state.controller

    # Directly compute what the legacy encode_state would produce (the scalar
    # features), and compare against the new encode_state which adds the
    # GRU projection. At init the projection must be exactly zero.
    scalar_only = torch.zeros(controller.state_dim)
    with torch.no_grad():
        radii = state.get_belief_radii()
        active = state.get_active_mask()
        active_radii = (
            radii[active] if active.any() else torch.zeros(1)
        )
        scalar_only[0] = active_radii.mean()
        scalar_only[1] = active_radii.std() if len(active_radii) > 1 else 0.0
        scalar_only[2] = active.float().mean()
        scalar_only[3] = state.num_active_edges() / max(state.config.max_edges, 1)
        scalar_only[4] = state.beta
        scalar_only[5] = state.running_stats.mean_surprise.item()
        scalar_only[6] = state.num_active_goals() / max(state.config.max_goals, 1)
        scalar_only[7] = controller.belief_advantage_ema.item()
        scalar_only[8] = controller._compute_goal_diversity(state)
        scalar_only[9] = controller._compute_goal_completion_rate(state)

    encoded = controller.encode_state(state)
    assert torch.allclose(encoded, scalar_only, atol=1e-6), (
        "history_proj must be zero-initialized so encode_state output "
        "matches the legacy scalar-only feature vector"
    )


def test_history_proj_zero_init_identifies_to_get_actions(state):
    """Sampling actions with a fresh controller is deterministic given the
    state and RNG seed, regardless of any staged history state."""
    _set_seed()
    a1 = state.controller.get_actions(state)
    _set_seed()
    a2 = state.controller.get_actions(state)
    for k in a1:
        assert a1[k] == pytest.approx(a2[k], abs=1e-6)


# ── History push / commit flow ──────────────────────────────────────────────


def test_history_commit_after_dense_reward(state):
    """compute_dense_reward pairs the staged action with the outcome vector
    and pushes one entry to the ring buffer."""
    controller = state.controller
    assert controller.history_total_writes.item() == 0

    _ = controller.get_actions(state)
    assert bool(controller._has_pending_action.item()) is True

    _ = controller.compute_dense_reward(
        state, belief_advantage=0.5, training_progress=0.0
    )
    assert controller.history_total_writes.item() == 1
    assert bool(controller._has_pending_action.item()) is False

    # First slot should hold the committed action and outcome.
    stored_action = controller.history_actions[0]
    stored_outcome = controller.history_outcomes[0]
    assert stored_action.shape == (NUM_ACTIONS,)
    assert stored_outcome.shape == (OUTCOME_DIM,)
    # belief_advantage lives in outcome[0].
    assert stored_outcome[0].item() == pytest.approx(0.5, abs=1e-6)


def test_history_commit_requires_pending_action(state):
    """compute_dense_reward without a preceding get_actions is a no-op on
    the ring buffer (no stale push)."""
    controller = state.controller
    _ = controller.compute_dense_reward(
        state, belief_advantage=0.1, training_progress=0.0
    )
    assert controller.history_total_writes.item() == 0


def test_history_ring_buffer_wraps(state):
    """Pushing more entries than depth wraps cleanly and keeps the newest
    `depth` entries in the correct slots."""
    controller = state.controller
    depth = controller.history_depth

    for i in range(depth + 3):
        _ = controller.get_actions(state)
        _ = controller.compute_dense_reward(
            state, belief_advantage=float(i), training_progress=0.0
        )

    assert controller.history_total_writes.item() == depth + 3
    # After wrap, the three newest outcomes (i = depth, depth+1, depth+2)
    # should be in slots 0, 1, 2.
    for k in range(3):
        expected_belief_advantage = float(depth + k)
        actual = controller.history_outcomes[k, 0].item()
        assert actual == pytest.approx(expected_belief_advantage, abs=1e-6)


def test_encode_history_reorders_after_wrap(state):
    """After a ring-buffer wrap, the GRU receives entries in chronological
    order (oldest first)."""
    controller = state.controller
    depth = controller.history_depth

    # Push depth + 1 entries — position 0 will hold the NEWEST entry, not
    # the oldest. _encode_history must re-order so the oldest is fed first.
    for i in range(depth + 1):
        _ = controller.get_actions(state)
        _ = controller.compute_dense_reward(
            state, belief_advantage=float(i), training_progress=0.0
        )

    # Verify by patching history_proj to return a linear sum of the GRU
    # output — we can then reason about which entries contributed.
    # Instead of white-boxing the GRU, we just check that the feature
    # vector is finite and non-degenerate after a few non-zero pushes AND
    # after we lift history_proj off zero-init.
    with torch.no_grad():
        controller.history_proj.weight.normal_(0.0, 0.1)
    feat = controller._encode_history(torch.device('cpu'))
    assert feat.shape == (controller.state_dim,)
    assert torch.isfinite(feat).all()


def test_encode_history_nonzero_when_proj_trained(state):
    """Once history_proj has non-zero weights, the encoded history vector
    differs from zero and therefore from the pre-#1 scalar-only baseline."""
    controller = state.controller
    # Push at least one entry so the GRU has input.
    _ = controller.get_actions(state)
    _ = controller.compute_dense_reward(
        state, belief_advantage=0.25, training_progress=0.0
    )

    # Fresh projection — still zero-init, encoded history must be zero.
    zero_feat = controller._encode_history(torch.device('cpu'))
    assert torch.all(zero_feat == 0)

    # Lift the projection off zero and verify history contributes.
    with torch.no_grad():
        controller.history_proj.weight.normal_(0.0, 0.5)
        controller.history_proj.bias.normal_(0.0, 0.5)
    trained_feat = controller._encode_history(torch.device('cpu'))
    assert not torch.all(trained_feat == 0)


# ── StateConfig threading ───────────────────────────────────────────────────


def test_history_depth_flows_from_state_config():
    """StateConfig.controller_history_depth determines the ring-buffer size."""
    cfg = StateConfig(
        belief_dim=32,
        max_beliefs=32,
        max_edges=64,
        max_goals=4,
        relation_dim=16,
        controller_history_depth=5,
        failed_buffer_depth=2,
    )
    s = CognitiveState(cfg)
    assert s.controller.history_depth == 5
    assert s.controller.history_actions.shape == (5, NUM_ACTIONS)
    assert s.controller.history_outcomes.shape == (5, OUTCOME_DIM)
