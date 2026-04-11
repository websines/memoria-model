"""Tests for A1: Tentative Belief Mode (Internal Autoresearch Loop)."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.provisional import (
    evaluate_provisional_beliefs,
    EVICT_FE_NOT_IMPROVED,
    EVICT_PRECISION_COLLAPSED,
    EVICT_NEVER_READ,
    PROMOTED,
)


def _simulate_post_alloc_reads(state, slot: int, n: int, start_step: int):
    """Simulate `n` post-allocation reads of a provisional belief.

    Read events only count toward the search/eval split when they happen
    strictly after the allocation step. Tests that exercise promotion need
    to call this after allocation to lift the `provisional_min_reads`
    criterion above the default (1).
    """
    indices = torch.tensor([slot], dtype=torch.long)
    for i in range(n):
        state.touch_beliefs(indices, step=start_step + i + 1)


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


def test_provisional_allocation(state):
    """Provisional beliefs are tracked correctly."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=10, current_fe=5.0)
    assert slot >= 0
    assert state.belief_provisional[slot].item() is True
    assert state.belief_provisional_step[slot].item() == 10.0
    assert state.belief_provisional_fe[slot].item() == 5.0
    assert state.belief_provisional_radius[slot].item() == pytest.approx(2.0, abs=0.01)


def test_provisional_no_access_count(state):
    """Provisional beliefs don't increment access_count when touched."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=0)

    indices = torch.tensor([slot], dtype=torch.long)
    state.touch_beliefs(indices, step=1)
    state.touch_beliefs(indices, step=2)
    state.touch_beliefs(indices, step=3)

    assert state.belief_access_count[slot].item() == 0  # no reinforcement
    assert state.belief_last_accessed[slot].item() == 3.0  # recency updated


def test_committed_access_count(state):
    """Committed (non-provisional) beliefs DO increment access_count."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=False, step=0)

    indices = torch.tensor([slot], dtype=torch.long)
    state.touch_beliefs(indices, step=1)
    state.touch_beliefs(indices, step=2)

    assert state.belief_access_count[slot].item() == 2


def test_provisional_promotion(state):
    """Provisional belief is promoted when FE decreased, precision held, and it was read."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=0, current_fe=10.0)

    # Simulate post-allocation reads on held-out data (search/eval split).
    _simulate_post_alloc_reads(state, slot, n=3, start_step=0)

    # Simulate: FE went down, precision still good
    stats = evaluate_provisional_beliefs(state, current_step=100, current_fe=5.0)

    assert stats['promoted'] == 1
    assert stats['evicted'] == 0
    assert state.belief_provisional[slot].item() is False  # now committed
    assert state.num_active_beliefs() == 1  # still exists
    # Promoted beliefs reset their read counter.
    assert state.belief_provisional_reads[slot].item() == 0.0


def test_provisional_eviction_fe_increased(state):
    """Provisional belief is evicted when FE increased (despite being read)."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=0, current_fe=5.0)

    # Simulate post-allocation reads — the belief WAS tested, but FE still rose.
    _simulate_post_alloc_reads(state, slot, n=3, start_step=0)

    # Simulate: FE went UP (hypothesis made things worse)
    stats = evaluate_provisional_beliefs(state, current_step=100, current_fe=10.0)

    assert stats['evicted'] == 1
    assert stats['evicted_fe_not_improved'] == 1
    assert stats['promoted'] == 0
    assert state.num_active_beliefs() == 0  # evicted


def test_provisional_eviction_precision_dropped(state):
    """Provisional belief is evicted when its precision dropped too much."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=0, current_fe=10.0)

    # Read the belief — rule out the never-read path so precision is the
    # sole reason for eviction.
    _simulate_post_alloc_reads(state, slot, n=3, start_step=0)

    # Simulate: gradient shrunk the belief to near-zero (useless hypothesis)
    with torch.no_grad():
        state.beliefs.data[slot] *= 0.01  # radius now ~0.02 (was 2.0)

    # FE decreased (good) but precision tanked (bad)
    stats = evaluate_provisional_beliefs(state, current_step=100, current_fe=5.0)

    assert stats['evicted'] == 1
    assert stats['evicted_precision_collapsed'] == 1


def test_provisional_eviction_never_read(state):
    """Hypothesis that was never retrieved post-allocation is evicted.

    This is the Meta-Harness search/eval split: a hypothesis whose eval
    window elapsed without any post-allocation reads was never tested on
    data that did not spawn it, so we cannot attribute the (possibly
    improving) global FE to this belief. Promotion in that regime would
    just ride training drift. Evict instead.
    """
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=0, current_fe=10.0)

    # No post-allocation reads — belief just sat in state unused.
    # FE would promote it by the old criterion; the new criterion evicts.
    stats = evaluate_provisional_beliefs(state, current_step=100, current_fe=5.0)

    assert stats['evicted'] == 1
    assert stats['evicted_never_read'] == 1
    assert stats['promoted'] == 0
    assert state.num_active_beliefs() == 0


def test_provisional_read_counter_ignores_alloc_step_touches(state):
    """Reads that happen on the allocation step itself do not count as held-out tests.

    A touch at step == alloc_step could happen inside the same forward pass
    that triggered the hypothesis generation — that's exactly the kind of
    confounded read we're splitting against. Only strictly-later steps count.
    """
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=50, current_fe=10.0)

    indices = torch.tensor([slot], dtype=torch.long)
    # Read at the same step as allocation — must NOT count.
    state.touch_beliefs(indices, step=50)
    assert state.belief_provisional_reads[slot].item() == 0.0

    # Read at a strictly later step — counts.
    state.touch_beliefs(indices, step=51)
    assert state.belief_provisional_reads[slot].item() == 1.0


def test_provisional_not_evaluated_too_early(state):
    """Provisional beliefs within eval window are not evaluated."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    state.allocate_belief(b, provisional=True, step=95, current_fe=10.0)

    # Only 5 steps elapsed, eval window is ~10
    stats = evaluate_provisional_beliefs(state, current_step=100, current_fe=5.0)

    assert stats['still_provisional'] == 1
    assert stats['promoted'] == 0
    assert stats['evicted'] == 0


def test_provisional_deallocate_clears_flags(state):
    """Deallocating a provisional belief clears all provisional metadata."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    slot = state.allocate_belief(b, provisional=True, step=10, current_fe=5.0)

    state.deallocate_belief(slot)

    assert state.belief_provisional[slot].item() is False
    assert state.belief_provisional_step[slot].item() == 0.0
    assert state.belief_provisional_fe[slot].item() == 0.0
    assert state.belief_provisional_radius[slot].item() == 0.0


def test_checkpoint_roundtrip_provisional(state):
    """Provisional state survives serialization."""
    b = make_belief([1, 0, 0], 2.0, state.config.belief_dim)
    state.allocate_belief(b, provisional=True, step=10, current_fe=5.0)

    checkpoint = state.state_dict_cognitive(compress=False)

    state2 = CognitiveState(state.config)
    state2.load_state_cognitive(checkpoint)

    assert state2.belief_provisional[0].item() is True
    assert state2.belief_provisional_fe[0].item() == 5.0
