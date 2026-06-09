"""Tests for D2b: the selected-action consequence loop in pass2.

The EFE policy (D2) selects an action type each step and persists it to
state.last_action_idx. At the START of the NEXT run_pass2, that action
modulates the structural-operation rates so the policy's decision is
behaviourally real (not a logged-only label). This is an INTERNAL control
consequence — a fixed-corpus next-token trainer has no external environment.

These tests assert:
  1. EXPLORE makes allocation strictly more aggressive than WAIT under an
     otherwise-identical state (the consequence is behavioral, not cosmetic).
  2. CONSOLIDATE forces the consolidation gate on.
  3. The selected action persists through a state save/load roundtrip.
"""

import torch
import pytest

from memoria.core.state import CognitiveState, StateConfig
from memoria.interface.write_path import WriteCandidate
from memoria.cognition.pass2 import run_pass2
from memoria.agency.action_selection import ACTION_TYPES
from memoria.agency.daemon import ActionType


@pytest.fixture
def config():
    return StateConfig(
        belief_dim=32, max_beliefs=128, max_edges=256, max_goals=8,
        relation_dim=16,
    )


def _fresh_state(config):
    state = CognitiveState(config)
    # Make the candidate-subsample gate trip on a small candidate set:
    # n_candidates_for_subsample = max(3, base_max_candidates // 10).
    state.running_stats.base_max_candidates = 8
    return state


def _make_candidates(n: int, dim: int) -> list[WriteCandidate]:
    """n unmatched candidates with distinct directions → each is a new belief."""
    cands = []
    for i in range(n):
        v = torch.zeros(dim)
        v[i % dim] = 1.0
        v[(i + 1) % dim] = 0.5
        v = torch.nn.functional.normalize(v, dim=0) * 2.0
        cands.append(WriteCandidate(
            belief_vector=v, matched_slot=-1, match_similarity=0.0,
            source_position=i, source_layer=0,
        ))
    return cands


def _fixed_actions_patch(state, allocate_rate: float = 1.0):
    """Pin the controller's actions so the only variable is the D2b modulation.

    allocate_rate starts at its upper bound: EXPLORE keeps it >=0.9 (no
    subsample), WAIT damps it toward 0 (subsamples candidates away).
    """
    def get_actions(_state):
        return {
            'allocate_rate': allocate_rate,
            'merge_threshold': _state.running_stats.merge_similarity_threshold,
            'prune_threshold': _state.running_stats.hard_cleanup_precision_threshold,
            'connect_rate': 1.0,
            'goal_rate': 0.0,
            'strategy_scale': 1.0,
        }
    state.controller.get_actions = get_actions  # type: ignore[assignment]


ACTION_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
N_CANDS = 8


def test_explore_allocates_more_than_wait(config):
    """EXPLORE → more beliefs allocated this step than WAIT (behavioral)."""
    # --- EXPLORE branch ---
    state_explore = _fresh_state(config)
    _fixed_actions_patch(state_explore, allocate_rate=1.0)
    state_explore.last_action_idx.fill_(ACTION_IDX[ActionType.EXPLORE])
    stats_explore = run_pass2(
        state_explore, _make_candidates(N_CANDS, config.belief_dim), [],
        current_step=1, is_sequence_boundary=False,
    )

    # --- WAIT branch (identical fresh state + candidates) ---
    state_wait = _fresh_state(config)
    _fixed_actions_patch(state_wait, allocate_rate=1.0)
    state_wait.last_action_idx.fill_(ACTION_IDX[ActionType.WAIT])
    stats_wait = run_pass2(
        state_wait, _make_candidates(N_CANDS, config.belief_dim), [],
        current_step=1, is_sequence_boundary=False,
    )

    explore_alloc = stats_explore['belief_new_allocations']
    wait_alloc = stats_wait['belief_new_allocations']

    # The consequence must be real: EXPLORE keeps all candidates, WAIT
    # subsamples them away, so strictly more beliefs are committed.
    assert explore_alloc > wait_alloc, (
        f"EXPLORE allocated {explore_alloc}, WAIT allocated {wait_alloc}; "
        "action consequence is not behavioral"
    )
    assert state_explore.num_active_beliefs() > state_wait.num_active_beliefs()

    # And the applied modulation is recorded in stats (observable).
    assert stats_explore['action_consequence'][0] == 'EXPLORE'
    assert stats_wait['action_consequence'][0] == 'WAIT'
    # EXPLORE pushed allocate_rate up (>= original 1.0 stays 1.0); WAIT pushed
    # it down strictly below the 0.9 subsample gate.
    assert stats_wait['action_consequence'][2] < 0.9


def test_consolidate_forces_consolidation_gate(config):
    """CONSOLIDATE → need_consolidation forced True even off-interval."""
    state = _fresh_state(config)
    _fixed_actions_patch(state)
    # Pick a step that is NOT on the soft-consolidation interval so the gate
    # would be OFF by default; the action must turn it on.
    soft_interval = state.running_stats.soft_consolidation_interval
    off_interval_step = soft_interval + 1
    assert off_interval_step % max(soft_interval, 1) != 0

    state.last_action_idx.fill_(ACTION_IDX[ActionType.CONSOLIDATE])
    stats = run_pass2(
        state, _make_candidates(N_CANDS, config.belief_dim), [],
        current_step=off_interval_step, is_sequence_boundary=False,
    )
    assert stats['pass2_skip']['consolidation'] is False
    assert stats['action_consequence'] == ('CONSOLIDATE', 'need_consolidation', True)


def test_nominal_action_no_modulation(config):
    """RESPOND → nominal, no rate modulation applied."""
    state = _fresh_state(config)
    _fixed_actions_patch(state)
    state.last_action_idx.fill_(ACTION_IDX[ActionType.RESPOND])
    stats = run_pass2(
        state, _make_candidates(N_CANDS, config.belief_dim), [],
        current_step=1, is_sequence_boundary=False,
    )
    assert stats['action_consequence'][0] == 'RESPOND'
    assert stats['action_consequence'][1] == 'nominal'


def test_no_prior_action_is_noop(config):
    """Default last_action_idx == -1 → no consequence applied."""
    state = _fresh_state(config)
    assert state.last_action_idx.item() == -1
    _fixed_actions_patch(state)
    stats = run_pass2(
        state, _make_candidates(N_CANDS, config.belief_dim), [],
        current_step=1, is_sequence_boundary=False,
    )
    assert stats['action_consequence'] is None


def test_action_selection_writes_last_action_idx(config):
    """Running pass2 with the real selector persists a valid action index."""
    state = _fresh_state(config)
    assert state.last_action_idx.item() == -1
    run_pass2(
        state, _make_candidates(N_CANDS, config.belief_dim), [],
        current_step=1, is_sequence_boundary=False,
    )
    idx = state.last_action_idx.item()
    assert 0 <= idx < len(ACTION_TYPES)


def test_last_action_idx_survives_save_load(config):
    """The selected action persists through a state_dict roundtrip."""
    state = _fresh_state(config)
    state.last_action_idx.fill_(ACTION_IDX[ActionType.CONSOLIDATE])

    ckpt = state.state_dict_cognitive(compress=False)
    assert 'last_action_idx' in ckpt

    restored = CognitiveState(config)
    assert restored.last_action_idx.item() == -1  # fresh default
    restored.load_state_cognitive(ckpt)
    assert restored.last_action_idx.item() == ACTION_IDX[ActionType.CONSOLIDATE]
    assert ACTION_TYPES[restored.last_action_idx.item()] == ActionType.CONSOLIDATE
