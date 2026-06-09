"""A4b: self-modification paths must be certified by the SGM safety gate.

The SafetyGate (A4) certifies self-modifications before they commit. These
tests assert that the two live self-modification paths route through
`state.safety_gate`:

  1. C2 learned belief update (pass2 block 13)
  2. C3 structural plasticity (run_structural_plasticity)

For each path: a clearly-HARMFUL proposed modification is REJECTED and rolled
back (beliefs unchanged); a clearly-BENEFICIAL one is ACCEPTED (beliefs change).

The improvement signal in both paths is the reduction in local relational free
energy (1 - cos to a belief's edge neighbours), fed as e-value samples to the
gate. We construct geometry so the sign of that signal is unambiguous.
"""

import pytest

# The SafetyGate depends on the `expectation` library (git-pinned). Skip the
# whole module cleanly if it (or the planning MCTS dep pulled in by pass2)
# isn't importable in this environment.
pytest.importorskip("expectation")

import torch
import torch.nn.functional as F

from memoria.core.state import CognitiveState, StateConfig
from memoria.cognition.pass2 import (
    run_pass2,
    _local_fe_improvement,
    _gate_self_modification,
)
from memoria.cognition.structural_plasticity import (
    StructuralPlasticity,
    run_structural_plasticity,
    _neighbour_beliefs,
    _disagreement,
)
from memoria.interface.write_path import WriteCandidate


DIM = 32


@pytest.fixture
def config():
    return StateConfig(
        belief_dim=DIM, max_beliefs=64, max_edges=256, max_goals=8, relation_dim=16,
    )


@pytest.fixture
def state(config):
    return CognitiveState(config)


def _vec(direction, precision, dim=DIM):
    v = torch.zeros(dim)
    for i, val in enumerate(direction):
        v[i] = val
    return F.normalize(v, dim=0) * precision


# ──────────────────────────────────────────────────────────────────────────
# Improvement-signal sanity
# ──────────────────────────────────────────────────────────────────────────

def test_local_fe_improvement_sign():
    """Improvement = disagreement_before - disagreement_after, isolated excluded."""
    # belief 0: before points at +x, after rotates to +y, context is +y → improves.
    # belief 1: isolated (zero context) → excluded from the sample set.
    before = torch.stack([_vec([1, 0, 0], 1.0), _vec([1, 0, 0], 1.0)])
    after = torch.stack([_vec([0, 1, 0], 1.0), _vec([1, 0, 0], 1.0)])
    ctx = torch.stack([_vec([0, 1, 0], 1.0), torch.zeros(DIM)])
    imp = _local_fe_improvement(before, after, ctx)
    assert len(imp) == 1  # isolated belief excluded
    assert imp[0] > 0.9  # 1 - cos(x,y)=1 before, ~0 after → improvement ≈ 1


def test_gate_rejects_empty_sample(state):
    """A modification with no measurable sample cannot be certified."""
    accepted, n = _gate_self_modification(state, "noop", [])
    assert not accepted
    assert n == 0
    # No budget spent on an empty evaluation.
    assert state.safety_gate.alpha_spent_total.item() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# C2: learned belief update gating (end-to-end through pass2)
# ──────────────────────────────────────────────────────────────────────────

def _setup_matched_beliefs_with_neighbours(state, n_beliefs=8):
    """n beliefs at +x, each with a +y neighbour, plus matched write candidates.

    Returns (slots, candidates) ready for run_pass2. Each neighbour makes a
    belief's edge-context point at +y, so a delta toward +y is beneficial and a
    delta toward -x is harmful. The one-sided sequential e-test needs several
    samples to certify, so we supply a batch of edited beliefs (one improvement
    sample each).
    """
    slots = []
    candidates = []
    for _ in range(n_beliefs):
        slot = state.allocate_belief(_vec([1, 0, 0], 1.0))
        neighbour = state.allocate_belief(_vec([0, 1, 0], 1.0))
        # Pin the neighbour so other pass2 blocks (message passing / dream)
        # can't rotate the relational context out from under the gate — the
        # edge-context must stay at +y for the improvement sign to be defined.
        with torch.no_grad():
            state.immutable_beliefs[neighbour] = True
        state.allocate_edge(slot, neighbour, torch.zeros(state.config.relation_dim), weight=1.0)
        slots.append(slot)
        candidates.append(WriteCandidate(
            belief_vector=_vec([1, 0.1, 0], 0.5), matched_slot=slot,
            match_similarity=0.95, source_position=0, source_layer=0,
        ))
    return slots, candidates


def _force_gate_open(state):
    # Open the learned-update gate fully so the learned delta is what gets
    # certified (gate near 0 at init would make the proposed edit a no-op).
    with torch.no_grad():
        state.meta_params._update_fn_gate.fill_(10.0)  # sigmoid(10) ≈ 1


class _FixedDelta(torch.nn.Module):
    """Stand-in learned_update returning a fixed delta toward a target angle."""

    def __init__(self, target_angle: torch.Tensor):
        super().__init__()
        self.target = target_angle

    def forward(self, beliefs, observation, precisions, errors, edge_context):
        n = beliefs.shape[0]
        # delta = (target_radius * target_angle) - current belief → moves belief
        # onto the target direction at the same radius.
        radius = beliefs.norm(dim=-1, keepdim=True)
        target = self.target.to(beliefs.device).unsqueeze(0) * radius
        delta = target - beliefs
        prec_scale = torch.ones(n, device=beliefs.device)
        merge = torch.zeros(n, device=beliefs.device)
        return delta, prec_scale, merge


def test_learned_update_beneficial_accepted(state):
    """A clearly free-energy-reducing learned update is certified by the gate.

    Exercises the exact wiring used by pass2 block 13: build per-edit improvement
    samples with `_local_fe_improvement` (beliefs rotated ONTO their relational
    context → strong positive improvement) and route them through the real
    `state.safety_gate` via `_gate_self_modification`. (The full run_pass2 loop's
    emergent edge creation makes the relational context nondeterministic, so we
    drive the decision logic directly here; the rejection/rollback half is tested
    end-to-end through run_pass2 below.)
    """
    n = 8
    before = torch.stack([_vec([1, 0, 0], 1.0) for _ in range(n)])
    # Context = +y; "after" rotates each belief onto +y → disagreement 1 → 0.
    after = torch.stack([_vec([0, 1, 0], 1.0) for _ in range(n)])
    ctx = torch.stack([_vec([0, 1, 0], 1.0) for _ in range(n)])

    improvements = _local_fe_improvement(before, after, ctx)
    assert len(improvements) == n
    assert min(improvements) > 0.9  # every edit strongly reduces local FE

    accepted, n_eval = _gate_self_modification(state, "c2_beneficial", improvements)
    assert accepted
    assert n_eval == n
    # Acceptance spends global error budget (a real certification occurred).
    assert state.safety_gate.alpha_spent_total.item() > 0
    assert state.safety_gate.n_accepted.item() == 1


def test_learned_update_harmful_rejected_and_rolled_back(state):
    slots, candidates = _setup_matched_beliefs_with_neighbours(state)
    _force_gate_open(state)
    # Harmful: push beliefs AWAY from neighbours (toward -x), increasing local FE.
    state.learned_update = _FixedDelta(_vec([-1, 0, 0], 1.0))
    before = state.beliefs.data[slots].clone()

    stats = run_pass2(state, candidates, slots, current_step=1)

    assert stats.get("learned_update_rejected") == 1
    assert stats.get("learned_update_applied", 0) == 0
    after = state.beliefs.data[slots]
    # Rolled back: the harmful learned delta (which would flip beliefs to -x)
    # was NOT committed — beliefs still point at +x. (Other legitimate pass2
    # blocks may nudge them microscopically; the rejected delta is large and
    # absent.) Compare against the +x direction the harmful delta would destroy.
    cos_x_before = F.cosine_similarity(before, _vec([1, 0, 0], 1.0).unsqueeze(0)).mean().item()
    cos_x_after = F.cosine_similarity(after, _vec([1, 0, 0], 1.0).unsqueeze(0)).mean().item()
    assert cos_x_after > 0.99
    assert abs(cos_x_after - cos_x_before) < 1e-2


# ──────────────────────────────────────────────────────────────────────────
# C3: structural plasticity gating
# ──────────────────────────────────────────────────────────────────────────

def _force_prune_candidate(state, plasticity, slot):
    """Make `slot` a prune candidate: low frequency, low radius, prune_net→1."""
    with torch.no_grad():
        # prune_net must fire: set its bias high so sigmoid(logit) > 0.5.
        plasticity.prune_net[-1].bias.fill_(10.0)
        # split_net must NOT fire.
        plasticity.split_net[-1].bias.fill_(-10.0)
    # Drive _total_steps up without bumping this slot's activation_count, so
    # frequency stays ~0 (< prune_threshold) and the belief looks "dead".
    other = state.allocate_belief(_vec([1, 0, 0], 5.0))
    for _ in range(50):
        plasticity.record_activation(torch.tensor([other]))
    return other


def test_structural_prune_harmful_rejected(state):
    """Pruning a WELL-integrated belief is rejected (no FE reduction)."""
    plasticity = StructuralPlasticity(belief_dim=DIM, max_beliefs=state.config.max_beliefs)
    # Target belief aligned WITH its neighbours → low disagreement → pruning it
    # does not reduce local relation energy → improvement samples ≈ 0 → reject.
    slot = state.allocate_belief(_vec([0, 1, 0], 0.05))  # low radius (dead-ish)
    for k in range(4):
        n = state.allocate_belief(_vec([0, 1, 0], 1.0))  # same direction = aligned
        state.allocate_edge(slot, n, torch.zeros(state.config.relation_dim), weight=1.0)
    _force_prune_candidate(state, plasticity, slot)

    # Sanity: the helper sees aligned neighbours → near-zero disagreement.
    neigh = _neighbour_beliefs(state, slot)
    assert neigh is not None and neigh.shape[0] == 4
    assert _disagreement(state.beliefs.data[slot], neigh).abs().max().item() < 0.05

    active_before = state.num_active_beliefs()
    stats = run_structural_plasticity(state, plasticity)
    assert stats["prunes_executed"] == 0
    assert stats["prunes_rejected"] >= 1
    assert state.num_active_beliefs() == active_before  # nothing removed


def test_structural_prune_beneficial_accepted(state):
    """Pruning a POORLY-integrated dead belief is accepted (FE reduction)."""
    plasticity = StructuralPlasticity(belief_dim=DIM, max_beliefs=state.config.max_beliefs)
    # Target belief points AGAINST all its neighbours → high disagreement →
    # removing it reduces local relation energy → strong improvement samples.
    slot = state.allocate_belief(_vec([1, 0, 0], 0.05))  # low radius (dead-ish)
    # Many neighbours pointing the opposite way → many high-disagreement samples
    # (need enough samples for the one-sided e-test to certify).
    for k in range(8):
        n = state.allocate_belief(_vec([-1, 0, 0], 1.0))
        state.allocate_edge(slot, n, torch.zeros(state.config.relation_dim), weight=1.0)
    _force_prune_candidate(state, plasticity, slot)

    neigh = _neighbour_beliefs(state, slot)
    assert neigh is not None and neigh.shape[0] == 8
    # Disagreement ≈ 1 - cos(+x, -x) = 2 for every neighbour.
    assert _disagreement(state.beliefs.data[slot], neigh).min().item() > 1.5

    stats = run_structural_plasticity(state, plasticity)
    assert stats["prunes_executed"] >= 1
    # The dead belief slot is now deallocated.
    assert state.beliefs.data[slot].norm().item() < 1e-3
