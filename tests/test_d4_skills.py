"""Tests for D4: Skill Crystallization + Disentanglement."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.agency.skills import (
    SkillBank,
    SkillDetector,
    SkillComposer,
    run_skill_step,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def bank():
    return SkillBank(belief_dim=64, max_skills=32)


@pytest.fixture
def detector():
    return SkillDetector(belief_dim=64, buffer_size=64)


@pytest.fixture
def composer():
    return SkillComposer(belief_dim=64)


def test_skill_bank_init(bank):
    """Skill bank initializes empty."""
    assert bank.num_active_skills() == 0
    assert bank.skill_embeddings.shape == (32, 64)


def test_allocate_skill(bank):
    """Can allocate a skill."""
    vec = torch.randn(64)
    slot = bank.allocate_skill(vec, step=10)
    assert slot >= 0
    assert bank.num_active_skills() == 1
    assert bank.skill_active[slot].item()
    assert bank.skill_created_step[slot].item() == 10.0


def test_deallocate_skill(bank):
    """Can deallocate a skill."""
    slot = bank.allocate_skill(torch.randn(64))
    bank.deallocate_skill(slot)
    assert bank.num_active_skills() == 0
    assert not bank.skill_active[slot].item()


def test_get_active_skills(bank):
    """Active skills returns correct data."""
    bank.allocate_skill(torch.randn(64))
    bank.allocate_skill(torch.randn(64))
    indices, embeddings = bank.get_active_skills()
    assert len(indices) == 2
    assert embeddings.shape == (2, 64)


def test_update_utility(bank):
    """Utility updates via EMA."""
    slot = bank.allocate_skill(torch.randn(64))
    assert bank.skill_utility[slot].item() == 0.5  # initial

    bank.update_utility(slot, 1.0)
    assert bank.skill_utility[slot].item() > 0.5  # increased
    assert bank.skill_use_count[slot].item() == 1


def test_skill_bank_full(bank):
    """Returns -1 when full."""
    for _ in range(32):
        bank.allocate_skill(torch.randn(64))
    assert bank.num_active_skills() == 32
    assert bank.allocate_skill(torch.randn(64)) == -1


def test_detector_init(detector):
    """Detector initializes with empty buffer."""
    assert detector._buffer_count.item() == 0


def test_record_pattern(detector):
    """Recording patterns fills the buffer."""
    detector.record_pattern(torch.randn(64), reward=1.0)
    assert detector._buffer_count.item() == 1


def test_record_pattern_ignores_negative(detector):
    """Only positive rewards are recorded."""
    detector.record_pattern(torch.randn(64), reward=-0.5)
    assert detector._buffer_count.item() == 0


def test_detect_skills_empty(detector):
    """No skills detected with empty buffer."""
    skills = detector.detect_skills(
        similarity_threshold=torch.tensor(0.8),
        detection_threshold=torch.tensor(1.0),
    )
    assert len(skills) == 0


def test_detect_skills_with_patterns(detector):
    """Skills detected from recurring patterns."""
    # Record similar patterns repeatedly
    base = torch.randn(64)
    base = base / base.norm()
    for _ in range(20):
        pattern = base + torch.randn(64) * 0.05  # slight noise
        detector.record_pattern(pattern, reward=1.0)

    skills = detector.detect_skills(
        similarity_threshold=torch.tensor(0.5),
        detection_threshold=torch.tensor(0.5),
    )
    # Should detect at least one cluster
    assert len(skills) >= 1
    assert skills[0].shape == (64,)


def test_composer_two_skills(composer):
    """Composing two skills produces valid output."""
    a = torch.randn(64)
    b = torch.randn(64)
    composed, compat = composer.compose(a, b)
    assert composed.shape == (64,)
    assert 0 <= compat.item() <= 1


def test_composer_multiple(composer):
    """Composing multiple skills works."""
    skills = torch.randn(4, 64)
    result = composer.compose_multiple(skills)
    assert result.shape == (64,)


def test_composer_single(composer):
    """Composing one skill returns it directly."""
    skills = torch.randn(1, 64)
    result = composer.compose_multiple(skills)
    assert torch.allclose(result, skills[0])


def test_composer_empty(composer):
    """Composing no skills returns zero."""
    skills = torch.zeros(0, 64)
    result = composer.compose_multiple(skills)
    assert result.abs().sum().item() == 0


def test_run_skill_step(state):
    """Full skill step runs without error."""
    stats = run_skill_step(
        state,
        state.skill_bank,
        state.skill_detector,
        state.skill_composer,
        current_step=100,
    )
    assert 'skills_detected' in stats
    assert 'skills_crystallized' in stats
    assert 'skills_pruned' in stats
    assert 'total_active_skills' in stats


def test_skill_dedup(state):
    """Duplicate skills are not crystallized twice."""
    bank = state.skill_bank
    det = state.skill_detector

    # Record identical patterns
    base = torch.randn(64)
    for _ in range(20):
        det.record_pattern(base + torch.randn(64) * 0.01, reward=1.0)

    stats1 = run_skill_step(state, bank, det, state.skill_composer, 100)
    n_first = bank.num_active_skills()

    # Record same patterns again
    for _ in range(20):
        det.record_pattern(base + torch.randn(64) * 0.01, reward=1.0)

    stats2 = run_skill_step(state, bank, det, state.skill_composer, 200)
    n_second = bank.num_active_skills()

    # Second round should not add duplicates
    assert n_second <= n_first + 1  # allow at most 1 new (noise variation)


def test_skill_pruning(state):
    """Low-utility old skills are pruned."""
    bank = state.skill_bank
    slot = bank.allocate_skill(torch.randn(64), step=0)
    bank.skill_utility[slot] = 0.01  # very low utility
    bank.skill_use_count[slot] = 20  # used enough times
    bank.skill_created_step[slot] = 0  # old

    stats = run_skill_step(
        state, bank, state.skill_detector, state.skill_composer, current_step=200,
    )
    assert stats['skills_pruned'] >= 1


def test_skills_integrated(state):
    """Skill modules are attached to state."""
    assert hasattr(state, 'skill_bank')
    assert isinstance(state.skill_bank, SkillBank)
    assert hasattr(state, 'skill_detector')
    assert isinstance(state.skill_detector, SkillDetector)
    assert hasattr(state, 'skill_composer')
    assert isinstance(state.skill_composer, SkillComposer)


def test_metaparams_skills(state):
    """MetaParams has skill thresholds."""
    d = state.meta_params.skill_detection_threshold
    s = state.meta_params.skill_similarity_threshold
    assert d.item() > 0  # softplus
    assert 0 < s.item() < 1  # sigmoid
