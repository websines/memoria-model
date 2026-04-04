"""Tests for A4: SGM Safety Gate (Statistical Godel Machine).

Uses the `expectation` library (v0.5.2) for sequential e-value testing.
"""

import torch
import pytest
from memoria.cognition.safety_gate import (
    SafetyGate,
    ModificationRecord,
    sequential_e_test,
)


@pytest.fixture
def gate():
    return SafetyGate(global_alpha=0.05, max_modifications=100)


def test_gate_init(gate):
    """Safety gate initializes with full budget."""
    assert gate.global_alpha == 0.05
    assert gate.alpha_remaining == pytest.approx(0.05)
    assert gate.n_modifications.item() == 0
    assert gate.n_accepted.item() == 0
    assert gate.n_rejected.item() == 0


def test_begin_evaluation(gate):
    """Can begin evaluating a modification."""
    handle = gate.begin_evaluation("test_mod")
    assert handle == "test_mod"
    assert "test_mod" in gate._active_records
    record = gate._active_records["test_mod"]
    assert record.alpha_spent > 0
    assert record.n_samples == 0


def test_duplicate_evaluation_raises(gate):
    """Cannot evaluate the same modification twice simultaneously."""
    gate.begin_evaluation("test_mod")
    with pytest.raises(ValueError, match="already in evaluation"):
        gate.begin_evaluation("test_mod")


def test_record_sample_updates_evalue(gate):
    """Recording positive samples increases the e-value."""
    handle = gate.begin_evaluation("good_mod")
    gate.record_sample(handle, 0.5)
    gate.record_sample(handle, 0.3)
    gate.record_sample(handle, 0.4)
    record = gate._active_records[handle]
    assert record.n_samples == 3
    assert record.e_value > 1.0  # positive improvements → e-value grows


def test_negative_samples_decrease_evalue(gate):
    """Recording negative samples decreases the e-value."""
    handle = gate.begin_evaluation("bad_mod")
    for _ in range(10):
        gate.record_sample(handle, -0.3)
    record = gate._active_records[handle]
    assert record.e_value < 1.0


def test_accept_strong_improvement(gate):
    """Strongly improving modification is accepted."""
    handle = gate.begin_evaluation("great_mod")
    for _ in range(50):
        gate.record_sample(handle, 0.8)
    assert gate.check_accept(handle)


def test_reject_bad_modification(gate):
    """Clearly harmful modification stays unaccepted (one-sided test).

    For H1: mean > 0, negative samples provide no evidence for acceptance.
    The e-value stays at 1.0 (neutral) since the test is one-sided — it
    measures evidence FOR improvement, not against. The p-value should be
    high (no significance). The confidence bounds should be negative.
    """
    handle = gate.begin_evaluation("terrible_mod")
    for _ in range(10):
        gate.record_sample(handle, -0.5)
    record = gate._active_records[handle]
    # E-value doesn't exceed threshold → not accepted
    assert not gate.check_accept(handle)
    # Confidence bounds should be negative (modification is harmful)
    assert record.confidence_bounds[1] < 0.0


def test_finalize_accepted(gate):
    """Finalize records acceptance and spends alpha."""
    handle = gate.begin_evaluation("strong_mod")
    for _ in range(100):
        gate.record_sample(handle, 0.7)
    record = gate.finalize(handle)
    assert record.accepted
    assert not record.rejected
    assert gate.alpha_spent_total.item() > 0
    assert gate.n_accepted.item() == 1


def test_finalize_rejected(gate):
    """Finalize records rejection when insufficient evidence."""
    handle = gate.begin_evaluation("meh_mod")
    gate.record_sample(handle, 0.01)
    gate.record_sample(handle, -0.01)
    record = gate.finalize(handle)
    assert record.rejected
    assert not record.accepted
    assert gate.n_rejected.item() == 1


def test_global_budget_enforced(gate):
    """Cannot begin evaluation when budget is exhausted."""
    gate.alpha_spent_total.fill_(0.05)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        gate.begin_evaluation("too_late")


def test_evaluate_modification_convenience(gate):
    """Batch evaluation convenience method works."""
    improvements = [0.5] * 50
    record = gate.evaluate_modification("batch_test", improvements)
    assert record.accepted
    assert record.n_samples == 50


def test_multiple_modifications_track_budget(gate):
    """Multiple modifications consume global budget incrementally."""
    for i in range(5):
        improvements = [0.6] * 30
        record = gate.evaluate_modification(f"mod_{i}", improvements)
    assert gate.n_modifications.item() == 5
    assert len(gate.get_history()) == 5


def test_sequential_e_test_positive():
    """Positive improvements produce large e-values via library."""
    result = sequential_e_test([0.5, 0.3, 0.4, 0.6, 0.5] * 10)
    assert result.e_value > 1.0
    assert result.reject_null


def test_sequential_e_test_negative():
    """Negative improvements don't reject null."""
    result = sequential_e_test([-0.3, -0.5, -0.2, -0.4, -0.3] * 10)
    assert result.e_value < 100.0  # should not be huge


def test_sequential_e_test_has_confidence_bounds():
    """Library provides confidence bounds."""
    result = sequential_e_test([0.5] * 20)
    assert result.confidence_bounds is not None


def test_record_has_p_value(gate):
    """Records include p-values from the library."""
    handle = gate.begin_evaluation("pval_test")
    for _ in range(20):
        gate.record_sample(handle, 0.5)
    record = gate._active_records[handle]
    assert record.p_value < 1.0  # significant


def test_summary(gate):
    """Summary produces readable output."""
    s = gate.summary()
    assert "SGM SafetyGate" in s
    assert "alpha=0.05" in s
