"""A4: Statistical Godel Machine (SGM) safety gate for self-modification.

Before deploying any self-modified update rule (Phase C: SRWM, meta-learned
Pass 2, structural plasticity), run a statistical confidence test. Only accept
modifications certified superior at a chosen confidence level.

Uses the `expectation` library (v0.5.2, Rust backend) for proper sequential
e-value testing with adaptive lambda strategies, empirical variance estimation,
and confidence sequences. We handle the global error budget and modification
lifecycle; the library handles the statistics.

Reference: Statistical Godel Machine (arXiv:2510.10232)
Reference: E-values: Calibration, Combination, and Applications (Vovk & Wang, 2021)
Reference: expectation library (github.com/jakorostami/expectation)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from expectation.seqtest.sequential_e_testing import (
    SequentialTesting,
    SequentialTestResult,
)
from expectation.modules.hypothesistesting import EValueConfig


@dataclass
class ModificationRecord:
    """Record of a single self-modification attempt."""
    name: str                           # human-readable description
    e_value: float = 1.0                # current e-value from sequential test
    n_samples: int = 0                  # number of evaluation samples so far
    alpha_spent: float = 0.0            # local alpha budget allocated
    accepted: bool = False              # whether modification was accepted
    rejected: bool = False              # whether modification was rejected
    confidence_bounds: tuple[float, float] = (0.0, 0.0)  # from library
    p_value: float = 1.0               # sequential p-value


class SafetyGate(nn.Module):
    """SGM safety gate for bounding self-modification risk.

    Manages a global error budget and provides sequential e-value testing for
    candidate modifications. Each modification attempt consumes a portion of
    the error budget via a harmonic alpha-spending function.

    The actual sequential testing is delegated to `expectation.SequentialTesting`
    which implements proper e-processes with adaptive lambda strategies and
    empirical variance estimation.

    Usage:
        gate = SafetyGate(global_alpha=0.05)
        handle = gate.begin_evaluation("new_consolidation_rule")
        for sample in evaluation_samples:
            improvement = old_fe - new_fe  # positive = improvement
            gate.record_sample(handle, improvement)
            if gate.check_accept(handle):
                break  # modification certified
            if gate.check_reject(handle):
                break  # evidence against
        result = gate.finalize(handle)
    """

    def __init__(self, global_alpha: float = 0.05, max_modifications: int = 100):
        super().__init__()
        self.global_alpha = global_alpha
        self.max_modifications = max_modifications

        # Track budget consumption
        self.register_buffer('alpha_spent_total', torch.tensor(0.0))
        self.register_buffer('n_modifications', torch.tensor(0, dtype=torch.long))
        self.register_buffer('n_accepted', torch.tensor(0, dtype=torch.long))
        self.register_buffer('n_rejected', torch.tensor(0, dtype=torch.long))

        # Active sequential tests (name → SequentialTesting instance)
        self._active_tests: dict[str, SequentialTesting] = {}
        self._active_records: dict[str, ModificationRecord] = {}
        self._history: list[ModificationRecord] = []

    @property
    def alpha_remaining(self) -> float:
        """Remaining global error budget."""
        return max(0.0, self.global_alpha - self.alpha_spent_total.item())

    def _alpha_spending(self, k: int) -> float:
        """Harmonic alpha spending for the k-th modification.

        alpha_k = alpha / (k * H_K) where H_K = sum(1/i for i=1..K).

        Gives more budget to early modifications while guaranteeing the total
        never exceeds global_alpha. Optimal for sequential testing (Shafer & Vovk).
        """
        K = self.max_modifications
        if K <= 1:
            return self.alpha_remaining

        H_K = sum(1.0 / i for i in range(1, K + 1))
        alpha_k = self.global_alpha / (k * H_K)
        return min(alpha_k, self.alpha_remaining)

    def begin_evaluation(self, name: str) -> str:
        """Start evaluating a candidate modification.

        Creates a SequentialTesting instance from the `expectation` library
        to handle the e-value computation with proper adaptive strategies.

        Args:
            name: human-readable name for this modification

        Returns:
            handle (name) for subsequent calls

        Raises:
            RuntimeError if global budget is exhausted
        """
        if self.alpha_remaining <= 0:
            raise RuntimeError(
                f"SGM global error budget exhausted "
                f"(spent {self.alpha_spent_total.item():.4f} / {self.global_alpha}). "
                f"No more modifications allowed."
            )

        if name in self._active_records:
            raise ValueError(f"Modification '{name}' already in evaluation")

        k = self.n_modifications.item() + 1
        alpha_local = self._alpha_spending(k)

        # Create sequential test: H0: mean improvement <= 0, H1: mean > 0
        # The library handles adaptive lambda, empirical variance, e-processes.
        # allow_infinite=True: strong evidence can produce very large e-values.
        test = SequentialTesting(
            test_type='mean',
            null_value=0.0,
            alternative='greater',
            use_empirical_variance=True,
            config=EValueConfig(allow_infinite=True),
        )

        record = ModificationRecord(name=name, alpha_spent=alpha_local)
        self._active_tests[name] = test
        self._active_records[name] = record
        self.n_modifications += 1

        return name

    def record_sample(self, handle: str, improvement: float):
        """Record one evaluation sample for a modification.

        Delegates to the `expectation` library's SequentialTesting.update().
        The library computes e-values using proper adaptive strategies.

        Args:
            handle: modification handle from begin_evaluation
            improvement: metric_old - metric_new (positive = new is better)
        """
        test = self._active_tests[handle]
        record = self._active_records[handle]
        record.n_samples += 1

        # The library handles all the e-value math
        result: SequentialTestResult = test.update([improvement])
        record.e_value = result.e_value
        record.p_value = result.p_value if result.p_value is not None else 1.0
        if result.confidence_bounds is not None:
            record.confidence_bounds = result.confidence_bounds

    def check_accept(self, handle: str) -> bool:
        """Check if the modification should be accepted.

        Returns True if the e-value exceeds 1/alpha_local.
        """
        record = self._active_records[handle]
        if record.alpha_spent <= 0:
            return False
        threshold = 1.0 / record.alpha_spent
        return record.e_value >= threshold

    def check_reject(self, handle: str, rejection_threshold: float = 0.01) -> bool:
        """Check if the modification should be rejected early.

        Rejects when the e-value is very low (strong evidence against improvement).
        """
        record = self._active_records[handle]
        if record.n_samples < 5:
            return False
        return record.e_value < rejection_threshold

    def finalize(self, handle: str) -> ModificationRecord:
        """Finalize evaluation of a modification.

        Returns the final ModificationRecord with accept/reject decision.
        """
        record = self._active_records.pop(handle)
        self._active_tests.pop(handle)

        threshold = 1.0 / record.alpha_spent if record.alpha_spent > 0 else float('inf')
        if record.e_value >= threshold:
            record.accepted = True
            self.n_accepted += 1
            self.alpha_spent_total += record.alpha_spent
        else:
            record.rejected = True
            self.n_rejected += 1

        self._history.append(record)
        return record

    def evaluate_modification(
        self,
        name: str,
        improvements: list[float],
    ) -> ModificationRecord:
        """Convenience: evaluate a batch of samples at once.

        Args:
            name: modification name
            improvements: list of metric_old - metric_new values

        Returns:
            ModificationRecord with accept/reject decision
        """
        handle = self.begin_evaluation(name)
        for x in improvements:
            self.record_sample(handle, x)
        return self.finalize(handle)

    def get_history(self) -> list[ModificationRecord]:
        """Get all finalized modification records."""
        return list(self._history)

    def summary(self) -> str:
        """Human-readable summary of safety gate state."""
        return (
            f"SGM SafetyGate: "
            f"alpha={self.global_alpha}, "
            f"spent={self.alpha_spent_total.item():.4f}, "
            f"remaining={self.alpha_remaining:.4f}, "
            f"modifications={self.n_modifications.item()} "
            f"(accepted={self.n_accepted.item()}, rejected={self.n_rejected.item()})"
        )


def sequential_e_test(
    improvements: list[float],
    alternative: str = 'greater',
) -> SequentialTestResult:
    """Standalone sequential e-value test using the `expectation` library.

    Convenience function for one-shot testing without the full SafetyGate
    lifecycle. Returns the library's SequentialTestResult with e-value,
    p-value, confidence bounds, and reject_null decision.

    Args:
        improvements: list of improvement values (positive = better)
        alternative: 'greater', 'less', or 'two_sided'

    Returns:
        SequentialTestResult from the expectation library
    """
    test = SequentialTesting(
        test_type='mean',
        null_value=0.0,
        alternative=alternative,
        use_empirical_variance=True,
        config=EValueConfig(allow_infinite=True),
    )
    return test.update(improvements)
