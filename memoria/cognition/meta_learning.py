"""Meta-learning: β computation and SPSA self-tuning of cognitive parameters.

β = H / (|E| + H + ε) — computed from actual state, not a hyperparameter.

SPSA (Simultaneous Perturbation Stochastic Approximation) tunes cognitive
parameters by minimizing free energy over multi-step evaluation windows:
- Apply +Δ perturbation, let Pass 2 run for N steps, measure mean F(+Δ)
- Apply -Δ perturbation, let Pass 2 run for N steps, measure mean F(-Δ)
- Estimate gradient from the two means, update parameters

This gives the perturbed thresholds time to causally affect the state via
Pass 2 operations, unlike the original single-step approach where F(+Δ)
and F(-Δ) measured the same state.

Ported from: prototype-research/src/dynamics/meta_learning.rs
"""

import torch
from torch import Tensor

from ..core.state import CognitiveState
from ..core.free_energy import compute_free_energy
from ..core.polar import EPSILON


def compute_beta(state: CognitiveState, temperature: float = 5.0) -> float:
    """Compute β from current state (already done inside compute_free_energy).

    This is a convenience wrapper that returns just β.
    β is also written to state.meta[0] as a side effect.

    Args:
        state: cognitive state
        temperature: for energy computation

    Returns:
        β ∈ [0, 1]
    """
    result = compute_free_energy(state, temperature)
    return result['beta'].item()


class SPSAController:
    """Multi-step SPSA controller for tuning cognitive meta-parameters.

    Instead of perturbing and evaluating in the same step (which measures the
    same state for both +Δ and -Δ), this controller spans multiple steps:

    1. idle: wait for interval
    2. eval_plus: apply +Δ, accumulate free energy over eval_window steps
    3. eval_minus: apply -Δ, accumulate free energy over eval_window steps
    4. compute gradient from mean F(+Δ) vs mean F(-Δ), update params, return to idle

    Tunes: reconsolidation_threshold (meta[4]), match_threshold (meta[5]),
    goal_dedup_threshold (meta[6]).
    """

    TUNABLE_START = 4
    # Clamp ranges for each tunable parameter
    CLAMP_RANGES = [
        (0.1, 0.9),    # reconsolidation_threshold
        (0.3, 0.95),   # match_threshold
        (0.2, 0.9),    # goal_dedup_threshold
        (0.0, 1.0),    # generic fallback
    ]

    def __init__(
        self,
        interval: int = 100,
        eval_window: int = 10,
        perturbation_scale: float = 0.01,
        step_size: float = 0.001,
    ):
        self.interval = interval
        self.eval_window = eval_window
        self.perturbation_scale = perturbation_scale
        self.step_size = step_size

        self._phase = 'idle'          # 'idle', 'eval_plus', 'eval_minus'
        self._original_values = None  # saved before perturbation
        self._delta = None            # perturbation direction
        self._steps_in_phase = 0
        self._fe_accumulator = 0.0
        self._fe_plus_mean = 0.0
        self._tunable_end = 0

    @property
    def phase(self) -> str:
        return self._phase

    def step(self, state: CognitiveState, current_step: int, temperature: float = 5.0) -> bool:
        """Called every step from Pass 2. Returns True if parameters were updated.

        Args:
            state: cognitive state (meta params may be modified)
            current_step: current training step
            temperature: for free energy computation

        Returns:
            True if SPSA completed a full cycle and updated parameters this step
        """
        if self._phase == 'idle':
            if current_step > 0 and current_step % self.interval == 0:
                self._start_plus_phase(state)
            return False

        # Accumulate free energy observation
        fe = compute_free_energy(state, temperature)['free_energy'].item()
        self._fe_accumulator += fe
        self._steps_in_phase += 1

        if self._phase == 'eval_plus':
            if self._steps_in_phase >= self.eval_window:
                self._fe_plus_mean = self._fe_accumulator / self.eval_window
                self._start_minus_phase(state)
            return False

        elif self._phase == 'eval_minus':
            if self._steps_in_phase >= self.eval_window:
                fe_minus_mean = self._fe_accumulator / self.eval_window
                self._apply_update(state, fe_minus_mean)
                return True
            return False

        return False

    def _start_plus_phase(self, state: CognitiveState):
        """Begin +Δ evaluation: save originals, apply positive perturbation."""
        self._tunable_end = min(self.TUNABLE_START + 4, state.config.meta_dim)
        n = self._tunable_end - self.TUNABLE_START
        if n <= 0:
            return

        device = state.meta.device
        with torch.no_grad():
            self._original_values = state.meta.data[self.TUNABLE_START:self._tunable_end].clone()
            self._delta = torch.where(
                torch.rand(n, device=device) > 0.5,
                torch.ones(n, device=device),
                -torch.ones(n, device=device),
            ) * self.perturbation_scale

            state.meta.data[self.TUNABLE_START:self._tunable_end] = self._original_values + self._delta

        self._phase = 'eval_plus'
        self._steps_in_phase = 0
        self._fe_accumulator = 0.0

    def _start_minus_phase(self, state: CognitiveState):
        """Switch to -Δ evaluation: apply negative perturbation."""
        with torch.no_grad():
            state.meta.data[self.TUNABLE_START:self._tunable_end] = self._original_values - self._delta

        self._phase = 'eval_minus'
        self._steps_in_phase = 0
        self._fe_accumulator = 0.0

    def _apply_update(self, state: CognitiveState, fe_minus_mean: float):
        """Compute gradient from mean F(+Δ) vs mean F(-Δ), update and clamp."""
        with torch.no_grad():
            gradient = (self._fe_plus_mean - fe_minus_mean) / (2.0 * self._delta + EPSILON)
            new_values = self._original_values - self.step_size * gradient

            # Clamp each parameter to its valid range
            n = self._tunable_end - self.TUNABLE_START
            for i in range(n):
                lo, hi = self.CLAMP_RANGES[min(i, len(self.CLAMP_RANGES) - 1)]
                new_values[i] = new_values[i].clamp(lo, hi)

            state.meta.data[self.TUNABLE_START:self._tunable_end] = new_values

        # Reset to idle
        self._phase = 'idle'
        self._original_values = None
        self._delta = None

    def state_dict(self) -> dict:
        """Serialize controller state for checkpointing."""
        return {
            'phase': self._phase,
            'original_values': self._original_values,
            'delta': self._delta,
            'steps_in_phase': self._steps_in_phase,
            'fe_accumulator': self._fe_accumulator,
            'fe_plus_mean': self._fe_plus_mean,
            'tunable_end': self._tunable_end,
        }

    def load_state_dict(self, d: dict):
        """Restore controller state from checkpoint."""
        self._phase = d.get('phase', 'idle')
        self._original_values = d.get('original_values')
        self._delta = d.get('delta')
        self._steps_in_phase = d.get('steps_in_phase', 0)
        self._fe_accumulator = d.get('fe_accumulator', 0.0)
        self._fe_plus_mean = d.get('fe_plus_mean', 0.0)
        self._tunable_end = d.get('tunable_end', 0)


def apply_sequence_boundary_decay(state: CognitiveState, decay_factor: float | None = None):
    """Apply exponential decay to belief precision at sequence boundaries.

    Vectorized: applies decay to all active non-immutable beliefs in one op.
    Beliefs not reinforced fade over ~20 sequences (0.95^20 ≈ 0.36).

    Args:
        state: cognitive state
        decay_factor: multiplicative decay per sequence boundary
            (defaults to state.meta_params.precision_decay_factor)
    """
    if decay_factor is None:
        decay_factor = state.meta_params.precision_decay_factor.item()

    with torch.no_grad():
        active = state.get_active_mask()
        mutable = active & ~state.immutable_beliefs

        if mutable.any():
            # Scale all mutable belief vectors by decay_factor (shrinks radius, preserves angle)
            state.beliefs.data[mutable] *= decay_factor

            # Zero out beliefs that decayed below epsilon
            decayed_radii = state.beliefs.data[mutable].norm(dim=-1)
            dead = decayed_radii < EPSILON
            if dead.any():
                mutable_indices = mutable.nonzero(as_tuple=False).squeeze(-1)
                state.beliefs.data[mutable_indices[dead]] = 0.0

        # Increment consolidation timer
        state.meta.data[2] += 1.0
