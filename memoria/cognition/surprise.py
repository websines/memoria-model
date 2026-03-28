"""Surprise computation: how much do new observations violate existing beliefs?

Surprise = prediction_error × observation_precision

High-precision observation contradicting low-precision belief → high surprise → reconsolidate
Low-precision observation contradicting high-precision belief → low surprise → incremental update

Ported from: prototype-research/src/dynamics/surprise.rs
Reference: Titans (arxiv.org/abs/2501.00663) — surprise-driven memorization
"""

import torch
from torch import Tensor
from dataclasses import dataclass

from ..core.state import CognitiveState
from ..core.polar import angular_distance, angular_similarity, EPSILON
from ..interface.write_path import WriteCandidate


@dataclass
class SurpriseResult:
    """Result of surprise computation for a single write candidate."""
    slot: int                  # belief slot index (-1 if new)
    surprise: float            # prediction_error × observation_precision
    gain: float                # Kalman-like gain: how much to shift
    should_reconsolidate: bool # gain > threshold → full rewrite
    observation: Tensor        # the observation vector
    is_new: bool               # no matching belief found


def compute_surprise_batch(
    candidates: list[WriteCandidate],
    state: CognitiveState,
) -> list[SurpriseResult]:
    """Compute surprise for all write candidates against existing beliefs.

    For matched candidates: surprise = angular_distance × obs_precision
    For new candidates: surprise = obs_precision (everything is new → maximum prediction error)

    Args:
        candidates: write candidates from state interface layers
        state: current cognitive state

    Returns:
        List of SurpriseResult, one per candidate
    """
    if not candidates:
        return []

    reconsolidation_threshold = state.reconsolidation_threshold
    results = []

    for c in candidates:
        obs_radius = c.belief_vector.norm().item()

        if c.matched_slot >= 0:
            # Matched an existing belief — compute surprise from disagreement
            existing = state.beliefs.data[c.matched_slot]
            existing_radius = existing.norm().item()

            if existing_radius < EPSILON:
                # Slot was deallocated between write and now — treat as new
                results.append(SurpriseResult(
                    slot=-1, surprise=obs_radius, gain=1.0,
                    should_reconsolidate=True, observation=c.belief_vector,
                    is_new=True,
                ))
                continue

            # Angular distance between observation and existing belief
            existing_angle = existing / max(existing_radius, EPSILON)
            obs_angle = c.belief_vector / max(obs_radius, EPSILON)
            pred_error = angular_distance(
                existing_angle.unsqueeze(0), obs_angle.unsqueeze(0)
            ).item()
            # pred_error ∈ [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite

            # Kalman-like gain
            # belief_precision = existing_radius (in polar form, radius IS precision)
            # obs_precision = obs_radius
            belief_precision = existing_radius
            obs_precision = obs_radius
            total_precision = belief_precision + obs_precision
            gain = obs_precision / total_precision if total_precision > EPSILON else 0.5

            # Surprise = prediction error × observation precision
            surprise = pred_error * obs_precision

            results.append(SurpriseResult(
                slot=c.matched_slot,
                surprise=surprise,
                gain=gain,
                should_reconsolidate=gain > reconsolidation_threshold,
                observation=c.belief_vector,
                is_new=False,
            ))
        else:
            # New observation — no matching belief
            # Everything is surprising (prediction error = 1.0)
            results.append(SurpriseResult(
                slot=-1,
                surprise=obs_radius,  # surprise = 1.0 × obs_precision
                gain=1.0,             # no prior → full acceptance
                should_reconsolidate=True,
                observation=c.belief_vector,
                is_new=True,
            ))

    return results
