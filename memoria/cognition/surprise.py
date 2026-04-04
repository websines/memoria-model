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
from ..core.polar import EPSILON
from ..interface.write_path import WriteCandidate


@dataclass
class SurpriseResult:
    """Result of surprise computation for a single write candidate."""
    slot: int                  # belief slot index (-1 if new)
    surprise: float            # prediction_error × observation_precision
    gain: float                # Kalman-like gain: how much to shift (MESU-modulated)
    should_reconsolidate: bool # gain > threshold → full rewrite
    observation: Tensor        # the observation vector
    is_new: bool               # no matching belief found
    mesu_gain_raw: float = 0.0 # pre-MESU gain (for diagnostics)


def compute_surprise_batch(
    candidates: list[WriteCandidate],
    state: CognitiveState,
) -> list[SurpriseResult]:
    """Compute surprise for all write candidates against existing beliefs.

    Vectorized: batches all matched candidates into a single tensor op,
    avoiding per-candidate Python loops and CPU-GPU syncs.

    Args:
        candidates: write candidates from state interface layers
        state: current cognitive state

    Returns:
        List of SurpriseResult, one per candidate
    """
    if not candidates:
        return []

    N = len(candidates)
    device = state.beliefs.device
    reconsolidation_threshold = state.reconsolidation_threshold

    # Stack all candidate data into tensors
    obs_all = torch.stack([c.belief_vector for c in candidates]).to(device)  # [N, D]
    slots = torch.tensor([c.matched_slot for c in candidates], device=device)  # [N]

    matched_mask = slots >= 0
    n_matched = matched_mask.sum().item()

    # Pre-allocate result arrays on CPU
    surprise_out = torch.zeros(N)
    gain_out = torch.full((N,), 1.0)
    gain_raw_out = torch.full((N,), 1.0)  # A2: pre-MESU gain for diagnostics
    recon_out = torch.ones(N, dtype=torch.bool)  # default True (for new)
    is_new_out = torch.ones(N, dtype=torch.bool)  # default True
    slot_out = slots.cpu().clone()

    if n_matched > 0:
        matched_idx = matched_mask.nonzero(as_tuple=False).squeeze(-1)
        matched_slots = slots[matched_idx]  # [M]
        matched_obs = obs_all[matched_idx]  # [M, D]

        existing = state.beliefs.data[matched_slots]  # [M, D]
        existing_radii = existing.norm(dim=-1)  # [M]
        obs_radii = matched_obs.norm(dim=-1).clamp(min=EPSILON)  # [M]

        # Deallocated slots (radius ≈ 0) → treat as new
        deallocated = existing_radii < EPSILON
        valid = ~deallocated
        n_valid = valid.sum().item()

        # Handle deallocated slots → mark as new
        if deallocated.any():
            dealloc_global = matched_idx[deallocated].cpu()
            slot_out[dealloc_global] = -1
            surprise_out[dealloc_global] = obs_radii[deallocated].cpu()
            # gain and recon already defaulted to 1.0 and True

        if n_valid > 0:
            v_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            v_existing = existing[v_idx]
            v_existing_radii = existing_radii[v_idx].clamp(min=EPSILON)
            v_obs = matched_obs[v_idx]
            v_obs_radii = obs_radii[v_idx]

            # Angular distance (vectorized)
            existing_angles = v_existing / v_existing_radii.unsqueeze(-1)
            obs_angles = v_obs / v_obs_radii.unsqueeze(-1)
            cos_sim = (existing_angles * obs_angles).sum(dim=-1).clamp(-1.0, 1.0)
            pred_error = 1.0 - cos_sim  # [V], range [0, 2]

            # Kalman-like gain (base)
            total_precision = v_existing_radii + v_obs_radii
            gain_raw = v_obs_radii / total_precision  # [V]

            # ── A2: MESU — modulate gain by precision variance ──
            # High variance → belief is uncertain → amplify gain (more willing to change)
            # Low variance → belief is confident → dampen gain (resists updates)
            # Reference: MESU (arXiv:2312.10153), Palimpsa (arXiv:2602.09075)
            v_matched_slots = matched_slots[v_idx]
            precision_var = state.belief_precision_var[v_matched_slots]
            gain_boost = state.meta_params.mesu_gain_boost.item()
            mesu_factor = (1.0 + precision_var * gain_boost).clamp(max=3.0)
            gain = (gain_raw * mesu_factor).clamp(max=1.0)

            # Surprise
            surprise = pred_error * v_obs_radii  # [V]

            # Write back to output arrays
            global_idx = matched_idx[v_idx].cpu()
            surprise_out[global_idx] = surprise.cpu()
            gain_out[global_idx] = gain.cpu()
            recon_out[global_idx] = (gain > reconsolidation_threshold).cpu()
            is_new_out[global_idx] = False

            # Store raw gain for diagnostics
            gain_raw_out[global_idx] = gain_raw.cpu()

    # Build result list (cheap — just reads from pre-computed tensors)
    results = []
    for i in range(N):
        results.append(SurpriseResult(
            slot=slot_out[i].item(),
            surprise=surprise_out[i].item(),
            gain=gain_out[i].item(),
            should_reconsolidate=recon_out[i].item(),
            observation=obs_all[i],
            is_new=is_new_out[i].item(),
            mesu_gain_raw=gain_raw_out[i].item(),
        ))

    return results
