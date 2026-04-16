"""Write path: buffer observations from transformer hidden stream into cognitive state.

New observations are projected into belief space, matched against existing beliefs,
and buffered as write candidates. Actual updates happen in pass 2.

The write path does NOT commit changes during the forward pass — it produces
candidates that pass 2 evaluates with precision-weighted revision.

Key design: precision-gated updates. High-precision existing beliefs resist change.
Low-precision beliefs accept updates readily. This is the Kalman-like gain from
Memoria's belief_update.rs, implemented as tensor ops.

Reference: prototype-research/src/aif/belief_update.rs
Reference: DeltaNet (arxiv.org/abs/2310.18020) — error-correcting state updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, belief_is_active, EPSILON
from .read_path import BeliefCache


@dataclass
class WriteCandidate:
    """A buffered observation ready for pass 2 evaluation."""
    belief_vector: Tensor          # [D] cartesian (radius = observation precision)
    matched_slot: int              # index of matched existing belief, or -1 if new
    match_similarity: float        # cosine sim with matched belief (0 if new)
    source_position: int           # token position that generated this candidate
    source_layer: int              # which state interface layer produced this


def pack_candidates(candidates: list[WriteCandidate], *, device: torch.device | str) -> Tensor:
    """Pack candidates into a single tensor for distributed gather.

    Vectorized: stacks belief vectors and metadata in one shot.

    Args:
        candidates: list of WriteCandidate from forward pass.
        device: target device for the empty-case tensor (must match NCCL rank device).

    Returns:
        [N, D+3] tensor where last 3 cols are (matched_slot, match_similarity, source_layer).
        Returns empty [0, 0] tensor if no candidates.
    """
    if not candidates:
        return torch.zeros(0, 0, device=device)
    D = candidates[0].belief_vector.shape[0]
    device = candidates[0].belief_vector.device
    N = len(candidates)

    # Stack belief vectors (detach only here for distributed transport)
    beliefs = torch.stack([c.belief_vector.detach() for c in candidates])  # [N, D]

    # Build metadata columns
    meta = torch.tensor(
        [[c.matched_slot, c.match_similarity, c.source_layer] for c in candidates],
        dtype=beliefs.dtype, device=device,
    )  # [N, 3]

    return torch.cat([beliefs, meta], dim=-1)  # [N, D+3]


def unpack_candidates(packed: Tensor, belief_dim: int) -> list[WriteCandidate]:
    """Unpack tensor back into WriteCandidate list.

    Vectorized: slices tensor columns then builds objects.
    """
    if packed.numel() == 0:
        return []

    beliefs = packed[:, :belief_dim]               # [N, D]
    slots = packed[:, belief_dim].long()            # [N]
    sims = packed[:, belief_dim + 1]                # [N]
    layers = packed[:, belief_dim + 2].long()       # [N]

    # Move to CPU once for all .item() calls
    slots_cpu = slots.cpu()
    sims_cpu = sims.cpu()
    layers_cpu = layers.cpu()

    candidates = []
    for i in range(packed.shape[0]):
        candidates.append(WriteCandidate(
            belief_vector=beliefs[i],
            matched_slot=slots_cpu[i].item(),
            match_similarity=sims_cpu[i].item(),
            source_position=i,
            source_layer=layers_cpu[i].item(),
        ))
    return candidates


class WritePath(nn.Module):
    """Project hidden states into belief space and match against existing beliefs.

    Produces WriteCandidate objects for pass 2 to process.
    Does NOT modify cognitive state during forward pass.
    """

    def __init__(self, hidden_dim: int, belief_dim: int):
        """
        Args:
            hidden_dim: transformer hidden dimension
            belief_dim: belief representation dimension (D)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim

        # Project hidden → observation in belief space
        self.obs_proj = nn.Linear(hidden_dim, belief_dim, bias=False)

        # Write gate: learned binary decision "is this worth storing?"
        # Bias = 0.0 (maximum-entropy initialization).
        # sigmoid(0) = 0.5: the uniform prior over {write, don't write}.
        # Zero is the unique non-arbitrary init for a learned binary gate --
        # any nonzero value encodes an unjustified prior on write frequency.
        # The network learns to discriminate via gradients from the
        # free-energy objective; no handcrafted gate rate is needed.
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        nn.init.constant_(self.write_gate[-1].bias, 0.0)

        # Precision estimator (only used when gate is open)
        self.precision_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        hidden: Tensor,
        state: CognitiveState,
        layer_idx: int = 0,
        belief_cache: 'BeliefCache | None' = None,
    ) -> tuple[list[WriteCandidate], Tensor]:
        """Project hidden states and match against existing beliefs.

        Args:
            hidden: [B, T, hidden_dim] transformer hidden states
            state: the cognitive state to match against
            layer_idx: which state interface layer this is (for tracking)
            belief_cache: pre-computed active belief data (avoids redundant computation)

        Returns:
            candidates: list of WriteCandidate objects for pass 2
            obs_vectors: [B, T, D] observation projections (in computation graph, for L_fe)
        """
        B, T, H = hidden.shape
        D = self.belief_dim

        # Project to belief space: [B, T, D] — stays in graph for differentiable L_fe
        obs_vectors = self.obs_proj(hidden)

        # Write gate: should this position produce a candidate at all?
        gate = torch.sigmoid(self.write_gate(hidden))  # [B, T, 1]

        # Precision (only meaningful where gate is open)
        obs_precision = self.precision_head(hidden)     # [B, T, 1]
        gated_precision = gate * obs_precision           # [B, T, 1]

        # Scale observation vectors by gated precision (set radius)
        obs_angles = F.normalize(obs_vectors, dim=-1, eps=EPSILON)  # [B, T, D]
        obs_beliefs = obs_angles * gated_precision                   # [B, T, D]

        # Batch matching across all B*T positions in one matmul
        candidates = self._match_and_buffer_batched(
            obs_beliefs, state, layer_idx, B, T, belief_cache=belief_cache,
        )

        # Return precision-gated observations for L_fe (not raw obs_vectors).
        # This connects write_gate + precision_head to the computation graph
        # so DDP doesn't flag them as unused parameters.
        return candidates, obs_beliefs

    def _match_and_buffer_batched(
        self,
        obs_beliefs: Tensor,
        state: CognitiveState,
        layer_idx: int,
        B: int,
        T: int,
        belief_cache: 'BeliefCache | None' = None,
    ) -> list[WriteCandidate]:
        """Match all B*T observations against existing beliefs in one matmul.

        Args:
            obs_beliefs: [B, T, D] observation vectors (radius = precision)
            state: cognitive state
            layer_idx: layer index for tracking
            B: batch size (for position reconstruction)
            T: sequence length
            belief_cache: pre-computed active belief data (avoids redundant computation)

        Returns:
            List of WriteCandidates
        """
        D = obs_beliefs.shape[-1]
        device = obs_beliefs.device
        obs_flat = obs_beliefs.reshape(-1, D)  # [B*T, D]
        N = obs_flat.shape[0]

        match_threshold = state.match_threshold

        obs_radii = obs_flat.norm(dim=-1)  # [N]
        meaningful = obs_radii > 0.05

        # Use belief cache if available, otherwise compute from state
        has_active = False
        if belief_cache is not None:
            has_active = belief_cache.n_active > 0
            if has_active:
                active_indices = belief_cache.indices
                active_angles = belief_cache.angles
        else:
            active_mask = state.get_active_mask()
            has_active = active_mask.any()
            if has_active:
                active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
                active_beliefs = state.beliefs[active_indices]
                active_radii = active_beliefs.norm(dim=-1).clamp(min=EPSILON)
                active_angles = active_beliefs / active_radii.unsqueeze(-1)

        if has_active and meaningful.any():
            obs_angles = obs_flat / obs_radii.unsqueeze(-1).clamp(min=EPSILON)

            # One big matmul: [B*T, N_active]
            similarities = obs_angles @ active_angles.T
            best_sims, best_local = similarities.max(dim=-1)
            best_global = active_indices[best_local]

            matched = meaningful & (best_sims > match_threshold)
        else:
            best_sims = torch.zeros(N, device=device)
            best_global = torch.full((N,), -1, dtype=torch.long, device=device)
            matched = torch.zeros(N, dtype=torch.bool, device=device)

        meaningful_idx = meaningful.nonzero(as_tuple=False).squeeze(-1)
        if len(meaningful_idx) == 0:
            return []

        obs_for_candidates = obs_flat

        slots = torch.where(matched, best_global, torch.tensor(-1, dtype=torch.long, device=device))
        sims_out = torch.where(matched, best_sims, torch.zeros_like(best_sims))

        m_idx = meaningful_idx.cpu()
        m_slots = slots[meaningful_idx].cpu()
        m_sims = sims_out[meaningful_idx].cpu()

        candidates = []
        for i in range(len(m_idx)):
            flat_pos = m_idx[i].item()
            t = flat_pos % T  # position within sequence
            candidates.append(WriteCandidate(
                belief_vector=obs_for_candidates[flat_pos].detach(),
                matched_slot=m_slots[i].item(),
                match_similarity=m_sims[i].item(),
                source_position=t,
                source_layer=layer_idx,
            ))

        return candidates
