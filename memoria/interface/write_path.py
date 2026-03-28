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


@dataclass
class WriteCandidate:
    """A buffered observation ready for pass 2 evaluation."""
    belief_vector: Tensor          # [D] cartesian (radius = observation precision)
    matched_slot: int              # index of matched existing belief, or -1 if new
    match_similarity: float        # cosine sim with matched belief (0 if new)
    source_position: int           # token position that generated this candidate
    source_layer: int              # which state interface layer produced this


def pack_candidates(candidates: list[WriteCandidate]) -> Tensor:
    """Pack candidates into a single tensor for distributed gather.

    Returns:
        [N, D+3] tensor where last 3 cols are (matched_slot, match_similarity, source_layer).
        Returns empty [0, 0] tensor if no candidates.
    """
    if not candidates:
        return torch.zeros(0, 0)
    D = candidates[0].belief_vector.shape[0]
    packed = torch.zeros(len(candidates), D + 3, device=candidates[0].belief_vector.device)
    for i, c in enumerate(candidates):
        packed[i, :D] = c.belief_vector
        packed[i, D] = c.matched_slot
        packed[i, D + 1] = c.match_similarity
        packed[i, D + 2] = c.source_layer
    return packed


def unpack_candidates(packed: Tensor, belief_dim: int) -> list[WriteCandidate]:
    """Unpack tensor back into WriteCandidate list."""
    if packed.numel() == 0:
        return []
    candidates = []
    for i in range(packed.shape[0]):
        candidates.append(WriteCandidate(
            belief_vector=packed[i, :belief_dim],
            matched_slot=int(packed[i, belief_dim].item()),
            match_similarity=packed[i, belief_dim + 1].item(),
            source_position=i,  # position lost in packing, use index
            source_layer=int(packed[i, belief_dim + 2].item()),
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
        # Initialized with negative bias so gate starts mostly closed (~12% open),
        # preventing state flooding before the model learns what's useful.
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Init gate bias negative so sigmoid starts near 0
        nn.init.constant_(self.write_gate[-1].bias, -2.0)

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
    ) -> list[WriteCandidate]:
        """Project hidden states and match against existing beliefs.

        We produce one write candidate per token position (taking the mean over batch).
        In practice, not every token generates a meaningful observation — pass 2
        filters by surprise magnitude.

        Args:
            hidden: [B, T, hidden_dim] transformer hidden states
            state: the cognitive state to match against
            layer_idx: which state interface layer this is (for tracking)

        Returns:
            List of WriteCandidate objects for pass 2
        """
        B, T, H = hidden.shape

        # Project to belief space: [B, T, D]
        obs_vectors = self.obs_proj(hidden)

        # Write gate: should this position produce a candidate at all?
        gate = torch.sigmoid(self.write_gate(hidden))  # [B, T, 1]

        # Precision (only meaningful where gate is open)
        obs_precision = self.precision_head(hidden)     # [B, T, 1]
        gated_precision = gate * obs_precision           # [B, T, 1]

        # Mean over batch for state updates (state is shared across batch)
        obs_mean = obs_vectors.mean(dim=0)           # [T, D]
        prec_mean = gated_precision.mean(dim=0)      # [T, 1]

        # Scale observation vectors by gated precision (set radius)
        obs_angles = F.normalize(obs_mean, dim=-1, eps=EPSILON)  # [T, D]
        obs_beliefs = obs_angles * prec_mean                      # [T, D] (radius = gated precision)

        # Match against existing beliefs
        candidates = self._match_and_buffer(obs_beliefs, state, layer_idx)

        return candidates

    def _match_and_buffer(
        self,
        observations: Tensor,
        state: CognitiveState,
        layer_idx: int,
    ) -> list[WriteCandidate]:
        """Match observations against existing beliefs, produce candidates.

        Args:
            observations: [T, D] observation vectors (radius = precision)
            state: cognitive state
            layer_idx: layer index for tracking

        Returns:
            List of WriteCandidates
        """
        T, D = observations.shape
        candidates = []

        active_mask = state.get_active_mask()
        match_threshold = state.match_threshold

        if active_mask.any():
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
            active_beliefs = state.beliefs.data[active_indices]  # [N_active, D]
            active_radii = active_beliefs.norm(dim=-1).clamp(min=EPSILON)
            active_angles = active_beliefs / active_radii.unsqueeze(-1)  # [N_active, D]

            # Observation angles
            obs_radii = observations.norm(dim=-1).clamp(min=EPSILON)
            obs_angles = observations / obs_radii.unsqueeze(-1)  # [T, D]

            # Cosine similarity: [T, N_active]
            similarities = obs_angles @ active_angles.T

            # Best match per observation
            best_sims, best_local = similarities.max(dim=-1)  # [T], [T]
            best_global = active_indices[best_local]           # map to global slot indices
        else:
            best_sims = torch.zeros(T, device=observations.device)
            best_global = torch.full((T,), -1, dtype=torch.long, device=observations.device)

        # Only keep observations where the gate produced meaningful precision
        obs_radii = observations.norm(dim=-1)
        meaningful = obs_radii > 0.05  # gate-filtered: higher threshold since gate does heavy lifting

        for t in range(T):
            if not meaningful[t]:
                continue

            sim = best_sims[t].item() if active_mask.any() else 0.0

            if sim > match_threshold:
                # Match found → update candidate
                candidates.append(WriteCandidate(
                    belief_vector=observations[t].detach(),
                    matched_slot=best_global[t].item(),
                    match_similarity=sim,
                    source_position=t,
                    source_layer=layer_idx,
                ))
            else:
                # No match → new belief candidate
                candidates.append(WriteCandidate(
                    belief_vector=observations[t].detach(),
                    matched_slot=-1,
                    match_similarity=0.0,
                    source_position=t,
                    source_layer=layer_idx,
                ))

        return candidates
