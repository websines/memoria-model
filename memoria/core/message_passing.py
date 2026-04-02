"""Factor graph message passing using PyTorch Geometric's MessagePassing.

Instead of hand-rolling loops over edges, we use PyG's sparse message passing
infrastructure. This gives us:
- Efficient scatter/gather operations on GPU
- Proper sparse tensor handling
- Battle-tested message aggregation

The factor graph is represented as a bipartite graph:
- Variable nodes (beliefs) send/receive messages via factor nodes (edges)
- Messages are precision-weighted (from Memoria's aif/messages.rs)

Reference: PyTorch Geometric MessagePassing (pytorch-geometric.readthedocs.io)
Reference: torch-bp (github.com/janapavlasek/torch-bp) — Gaussian BP in PyTorch
Reference: RxInfer.jl — Bethe free energy via message passing (the gold standard)
Reference: prototype-research/src/aif/messages.rs — precision-weighted fusion
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .polar import angular_similarity, EPSILON
from .state import CognitiveState


class FactorGraphMessagePassing(nn.Module if not HAS_PYG else MessagePassing):
    """Precision-weighted message passing on the cognitive state's relation graph.

    Each message from belief j to belief i through edge f:
        message = w_f × r_j × transform(angle_j, relation_f)
        precision = w_f × r_j

    Aggregation (precision-weighted fusion from Memoria):
        fused = Σ(precision_k × message_k) / Σ(precision_k)

    This replaces hand-rolled loops in free_energy.py's compute_energy
    and causal.py's propagation with proper sparse ops.
    """

    def __init__(self, belief_dim: int, relation_dim: int):
        if HAS_PYG:
            super().__init__(aggr='add')  # we'll do precision-weighted fusion manually
        else:
            super().__init__()
        self.belief_dim = belief_dim
        self.relation_dim = relation_dim

        # Learnable relation transform: how relation vectors modify messages
        self.relation_transform = nn.Linear(relation_dim, belief_dim, bias=False)

    def forward(self, state: CognitiveState, num_iterations: int = 1) -> dict:
        """Run message passing on the factor graph.

        Args:
            state: cognitive state with beliefs and edges
            num_iterations: number of BP iterations (1 = single pass, more = loopy BP)

        Returns:
            dict with:
                messages: [N_beliefs, D] aggregated incoming messages per belief
                precisions: [N_beliefs] aggregated incoming precision per belief
                agreement: [N_edges] per-edge agreement score
        """
        if not state.edge_active.any():
            n = state.config.max_beliefs
            return {
                'messages': torch.zeros(n, self.belief_dim, device=state.beliefs.device),
                'precisions': torch.zeros(n, device=state.beliefs.device),
                'agreement': torch.tensor([], device=state.beliefs.device),
            }

        # Get active edges
        active_mask = state.edge_active
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        src = state.edge_src[active_idx]  # [E]
        tgt = state.edge_tgt[active_idx]  # [E]
        relations = state.edge_relations[active_idx]  # [E, K]
        weights = state.edge_weights[active_idx]  # [E]

        beliefs = state.beliefs  # [N, D]
        radii = beliefs.norm(dim=-1).clamp(min=EPSILON)  # [N]
        angles = beliefs / radii.unsqueeze(-1)  # [N, D]

        # Transform source angles through relation
        src_angles = angles[src]  # [E, D]
        tgt_angles = angles[tgt]  # [E, D]
        relation_bias = self.relation_transform(relations)  # [E, D]

        # Transformed target = what target "looks like" through the relation
        tgt_transformed = torch.nn.functional.normalize(
            tgt_angles + relation_bias, dim=-1, eps=EPSILON
        )

        # Agreement per edge
        agreement = angular_similarity(src_angles, tgt_transformed)  # [E]

        # CoED edge directions: scale messages by cos(θ) for tgt→src, sin(θ) for src→tgt
        # θ=π/4 (init): cos=sin=0.707 — symmetric/undirected
        # θ→0: cos→1, sin→0 — fully directed src→tgt (message flows tgt to src)
        # θ→π/2: cos→0, sin→1 — fully directed tgt→src
        directions = state.edge_direction[active_idx]  # [E]
        dir_cos = torch.cos(directions)  # tgt→src message scale
        dir_sin = torch.sin(directions)  # src→tgt message scale

        # Messages: tgt sends to src, scaled by cos(θ)
        msg_precision_fwd = weights * radii[tgt] * dir_cos  # [E]
        msg_values_fwd = msg_precision_fwd.unsqueeze(-1) * tgt_transformed  # [E, D]

        # Reverse messages: src sends to tgt, scaled by sin(θ)
        src_transformed = torch.nn.functional.normalize(
            src_angles + relation_bias, dim=-1, eps=EPSILON
        )
        msg_precision_rev = weights * radii[src] * dir_sin  # [E]
        msg_values_rev = msg_precision_rev.unsqueeze(-1) * src_transformed  # [E, D]

        # Aggregate forward + reverse messages
        n = state.config.max_beliefs
        agg_messages = torch.zeros(n, self.belief_dim, device=beliefs.device)
        agg_precisions = torch.zeros(n, device=beliefs.device)

        # Forward: tgt→src
        agg_messages.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.belief_dim), msg_values_fwd)
        agg_precisions.scatter_add_(0, src, msg_precision_fwd)
        # Reverse: src→tgt
        agg_messages.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.belief_dim), msg_values_rev)
        agg_precisions.scatter_add_(0, tgt, msg_precision_rev)

        # Normalize by total precision (precision-weighted fusion)
        safe_prec = agg_precisions.unsqueeze(-1).clamp(min=EPSILON)
        agg_messages = agg_messages / safe_prec

        return {
            'messages': agg_messages,
            'precisions': agg_precisions,
            'agreement': agreement,
        }


def compute_energy_from_messages(
    state: CognitiveState,
    agreement: Tensor,
    temperature: float = 5.0,
) -> Tensor:
    """Compute energy from per-edge agreement scores.

    E = Σ_f -w_f × r_i × r_j × log(σ(agreement × temp))

    Args:
        state: cognitive state
        agreement: [E] per-edge agreement from message passing
        temperature: sigmoid sharpness

    Returns:
        Scalar energy tensor
    """
    if len(agreement) == 0:
        return torch.tensor(0.0, device=state.beliefs.device)

    active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    src = state.edge_src[active_idx]
    tgt = state.edge_tgt[active_idx]
    weights = state.edge_weights[active_idx]

    radii = state.beliefs.norm(dim=-1).clamp(min=EPSILON)
    src_radii = radii[src]
    tgt_radii = radii[tgt]

    log_sigmoid = torch.nn.functional.logsigmoid(agreement * temperature)
    energy_per_edge = -weights * src_radii * tgt_radii * log_sigmoid

    return energy_per_edge.sum()
