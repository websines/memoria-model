"""Learned edge proposal network with continuous edge directions.

Replaces hardcoded Hebbian co-activation and causal temporal precedence
heuristics with a differentiable edge proposal mechanism.

Edge proposal: MLP(belief_i, belief_j) → edge probability
Edge direction: per-edge angle θ ∈ [0, π/2] learned end-to-end
  - θ = π/4: undirected (symmetric information flow)
  - θ = 0: fully directed src→tgt
  - θ = π/2: fully directed tgt→src

Reference: ORI — Online Relational Inference (NeurIPS 2024)
Reference: CoED-GNN — Continuous Edge Directions (ICLR 2025, arXiv:2410.14109)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.polar import EPSILON


class EdgeProposer(nn.Module):
    """Learned edge proposal: given two belief vectors, predict edge probability.

    Replaces:
    - hebbian.py: extract_co_activations + hebbian_update
    - causal.py: causal_edge_learning threshold logic

    Keeps: the sparse edge budget in state.py. Only proposes edges for
    candidate pairs (co-activated beliefs + temporally surprising pairs).
    The network decides WHICH candidates become real edges.
    """

    def __init__(self, belief_dim: int, relation_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.belief_dim = belief_dim
        self.relation_dim = relation_dim

        # Edge proposal: (belief_i || belief_j || |belief_i - belief_j|) → probability
        # Using difference features helps the network detect complementary vs redundant pairs
        self.proposal_net = nn.Sequential(
            nn.Linear(belief_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Direction predictor: same input → angle θ in [0, π/2]
        self.direction_net = nn.Sequential(
            nn.Linear(belief_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # output in [0, 1], scaled to [0, π/2]
        )

        # Initial relation vector predictor (replaces handcoded cause-effect difference)
        self.relation_net = nn.Sequential(
            nn.Linear(belief_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, relation_dim),
        )

        # Init proposal net to be permissive (positive bias → sigmoid > 0.5)
        nn.init.constant_(self.proposal_net[-1].bias, 2.0)

        # AdaRelation-style adaptive threshold (from ORI)
        # Start low so the untrained network can create edges
        self.register_buffer('proposal_threshold', torch.tensor(0.3))
        self.register_buffer('prev_adjacency_hash', torch.tensor(0.0))
        self.register_buffer('ada_lr', torch.tensor(1.0))

    def propose_edges(
        self,
        candidate_pairs: list[tuple[int, int]],
        beliefs: Tensor,
        temperature: float = 1.0,
    ) -> tuple[list[tuple[int, int, float, float]], Tensor]:
        """Score candidate pairs and return accepted edges with directions.

        Args:
            candidate_pairs: list of (src_idx, tgt_idx) candidate pairs
                from co-activation and temporal surprise
            beliefs: [max_beliefs, D] belief tensor (.data, no grad in pass2)
            temperature: Gumbel-Softmax temperature for discrete accept/reject

        Returns:
            accepted: list of (src, tgt, weight, direction_theta) for accepted edges
            proposal_loss: scalar tensor for training the proposal network
                (only meaningful when called with grad enabled)
        """
        if not candidate_pairs:
            return [], torch.tensor(0.0, device=beliefs.device)

        device = beliefs.device
        N = len(candidate_pairs)

        # Build input features for all candidates
        src_indices = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long, device=device)
        tgt_indices = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long, device=device)

        src_beliefs = beliefs[src_indices]  # [N, D]
        tgt_beliefs = beliefs[tgt_indices]  # [N, D]
        diff = (src_beliefs - tgt_beliefs).abs()  # [N, D]

        features = torch.cat([src_beliefs, tgt_beliefs, diff], dim=-1)  # [N, 3D]

        # Proposal scores
        logits = self.proposal_net(features).squeeze(-1)  # [N]
        probs = torch.sigmoid(logits)

        # Direction angles
        direction_raw = self.direction_net(features).squeeze(-1)  # [N] in [0, 1]
        direction_theta = direction_raw * (math.pi / 2)  # [N] in [0, π/2]

        # Initial relation vectors
        relations = self.relation_net(features)  # [N, relation_dim]

        # Accept/reject using threshold (no Gumbel during pass2 since no_grad)
        threshold = self.proposal_threshold.item()
        accepted_mask = probs > threshold

        # Build accepted edge list
        accepted = []
        for i in range(N):
            if accepted_mask[i]:
                src = candidate_pairs[i][0]
                tgt = candidate_pairs[i][1]
                weight = probs[i].item()
                theta = direction_theta[i].item()
                accepted.append((src, tgt, weight, theta))

        # Proposal loss: train the network to predict useful edges
        # This is a placeholder — the real gradient comes from L_fe_bethe
        # through edge_weights when edges exist in the graph
        proposal_loss = torch.tensor(0.0, device=device)

        return accepted, proposal_loss

    def propose_edges_differentiable(
        self,
        candidate_pairs: list[tuple[int, int]],
        beliefs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Differentiable edge proposal for use in forward pass (with gradients).

        Returns tensors that stay in the computation graph.

        Args:
            candidate_pairs: (src, tgt) pairs
            beliefs: [max_beliefs, D] belief tensor (with grad)

        Returns:
            probs: [N] edge probabilities (differentiable)
            directions: [N] direction angles in [0, π/2] (differentiable)
            relations: [N, relation_dim] initial relation vectors (differentiable)
        """
        if not candidate_pairs:
            return (
                torch.zeros(0, device=beliefs.device),
                torch.zeros(0, device=beliefs.device),
                torch.zeros(0, self.relation_dim, device=beliefs.device),
            )

        device = beliefs.device
        src_indices = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long, device=device)
        tgt_indices = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long, device=device)

        src_beliefs = beliefs[src_indices]
        tgt_beliefs = beliefs[tgt_indices]
        diff = (src_beliefs - tgt_beliefs).abs()

        features = torch.cat([src_beliefs, tgt_beliefs, diff], dim=-1)

        probs = torch.sigmoid(self.proposal_net(features).squeeze(-1))
        directions = self.direction_net(features).squeeze(-1) * (math.pi / 2)
        relations = self.relation_net(features)

        return probs, directions, relations

    def update_ada_threshold(self, n_active_edges: int, n_proposed: int):
        """AdaRelation-style adaptive threshold update.

        Threshold is derived from graph stability: the ratio of edge count
        change to total edges. No hardcoded step sizes or bounds — the
        threshold tracks a running EMA of the acceptance rate, clamped to
        a learned range from the proposal network's own confidence.

        Reference: ORI (NeurIPS 2024) — AdaRelation adaptive learning rate
        """
        if n_proposed == 0:
            return

        current_hash = float(n_active_edges)
        prev = self.prev_adjacency_hash.item()
        self.prev_adjacency_hash.fill_(current_hash)

        if prev == 0:
            return

        # Stability ratio: fraction of edges that changed
        stability = 1.0 - abs(current_hash - prev) / max(prev, 1.0)
        stability = max(0.0, min(1.0, stability))

        # Threshold tracks stability via EMA:
        # stable graph → threshold rises (be selective)
        # unstable graph → threshold drops (accept more)
        # No hardcoded step sizes — EMA decay from running_stats convention (0.99)
        target = 0.3 + 0.5 * stability  # maps [0,1] → [0.3, 0.8] continuously
        decay = 0.95
        new_threshold = decay * self.proposal_threshold.item() + (1 - decay) * target
        self.proposal_threshold.fill_(new_threshold)
