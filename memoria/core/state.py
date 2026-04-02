"""Cognitive State: the persistent structured tensor that holds the model's world model.

Four regions:
- Belief region: world model in polar form (radius = precision, angle = content)
- Relation region: causal/associative edges between beliefs (hard indices, soft weights)
- Goal region: Telos goals with embeddings + metadata
- Meta region: β, accumulated surprise, cognitive parameters

The state persists across sequences and evolves through pass 2 updates.
Active belief count is dynamic — radius=0 means empty slot.
Max capacity is configurable; cost scales with active count, not max capacity.
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from .polar import belief_is_active, EPSILON
from .meta_params import MetaParams
from .running_stats import RunningStats
from ..cognition.telos_module import TelosModule, NUM_STATUS


@dataclass
class StateConfig:
    """Configuration for cognitive state dimensions."""
    belief_dim: int = 256          # D: representation dimension per belief
    max_beliefs: int = 4096        # configurable max capacity
    max_edges: int = 16384         # 4× beliefs
    max_goals: int = 64            # concurrent goals
    relation_dim: int = 64         # K: relation representation dimension
    goal_metadata_dim: int = 8     # G: priority, urgency, progress, status, depth, surprise, created, deadline
    meta_dim: int = 32             # M: meta parameters


class CognitiveState(nn.Module):
    """The model's persistent cognitive state.

    All regions are stored as tensors. Active beliefs have radius > 0.
    Empty slots have radius ≈ 0 and are available for allocation.

    This module does NOT include the update logic (that's in cognition/pass2.py).
    It only holds the state and provides accessors.
    """

    def __init__(self, config: StateConfig):
        super().__init__()
        self.config = config

        # ── Belief Region ──
        # Stored in cartesian form internally. Polar decomposition on access.
        # Radius = precision, angle = content direction.
        self.beliefs = nn.Parameter(
            torch.zeros(config.max_beliefs, config.belief_dim),
            requires_grad=True,  # Updated by pass 2, not by optimizer
        )

        # ── Relation Region ──
        # Hard indices (source, target) + soft relation vector + soft weight
        self.register_buffer('edge_src', torch.zeros(config.max_edges, dtype=torch.long))
        self.register_buffer('edge_tgt', torch.zeros(config.max_edges, dtype=torch.long))
        self.edge_relations = nn.Parameter(
            torch.zeros(config.max_edges, config.relation_dim),
            requires_grad=True,
        )
        self.edge_weights = nn.Parameter(
            torch.zeros(config.max_edges),
            requires_grad=True,
        )
        # Track which edges are active (allocated)
        self.register_buffer('edge_active', torch.zeros(config.max_edges, dtype=torch.bool))
        # Causal edge observation counts (0 = Hebbian/associative, >0 = causal with N observations)
        self.register_buffer('edge_causal_obs', torch.zeros(config.max_edges))

        # ── Goal Region (Telos) ──
        # Goal embedding (same space as beliefs) + metadata
        self.goal_embeddings = nn.Parameter(
            torch.zeros(config.max_goals, config.belief_dim),
            requires_grad=True,  # differentiable — trained by L_fe_bethe telos energy
        )
        self.goal_metadata = nn.Parameter(
            torch.zeros(config.max_goals, config.goal_metadata_dim),
            requires_grad=False,  # structural metadata (created_step, etc.)
        )
        # Metadata layout: [priority, urgency, progress, status_legacy, depth, surprise_accum, created_step, deadline]
        # Status encoding (legacy float): kept for backward compat but overridden by goal_status_logits

        # Gumbel-Softmax status: [max_goals, 6] logits over {empty, proposed, active, stalled, completed, failed}
        self.register_buffer(
            'goal_status_logits',
            torch.zeros(config.max_goals, NUM_STATUS),
        )
        # Initialize all goals as empty (high logit on STATUS_EMPTY)
        with torch.no_grad():
            self.goal_status_logits[:, 0] = 10.0  # strong prior on empty

        # ── Meta Region ──
        self.meta = nn.Parameter(
            torch.zeros(config.meta_dim),
            requires_grad=False,
        )
        # Meta layout (first few slots have defined meaning):
        # [0] = β (exploration/exploitation, computed from state)
        # [1] = accumulated_surprise
        # [2] = consolidation_timer (steps since last consolidation)
        # [3] = learning_rate_modulation
        # [4] = reconsolidation_threshold (default 0.3)
        # [5] = match_threshold (default 0.7)
        # [6] = goal_dedup_threshold (default 0.5, SPSA-tunable)
        # [7:] = SPSA-tunable parameters

        # Initialize meta defaults
        with torch.no_grad():
            self.meta[0] = 1.0    # β = 1.0 (maximum exploration, no data yet)
            self.meta[4] = 0.3    # reconsolidation threshold
            self.meta[5] = 0.7    # match threshold
            self.meta[6] = 0.5    # goal dedup threshold (SPSA-tunable)

        # ── Recency Tracking ──
        # Track when beliefs were last accessed (read or written) and how often
        self.register_buffer(
            'belief_last_accessed', torch.zeros(config.max_beliefs)
        )
        self.register_buffer(
            'belief_access_count', torch.zeros(config.max_beliefs)
        )

        # ── Causal Learning ──
        # Previous step's per-belief surprise (for temporal precedence detection)
        self.register_buffer(
            'belief_prev_surprise', torch.zeros(config.max_beliefs)
        )

        # ── Kernel Rules (immutability masks) ──
        self.register_buffer(
            'immutable_beliefs', torch.zeros(config.max_beliefs, dtype=torch.bool)
        )
        self.register_buffer(
            'immutable_edges', torch.zeros(config.max_edges, dtype=torch.bool)
        )
        self.register_buffer(
            'immutable_goals', torch.zeros(config.max_goals, dtype=torch.bool)
        )

        # ── Learnable meta-parameters (replace 15 magic numbers) ──
        self.meta_params = MetaParams()

        # ── Running statistics (replace 10 hardcoded thresholds) ──
        self.running_stats = RunningStats()

        # ── Differentiable Telos (learned goal system) ──
        self.telos = TelosModule(
            belief_dim=config.belief_dim,
            max_goals=config.max_goals,
        )

    # ── Belief Region Accessors ──

    def get_belief_radii(self) -> Tensor:
        """Get precision (radius) of all beliefs."""
        return self.beliefs.norm(dim=-1)

    def get_belief_angles(self) -> Tensor:
        """Get content direction (unit vector) of all beliefs."""
        radii = self.get_belief_radii().unsqueeze(-1).clamp(min=EPSILON)
        return self.beliefs / radii

    def get_active_mask(self) -> Tensor:
        """Boolean mask of active (non-empty) beliefs."""
        return belief_is_active(self.get_belief_radii())

    def get_active_beliefs(self) -> tuple[Tensor, Tensor]:
        """Get only active beliefs as (indices, belief_vectors).

        Returns:
            indices: [N_active] long tensor of active slot indices
            beliefs: [N_active, D] belief vectors (cartesian)
        """
        mask = self.get_active_mask()
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return indices, self.beliefs[indices]

    def num_active_beliefs(self) -> int:
        """Count of active beliefs."""
        return self.get_active_mask().sum().item()

    def allocate_belief(self, belief_vector: Tensor) -> int:
        """Allocate a new belief in the first empty slot.

        Args:
            belief_vector: [D] tensor (cartesian form, radius encodes precision)

        Returns:
            slot index, or -1 if full
        """
        radii = self.get_belief_radii()
        empty = (~belief_is_active(radii)).nonzero(as_tuple=False)
        if len(empty) == 0:
            return -1
        slot = empty[0].item()
        with torch.no_grad():
            self.beliefs.data[slot] = belief_vector
        return slot

    def touch_beliefs(self, indices: Tensor, step: int):
        """Update recency tracking for accessed beliefs.

        Args:
            indices: [N] long tensor of belief indices that were accessed
            step: current step number
        """
        if len(indices) == 0:
            return
        with torch.no_grad():
            self.belief_last_accessed[indices] = float(step)
            self.belief_access_count[indices] += 1

    def deallocate_belief(self, index: int):
        """Free a belief slot (set to zero)."""
        if self.immutable_beliefs[index]:
            return  # kernel rule: cannot deallocate
        with torch.no_grad():
            self.beliefs.data[index].zero_()
            self.belief_last_accessed[index] = 0.0
            self.belief_access_count[index] = 0.0

    # ── Relation Region Accessors ──

    def get_active_edges(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get active edges.

        Returns:
            src: [N_active_edges] source belief indices
            tgt: [N_active_edges] target belief indices
            relations: [N_active_edges, K] relation vectors
            weights: [N_active_edges] edge weights
        """
        mask = self.edge_active
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return (
            self.edge_src[indices],
            self.edge_tgt[indices],
            self.edge_relations[indices],
            self.edge_weights[indices],
        )

    def allocate_edge(self, src: int, tgt: int, relation: Tensor, weight: float = 0.1) -> int:
        """Allocate a new edge.

        Args:
            src: source belief index
            tgt: target belief index
            relation: [K] relation representation
            weight: initial edge weight

        Returns:
            edge index, or -1 if full
        """
        empty = (~self.edge_active).nonzero(as_tuple=False)
        if len(empty) == 0:
            return -1
        slot = empty[0].item()
        with torch.no_grad():
            self.edge_src[slot] = src
            self.edge_tgt[slot] = tgt
            self.edge_relations.data[slot] = relation
            self.edge_weights.data[slot] = weight
            self.edge_active[slot] = True
        return slot

    def deallocate_edge(self, index: int):
        """Free an edge slot."""
        if self.immutable_edges[index]:
            return
        with torch.no_grad():
            self.edge_active[index] = False
            self.edge_weights.data[index] = 0.0
            self.edge_causal_obs[index] = 0.0

    def num_active_edges(self) -> int:
        return self.edge_active.sum().item()

    # ── Goal Region Accessors ──

    def get_active_goals(self) -> tuple[Tensor, Tensor, Tensor]:
        """Get truly active goals (proposed or active status only).

        Excludes completed, failed, and stalled goals so they don't
        influence retrieval bias, free energy, or duplicate detection.

        Returns:
            indices: [N_active_goals]
            embeddings: [N_active_goals, D]
            metadata: [N_active_goals, G]
        """
        # Gumbel-Softmax status: goal is active if proposed (1) or active (2) has highest prob
        status_probs = torch.softmax(self.goal_status_logits, dim=-1)
        best_status = status_probs.argmax(dim=-1)
        mask = (best_status == 1) | (best_status == 2)  # proposed or active
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return indices, self.goal_embeddings[indices], self.goal_metadata.data[indices]

    def get_all_allocated_goals(self) -> tuple[Tensor, Tensor, Tensor]:
        """Get all non-empty goals (any non-empty status). For slot management only."""
        status_probs = torch.softmax(self.goal_status_logits, dim=-1)
        best_status = status_probs.argmax(dim=-1)
        mask = best_status != 0  # not empty
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return indices, self.goal_embeddings[indices], self.goal_metadata.data[indices]

    def num_active_goals(self) -> int:
        status_probs = torch.softmax(self.goal_status_logits, dim=-1)
        best_status = status_probs.argmax(dim=-1)
        return ((best_status == 1) | (best_status == 2)).sum().item()

    # ── Meta Accessors ──

    @property
    def beta(self) -> float:
        """Current exploration/exploitation parameter."""
        return self.meta.data[0].item()

    @property
    def accumulated_surprise(self) -> float:
        return self.meta.data[1].item()

    @property
    def reconsolidation_threshold(self) -> float:
        return self.meta.data[4].item()

    @property
    def match_threshold(self) -> float:
        return self.meta.data[5].item()

    # ── Serialization ──

    def state_dict_cognitive(self) -> dict:
        """Serialize cognitive state for checkpointing."""
        return {
            'beliefs': self.beliefs.data.clone(),
            'edge_src': self.edge_src.clone(),
            'edge_tgt': self.edge_tgt.clone(),
            'edge_relations': self.edge_relations.data.clone(),
            'edge_weights': self.edge_weights.data.clone(),
            'edge_active': self.edge_active.clone(),
            'edge_causal_obs': self.edge_causal_obs.clone(),
            'goal_embeddings': self.goal_embeddings.data.clone(),
            'goal_metadata': self.goal_metadata.data.clone(),
            'meta': self.meta.data.clone(),
            'immutable_beliefs': self.immutable_beliefs.clone(),
            'immutable_edges': self.immutable_edges.clone(),
            'immutable_goals': self.immutable_goals.clone(),
            'belief_last_accessed': self.belief_last_accessed.clone(),
            'belief_access_count': self.belief_access_count.clone(),
            'belief_prev_surprise': self.belief_prev_surprise.clone(),
            'goal_status_logits': self.goal_status_logits.clone(),
            'meta_params': self.meta_params.state_dict(),
            'running_stats': {k: v.clone() for k, v in self.running_stats._buffers.items()},
            'telos': self.telos.state_dict(),
        }

    def load_state_cognitive(self, state: dict):
        """Restore cognitive state from checkpoint."""
        with torch.no_grad():
            self.beliefs.data.copy_(state['beliefs'])
            self.edge_src.copy_(state['edge_src'])
            self.edge_tgt.copy_(state['edge_tgt'])
            self.edge_relations.data.copy_(state['edge_relations'])
            self.edge_weights.data.copy_(state['edge_weights'])
            self.edge_active.copy_(state['edge_active'])
            if 'edge_causal_obs' in state:
                self.edge_causal_obs.copy_(state['edge_causal_obs'])
            self.goal_embeddings.data.copy_(state['goal_embeddings'])
            self.goal_metadata.data.copy_(state['goal_metadata'])
            self.meta.data.copy_(state['meta'])
            self.immutable_beliefs.copy_(state['immutable_beliefs'])
            self.immutable_edges.copy_(state['immutable_edges'])
            self.immutable_goals.copy_(state['immutable_goals'])
            if 'belief_last_accessed' in state:
                self.belief_last_accessed.copy_(state['belief_last_accessed'])
                self.belief_access_count.copy_(state['belief_access_count'])
            if 'belief_prev_surprise' in state:
                self.belief_prev_surprise.copy_(state['belief_prev_surprise'])
            if 'goal_status_logits' in state:
                self.goal_status_logits.copy_(state['goal_status_logits'])
            if 'meta_params' in state:
                self.meta_params.load_state_dict(state['meta_params'])
            if 'running_stats' in state:
                for k, v in state['running_stats'].items():
                    if hasattr(self.running_stats, k):
                        getattr(self.running_stats, k).copy_(v)
            if 'telos' in state:
                self.telos.load_state_dict(state['telos'])

    # ── Summary ──

    def summary(self) -> str:
        """Human-readable state summary."""
        return (
            f"CognitiveState: "
            f"{self.num_active_beliefs()}/{self.config.max_beliefs} beliefs, "
            f"{self.num_active_edges()}/{self.config.max_edges} edges, "
            f"{self.num_active_goals()}/{self.config.max_goals} goals, "
            f"β={self.beta:.3f}"
        )
