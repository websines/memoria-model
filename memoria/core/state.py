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
from ..cognition.edge_proposal import EdgeProposer
from ..cognition.cognitive_controller import CognitiveController


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
        # [6] = goal_dedup_threshold (default 0.5)
        # [7:] = reserved

        # Initialize meta defaults
        with torch.no_grad():
            self.meta[0] = 1.0    # β = 1.0 (maximum exploration, no data yet)
            self.meta[4] = 0.3    # reconsolidation threshold
            self.meta[5] = 0.7    # match threshold
            self.meta[6] = 0.5    # goal dedup threshold

        # ── Recency Tracking ──
        # Track when beliefs were last accessed (read or written) and how often
        self.register_buffer(
            'belief_last_accessed', torch.zeros(config.max_beliefs)
        )
        self.register_buffer(
            'belief_access_count', torch.zeros(config.max_beliefs)
        )

        # ── Belief Abstraction Levels ──
        # Level 0: raw observations (from write path)
        # Level 1: reinforced beliefs (survived multiple updates)
        # Level 2: abstract beliefs (created by consolidation merges)
        # Level 3: core beliefs (high confidence, high access count, persist across sessions)
        self.register_buffer('belief_level', torch.zeros(config.max_beliefs, dtype=torch.int8))

        # ── Source Chain Tracking (MemOS-inspired provenance) ──
        # Each belief can have up to 4 source belief indices (-1 = no source)
        self.register_buffer('belief_sources', torch.full((config.max_beliefs, 4), -1, dtype=torch.long))
        # source_type: 0=observation, 1=merge, 2=promotion, 3=ttt_update
        self.register_buffer('belief_source_type', torch.zeros(config.max_beliefs, dtype=torch.int8))
        self.register_buffer('belief_created_step', torch.zeros(config.max_beliefs))

        # ── Per-Belief Adaptive Learning Rate (RWKV-7 inspired) ──
        # LR scale = EMA of inverse surprise. High surprise → lower LR (caution).
        # Low surprise → higher LR (confident, can update faster).
        self.register_buffer('belief_lr_scale', torch.ones(config.max_beliefs))

        # ── Tentative Belief Mode (A1: Internal Autoresearch Loop) ──
        # Provisional beliefs participate in forward passes but don't build
        # reinforcement. After eval_window steps, they're evaluated: promote
        # if FE decreased AND precision held, evict otherwise.
        # Reference: BrainCL (arXiv:2504.14727) wake/sleep staging
        self.register_buffer(
            'belief_provisional', torch.zeros(config.max_beliefs, dtype=torch.bool)
        )
        self.register_buffer('belief_provisional_step', torch.zeros(config.max_beliefs))
        self.register_buffer('belief_provisional_fe', torch.zeros(config.max_beliefs))
        self.register_buffer('belief_provisional_radius', torch.zeros(config.max_beliefs))

        # ── MESU Precision Variance (A2: Bayesian Metaplasticity) ──
        # Per-belief uncertainty about its own precision. High variance → high
        # plasticity (willing to change), low variance → resists updates.
        # Windowed posterior: reinforcement_count caps how much variance can shrink.
        # Reference: MESU (arXiv:2312.10153, Nature Comms 2025)
        # Reference: Palimpsa (arXiv:2602.09075) — MESU for attention states
        self.register_buffer('belief_precision_var', torch.ones(config.max_beliefs))
        self.register_buffer(
            'belief_reinforcement_count', torch.zeros(config.max_beliefs, dtype=torch.long)
        )

        # ── E3: Empirical Precision Recalibration ──
        # Track prediction accuracy per belief for recalibration.
        self.register_buffer(
            'belief_confirmed_count', torch.zeros(config.max_beliefs, dtype=torch.long)
        )
        self.register_buffer(
            'belief_contradicted_count', torch.zeros(config.max_beliefs, dtype=torch.long)
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

        # ── Learned edge proposal (replaces Hebbian/causal heuristics) ──
        self.edge_proposal = EdgeProposer(
            belief_dim=config.belief_dim,
            relation_dim=config.relation_dim,
        )

        # ── Continuous edge directions (CoED, ICLR 2025) ──
        # θ=π/4 undirected, θ=0 directed src→tgt, θ=π/2 directed tgt→src
        import math
        self.edge_direction = nn.Parameter(
            torch.full((config.max_edges,), math.pi / 4),  # init undirected
            requires_grad=True,
        )

        # ── SEAL-style cognitive controller ──
        self.controller = CognitiveController(belief_dim=config.belief_dim)

        # ── SleepGate (learned sleep cycle, arXiv:2603.14517) ──
        from ..cognition.sleep import SleepGate
        self.sleep_gate = SleepGate(belief_dim=config.belief_dim)

        # ── Factor graph message passing (for dream phase + belief shift) ──
        # Lazy import to avoid circular: state ↔ message_passing
        from .message_passing import FactorGraphMessagePassing
        self.message_passing = FactorGraphMessagePassing(
            belief_dim=config.belief_dim,
            relation_dim=config.relation_dim,
        )

        # ── Internal Autoresearch Loop ──
        from ..cognition.autoresearch import HypothesisGenerator, HypothesisTracker
        self.hypothesis_gen = HypothesisGenerator(belief_dim=config.belief_dim)
        self.hypothesis_tracker = HypothesisTracker(max_goals=config.max_goals)

        # ── A4: SGM Safety Gate ──
        from ..cognition.safety_gate import SafetyGate
        self.safety_gate = SafetyGate(global_alpha=0.05, max_modifications=100)

        # ── B1-B4: Planning priors (populated by planning step) ──
        self.register_buffer('_planning_pref_messages',
                             torch.zeros(config.max_beliefs, config.belief_dim))
        self.register_buffer('_planning_pref_precisions',
                             torch.zeros(config.max_beliefs))
        self.register_buffer('_planning_epist_precisions',
                             torch.zeros(config.max_beliefs))

        # ── C1: Self-Referential Weight Matrix ──
        from ..cognition.srwm import SRWM
        # n_meta_params matches the total count of MetaParams properties
        self.srwm = SRWM(state_dim=min(config.belief_dim, 64), n_meta_params=62, rank=32)

        # ── C2: Meta-Learned Update Function ──
        from ..cognition.learned_update import LearnedUpdateFunction
        self.learned_update = LearnedUpdateFunction(belief_dim=config.belief_dim)

        # ── C3: Structural Plasticity ──
        from ..cognition.structural_plasticity import StructuralPlasticity
        self.structural_plasticity = StructuralPlasticity(
            belief_dim=config.belief_dim, max_beliefs=config.max_beliefs,
        )

        # ── C4: Adaptive Depth ──
        from ..cognition.adaptive_depth import AdaptiveDepth
        self.adaptive_depth = AdaptiveDepth(belief_dim=config.belief_dim, max_depth=8)

        # ── D1: Daemon Loop ──
        from ..agency.daemon import DaemonLoop
        self.daemon = DaemonLoop(belief_dim=config.belief_dim)

        # ── D2: Action Selection ──
        from ..agency.action_selection import ActionSelector
        self.action_selector = ActionSelector(belief_dim=config.belief_dim)

        # ── D3: Curiosity Module ──
        from ..agency.curiosity import CuriosityModule
        self.curiosity = CuriosityModule(belief_dim=config.belief_dim)

        # ── D4: Skill Bank ──
        from ..agency.skills import SkillBank, SkillDetector, SkillComposer
        self.skill_bank = SkillBank(belief_dim=config.belief_dim, max_skills=128)
        self.skill_detector = SkillDetector(belief_dim=config.belief_dim)
        self.skill_composer = SkillComposer(belief_dim=config.belief_dim)

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

    def allocate_belief(self, belief_vector: Tensor, source_type: int = 0,
                        source_ids: list[int] | None = None, step: int = 0,
                        provisional: bool = False, current_fe: float = 0.0) -> int:
        """Allocate a new belief in the first empty slot.

        Args:
            belief_vector: [D] tensor (cartesian form, radius encodes precision)
            source_type: provenance type (0=observation, 1=merge, 2=promotion, 3=ttt_update)
            source_ids: optional list of up to 4 source belief indices
            step: current training/inference step for creation metadata
            provisional: if True, mark as tentative (evaluated after K steps)
            current_fe: current global free energy (for provisional evaluation)

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
            # Set provenance
            self.belief_level[slot] = 0  # raw observation
            self.belief_source_type[slot] = source_type
            self.belief_created_step[slot] = float(step)
            self.belief_sources[slot] = -1  # reset all sources
            if source_ids:
                for i, sid in enumerate(source_ids[:4]):
                    self.belief_sources[slot, i] = sid
            # A1: Tentative belief tracking
            self.belief_provisional[slot] = provisional
            if provisional:
                self.belief_provisional_step[slot] = float(step)
                self.belief_provisional_fe[slot] = current_fe
                self.belief_provisional_radius[slot] = belief_vector.norm().item()
            # A2: MESU — new beliefs start with high variance (uncertain)
            self.belief_precision_var[slot] = 1.0
            self.belief_reinforcement_count[slot] = 0
        return slot

    def touch_beliefs(self, indices: Tensor, step: int):
        """Update recency tracking for accessed beliefs.

        A1: Provisional beliefs update recency but do NOT increment access_count.
        This prevents tentative hypotheses from building reinforcement before
        they've been evaluated. Only committed beliefs accumulate access credit.

        Args:
            indices: [N] long tensor of belief indices that were accessed
            step: current step number
        """
        if len(indices) == 0:
            return
        with torch.no_grad():
            self.belief_last_accessed[indices] = float(step)
            # Only increment access count for non-provisional beliefs
            committed = ~self.belief_provisional[indices]
            if committed.any():
                committed_indices = indices[committed]
                self.belief_access_count[committed_indices] += 1

    def deallocate_belief(self, index: int):
        """Free a belief slot (set to zero)."""
        if self.immutable_beliefs[index]:
            return  # kernel rule: cannot deallocate
        with torch.no_grad():
            self.beliefs.data[index].zero_()
            self.belief_last_accessed[index] = 0.0
            self.belief_access_count[index] = 0.0
            self.belief_level[index] = 0
            self.belief_sources[index] = -1
            self.belief_source_type[index] = 0
            self.belief_created_step[index] = 0.0
            self.belief_lr_scale[index] = 1.0
            # A1: Clear provisional tracking
            self.belief_provisional[index] = False
            self.belief_provisional_step[index] = 0.0
            self.belief_provisional_fe[index] = 0.0
            self.belief_provisional_radius[index] = 0.0
            # A2: Reset MESU variance
            self.belief_precision_var[index] = 1.0
            self.belief_reinforcement_count[index] = 0

    def promote_belief(self, index: int):
        """Promote a belief to the next abstraction level based on evidence.

        Criteria are derived from running statistics via running_stats.promotion_thresholds(),
        not hardcoded. Relative multipliers [0.3, 0.6, 1.0] × mean_precision for radius and
        [0.3, 1.0, 3.0] × mean_access_count for access express relative difficulty per tier.
        """
        current = self.belief_level[index].item()
        if current >= 3:
            return  # already at max level
        radius = self.beliefs.data[index].norm().item()
        access = self.belief_access_count[index].item()

        min_r, min_a = self.running_stats.promotion_thresholds(current)
        if radius >= min_r and access >= min_a:
            self.belief_level[index] = current + 1

    def propagate_confidence(self, changed_indices: Tensor, old_radii: Tensor, influence: float | None = None):
        """Propagate confidence changes through source chains (MemOS-inspired).

        When source beliefs change confidence, derived beliefs update proportionally.
        Influence is a learned parameter (meta_params.confidence_propagation_influence),
        not a hardcoded constant.

        Args:
            changed_indices: [N] long tensor of belief indices whose confidence changed
            old_radii: [N] float tensor of their radii before the change
            influence: override for learned influence (None = use meta_params)
        """
        if influence is None:
            influence = self.meta_params.confidence_propagation_influence.item()
        if len(changed_indices) == 0:
            return
        with torch.no_grad():
            new_radii = self.beliefs.data[changed_indices].norm(dim=-1)
            delta_radii = new_radii - old_radii  # positive = gained confidence

            for i, idx in enumerate(changed_indices.tolist()):
                if abs(delta_radii[i].item()) < 1e-4:
                    continue
                # Find all beliefs that cite idx as a source
                has_source = (self.belief_sources == idx).any(dim=-1)  # [max_beliefs]
                has_source = has_source & self.get_active_mask() & ~self.immutable_beliefs
                if not has_source.any():
                    continue
                derived_idx = has_source.nonzero(as_tuple=False).squeeze(-1)
                for d in derived_idx.tolist():
                    current = self.beliefs.data[d]
                    current_r = current.norm().clamp(min=1e-10)
                    # More sources = less influence per source
                    n_sources = (self.belief_sources[d] >= 0).sum().item()
                    scaled_influence = influence / max(n_sources, 1)
                    new_r = current_r + delta_radii[i] * scaled_influence
                    new_r = new_r.clamp(min=0.0)
                    # Scale vector to new radius (preserve direction)
                    if current_r > 1e-10:
                        self.beliefs.data[d] = current * (new_r / current_r)

    def update_belief_lr_scale(self, indices: Tensor, surprise_values: Tensor, decay: float = 0.95):
        """Update per-belief adaptive learning rates based on surprise (RWKV-7 style).

        High surprise → scale down (be cautious with volatile beliefs).
        Low surprise → scale up (confident updates for stable beliefs).

        Args:
            indices: [N] long tensor of belief indices to update
            surprise_values: [N] float tensor of surprise magnitudes for each belief
            decay: EMA decay factor (default 0.95)
        """
        if len(indices) == 0:
            return
        with torch.no_grad():
            # Normalize surprise to [0, 1] range using sigmoid
            norm_surprise = torch.sigmoid(surprise_values - surprise_values.mean())
            # Learned scale bounds: low surprise → high LR, high surprise → cautious LR
            scale_high = self.meta_params.lr_scale_high.item()
            scale_low = self.meta_params.lr_scale_low.item()
            new_scale = scale_high * (1.0 - norm_surprise) + scale_low * norm_surprise
            # EMA update
            self.belief_lr_scale[indices] = decay * self.belief_lr_scale[indices] + (1 - decay) * new_scale

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

    def state_dict_cognitive(self, compress: bool = True) -> dict:
        """Serialize cognitive state for checkpointing.

        Args:
            compress: If True, quantize beliefs and edge relations to 3-bit
                using PolarQuant. Reduces checkpoint size ~3.3x for these tensors.
                The quantization is lossless enough for checkpoint restore
                (97%+ cosine similarity).
        """
        if compress:
            from .quantize import QuantizedBeliefStore, PolarQuantizer

            # Compress beliefs: polar decomposition (angle quantized, radius full precision)
            bs = QuantizedBeliefStore(self.config.belief_dim, bits=3)
            belief_compressed = bs.compress_beliefs(self.beliefs.data)

            # Compress edge relations (same approach, different dim)
            rel_q = PolarQuantizer(self.config.relation_dim, bits=3, rotate=False)
            rel_codes, rel_scale = rel_q.quantize(self.edge_relations.data)

            beliefs_data = {
                'compressed': True,
                'angle_codes': belief_compressed['angle_codes'],
                'angle_scale': belief_compressed['angle_scale'],
                'radii': belief_compressed['radii'],
            }
            relations_data = {
                'compressed': True,
                'codes': rel_codes,
                'scale': rel_scale,
            }
        else:
            beliefs_data = {'compressed': False, 'data': self.beliefs.data.clone()}
            relations_data = {'compressed': False, 'data': self.edge_relations.data.clone()}

        return {
            'beliefs': beliefs_data,
            'edge_src': self.edge_src.clone(),
            'edge_tgt': self.edge_tgt.clone(),
            'edge_relations': relations_data,
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
            'belief_level': self.belief_level.clone(),
            'belief_sources': self.belief_sources.clone(),
            'belief_source_type': self.belief_source_type.clone(),
            'belief_created_step': self.belief_created_step.clone(),
            'belief_lr_scale': self.belief_lr_scale.clone(),
            'belief_provisional': self.belief_provisional.clone(),
            'belief_provisional_step': self.belief_provisional_step.clone(),
            'belief_provisional_fe': self.belief_provisional_fe.clone(),
            'belief_provisional_radius': self.belief_provisional_radius.clone(),
            'belief_precision_var': self.belief_precision_var.clone(),
            'belief_reinforcement_count': self.belief_reinforcement_count.clone(),
            'belief_confirmed_count': self.belief_confirmed_count.clone(),
            'belief_contradicted_count': self.belief_contradicted_count.clone(),
            'goal_status_logits': self.goal_status_logits.clone(),
            'meta_params': self.meta_params.state_dict(),
            'running_stats': {k: v.clone() for k, v in self.running_stats._buffers.items()},
            'telos': self.telos.state_dict(),
            'edge_direction': self.edge_direction.data.clone(),
            'edge_proposal': self.edge_proposal.state_dict(),
            'controller': self.controller.state_dict(),
            'sleep_gate': self.sleep_gate.state_dict(),
            'message_passing': self.message_passing.state_dict(),
            'hypothesis_gen': self.hypothesis_gen.state_dict(),
            'hypothesis_tracker': {
                'hypothesis_count': self.hypothesis_tracker.hypothesis_count.clone(),
                'hypothesis_promoted': self.hypothesis_tracker.hypothesis_promoted.clone(),
                'hypothesis_evicted': self.hypothesis_tracker.hypothesis_evicted.clone(),
                'goal_success_ema': self.hypothesis_tracker.goal_success_ema.clone(),
            },
            'safety_gate': {
                'alpha_spent_total': self.safety_gate.alpha_spent_total.clone(),
                'n_modifications': self.safety_gate.n_modifications.clone(),
                'n_accepted': self.safety_gate.n_accepted.clone(),
                'n_rejected': self.safety_gate.n_rejected.clone(),
            },
            '_planning_pref_messages': self._planning_pref_messages.clone(),
            '_planning_pref_precisions': self._planning_pref_precisions.clone(),
            '_planning_epist_precisions': self._planning_epist_precisions.clone(),
            # C1: SRWM
            'srwm': self.srwm.state_dict(),
            'srwm_W_fast': self.srwm.W_fast.clone(),
            # C2: Learned Update
            'learned_update': self.learned_update.state_dict(),
            # C3: Structural Plasticity
            'structural_plasticity': self.structural_plasticity.state_dict(),
            'structural_plasticity_buffers': {
                'activation_count': self.structural_plasticity.activation_count.clone(),
                'activation_entropy': self.structural_plasticity.activation_entropy.clone(),
                'context_signatures': self.structural_plasticity.context_signatures.clone(),
                '_total_steps': self.structural_plasticity._total_steps.clone(),
            },
            # C4: Adaptive Depth
            'adaptive_depth': self.adaptive_depth.state_dict(),
            # D1: Daemon
            'daemon': self.daemon.state_dict(),
            # D2: Action Selector
            'action_selector': self.action_selector.state_dict(),
            # D3: Curiosity
            'curiosity': self.curiosity.state_dict(),
            # D4: Skills
            'skill_bank': self.skill_bank.state_dict(),
            'skill_bank_buffers': {
                'skill_active': self.skill_bank.skill_active.clone(),
                'skill_utility': self.skill_bank.skill_utility.clone(),
                'skill_use_count': self.skill_bank.skill_use_count.clone(),
                'skill_created_step': self.skill_bank.skill_created_step.clone(),
                'skill_last_used': self.skill_bank.skill_last_used.clone(),
            },
            'skill_detector': self.skill_detector.state_dict(),
            'skill_composer': self.skill_composer.state_dict(),
        }

    def load_state_cognitive(self, state: dict):
        """Restore cognitive state from checkpoint.

        Handles both compressed (PolarQuant) and uncompressed belief/relation formats.
        """
        with torch.no_grad():
            # Beliefs: compressed or raw
            beliefs_data = state['beliefs']
            if isinstance(beliefs_data, dict) and beliefs_data.get('compressed'):
                from .quantize import QuantizedBeliefStore
                bs = QuantizedBeliefStore(self.config.belief_dim, bits=3)
                self.beliefs.data.copy_(bs.decompress_beliefs(beliefs_data))
            elif isinstance(beliefs_data, dict):
                self.beliefs.data.copy_(beliefs_data['data'])
            else:
                # Legacy format: raw tensor
                self.beliefs.data.copy_(beliefs_data)

            self.edge_src.copy_(state['edge_src'])
            self.edge_tgt.copy_(state['edge_tgt'])

            # Edge relations: compressed or raw
            rel_data = state['edge_relations']
            if isinstance(rel_data, dict) and rel_data.get('compressed'):
                from .quantize import PolarQuantizer
                rel_q = PolarQuantizer(self.config.relation_dim, bits=3, rotate=False)
                self.edge_relations.data.copy_(rel_q.dequantize(rel_data['codes'], rel_data['scale']))
            elif isinstance(rel_data, dict):
                self.edge_relations.data.copy_(rel_data['data'])
            else:
                # Legacy format: raw tensor
                self.edge_relations.data.copy_(rel_data)

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
            if 'belief_level' in state:
                self.belief_level.copy_(state['belief_level'])
            if 'belief_sources' in state:
                self.belief_sources.copy_(state['belief_sources'])
            if 'belief_source_type' in state:
                self.belief_source_type.copy_(state['belief_source_type'])
            if 'belief_created_step' in state:
                self.belief_created_step.copy_(state['belief_created_step'])
            if 'belief_lr_scale' in state:
                self.belief_lr_scale.copy_(state['belief_lr_scale'])
            if 'belief_provisional' in state:
                self.belief_provisional.copy_(state['belief_provisional'])
                self.belief_provisional_step.copy_(state['belief_provisional_step'])
                self.belief_provisional_fe.copy_(state['belief_provisional_fe'])
                self.belief_provisional_radius.copy_(state['belief_provisional_radius'])
            if 'belief_precision_var' in state:
                self.belief_precision_var.copy_(state['belief_precision_var'])
                self.belief_reinforcement_count.copy_(state['belief_reinforcement_count'])
            if 'belief_confirmed_count' in state:
                self.belief_confirmed_count.copy_(state['belief_confirmed_count'])
                self.belief_contradicted_count.copy_(state['belief_contradicted_count'])
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
            if 'edge_direction' in state:
                self.edge_direction.data.copy_(state['edge_direction'])
            if 'edge_proposal' in state:
                self.edge_proposal.load_state_dict(state['edge_proposal'])
            if 'controller' in state:
                self.controller.load_state_dict(state['controller'])
            if 'sleep_gate' in state:
                self.sleep_gate.load_state_dict(state['sleep_gate'])
            if 'message_passing' in state:
                self.message_passing.load_state_dict(state['message_passing'])
            if 'hypothesis_gen' in state:
                self.hypothesis_gen.load_state_dict(state['hypothesis_gen'])
            if 'hypothesis_tracker' in state:
                ht = state['hypothesis_tracker']
                self.hypothesis_tracker.hypothesis_count.copy_(ht['hypothesis_count'])
                self.hypothesis_tracker.hypothesis_promoted.copy_(ht['hypothesis_promoted'])
                self.hypothesis_tracker.hypothesis_evicted.copy_(ht['hypothesis_evicted'])
                self.hypothesis_tracker.goal_success_ema.copy_(ht['goal_success_ema'])
            if 'safety_gate' in state:
                sg = state['safety_gate']
                self.safety_gate.alpha_spent_total.copy_(sg['alpha_spent_total'])
                self.safety_gate.n_modifications.copy_(sg['n_modifications'])
                self.safety_gate.n_accepted.copy_(sg['n_accepted'])
                self.safety_gate.n_rejected.copy_(sg['n_rejected'])
            if '_planning_pref_messages' in state:
                self._planning_pref_messages.copy_(state['_planning_pref_messages'])
                self._planning_pref_precisions.copy_(state['_planning_pref_precisions'])
                self._planning_epist_precisions.copy_(state['_planning_epist_precisions'])
            # C1: SRWM
            if 'srwm' in state:
                self.srwm.load_state_dict(state['srwm'])
            if 'srwm_W_fast' in state:
                self.srwm.W_fast.copy_(state['srwm_W_fast'])
            # C2: Learned Update
            if 'learned_update' in state:
                self.learned_update.load_state_dict(state['learned_update'])
            # C3: Structural Plasticity
            if 'structural_plasticity' in state:
                self.structural_plasticity.load_state_dict(state['structural_plasticity'])
            if 'structural_plasticity_buffers' in state:
                sp = state['structural_plasticity_buffers']
                self.structural_plasticity.activation_count.copy_(sp['activation_count'])
                self.structural_plasticity.activation_entropy.copy_(sp['activation_entropy'])
                self.structural_plasticity.context_signatures.copy_(sp['context_signatures'])
                self.structural_plasticity._total_steps.copy_(sp['_total_steps'])
            # C4: Adaptive Depth
            if 'adaptive_depth' in state:
                self.adaptive_depth.load_state_dict(state['adaptive_depth'])
            # D1: Daemon
            if 'daemon' in state:
                self.daemon.load_state_dict(state['daemon'])
            # D2: Action Selector
            if 'action_selector' in state:
                self.action_selector.load_state_dict(state['action_selector'])
            # D3: Curiosity
            if 'curiosity' in state:
                self.curiosity.load_state_dict(state['curiosity'])
            # D4: Skills
            if 'skill_bank' in state:
                self.skill_bank.load_state_dict(state['skill_bank'])
            if 'skill_bank_buffers' in state:
                sb = state['skill_bank_buffers']
                self.skill_bank.skill_active.copy_(sb['skill_active'])
                self.skill_bank.skill_utility.copy_(sb['skill_utility'])
                self.skill_bank.skill_use_count.copy_(sb['skill_use_count'])
                self.skill_bank.skill_created_step.copy_(sb['skill_created_step'])
                self.skill_bank.skill_last_used.copy_(sb['skill_last_used'])
            if 'skill_detector' in state:
                self.skill_detector.load_state_dict(state['skill_detector'])
            if 'skill_composer' in state:
                self.skill_composer.load_state_dict(state['skill_composer'])

    # ── Summary ──

    def summary(self) -> str:
        """Human-readable state summary."""
        active = self.get_active_mask()
        levels = self.belief_level[active]
        level_counts = [(levels == i).sum().item() for i in range(4)]
        n_provisional = self.belief_provisional[active].sum().item() if active.any() else 0
        mean_var = self.belief_precision_var[active].mean().item() if active.any() else 1.0
        return (
            f"CognitiveState: "
            f"{self.num_active_beliefs()}/{self.config.max_beliefs} beliefs "
            f"(L0:{level_counts[0]} L1:{level_counts[1]} L2:{level_counts[2]} L3:{level_counts[3]}, "
            f"prov:{n_provisional}), "
            f"{self.num_active_edges()}/{self.config.max_edges} edges, "
            f"{self.num_active_goals()}/{self.config.max_goals} goals, "
            f"β={self.beta:.3f}, σ²={mean_var:.3f}"
        )
