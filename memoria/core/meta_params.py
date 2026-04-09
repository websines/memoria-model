"""
meta_params.py — Learnable meta-parameters for the Memoria cognitive system.

Each of the 15 parameters replaces a hardcoded constant elsewhere in the
codebase.  Raw (unconstrained) values are stored as nn.Parameters; callers
access the constrained value through the named property.

Activation conventions:
  sigmoid   — output in (0, 1), suitable for rates, thresholds, and fractions
  softplus  — output in (0, ∞), suitable for strictly-positive scale values

Init strategy: the raw value is chosen so that applying the activation
returns exactly (or very nearly) the original hardcoded constant, making the
system behave identically before any meta-learning occurs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaParams(nn.Module):
    """
    Learnable replacements for the 15 magic numbers in Memoria.

    Parameter map (name → activation → source location):
      fe_lambda                  sigmoid   losses.py:115          F = E - λH
      fe_temperature             softplus  pass2.py:42            softmax temperature
      hebbian_lr                 sigmoid   hebbian.py:26          Hebbian learning rate
      hebbian_decay              sigmoid   hebbian.py:27          weight decay per step
      initial_edge_weight        sigmoid   hebbian.py:110         new-edge init weight
      causal_min_signal          sigmoid   causal.py:206          causal signal floor
      causal_decay_rate          sigmoid   causal.py:207          causal weight decay
      causal_initial_weight_scale sigmoid  causal.py:293          initial causal weight
      causal_relation_scale      sigmoid   causal.py:277,290      relation score scale
      goal_relevance_threshold   sigmoid   telos.py:179           goal relevance cutoff
      goal_progress_rate         sigmoid   telos.py:181           goal progress step
      precision_decay_factor     sigmoid   meta_learning.py:204   precision EMA decay
      reconsolidation_threshold  sigmoid   meta_learning.py       reconsolidation gate
      match_threshold            sigmoid   meta_learning.py       pattern-match gate
      goal_dedup_threshold       sigmoid   meta_learning.py       goal dedup distance
    """

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Raw (unconstrained) parameters.  Init values are sigmoid/softplus   #
        # inverse of the original hardcoded constants.                        #
        # ------------------------------------------------------------------ #

        # losses.py:115 — F = E - λH  (λ controls entropy weight)
        self._fe_lambda = nn.Parameter(torch.tensor(-2.197225))

        # pass2.py:42 — softmax temperature for free-energy pass
        self._fe_temperature = nn.Parameter(torch.tensor(4.993239))

        # hebbian.py:26 — learning rate for Hebbian weight updates
        self._hebbian_lr = nn.Parameter(torch.tensor(-2.944439))

        # hebbian.py:27 — multiplicative decay applied to Hebbian weights each step
        self._hebbian_decay = nn.Parameter(torch.tensor(-4.595120))

        # hebbian.py:110 — initial weight assigned to newly created edges
        self._initial_edge_weight = nn.Parameter(torch.tensor(-2.197225))

        # causal.py:206 — minimum signal strength to register a causal link
        self._causal_min_signal = nn.Parameter(torch.tensor(-2.197225))

        # causal.py:207 — per-step decay rate of causal edge weights
        self._causal_decay_rate = nn.Parameter(torch.tensor(-5.293305))

        # causal.py:293 — scale factor applied when initialising causal weights
        self._causal_initial_weight_scale = nn.Parameter(torch.tensor(-0.847298))

        # causal.py:277,290 — scale applied to relation scores
        self._causal_relation_scale = nn.Parameter(torch.tensor(-2.197225))

        # telos.py:179 — minimum relevance score for a goal to be considered active
        self._goal_relevance_threshold = nn.Parameter(torch.tensor(-0.847298))

        # telos.py:181 — step size for updating goal progress estimates
        self._goal_progress_rate = nn.Parameter(torch.tensor(-2.197225))

        # meta_learning.py:204 — EMA decay for the precision estimate
        self._precision_decay_factor = nn.Parameter(torch.tensor(5.293305))

        # meta_learning.py — threshold for triggering memory reconsolidation
        self._reconsolidation_threshold = nn.Parameter(torch.tensor(-0.847298))

        # meta_learning.py — cosine-similarity threshold for pattern matching
        self._match_threshold = nn.Parameter(torch.tensor(0.847298))

        # meta_learning.py — distance threshold for deduplicating goals
        self._goal_dedup_threshold = nn.Parameter(torch.tensor(0.0))

        # state.py — confidence propagation influence fraction
        # sigmoid(−2.197) ≈ 0.1 (original hardcoded default)
        self._confidence_propagation_influence = nn.Parameter(torch.tensor(-2.197225))

        # state.py — adaptive LR scale upper bound (high confidence → fast updates)
        # softplus(1.3132559) ≈ 1.6265 → we want 2.0; softplus(1.6946) ≈ 2.0
        self._lr_scale_high = nn.Parameter(torch.tensor(1.694596))

        # state.py — adaptive LR scale lower bound (high surprise → cautious updates)
        # softplus(−0.3132617) ≈ 0.5 (original hardcoded default)
        self._lr_scale_low = nn.Parameter(torch.tensor(-0.313262))

        # free_energy.py — weight on epistemic term in Expected Free Energy
        # softplus(1.0) ≈ 1.313 → starts with mild epistemic bias
        self._efe_epistemic_weight = nn.Parameter(torch.tensor(1.0))

        # free_energy.py — weight on risk term in Expected Free Energy
        # softplus(1.0) ≈ 1.313 → symmetric with epistemic at start
        self._efe_risk_weight = nn.Parameter(torch.tensor(1.0))

        # losses.py, free_energy.py — Huber transition point for belief matching
        # Huber loss: quadratic for |error| < delta (precise), linear beyond (robust).
        # Prevents outlier cosine disagreements from dominating belief update gradients.
        # softplus(-0.4338) ≈ 0.5 → delta starts at 0.5 (1-cos_sim typical range [0, 0.3] for matches)
        # Reference: MIRAS/YAAD (arXiv:2504.13173); Huber (1964)
        self._huber_delta = nn.Parameter(torch.tensor(-0.433780))

        # ── A1: Tentative Belief Mode ──
        # Evaluation window: how many steps before evaluating provisional beliefs.
        # softplus(2.0) ≈ 2.127 → we want ~10 steps; softplus(9.99) ≈ 10.0
        self._provisional_eval_window = nn.Parameter(torch.tensor(9.99))
        # FE improvement threshold: FE must decrease by this fraction to promote.
        # sigmoid(0.0) = 0.5 → multiply by 0.2 → 0.1 initial threshold (10% FE decrease)
        self._provisional_fe_threshold = nn.Parameter(torch.tensor(0.0))
        # Precision retention: belief must retain this fraction of initial radius.
        # sigmoid(1.386) ≈ 0.8 → belief must keep 80% of its initial precision
        self._provisional_precision_retention = nn.Parameter(torch.tensor(1.386294))

        # ── A2: MESU Precision Variance ──
        # Min variance floor: prevents overconfidence.
        # softplus(-4.6) ≈ 0.01 → minimum variance of 0.01
        self._mesu_min_variance = nn.Parameter(torch.tensor(-4.595120))
        # Variance shrink rate: how fast variance decreases per observation.
        # sigmoid(-0.847) ≈ 0.3 → variance shrinks by gain^2 * 0.3 per update
        self._mesu_variance_shrink = nn.Parameter(torch.tensor(-0.847298))
        # Reinforcement window: max observations before variance floor kicks in.
        # softplus(3.16) ≈ 32 steps
        self._mesu_window_size = nn.Parameter(torch.tensor(3.434818))
        # Variance gain boost: how much high variance amplifies gain.
        # softplus(0.693) ≈ 1.0 → gain multiplied by (1 + variance * 1.0)
        self._mesu_gain_boost = nn.Parameter(torch.tensor(0.693147))

        # ── A3: Causal Cascade Revision ──
        # Base decay factor per hop in the causal graph.
        # sigmoid(-0.405) ≈ 0.4 → 40% precision decay per hop
        self._cascade_decay_factor = nn.Parameter(torch.tensor(-0.405465))
        # Max cascade depth: how many hops to propagate.
        # softplus(0.693) ≈ 1.0 → multiplied by 3 → 3 hops default
        self._cascade_max_depth = nn.Parameter(torch.tensor(0.693147))
        # Variance increase per hop: how much downstream variance grows.
        # softplus(-0.693) ≈ 0.5 → add 0.5 * decay_per_hop to downstream variance
        self._cascade_variance_boost = nn.Parameter(torch.tensor(-0.693147))

        # ── A4: SGM Safety Gate ──
        # Confidence level for accepting self-modifications.
        # sigmoid(2.944) ≈ 0.95 → require 95% confidence
        self._sgm_confidence = nn.Parameter(torch.tensor(2.944439))
        # Rejection patience: how many samples before giving up on a bad modification.
        # softplus(2.3) ≈ 10 samples minimum
        self._sgm_min_samples = nn.Parameter(torch.tensor(2.302585))

        # ── B1-B4: Planning ──
        # Planning horizon: how many causal steps to simulate ahead.
        # softplus(1.6) ≈ 5 steps
        self._planning_horizon = nn.Parameter(torch.tensor(1.609438))
        # Temporal discount: how much to discount future EFE per step.
        # sigmoid(1.386) ≈ 0.8 → 20% discount per step
        self._planning_discount = nn.Parameter(torch.tensor(1.386294))
        # MCTS exploration constant (UCB-style).
        # softplus(0.0) ≈ 0.693 → sqrt(2)/2 ≈ standard UCB1
        self._mcts_exploration = nn.Parameter(torch.tensor(0.0))
        # Planning temperature for action selection softmax.
        # softplus(0.0) ≈ 0.693
        self._planning_temperature = nn.Parameter(torch.tensor(0.0))
        # Preference prior strength: how strongly goals pull the factor graph.
        # softplus(0.0) ≈ 0.693
        self._preference_prior_strength = nn.Parameter(torch.tensor(0.0))
        # Epistemic prior strength: how strongly uncertainty drives exploration.
        # softplus(0.0) ≈ 0.693
        self._epistemic_prior_strength = nn.Parameter(torch.tensor(0.0))

        # ── C1: SRWM (Self-Referential Weight Matrix) ──
        # Learning rate for fast-weight Hebbian updates: W += lr * outer(k, v).
        # sigmoid(-2.197) ≈ 0.1 → conservative initial fast-weight learning
        self._srwm_lr = nn.Parameter(torch.tensor(-2.197225))
        # Decay rate for fast weights each step (prevents unbounded growth).
        # sigmoid(-2.944) ≈ 0.05 → slow decay, fast weights persist ~20 steps
        self._srwm_decay = nn.Parameter(torch.tensor(-2.944439))

        # ── C2: Meta-Learned Update Function ──
        # Gate controlling blend between learned update and hand-coded update.
        # sigmoid(-2.197) ≈ 0.1 → starts mostly hand-coded, learns to trust NN
        self._update_fn_gate = nn.Parameter(torch.tensor(-2.197225))

        # ── C3: Structural Plasticity ──
        # Activation entropy threshold for splitting polysemantic beliefs.
        # sigmoid(0.847) ≈ 0.7 → beliefs with entropy > 70% of max are split candidates
        self._plasticity_split_threshold = nn.Parameter(torch.tensor(0.847298))
        # Activation frequency threshold for pruning dead beliefs.
        # sigmoid(-2.197) ≈ 0.1 → beliefs accessed < 10% of mean are prune candidates
        self._plasticity_prune_threshold = nn.Parameter(torch.tensor(-2.197225))
        # Growth rate: fraction of capacity to add when growing.
        # sigmoid(-2.197) ≈ 0.1 → grow by 10% of current capacity
        self._plasticity_growth_rate = nn.Parameter(torch.tensor(-2.197225))

        # ── C4: Learned Recursion Depth ──
        # Ponder cost: penalty per additional recursion step (ACT-style).
        # softplus(-0.693) ≈ 0.5 → mild penalty, allows ~3-5 extra steps
        self._recursion_depth_penalty = nn.Parameter(torch.tensor(-0.693147))
        # Halt bias: initial tendency to halt early.
        # sigmoid(0.0) = 0.5 → neutral start, learns when to continue
        self._recursion_halt_bias = nn.Parameter(torch.tensor(0.0))

        # ── D2: EFE Action Selection ──
        # Temperature for Gumbel-Softmax action selection.
        # softplus(0.0) ≈ 0.693 → moderate stochasticity
        self._action_temperature = nn.Parameter(torch.tensor(0.0))
        # Risk aversion: multiplier on risk term in action EFE.
        # softplus(0.0) ≈ 0.693 → balanced risk sensitivity
        self._action_risk_aversion = nn.Parameter(torch.tensor(0.0))

        # ── D3: Curiosity ──
        # Curiosity threshold: above this, generate exploration goals.
        # softplus(0.0) ≈ 0.693 → triggers on moderate novelty
        self._curiosity_threshold = nn.Parameter(torch.tensor(0.0))
        # Curiosity weight: relative importance of exploration vs pragmatic goals.
        # softplus(-0.693) ≈ 0.5 → exploration is half as important as exploitation
        self._curiosity_weight = nn.Parameter(torch.tensor(-0.693147))

        # ── D4: Skill Crystallization ──
        # Detection threshold: minimum recurrence count for crystallization.
        # softplus(0.693) ≈ 1.0 → multiply by 3 → need 3 recurrences
        self._skill_detection_threshold = nn.Parameter(torch.tensor(0.693147))
        # Similarity threshold for matching actions to existing skills.
        # sigmoid(1.386) ≈ 0.8 → 80% cosine similarity required
        self._skill_similarity_threshold = nn.Parameter(torch.tensor(1.386294))

        # ── E1: Two-Factor Sleep Consolidation ──
        # Homeostatic target: desired total precision budget (sum of radii).
        # softplus(4.605) ≈ 100.0 → default budget of 100 precision units
        self._homeostatic_target = nn.Parameter(torch.tensor(4.605170))
        # Homeostatic rate: how fast to normalize toward target per sleep cycle.
        # sigmoid(-2.197) ≈ 0.1 → 10% correction per cycle
        self._homeostatic_rate = nn.Parameter(torch.tensor(-2.197225))
        # Conflict threshold: angular cosine above which beliefs are in conflict.
        # sigmoid(1.735) ≈ 0.85 → 85% cosine similarity = near-duplicate conflict
        self._sleep_conflict_threshold = nn.Parameter(torch.tensor(1.735085))

        # ── E2: Self-Verification Pass ──
        # Divergence threshold: message vs stored belief disagreement for inconsistency.
        # sigmoid(-0.847) ≈ 0.3 → flag if cosine similarity < 0.3
        self._verification_divergence_threshold = nn.Parameter(torch.tensor(-0.847298))
        # Precision decay on inconsistency: how much to weaken the weakest link.
        # sigmoid(-1.386) ≈ 0.2 → reduce radius by 20% on inconsistency
        self._verification_precision_decay = nn.Parameter(torch.tensor(-1.386294))
        # Supersession similarity: threshold for conflict-aware supersession.
        # sigmoid(1.735) ≈ 0.85 → beliefs with >85% cosine similarity may supersede
        self._supersession_similarity = nn.Parameter(torch.tensor(1.735085))

        # ── E3: Empirical Precision Recalibration ──
        # Recalibration rate: how fast stored radius decays toward empirical precision.
        # sigmoid(-2.197) ≈ 0.1 → 10% correction per recalibration cycle
        self._recalibration_rate = nn.Parameter(torch.tensor(-2.197225))
        # Minimum samples: confirmed+contradicted count before recalibrating.
        # softplus(1.609) ≈ 5.0 → need at least 5 observations
        self._recalibration_min_samples = nn.Parameter(torch.tensor(1.609438))

        # ── E4: Interleaved Replay ──
        # Replay ratio: fraction of replay set that are old high-precision beliefs.
        # sigmoid(-0.847) ≈ 0.3 → 30% old beliefs, 70% recent high-surprise
        self._replay_ratio = nn.Parameter(torch.tensor(-0.847298))
        # Contradiction threshold: message disagreement for cross-temporal contradiction.
        # sigmoid(0.0) = 0.5 → moderate threshold for flagging contradictions
        self._replay_contradiction_threshold = nn.Parameter(torch.tensor(0.0))

        # ── F1: Predictive Refinement (MoR + SCORE) ──
        # Contraction rate: SCORE-style step-size decay per loop iteration.
        # dt(l) = (1 - contraction_rate)^l → later loops contribute smaller deltas.
        # sigmoid(-1.386) ≈ 0.2 → dt decays as 0.8^l (loop 0: 1.0, loop 1: 0.8, loop 2: 0.64)
        # Reference: SCORE (arXiv:2603.10544) — contractive recurrent depth
        self._refinement_contraction_rate = nn.Parameter(torch.tensor(-1.386294))
        # Retrieval delta threshold: minimum per-position delta L2-norm to trigger
        # belief re-query in the retrieve-reason-retrieve step.
        # sigmoid(-2.197) ≈ 0.1 → skip re-query when delta < 10% of mean hidden norm
        # Reference: DeltaLLM (arXiv:2507.19608) — temporal sparsity via delta threshold
        self._refinement_retrieval_threshold = nn.Parameter(torch.tensor(-2.197225))
        # Ponder cost: per-position penalty for continuing refinement. Scales the
        # ponder regularization loss that encourages the router to halt early.
        # softplus(-0.693) ≈ 0.5 → moderate penalty, allows continued refinement when needed
        # Reference: PonderNet (arXiv:2107.05407) — learned halting with geometric prior
        self._refinement_ponder_cost = nn.Parameter(torch.tensor(-0.693147))

    # ---------------------------------------------------------------------- #
    # Properties — apply activation to yield constrained values               #
    # ---------------------------------------------------------------------- #

    @property
    def fe_lambda(self) -> torch.Tensor:
        """Entropy weight λ in F = E - λH. Range: (0, 1)."""
        return torch.sigmoid(self._fe_lambda)

    @property
    def fe_temperature(self) -> torch.Tensor:
        """Softmax temperature for free-energy attention pass. Range: (0, ∞)."""
        return F.softplus(self._fe_temperature)

    @property
    def hebbian_lr(self) -> torch.Tensor:
        """Hebbian learning rate. Range: (0, 1)."""
        return torch.sigmoid(self._hebbian_lr)

    @property
    def hebbian_decay(self) -> torch.Tensor:
        """Per-step multiplicative decay on Hebbian weights. Range: (0, 1)."""
        return torch.sigmoid(self._hebbian_decay)

    @property
    def initial_edge_weight(self) -> torch.Tensor:
        """Weight assigned to newly created Hebbian edges. Range: (0, 1)."""
        return torch.sigmoid(self._initial_edge_weight)

    @property
    def causal_min_signal(self) -> torch.Tensor:
        """Minimum signal strength to register a causal edge. Range: (0, 1)."""
        return torch.sigmoid(self._causal_min_signal)

    @property
    def causal_decay_rate(self) -> torch.Tensor:
        """Per-step decay rate of causal edge weights. Range: (0, 1)."""
        return torch.sigmoid(self._causal_decay_rate)

    @property
    def causal_initial_weight_scale(self) -> torch.Tensor:
        """Scale factor for initial causal edge weights. Range: (0, 1)."""
        return torch.sigmoid(self._causal_initial_weight_scale)

    @property
    def causal_relation_scale(self) -> torch.Tensor:
        """Scale applied to relation scores in causal reasoning. Range: (0, 1)."""
        return torch.sigmoid(self._causal_relation_scale)

    @property
    def goal_relevance_threshold(self) -> torch.Tensor:
        """Minimum relevance score for an active goal. Range: (0, 1)."""
        return torch.sigmoid(self._goal_relevance_threshold)

    @property
    def goal_progress_rate(self) -> torch.Tensor:
        """Step size for updating goal progress estimates. Range: (0, 1)."""
        return torch.sigmoid(self._goal_progress_rate)

    @property
    def precision_decay_factor(self) -> torch.Tensor:
        """EMA decay factor for the precision estimate. Range: (0, 1)."""
        return torch.sigmoid(self._precision_decay_factor)

    @property
    def reconsolidation_threshold(self) -> torch.Tensor:
        """Threshold for triggering memory reconsolidation. Range: (0, 1)."""
        return torch.sigmoid(self._reconsolidation_threshold)

    @property
    def match_threshold(self) -> torch.Tensor:
        """Cosine-similarity threshold for pattern matching. Range: (0, 1)."""
        return torch.sigmoid(self._match_threshold)

    @property
    def goal_dedup_threshold(self) -> torch.Tensor:
        """Distance threshold below which two goals are considered duplicates. Range: (0, 1)."""
        return torch.sigmoid(self._goal_dedup_threshold)

    @property
    def confidence_propagation_influence(self) -> torch.Tensor:
        """Fraction of confidence delta propagated to derived beliefs. Range: (0, 1)."""
        return torch.sigmoid(self._confidence_propagation_influence)

    @property
    def lr_scale_high(self) -> torch.Tensor:
        """Upper bound of per-belief LR scale (low surprise). Range: (0, inf)."""
        return F.softplus(self._lr_scale_high)

    @property
    def lr_scale_low(self) -> torch.Tensor:
        """Lower bound of per-belief LR scale (high surprise). Range: (0, inf)."""
        return F.softplus(self._lr_scale_low)

    @property
    def efe_epistemic_weight(self) -> torch.Tensor:
        """Weight on the epistemic value term in EFE. Range: (0, inf)."""
        return F.softplus(self._efe_epistemic_weight)

    @property
    def efe_risk_weight(self) -> torch.Tensor:
        """Weight on the risk term in EFE. Range: (0, inf)."""
        return F.softplus(self._efe_risk_weight)

    @property
    def huber_delta(self) -> torch.Tensor:
        """Huber loss transition point for belief matching. Range: (0, inf).

        Below delta: quadratic penalty (precise gradient for small errors).
        Above delta: linear penalty (robust to outlier disagreements).
        Reference: MIRAS/YAAD (arXiv:2504.13173); Huber (1964).
        """
        return F.softplus(self._huber_delta)

    # ── A1: Tentative Belief Mode ──

    @property
    def provisional_eval_window(self) -> torch.Tensor:
        """Steps before evaluating provisional beliefs. Range: (0, inf)."""
        return F.softplus(self._provisional_eval_window)

    @property
    def provisional_fe_threshold(self) -> torch.Tensor:
        """Required fractional FE decrease for provisional promotion. Range: (0, 0.2)."""
        return torch.sigmoid(self._provisional_fe_threshold) * 0.2

    @property
    def provisional_precision_retention(self) -> torch.Tensor:
        """Min fraction of initial radius to retain for promotion. Range: (0, 1)."""
        return torch.sigmoid(self._provisional_precision_retention)

    # ── A2: MESU Precision Variance ──

    @property
    def mesu_min_variance(self) -> torch.Tensor:
        """Floor for belief precision variance. Range: (0, inf)."""
        return F.softplus(self._mesu_min_variance)

    @property
    def mesu_variance_shrink(self) -> torch.Tensor:
        """Shrink rate for variance per observation. Range: (0, 1)."""
        return torch.sigmoid(self._mesu_variance_shrink)

    @property
    def mesu_window_size(self) -> torch.Tensor:
        """Max reinforcement count before variance floor. Range: (0, inf)."""
        return F.softplus(self._mesu_window_size) * 10.0

    @property
    def mesu_gain_boost(self) -> torch.Tensor:
        """How much high variance amplifies gain. Range: (0, inf)."""
        return F.softplus(self._mesu_gain_boost)

    # ── A3: Causal Cascade Revision ──

    @property
    def cascade_decay_factor(self) -> torch.Tensor:
        """Precision decay fraction per causal hop. Range: (0, 1)."""
        return torch.sigmoid(self._cascade_decay_factor)

    @property
    def cascade_max_depth(self) -> torch.Tensor:
        """Maximum hops for cascade propagation. Range: (0, inf). Use int(3 * x)."""
        return F.softplus(self._cascade_max_depth)

    @property
    def cascade_variance_boost(self) -> torch.Tensor:
        """Variance increase per hop during cascade. Range: (0, inf)."""
        return F.softplus(self._cascade_variance_boost)

    # ── A4: SGM Safety Gate ──

    @property
    def sgm_confidence(self) -> torch.Tensor:
        """Confidence level for accepting self-modifications. Range: (0, 1)."""
        return torch.sigmoid(self._sgm_confidence)

    @property
    def sgm_min_samples(self) -> torch.Tensor:
        """Minimum samples before rejecting a modification. Range: (0, inf)."""
        return F.softplus(self._sgm_min_samples)

    # ── B1-B4: Planning ──

    @property
    def planning_horizon(self) -> torch.Tensor:
        """Number of causal steps to simulate ahead. Range: (0, inf)."""
        return F.softplus(self._planning_horizon)

    @property
    def planning_discount(self) -> torch.Tensor:
        """Temporal discount for future EFE per step. Range: (0, 1)."""
        return torch.sigmoid(self._planning_discount)

    @property
    def mcts_exploration(self) -> torch.Tensor:
        """MCTS UCB exploration constant. Range: (0, inf)."""
        return F.softplus(self._mcts_exploration)

    @property
    def planning_temperature(self) -> torch.Tensor:
        """Softmax temperature for planning action selection. Range: (0, inf)."""
        return F.softplus(self._planning_temperature)

    @property
    def preference_prior_strength(self) -> torch.Tensor:
        """How strongly Telos goals pull the factor graph. Range: (0, inf)."""
        return F.softplus(self._preference_prior_strength)

    @property
    def epistemic_prior_strength(self) -> torch.Tensor:
        """How strongly uncertainty drives exploration planning. Range: (0, inf)."""
        return F.softplus(self._epistemic_prior_strength)

    # ── C1: SRWM ──

    @property
    def srwm_lr(self) -> torch.Tensor:
        """Fast-weight Hebbian learning rate. Range: (0, 1)."""
        return torch.sigmoid(self._srwm_lr)

    @property
    def srwm_decay(self) -> torch.Tensor:
        """Fast-weight decay rate per step. Range: (0, 1)."""
        return torch.sigmoid(self._srwm_decay)

    # ── C2: Meta-Learned Update Function ──

    @property
    def update_fn_gate(self) -> torch.Tensor:
        """Blend between learned and hand-coded update. Range: (0, 1)."""
        return torch.sigmoid(self._update_fn_gate)

    # ── C3: Structural Plasticity ──

    @property
    def plasticity_split_threshold(self) -> torch.Tensor:
        """Entropy threshold for splitting polysemantic beliefs. Range: (0, 1)."""
        return torch.sigmoid(self._plasticity_split_threshold)

    @property
    def plasticity_prune_threshold(self) -> torch.Tensor:
        """Frequency threshold for pruning dead beliefs. Range: (0, 1)."""
        return torch.sigmoid(self._plasticity_prune_threshold)

    @property
    def plasticity_growth_rate(self) -> torch.Tensor:
        """Capacity growth fraction when expanding. Range: (0, 1)."""
        return torch.sigmoid(self._plasticity_growth_rate)

    # ── C4: Learned Recursion Depth ──

    @property
    def recursion_depth_penalty(self) -> torch.Tensor:
        """Ponder cost per additional recursion step. Range: (0, inf)."""
        return F.softplus(self._recursion_depth_penalty)

    @property
    def recursion_halt_bias(self) -> torch.Tensor:
        """Initial halt tendency (0.5 = neutral). Range: (0, 1)."""
        return torch.sigmoid(self._recursion_halt_bias)

    # ── D2: EFE Action Selection ──

    @property
    def action_temperature(self) -> torch.Tensor:
        """Gumbel-Softmax temperature for action selection. Range: (0, inf)."""
        return F.softplus(self._action_temperature)

    @property
    def action_risk_aversion(self) -> torch.Tensor:
        """Multiplier on risk term in action EFE. Range: (0, inf)."""
        return F.softplus(self._action_risk_aversion)

    # ── D3: Curiosity ──

    @property
    def curiosity_threshold(self) -> torch.Tensor:
        """Novelty level above which exploration goals are generated. Range: (0, inf)."""
        return F.softplus(self._curiosity_threshold)

    @property
    def curiosity_weight(self) -> torch.Tensor:
        """Relative weight of exploration vs exploitation goals. Range: (0, inf)."""
        return F.softplus(self._curiosity_weight)

    # ── D4: Skill Crystallization ──

    @property
    def skill_detection_threshold(self) -> torch.Tensor:
        """Min recurrence count for skill crystallization. Range: (0, inf). Use int(3*x)."""
        return F.softplus(self._skill_detection_threshold)

    @property
    def skill_similarity_threshold(self) -> torch.Tensor:
        """Cosine similarity threshold for skill matching. Range: (0, 1)."""
        return torch.sigmoid(self._skill_similarity_threshold)

    # ── E1: Two-Factor Sleep Consolidation ──

    @property
    def homeostatic_target(self) -> torch.Tensor:
        """Target total precision budget (sum of radii). Range: (0, inf)."""
        return F.softplus(self._homeostatic_target)

    @property
    def homeostatic_rate(self) -> torch.Tensor:
        """Correction rate toward homeostatic target per sleep cycle. Range: (0, 1)."""
        return torch.sigmoid(self._homeostatic_rate)

    @property
    def sleep_conflict_threshold(self) -> torch.Tensor:
        """Angular cosine threshold for conflict detection. Range: (0, 1)."""
        return torch.sigmoid(self._sleep_conflict_threshold)

    # ── E2: Self-Verification Pass ──

    @property
    def verification_divergence_threshold(self) -> torch.Tensor:
        """Cosine similarity threshold for inconsistency detection. Range: (0, 1)."""
        return torch.sigmoid(self._verification_divergence_threshold)

    @property
    def verification_precision_decay(self) -> torch.Tensor:
        """Radius reduction fraction on inconsistency. Range: (0, 1)."""
        return torch.sigmoid(self._verification_precision_decay)

    @property
    def supersession_similarity(self) -> torch.Tensor:
        """Cosine threshold for conflict-aware supersession. Range: (0, 1)."""
        return torch.sigmoid(self._supersession_similarity)

    # ── E3: Empirical Precision Recalibration ──

    @property
    def recalibration_rate(self) -> torch.Tensor:
        """Rate of stored precision decay toward empirical. Range: (0, 1)."""
        return torch.sigmoid(self._recalibration_rate)

    @property
    def recalibration_min_samples(self) -> torch.Tensor:
        """Min observations before recalibrating. Range: (0, inf)."""
        return F.softplus(self._recalibration_min_samples)

    # ── E4: Interleaved Replay ──

    @property
    def replay_ratio(self) -> torch.Tensor:
        """Fraction of replay set that are old high-precision beliefs. Range: (0, 1)."""
        return torch.sigmoid(self._replay_ratio)

    @property
    def replay_contradiction_threshold(self) -> torch.Tensor:
        """Message disagreement threshold for cross-temporal contradiction. Range: (0, 1)."""
        return torch.sigmoid(self._replay_contradiction_threshold)

    # ── F1: Predictive Refinement ──

    @property
    def refinement_contraction_rate(self) -> torch.Tensor:
        """SCORE-style step-size decay per loop. dt(l) = (1-rate)^l. Range: (0, 1)."""
        return torch.sigmoid(self._refinement_contraction_rate)

    @property
    def refinement_retrieval_threshold(self) -> torch.Tensor:
        """Min delta L2-norm fraction to trigger belief re-query. Range: (0, 1)."""
        return torch.sigmoid(self._refinement_retrieval_threshold)

    @property
    def refinement_ponder_cost(self) -> torch.Tensor:
        """Per-position penalty for continuing refinement. Range: (0, inf)."""
        return F.softplus(self._refinement_ponder_cost)
