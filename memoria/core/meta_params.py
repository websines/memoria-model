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
