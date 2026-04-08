"""Bethe Free Energy and Expected Free Energy (EFE) over the cognitive state.

Three computations:
1. compute_bethe_free_energy(): Proper Bethe free energy on the belief factor graph.
   F_B = Σ_factors U_a - Σ_factors H_a + Σ_variables (d_i - 1) × H_i
   Uses Power Spherical entropy (closed-form via digamma, no Bessel functions).
   Fully differentiable — gradients flow into beliefs, edge weights, and relations.

2. compute_expected_free_energy(): Full Active Inference EFE decomposition.
   EFE = -pragmatic + w_e * epistemic + w_r * risk
   Weights w_e, w_r are learned nn.Parameters (via MetaParams).

3. compute_free_energy(): Legacy computation for beta/stats (kept for backward compat).

Reference: Yedidia, Freeman, Weiss "Constructing Free Energy Approximations" (2005)
Reference: De Cao & Aziz "Power Spherical Distribution" (2020) — entropy formula
Reference: Friston et al., "Active Inference and Epistemic Value"
Reference: "The Missing Reward" (arXiv:2508.05619)
Reference: RxInfer.jl (Bethe free energy on factor graphs)
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from .polar import angular_similarity, belief_is_active, EPSILON
from .state import CognitiveState
from .message_passing import FactorGraphMessagePassing, compute_energy_from_messages


def power_spherical_entropy(kappa: Tensor, d: int) -> Tensor:
    """Entropy of the Power Spherical distribution (De Cao & Aziz, 2020).

    Closed-form via digamma — no Bessel functions. Fully differentiable.

    The Power Spherical distribution models directional data on S^{d-1}:
        p(x; μ, κ) = C(κ,d) × (1 + μᵀx)^κ

    Args:
        kappa: [N] concentration parameters (= belief radii). Must be > 0.
        d: dimensionality of the belief vectors

    Returns:
        [N] entropy in nats, monotonically decreasing with kappa
    """
    alpha = (d - 1) / 2.0 + kappa       # [N]
    beta_val = (d - 1) / 2.0             # scalar

    log_norm = (
        (alpha + beta_val) * math.log(2)
        + beta_val * math.log(math.pi)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + beta_val)
    )
    entropy = log_norm - kappa * (
        math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + torch.tensor(beta_val, device=kappa.device))
    )
    return entropy


def compute_expected_free_energy(
    state: CognitiveState,
    retrieved_beliefs: Tensor | None = None,
    observation: Tensor | None = None,
) -> dict:
    """Full Expected Free Energy decomposition (Active Inference).

    EFE = -pragmatic + w_epistemic * epistemic + w_risk * risk

    Where:
    - pragmatic_value: alignment of current beliefs with active goals
      (how much does this state serve our objectives?)
    - epistemic_value: expected information gain from the current belief state
      measured by Power Spherical entropy (high entropy = more to learn)
    - risk: cosine disagreement between retrieved beliefs and current observations
      (how far is the model's prediction from what we actually see?)

    The combination weights w_epistemic and w_risk are learned nn.Parameters
    (softplus-constrained, stored in state.meta_params) — never hardcoded.

    Reference: Friston et al., "Active Inference and Epistemic Value"
    Reference: "The Missing Reward" (arXiv:2508.05619)

    Args:
        state: cognitive state with beliefs, goals, and meta_params
        retrieved_beliefs: [B, T, D] beliefs retrieved during read path (optional)
        observation: [B, T, D] current observation from write path (optional)

    Returns:
        dict with scalar tensors: efe, pragmatic, epistemic, risk
    """
    device = state.beliefs.device
    D = state.config.belief_dim

    active_mask = state.get_active_mask()
    if not active_mask.any():
        zero = torch.tensor(0.0, device=device)
        return {'efe': zero, 'pragmatic': zero, 'epistemic': zero, 'risk': zero}

    active_beliefs = state.beliefs[active_mask]          # [N_active, D]
    active_radii = active_beliefs.norm(dim=-1)            # [N_active]

    # ── Pragmatic value: goal alignment ──────────────────────────────────────
    # How well does the current belief state serve active goals?
    # Computed as the mean over active beliefs of their best-matching goal cosine sim,
    # weighted by goal priority. Range: [-1, 1]; higher = better goal alignment.
    indices, goal_embeds, goal_meta = state.get_active_goals()
    if len(indices) > 0:
        goal_angles = F.normalize(goal_embeds, dim=-1, eps=EPSILON)   # [G, D]
        belief_angles = F.normalize(active_beliefs, dim=-1, eps=EPSILON)  # [N, D]
        # Similarity matrix: [N_active, N_goals]
        sim = belief_angles @ goal_angles.T
        # Priority column vector: goal_meta[:, 0] = priority
        priority = goal_meta[:, 0].clamp(min=EPSILON)                 # [N_goals]
        # Weight similarities by goal priority; take best-matching goal per belief
        weighted_sim = sim * priority.unsqueeze(0)                    # [N, G]
        pragmatic = weighted_sim.max(dim=-1).values.mean()            # scalar
    else:
        pragmatic = torch.tensor(0.0, device=device)

    # ── Epistemic value: expected information gain ────────────────────────────
    # Measured by the Power Spherical entropy of active beliefs.
    # High entropy (uncertain beliefs, small radius) = high epistemic value.
    # The gradient through this term incentivises the system to seek states
    # that reduce uncertainty (the "Bayesian surprise" drive).
    kappa = active_radii.clamp(min=EPSILON)
    H_per_belief = power_spherical_entropy(kappa, D)      # [N_active]
    epistemic = H_per_belief.mean()                       # scalar (mean over active)

    # ── Risk: prediction-observation disagreement ─────────────────────────────
    # If we have retrieved beliefs (model prediction) and the actual observation,
    # risk is the mean cosine disagreement — how wrong is the model right now?
    # Range: [0, 2]; 0 = perfect agreement, 2 = complete opposition.
    if retrieved_beliefs is not None and observation is not None:
        ret_flat = retrieved_beliefs.reshape(-1, D)       # [N, D]
        obs_flat = observation.reshape(-1, D)             # [N, D]
        cos_sim = F.cosine_similarity(ret_flat, obs_flat, dim=-1)   # [N]
        # Huber loss: quadratic for small disagreements, linear for outliers.
        # Prevents catastrophic belief updates from spurious observation matches.
        # Reference: MIRAS/YAAD (arXiv:2504.13173); Huber (1964)
        disagreement = 1.0 - cos_sim                     # [N], range [0, 2]
        delta = float(state.meta_params.huber_delta)
        risk = F.huber_loss(
            disagreement, torch.zeros_like(disagreement),
            reduction='mean', delta=delta,
        )                                                 # scalar
    else:
        risk = torch.tensor(0.0, device=device)

    # ── Combine with learned weights from meta_params ─────────────────────────
    # w_epistemic and w_risk are softplus(raw_param) — strictly positive, no upper bound.
    # They are trained by the main optimizer alongside all other parameters.
    # EFE = -pragmatic + w_e * epistemic + w_r * risk
    # The negative sign on pragmatic: higher goal alignment = lower free energy (good).
    w_epistemic = state.meta_params.efe_epistemic_weight  # learned, (0, ∞)
    w_risk = state.meta_params.efe_risk_weight            # learned, (0, ∞)

    efe = -pragmatic + w_epistemic * epistemic + w_risk * risk

    return {
        'efe': efe,
        'pragmatic': pragmatic,
        'epistemic': epistemic,
        'risk': risk,
    }


def compute_bethe_free_energy(state: CognitiveState, temperature: float = 5.0) -> dict:
    """Proper Bethe free energy over the cognitive state factor graph.

    F_B = Σ_a U_a + Σ_i (d_i - 1) × H_i

    Where:
    - U_a = per-factor energy (disagreement between connected beliefs)
    - H_i = per-variable entropy (Power Spherical, based on belief precision)
    - d_i = degree of variable node i (number of edges connected to belief i)

    The (d_i - 1) counting correction prevents overcounting entropy for beliefs
    involved in multiple relations. Without it, highly-connected beliefs have
    their uncertainty weighted too heavily.

    Fully differentiable: gradients flow into beliefs, edge_weights, edge_relations.

    Args:
        state: cognitive state with differentiable beliefs and edges
        temperature: scales sigmoid sharpness in energy computation

    Returns:
        dict with free_energy (scalar), energy, entropy, beta, and per-component terms
    """
    device = state.beliefs.device
    D = state.config.belief_dim

    # --- Per-variable entropy H_i (Power Spherical) ---
    radii = state.beliefs.norm(dim=-1)   # [max_beliefs]
    active = belief_is_active(radii)

    if not active.any():
        zero = torch.tensor(0.0, device=device)
        return {
            'free_energy': zero, 'energy': zero, 'entropy': zero,
            'beta': torch.tensor(1.0, device=device),
        }

    active_indices = active.nonzero(as_tuple=False).squeeze(-1)
    active_radii = radii[active_indices]
    # Clamp kappa > 0 for numerical stability in digamma
    kappa = active_radii.clamp(min=1e-4)
    H_per_var = power_spherical_entropy(kappa, D)  # [N_active]

    # --- Compute degree d_i for each active belief ---
    src_idx, tgt_idx, relations, weights = state.get_active_edges()
    n_active = len(active_indices)

    if len(src_idx) > 0:
        # Map global belief indices to local active-belief indices
        # Build inverse map: global_idx -> local_idx
        idx_map = torch.full((state.config.max_beliefs,), -1, dtype=torch.long, device=device)
        idx_map[active_indices] = torch.arange(n_active, device=device)

        src_local = idx_map[src_idx]   # [N_edges]
        tgt_local = idx_map[tgt_idx]   # [N_edges]

        # Only count edges where both endpoints are active
        valid = (src_local >= 0) & (tgt_local >= 0)
        src_valid = src_local[valid]
        tgt_valid = tgt_local[valid]

        degree = torch.zeros(n_active, device=device)
        degree.scatter_add_(0, src_valid, torch.ones_like(src_valid, dtype=degree.dtype))
        degree.scatter_add_(0, tgt_valid, torch.ones_like(tgt_valid, dtype=degree.dtype))
    else:
        degree = torch.zeros(n_active, device=device)

    # --- Per-factor energy U_a ---
    if len(src_idx) > 0:
        src_beliefs = state.beliefs[src_idx]   # [N_edges, D]
        tgt_beliefs = state.beliefs[tgt_idx]   # [N_edges, D]

        src_radii = src_beliefs.norm(dim=-1).clamp(min=EPSILON)
        tgt_radii = tgt_beliefs.norm(dim=-1).clamp(min=EPSILON)

        src_angles = src_beliefs / src_radii.unsqueeze(-1)
        tgt_angles = tgt_beliefs / tgt_radii.unsqueeze(-1)

        # Transform target through relation vector
        K = relations.shape[-1]
        if K < D:
            tgt_transformed = tgt_angles.clone()
            tgt_transformed[:, :K] = tgt_transformed[:, :K] + relations
            tgt_transformed = torch.nn.functional.normalize(tgt_transformed, dim=-1, eps=EPSILON)
        else:
            tgt_transformed = torch.nn.functional.normalize(
                tgt_angles + relations[:, :D], dim=-1, eps=EPSILON
            )

        agreement = angular_similarity(src_angles, tgt_transformed)
        log_sigmoid = torch.nn.functional.logsigmoid(agreement * temperature)
        energy_per_factor = -weights * src_radii * tgt_radii * log_sigmoid

        total_energy = energy_per_factor.sum()
    else:
        total_energy = torch.tensor(0.0, device=device)

    # --- Telos energy (unfinished goals increase free energy) ---
    indices, embeddings, metadata = state.get_active_goals()
    if len(indices) > 0:
        priority = metadata[:, 0]
        progress = metadata[:, 2]
        goal_radii = embeddings.norm(dim=-1)
        importance = priority * goal_radii
        goal_distance = (1.0 - progress).clamp(min=0.0)
        telos_energy = (importance * goal_distance).sum()
    else:
        telos_energy = torch.tensor(0.0, device=device)

    # --- Bethe free energy ---
    # F_B = Σ_a U_a + Σ_i (d_i - 1) × H_i
    # Note: (d_i - 1) can be negative for isolated beliefs (d_i=0 → weight=-1),
    # which means isolated beliefs' entropy is SUBTRACTED (they increase uncertainty,
    # reducing free energy — encouraging exploration of unconnected beliefs).
    counting_correction = degree - 1.0
    total_entropy = (counting_correction * H_per_var).sum()

    full_energy = total_energy + telos_energy
    free_energy = full_energy + total_entropy  # F = E + (d-1)*H (sign convention)

    # β: exploration/exploitation balance
    H_raw = H_per_var.sum()
    beta = H_raw / (full_energy.abs() + H_raw + EPSILON)
    beta = beta.clamp(0.0, 1.0)

    with torch.no_grad():
        state.meta.data[0] = beta.item()

    return {
        'free_energy': free_energy,
        'energy': full_energy,
        'relation_energy': total_energy,
        'telos_energy': telos_energy,
        'entropy': H_raw,
        'bethe_entropy': total_entropy,
        'beta': beta,
    }


def compute_energy(state: CognitiveState, temperature: float = 5.0) -> Tensor:
    """Compute the energy term E over all active relations.

    For each active edge f connecting beliefs i, j:
        agreement = cosine_sim(angle_i, transform(angle_j, relation_f))
        E_f = -w_f × r_i × r_j × log(σ(agreement × temperature))

    High precision beliefs (large r) connected by strong edges (large w)
    that disagree (low agreement) → high energy → strong gradient to fix.

    Args:
        state: The cognitive state
        temperature: Scales the sigmoid sharpness. Higher = sharper agreement/disagreement.

    Returns:
        Scalar energy tensor (differentiable)
    """
    src_idx, tgt_idx, relations, weights = state.get_active_edges()

    if len(src_idx) == 0:
        return torch.tensor(0.0, device=state.beliefs.device)

    # Get belief vectors for edge endpoints
    src_beliefs = state.beliefs[src_idx]   # [N_edges, D]
    tgt_beliefs = state.beliefs[tgt_idx]   # [N_edges, D]

    # Radii (precision) of endpoints
    src_radii = src_beliefs.norm(dim=-1)   # [N_edges]
    tgt_radii = tgt_beliefs.norm(dim=-1)   # [N_edges]

    # Angles (content direction) of endpoints
    src_angles = src_beliefs / src_radii.unsqueeze(-1).clamp(min=EPSILON)
    tgt_angles = tgt_beliefs / tgt_radii.unsqueeze(-1).clamp(min=EPSILON)

    # Transform target angle through relation representation.
    # Simple version: relation vector acts as a bias/rotation on what "agreement" means.
    # For two beliefs to "agree" through a relation, the target's angle projected through
    # the relation should be similar to the source's angle.
    #
    # We use: agreement = cosine_sim(src_angle, tgt_angle + relation_projection)
    # where relation_projection is the relation vector projected into belief space.
    #
    # For relations with dim K < D, we pad with zeros (only first K dims are modulated).
    K = relations.shape[-1]
    D = src_angles.shape[-1]

    if K < D:
        # Project relation into belief space: first K dims are shifted
        tgt_transformed = tgt_angles.clone()
        tgt_transformed[:, :K] = tgt_transformed[:, :K] + relations
        # Re-normalize to unit sphere
        tgt_transformed = torch.nn.functional.normalize(tgt_transformed, dim=-1, eps=EPSILON)
    else:
        tgt_transformed = torch.nn.functional.normalize(tgt_angles + relations[:, :D], dim=-1, eps=EPSILON)

    # Agreement: cosine similarity between source and transformed target
    agreement = angular_similarity(src_angles, tgt_transformed)  # [N_edges]

    # Energy per edge: -w × r_i × r_j × log(σ(agreement × temp))
    log_sigmoid = torch.nn.functional.logsigmoid(agreement * temperature)  # [N_edges]
    energy_per_edge = -weights * src_radii * tgt_radii * log_sigmoid

    return energy_per_edge.sum()


def compute_entropy(state: CognitiveState) -> Tensor:
    """Compute the entropy term H over all active beliefs.

    Uses Power Spherical entropy: proper information-theoretic quantity
    that depends on both concentration (radius) and dimensionality.

    Only computed over active beliefs (radius > threshold).

    Args:
        state: The cognitive state

    Returns:
        Scalar entropy tensor (differentiable, always positive)
    """
    D = state.config.belief_dim
    radii = state.beliefs.norm(dim=-1)  # [N_beliefs]
    active = belief_is_active(radii)

    if not active.any():
        return torch.tensor(0.0, device=state.beliefs.device)

    active_radii = radii[active]
    kappa = active_radii.clamp(min=1e-4)
    entropy_per_belief = power_spherical_entropy(kappa, D)

    return entropy_per_belief.sum()


def compute_telos_energy(state: CognitiveState) -> Tensor:
    """Compute energy contribution from active goals (Telos preferences).

    Unachieved important goals increase free energy, driving the system toward action.

    For each active goal:
        importance = priority × goal_radius
        goal_distance = 1.0 - progress
        E_telos = importance × goal_distance

    Energy is always non-negative: important unfinished goals add pressure,
    completed goals (distance=0) contribute nothing.

    Args:
        state: The cognitive state

    Returns:
        Scalar energy tensor
    """
    indices, embeddings, metadata = state.get_active_goals()

    if len(indices) == 0:
        return torch.tensor(0.0, device=state.beliefs.device)

    priority = metadata[:, 0]    # [N_goals]
    progress = metadata[:, 2]    # [N_goals]

    # Goal importance from priority and embedding radius (confidence in the goal)
    goal_radii = embeddings.norm(dim=-1)  # [N_goals]
    importance = priority * goal_radii

    goal_distance = (1.0 - progress).clamp(min=0.0)

    # Energy is always positive: unfinished important goals increase free energy,
    # creating pressure to resolve them. Completed goals contribute nothing.
    energy_per_goal = importance * goal_distance

    return energy_per_goal.sum()


def compute_free_energy(state: CognitiveState, temperature: float = 5.0) -> dict:
    """Compute Bethe free energy over the entire cognitive state.

    F = E_relations + E_telos - H_beliefs

    Also computes β = H / (|E| + H + ε)

    Args:
        state: The cognitive state
        temperature: Temperature for energy computation

    Returns:
        Dict with:
            free_energy: scalar tensor (the loss)
            energy: scalar tensor (relation energy + telos energy)
            entropy: scalar tensor
            beta: scalar tensor (exploration/exploitation)
    """
    E_relations = compute_energy(state, temperature)
    E_telos = compute_telos_energy(state)
    H = compute_entropy(state)

    total_energy = E_relations + E_telos
    free_energy = total_energy - H

    # β: exploration/exploitation balance
    # Power Spherical entropy is negative for continuous distributions (more
    # concentrated → more negative). We use -H as a measure of confidence:
    # high confidence (-H large) → low β → exploit.
    # low confidence (-H small) → high β → explore.
    # Normalize per-unit so scales are comparable.
    n_active = max(state.num_active_beliefs(), 1)
    n_edges = max(state.num_active_edges(), 1)
    uncertainty = -H / n_active   # positive, smaller when more confident
    energy_density = total_energy.abs() / n_edges if state.num_active_edges() > 0 else torch.tensor(0.0, device=H.device)
    # When uncertainty dominates → explore. When energy dominates → exploit.
    beta = uncertainty / (energy_density + uncertainty + EPSILON)
    beta = beta.clamp(0.0, 1.0)

    # Update meta region with computed β
    with torch.no_grad():
        state.meta.data[0] = beta.item()

    return {
        'free_energy': free_energy,
        'energy': total_energy,
        'relation_energy': E_relations,
        'telos_energy': E_telos,
        'entropy': H,
        'beta': beta,
    }
