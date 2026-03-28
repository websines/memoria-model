"""Bethe Free Energy computation over the cognitive state.

F = E - H

Where:
- E (energy) = sum over relations of disagreement between connected beliefs.
  High-precision beliefs connected by strong edges that disagree → high energy.
- H (entropy) = sum over beliefs of uncertainty.
  High-radius beliefs → low entropy (confident). Low-radius → high entropy (uncertain).

Minimizing F forces:
- Beliefs to be consistent with each other (reduce E)
- Beliefs to maintain appropriate uncertainty (balance E and H)
- Overconfident wrong beliefs to be corrected (high E without low H)

β = H / (E + H + ε) — the exploration/exploitation parameter, computed from state.

Ported from: prototype-research/src/aif/free_energy.rs
Reference: RxInfer.jl (Bethe free energy on factor graphs)
"""

import torch
from torch import Tensor

from .polar import angular_similarity, belief_is_active, EPSILON
from .state import CognitiveState


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

    H_i = -log(r_i + ε)

    High precision (large radius) → low entropy → confident belief.
    Low precision (small radius) → high entropy → uncertain belief.

    Only computed over active beliefs (radius > threshold).

    Args:
        state: The cognitive state

    Returns:
        Scalar entropy tensor (differentiable)
    """
    radii = state.beliefs.norm(dim=-1)  # [N_beliefs]
    active = belief_is_active(radii)

    if not active.any():
        return torch.tensor(0.0, device=state.beliefs.device)

    active_radii = radii[active]
    entropy_per_belief = -torch.log(active_radii + EPSILON)

    return entropy_per_belief.sum()


def compute_telos_energy(state: CognitiveState) -> Tensor:
    """Compute energy contribution from active goals (Telos preferences).

    Unachieved important goals increase free energy, driving the system toward action.

    For each active goal:
        preference_precision = priority × confidence (approximated by goal embedding radius)
        goal_distance = 1.0 - progress
        E_telos = -log(preference_precision + ε) × goal_distance

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

    # Goal "precision" from embedding radius (confidence in the goal)
    goal_radii = embeddings.norm(dim=-1)  # [N_goals]
    preference_precision = priority * goal_radii

    goal_distance = (1.0 - progress).clamp(min=0.0)

    # Only contribute energy for goals with meaningful distance remaining
    energy_per_goal = -torch.log(preference_precision + EPSILON) * goal_distance

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
    # High entropy (uncertain) → high β → explore
    # High energy (inconsistent) → low β → exploit (fix inconsistencies)
    beta = H / (total_energy.abs() + H + EPSILON)
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
