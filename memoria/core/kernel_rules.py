"""Kernel rules: inviolable constraints on the cognitive state.

Hard masks applied as element-wise multiplication on update tensors.
If immutable_beliefs[i] = True, belief i cannot be modified.
Mathematically impossible to violate — the mask is inside the computation graph.

Ported from: prototype-research/src/types/kernel.rs
"""

import torch
from ..core.state import CognitiveState


def mark_belief_immutable(state: CognitiveState, index: int):
    """Mark a belief as immutable (kernel rule: cannot be modified or deleted)."""
    state.immutable_beliefs[index] = True


def mark_belief_mutable(state: CognitiveState, index: int):
    """Remove immutability from a belief."""
    state.immutable_beliefs[index] = False


def mark_edge_immutable(state: CognitiveState, index: int):
    """Mark an edge as immutable."""
    state.immutable_edges[index] = True


def mark_goal_immutable(state: CognitiveState, index: int):
    """Mark a goal as immutable (enterprise directive: cannot be abandoned)."""
    state.immutable_goals[index] = True


def verify_kernel_integrity(state: CognitiveState, snapshot: dict) -> list[str]:
    """Verify that no immutable beliefs/edges/goals were modified since snapshot.

    Args:
        state: current state
        snapshot: previous state_dict_cognitive() to compare against

    Returns:
        List of violation descriptions (empty if all good)
    """
    violations = []

    # Check immutable beliefs
    for i in range(state.config.max_beliefs):
        if state.immutable_beliefs[i]:
            current = state.beliefs.data[i]
            previous = snapshot['beliefs'][i]
            if not torch.allclose(current, previous, atol=1e-6):
                violations.append(f"Immutable belief {i} was modified")

    # Check immutable edges
    for i in range(state.config.max_edges):
        if state.immutable_edges[i]:
            current_w = state.edge_weights.data[i]
            previous_w = snapshot['edge_weights'][i]
            if abs(current_w.item() - previous_w.item()) > 1e-6:
                violations.append(f"Immutable edge {i} weight was modified")

    # Check immutable goals
    for i in range(state.config.max_goals):
        if state.immutable_goals[i]:
            current_status = state.goal_metadata.data[i, 3]
            previous_status = snapshot['goal_metadata'][i, 3]
            if abs(current_status.item() - previous_status.item()) > 1e-6:
                violations.append(f"Immutable goal {i} status was modified")

    return violations
