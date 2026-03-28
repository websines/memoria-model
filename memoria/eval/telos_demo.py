"""Telos evaluation: intrinsic goal generation from surprise.

Tests: does the model autonomously generate investigation goals when it
encounters persistent uncertainty in a domain?
"""

import torch
from ..model.memoria_model import MemoriaModel
from ..cognition.pass2 import run_pass2
from ..cognition.telos import STATUS_ACTIVE, STATUS_PROPOSED, STATUS
from ..data.tokenizer import get_tokenizer


def evaluate_telos(
    model: MemoriaModel,
    num_interactions: int = 100,
) -> dict:
    """Evaluate Telos goal generation.

    Feed the model a stream of information with deliberate gaps/contradictions.
    Check if intrinsic goals emerge.

    Returns:
        dict with goals_generated, goal_types, surprise_correlation
    """
    tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    model.eval()

    # Reset state
    with torch.no_grad():
        model.state.beliefs.data.zero_()
        model.state.edge_weights.data.zero_()
        model.state.edge_active.zero_()
        model.state.goal_metadata.data.zero_()
        model.state.goal_embeddings.data.zero_()
        model.state.meta.data[0] = 1.0  # β = max exploration
        model.state.meta.data[1] = 0.0  # accumulated surprise = 0

    goals_over_time = []
    surprise_over_time = []

    # Feed a stream with increasing contradiction/uncertainty
    statements = _generate_surprising_stream(num_interactions)

    for step, text in enumerate(statements):
        tokens = tokenizer.encode(text, return_tensors='pt').to(device)
        with torch.no_grad():
            result = model.forward(tokens)

        stats = run_pass2(
            model.state, result['candidates'], [],
            current_step=step, is_sequence_boundary=(step % 5 == 0),
        )
        model.detach_state()

        goals_over_time.append(model.state.num_active_goals())
        surprise_over_time.append(stats['total_surprise'])

    # Analyze
    total_goals = model.state.num_active_goals()
    _, _, goal_meta = model.state.get_active_goals()

    active_count = 0
    proposed_count = 0
    if len(goal_meta) > 0:
        for g in range(len(goal_meta)):
            status = goal_meta[g, STATUS].item()
            if abs(status - STATUS_ACTIVE) < 0.05:
                active_count += 1
            elif abs(status - STATUS_PROPOSED) < 0.05:
                proposed_count += 1

    return {
        'total_goals_generated': total_goals,
        'active_goals': active_count,
        'proposed_goals': proposed_count,
        'goals_over_time': goals_over_time,
        'surprise_over_time': surprise_over_time,
        'final_beta': model.state.beta,
        'goal_generation_triggered': total_goals > 0,
    }


def _generate_surprising_stream(n: int) -> list[str]:
    """Generate a stream with increasing surprise/contradiction."""
    import random
    statements = []

    # Phase 1: consistent facts (low surprise)
    for _ in range(n // 3):
        statements.append(random.choice([
            "Alice works at Acme as a software engineer.",
            "The Acme codebase uses Python and PostgreSQL.",
            "Bob manages the infrastructure team at Acme.",
        ]))

    # Phase 2: introduce contradictions (rising surprise)
    for _ in range(n // 3):
        statements.append(random.choice([
            "Alice actually works at Globex, not Acme.",
            "The Acme codebase was recently migrated to Rust.",
            "Bob was moved to the product team, not infrastructure.",
            "Acme is shutting down their PostgreSQL databases.",
        ]))

    # Phase 3: new unexplained domain (high surprise)
    for _ in range(n // 3):
        statements.append(random.choice([
            "The quantum computing division reported anomalous results.",
            "A new encryption protocol failed all validation tests.",
            "The satellite link experienced unexplained latency spikes.",
            "Core temperature readings don't match any known pattern.",
        ]))

    return statements
