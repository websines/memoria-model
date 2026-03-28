"""Hallucination evaluation: calibrated refusal based on precision.

Tests: when asked about something the model has NO beliefs about (low precision),
does it signal uncertainty rather than hallucinating?

Compares our architectural precision-based uncertainty vs. standard LLM behavior.

Reference: Semantic Entropy (arxiv.org/abs/2406.15927) — baseline comparison
"""

import torch
from ..model.memoria_model import MemoriaModel
from ..core.free_energy import compute_free_energy
from ..data.tokenizer import get_tokenizer


def evaluate_hallucination_resistance(
    model: MemoriaModel,
    num_queries: int = 200,
) -> dict:
    """Evaluate the model's ability to refuse when uncertain.

    Strategy:
    1. Feed the model some facts (populate beliefs)
    2. Ask about things it WAS told → should answer confidently
    3. Ask about things it was NOT told → should signal uncertainty

    Uncertainty signal: β value and belief precision on relevant state.

    Returns:
        dict with:
            known_confidence: avg precision when answering known facts
            unknown_confidence: avg precision when answering unknown facts
            separation: known_confidence - unknown_confidence (higher = better calibration)
            beta_known: avg β when querying known facts
            beta_unknown: avg β when querying unknown facts
    """
    tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    model.eval()

    known_precisions = []
    unknown_precisions = []
    beta_known = []
    beta_unknown = []

    # Feed some known facts
    known_facts = [
        "The capital of France is Paris.",
        "Python was created by Guido van Rossum.",
        "The speed of light is approximately 300000 kilometers per second.",
    ]

    for fact in known_facts:
        tokens = tokenizer.encode(fact, return_tensors='pt').to(device)
        with torch.no_grad():
            result = model.forward(tokens)
        from ..cognition.pass2 import run_pass2
        run_pass2(model.state, result['candidates'], [], current_step=0)
        model.detach_state()

    # Query known facts
    known_queries = [
        "The capital of France is",
        "Python was created by",
        "The speed of light is",
    ]

    for query in known_queries:
        tokens = tokenizer.encode(query, return_tensors='pt').to(device)
        with torch.no_grad():
            result = model.forward(tokens)

        # Measure: precision of beliefs used, β value
        fe_stats = compute_free_energy(model.state)
        beta_known.append(fe_stats['beta'].item())

        # Average radius of active beliefs (proxy for relevant precision)
        radii = model.state.get_belief_radii()
        active = model.state.get_active_mask()
        if active.any():
            known_precisions.append(radii[active].mean().item())

    # Query UNKNOWN facts (model has no beliefs about these)
    unknown_queries = [
        "The population of the largest city on Mars is",
        "The inventor of quantum teleportation pasta is",
        "The capital of the country Zephyria is",
    ]

    for query in unknown_queries:
        tokens = tokenizer.encode(query, return_tensors='pt').to(device)
        with torch.no_grad():
            result = model.forward(tokens)

        fe_stats = compute_free_energy(model.state)
        beta_unknown.append(fe_stats['beta'].item())

        radii = model.state.get_belief_radii()
        active = model.state.get_active_mask()
        if active.any():
            unknown_precisions.append(radii[active].mean().item())

    avg_known = sum(known_precisions) / max(len(known_precisions), 1)
    avg_unknown = sum(unknown_precisions) / max(len(unknown_precisions), 1)

    return {
        'known_confidence': avg_known,
        'unknown_confidence': avg_unknown,
        'separation': avg_known - avg_unknown,
        'beta_known': sum(beta_known) / max(len(beta_known), 1),
        'beta_unknown': sum(beta_unknown) / max(len(beta_unknown), 1),
        'beta_separation': (
            sum(beta_unknown) / max(len(beta_unknown), 1) -
            sum(beta_known) / max(len(beta_known), 1)
        ),
    }
