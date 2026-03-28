"""Improvement curve: performance vs experience (the hero figure).

Measures: does the model get better over time with same parameters?
Plot: X = number of interactions, Y = task accuracy.

The curve should rise for our model and stay flat for baselines.
This is THE primary metric for the paper.
"""

import torch
import json
from pathlib import Path

from ..model.memoria_model import MemoriaModel
from ..cognition.pass2 import run_pass2
from ..data.tokenizer import get_tokenizer
from ..data.synthetic import generate_belief_tracking


def evaluate_improvement_curve(
    model: MemoriaModel,
    total_interactions: int = 500,
    eval_every: int = 50,
    eval_queries: int = 20,
) -> dict:
    """Measure how model performance improves with experience.

    Protocol:
    1. Start with fresh cognitive state
    2. Feed interactions from a domain
    3. Every N interactions, evaluate accuracy on domain queries
    4. Record the curve

    Args:
        model: trained MemoriaModel
        total_interactions: total number of domain interactions
        eval_every: evaluate every N interactions
        eval_queries: number of queries per evaluation point

    Returns:
        dict with:
            interactions: list of interaction counts [0, 50, 100, ...]
            accuracies: list of accuracy at each checkpoint
            beliefs_count: list of active belief counts
            beta_values: list of β values
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
        model.state.meta.data[0] = 1.0
        model.state.meta.data[1] = 0.0

    # Generate domain data
    domain_texts = list(generate_belief_tracking(num_sequences=total_interactions + eval_queries))

    interactions_log = [0]
    accuracies_log = [_eval_accuracy(model, tokenizer, device, domain_texts[-eval_queries:])]
    beliefs_log = [model.state.num_active_beliefs()]
    beta_log = [model.state.beta]

    for i in range(total_interactions):
        text = domain_texts[i]
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True,
                                  max_length=model.config.transformer.sequence_len).to(device)

        with torch.no_grad():
            result = model.forward(tokens)

        read_indices = [c.matched_slot for c in result['candidates'] if c.matched_slot >= 0]
        run_pass2(model.state, result['candidates'], read_indices,
                  current_step=i, is_sequence_boundary=True)
        model.detach_state()

        if (i + 1) % eval_every == 0:
            acc = _eval_accuracy(model, tokenizer, device, domain_texts[-eval_queries:])
            interactions_log.append(i + 1)
            accuracies_log.append(acc)
            beliefs_log.append(model.state.num_active_beliefs())
            beta_log.append(model.state.beta)

    return {
        'interactions': interactions_log,
        'accuracies': accuracies_log,
        'beliefs_count': beliefs_log,
        'beta_values': beta_log,
    }


def _eval_accuracy(model, tokenizer, device, eval_texts: list[str]) -> float:
    """Quick accuracy check: does model assign reasonable probability to expected tokens?"""
    correct = 0
    total = 0

    for text in eval_texts:
        # Split text at "Answer:" to get query and expected answer
        if "Answer:" not in text:
            continue

        parts = text.split("Answer:")
        query = parts[0] + "Answer:"
        expected = parts[1].strip().split()[0] if parts[1].strip() else ""

        if not expected:
            continue

        tokens = tokenizer.encode(query, return_tensors='pt', truncation=True,
                                  max_length=model.config.transformer.sequence_len).to(device)

        with torch.no_grad():
            result = model.forward(tokens)

        logits = result['logits']
        next_probs = torch.softmax(logits[0, -1], dim=-1)

        expected_tokens = tokenizer.encode(" " + expected, add_special_tokens=False)
        if expected_tokens:
            prob = next_probs[expected_tokens[0]].item()
            if prob > 0.01:
                correct += 1
        total += 1

    return correct / max(total, 1)


def save_improvement_results(results: dict, path: str):
    """Save improvement curve results for plotting."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Improvement curve saved to {path}")
