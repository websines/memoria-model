"""Crossover evaluation: small model + experience vs large model fresh.

THE experiment that proves the thesis:
- 500M model with cognitive state after N interactions
- vs 10B model with no persistent state on the same tasks

If the curves cross, scaling laws are broken.
"""

import torch
import json
from pathlib import Path

from ..model.memoria_model import MemoriaModel
from ..cognition.pass2 import run_pass2
from ..data.tokenizer import get_tokenizer
from ..data.synthetic import generate_belief_tracking


def evaluate_crossover(
    memoria_model: MemoriaModel,
    baseline_model=None,
    baseline_name: str = "baseline",
    domain_interactions: int = 500,
    eval_checkpoints: list[int] | None = None,
    eval_queries: int = 50,
    output_path: str | None = None,
) -> dict:
    """Run the crossover experiment.

    Protocol:
    1. Both models start fresh
    2. Feed domain interactions to BOTH models
    3. At each checkpoint, evaluate both on held-out queries
    4. Record performance curves for both
    5. The crossover = when our model surpasses baseline

    For the baseline: if no model provided, we evaluate our model
    with cognitive state DISABLED (alpha=0, pass 2 disabled) as the
    ablation baseline. For the actual crossover experiment, pass a
    larger model (e.g., quantized 10B via HuggingFace).

    Args:
        memoria_model: our trained MemoriaModel
        baseline_model: a larger baseline model (optional)
        baseline_name: name for the baseline in results
        domain_interactions: number of domain interactions
        eval_checkpoints: when to evaluate [50, 100, 200, 500]
        eval_queries: queries per evaluation
        output_path: save results here

    Returns:
        dict with both curves
    """
    if eval_checkpoints is None:
        eval_checkpoints = [0, 25, 50, 100, 200, 300, 400, 500]

    tokenizer = get_tokenizer()
    device = next(memoria_model.parameters()).device

    # Generate domain data
    domain_texts = list(generate_belief_tracking(
        num_sequences=domain_interactions + eval_queries, seq_length=20
    ))
    eval_texts = domain_texts[-eval_queries:]
    train_texts = domain_texts[:-eval_queries]

    # Reset our model's state
    _reset_state(memoria_model)

    memoria_curve = []
    baseline_curve = []

    for checkpoint in eval_checkpoints:
        # Feed interactions up to this checkpoint
        for i in range(len(memoria_curve) > 0 and eval_checkpoints[len(memoria_curve) - 1] or 0, checkpoint):
            if i >= len(train_texts):
                break
            _feed_interaction(memoria_model, tokenizer, device, train_texts[i], step=i)

        # Evaluate our model
        memoria_acc = _evaluate_queries(memoria_model, tokenizer, device, eval_texts)
        memoria_curve.append({
            'interactions': checkpoint,
            'accuracy': memoria_acc,
            'beliefs': memoria_model.state.num_active_beliefs(),
            'edges': memoria_model.state.num_active_edges(),
            'goals': memoria_model.state.num_active_goals(),
            'beta': memoria_model.state.beta,
        })

        # Evaluate baseline (if provided)
        if baseline_model is not None:
            baseline_acc = _evaluate_queries_stateless(
                baseline_model, tokenizer, device, eval_texts
            )
        else:
            # Ablation: our model without cognitive state
            baseline_acc = _evaluate_queries_stateless(
                memoria_model, tokenizer, device, eval_texts
            )
        baseline_curve.append({
            'interactions': checkpoint,
            'accuracy': baseline_acc,
        })

        print(f"  Checkpoint {checkpoint}: memoria={memoria_acc:.3f}, {baseline_name}={baseline_acc:.3f}")

    # Find crossover point
    crossover_point = None
    for i in range(len(memoria_curve)):
        if memoria_curve[i]['accuracy'] > baseline_curve[i]['accuracy']:
            crossover_point = eval_checkpoints[i]
            break

    results = {
        'memoria_curve': memoria_curve,
        'baseline_curve': baseline_curve,
        'baseline_name': baseline_name,
        'crossover_point': crossover_point,
        'eval_checkpoints': eval_checkpoints,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Crossover results saved to {output_path}")

    return results


def _reset_state(model: MemoriaModel):
    """Reset cognitive state to empty."""
    with torch.no_grad():
        model.state.beliefs.data.zero_()
        model.state.edge_weights.data.zero_()
        model.state.edge_active.zero_()
        model.state.goal_metadata.data.zero_()
        model.state.goal_embeddings.data.zero_()
        model.state.meta.data[0] = 1.0
        model.state.meta.data[1] = 0.0


def _feed_interaction(model, tokenizer, device, text, step):
    """Feed one interaction and run pass 2."""
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True,
                              max_length=model.config.transformer.sequence_len).to(device)
    with torch.no_grad():
        result = model.forward(tokens)

    read_indices = [c.matched_slot for c in result['candidates'] if c.matched_slot >= 0]
    run_pass2(model.state, result['candidates'], read_indices,
              current_step=step, is_sequence_boundary=True)
    model.detach_state()


def _evaluate_queries(model, tokenizer, device, eval_texts):
    """Evaluate with cognitive state active."""
    correct = 0
    total = 0

    for text in eval_texts:
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

        probs = torch.softmax(result['logits'][0, -1], dim=-1)
        expected_tokens = tokenizer.encode(" " + expected, add_special_tokens=False)
        if expected_tokens and probs[expected_tokens[0]].item() > 0.01:
            correct += 1
        total += 1

    return correct / max(total, 1)


def _evaluate_queries_stateless(model, tokenizer, device, eval_texts):
    """Evaluate without cognitive state (pure transformer baseline).

    Temporarily zeros out the state so the model gets no belief context.
    """
    # Save state
    saved = model.state.state_dict_cognitive()

    # Zero state
    _reset_state(model)

    acc = _evaluate_queries(model, tokenizer, device, eval_texts)

    # Restore state
    model.state.load_state_cognitive(saved)

    return acc
