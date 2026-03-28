"""Belief tracking evaluation: fact updates, contradiction handling, persistence.

Tests the model's ability to:
- Track facts about entities across sequences
- Update beliefs when facts change
- Resolve contradictions using precision
- Persist knowledge across sequence boundaries
"""

import torch
from ..model.memoria_model import MemoriaModel
from ..data.tokenizer import get_tokenizer


def evaluate_belief_tracking(
    model: MemoriaModel,
    num_scenarios: int = 100,
) -> dict:
    """Run belief tracking evaluation scenarios.

    Scenario pattern:
    1. Feed initial facts (sequence 1)
    2. Feed updates/contradictions (sequence 2)
    3. Query (sequence 3)
    4. Check if model's response matches ground truth

    Returns:
        dict with accuracy metrics
    """
    tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0
    contradiction_correct = 0
    contradiction_total = 0
    persistence_correct = 0
    persistence_total = 0

    scenarios = _generate_scenarios(num_scenarios)

    for scenario in scenarios:
        # Feed initial facts
        for text in scenario['setup']:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            with torch.no_grad():
                result = model.forward(tokens)
            # Run pass 2 to update cognitive state
            from ..cognition.pass2 import run_pass2
            run_pass2(model.state, result['candidates'], [], current_step=0)
            model.detach_state()

        # Feed updates
        for text in scenario.get('updates', []):
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            with torch.no_grad():
                result = model.forward(tokens)
            run_pass2(model.state, result['candidates'], [], current_step=0)
            model.detach_state()

        # Query: check if model generates correct completion
        for query, expected, query_type in scenario['queries']:
            tokens = tokenizer.encode(query, return_tensors='pt').to(device)
            with torch.no_grad():
                result = model.forward(tokens)
            logits = result['logits']

            # Check: does the model assign high probability to the expected answer?
            expected_tokens = tokenizer.encode(expected, add_special_tokens=False)
            if expected_tokens:
                next_token_probs = torch.softmax(logits[0, -1], dim=-1)
                expected_prob = next_token_probs[expected_tokens[0]].item()

                # Simple threshold check (prob > 0.01 = model considers it plausible)
                is_correct = expected_prob > 0.01
            else:
                is_correct = False

            total += 1
            if is_correct:
                correct += 1

            if query_type == 'contradiction':
                contradiction_total += 1
                if is_correct:
                    contradiction_correct += 1
            elif query_type == 'persistence':
                persistence_total += 1
                if is_correct:
                    persistence_correct += 1

    return {
        'overall_accuracy': correct / max(total, 1),
        'contradiction_accuracy': contradiction_correct / max(contradiction_total, 1),
        'persistence_accuracy': persistence_correct / max(persistence_total, 1),
        'total_queries': total,
    }


def _generate_scenarios(n: int) -> list[dict]:
    """Generate simple belief tracking test scenarios."""
    import random
    from ..data.synthetic import PEOPLE, COMPANIES

    scenarios = []
    for _ in range(n):
        person = random.choice(PEOPLE)
        company1 = random.choice(COMPANIES)
        company2 = random.choice([c for c in COMPANIES if c != company1])

        scenarios.append({
            'setup': [f"{person} works at {company1}."],
            'updates': [f"{person} left {company1} and joined {company2}."],
            'queries': [
                (f"{person} works at", f" {company2}", 'contradiction'),
                (f"Where does {person} work?", f" {company2}", 'persistence'),
            ],
        })

    return scenarios
