"""Synthetic cognitive task generation.

Generates training data that exercises the cognitive state:
- Belief tracking: facts stated, updated, queried
- Contradiction handling: conflicting facts with varying precision
- Causal chains: A→B→C with interventional queries
- Precision calibration: facts from sources of different reliability

These are programmatically generated with controlled ground truth.
Small enough to store locally (a few hundred MB).

Reference: CausalARC (huggingface.co/datasets/jmaasch/causal_arc) — structure
Reference: BaRDa (huggingface.co/papers/2312.07527) — belief accuracy tasks
"""

import random
import json
from typing import Iterator


# ── Entity pools ──

PEOPLE = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank", "Iris", "Jack"]
COMPANIES = ["Acme", "Globex", "Initech", "Umbrella", "Cyberdyne", "Stark", "Wayne", "Oscorp"]
CITIES = ["New York", "London", "Tokyo", "Berlin", "Paris", "Sydney", "Toronto", "Mumbai"]
ROLES = ["engineer", "manager", "designer", "scientist", "analyst", "director", "intern", "CEO"]
LANGUAGES = ["Python", "Rust", "JavaScript", "Go", "TypeScript", "Java", "C++", "Ruby"]


def generate_belief_tracking(num_sequences: int = 1000, seq_length: int = 20) -> Iterator[str]:
    """Generate belief tracking tasks.

    Pattern:
    - State facts about entities
    - Later update some facts
    - Query the current state

    Example:
        Alice works at Acme. Bob works at Globex. Carol lives in Tokyo.
        Alice left Acme and joined Umbrella.
        Question: Where does Alice work? Answer: Umbrella
        Question: Where does Bob work? Answer: Globex
    """
    for _ in range(num_sequences):
        entities = random.sample(PEOPLE, k=min(5, len(PEOPLE)))
        facts = {}
        lines = []

        # Initial facts
        for person in entities:
            fact_type = random.choice(["works_at", "lives_in", "role"])
            if fact_type == "works_at":
                value = random.choice(COMPANIES)
                facts[(person, "works_at")] = value
                lines.append(f"{person} works at {value}.")
            elif fact_type == "lives_in":
                value = random.choice(CITIES)
                facts[(person, "lives_in")] = value
                lines.append(f"{person} lives in {value}.")
            else:
                value = random.choice(ROLES)
                facts[(person, "role")] = value
                lines.append(f"{person} is a {value}.")

        # Filler sentences (noise)
        for _ in range(random.randint(3, 8)):
            lines.append(random.choice([
                "The weather was good today.",
                "There was a meeting in the afternoon.",
                "The project deadline is next week.",
                "Several new features were shipped.",
                "The team discussed the roadmap.",
            ]))

        # Updates (change some facts)
        updated = {}
        for person, fact_type in random.sample(list(facts.keys()), k=min(2, len(facts))):
            if fact_type == "works_at":
                old = facts[(person, fact_type)]
                new = random.choice([c for c in COMPANIES if c != old])
                facts[(person, fact_type)] = new
                updated[(person, fact_type)] = (old, new)
                lines.append(f"{person} left {old} and joined {new}.")
            elif fact_type == "lives_in":
                old = facts[(person, fact_type)]
                new = random.choice([c for c in CITIES if c != old])
                facts[(person, fact_type)] = new
                updated[(person, fact_type)] = (old, new)
                lines.append(f"{person} moved from {old} to {new}.")
            elif fact_type == "role":
                old = facts[(person, fact_type)]
                new = random.choice([r for r in ROLES if r != old])
                facts[(person, fact_type)] = new
                updated[(person, fact_type)] = (old, new)
                lines.append(f"{person} was promoted from {old} to {new}.")

        # More filler
        for _ in range(random.randint(2, 5)):
            lines.append("The quarter ended with strong results.")

        # Queries
        for (person, fact_type), value in facts.items():
            if fact_type == "works_at":
                lines.append(f"Question: Where does {person} work? Answer: {value}")
            elif fact_type == "lives_in":
                lines.append(f"Question: Where does {person} live? Answer: {value}")
            elif fact_type == "role":
                lines.append(f"Question: What is {person}'s role? Answer: {value}")

        yield " ".join(lines)


def generate_contradiction_tasks(num_sequences: int = 500) -> Iterator[str]:
    """Generate tasks with contradictory information from different sources.

    Tests precision calibration: reliable vs unreliable sources.

    Example:
        A reliable source reports: Alice works at Acme.
        An unverified rumor says: Alice works at Globex.
        Question: Where does Alice most likely work? Answer: Acme
        Confidence: high (reliable source)
    """
    for _ in range(num_sequences):
        person = random.choice(PEOPLE)
        true_value = random.choice(COMPANIES)
        false_value = random.choice([c for c in COMPANIES if c != true_value])

        # Reliable source states truth
        reliable = random.choice([
            f"A reliable source confirms: {person} works at {true_value}.",
            f"Official records show {person} is employed at {true_value}.",
            f"{person}'s LinkedIn profile states they work at {true_value}.",
        ])

        # Unreliable source states falsehood
        unreliable = random.choice([
            f"An unverified rumor says {person} works at {false_value}.",
            f"Someone mentioned they heard {person} might be at {false_value}.",
            f"An anonymous source claims {person} is at {false_value}.",
        ])

        # Randomize order
        if random.random() > 0.5:
            text = f"{reliable} {unreliable}"
        else:
            text = f"{unreliable} {reliable}"

        text += f" Question: Where does {person} most likely work? Answer: {true_value}"
        text += f" Confidence: high (reliable source confirms)"

        yield text


def generate_causal_chains(num_sequences: int = 500) -> Iterator[str]:
    """Generate causal reasoning tasks.

    Pattern: A causes B. B causes C. Query about interventions.

    Tests: causal vs correlational reasoning, do-calculus.

    Example:
        Rain causes wet ground. Wet ground causes muddy shoes.
        Observation: The ground is wet.
        Question: Is it raining? Answer: Possibly (wet ground doesn't prove rain)
        Intervention: We turn on the sprinkler (forcing wet ground).
        Question: Did it rain? Answer: Unknown (sprinkler explains wet ground)
    """
    causal_triples = [
        ("rain", "wet ground", "muddy shoes"),
        ("studying", "knowledge", "good grades"),
        ("exercise", "fitness", "weight loss"),
        ("bug in code", "test failure", "delayed release"),
        ("high demand", "price increase", "lower sales"),
        ("new feature", "user growth", "revenue increase"),
        ("server load", "slow response", "user complaints"),
        ("code review", "fewer bugs", "stable release"),
    ]

    for _ in range(num_sequences):
        a, b, c = random.choice(causal_triples)

        lines = [
            f"{a.capitalize()} causes {b}.",
            f"{b.capitalize()} causes {c}.",
        ]

        # Different query types
        query_type = random.choice(["observation", "intervention", "counterfactual"])

        if query_type == "observation":
            lines.append(f"Observation: {b} is present.")
            lines.append(f"Question: Is {a} the cause? Answer: Possibly, but {b} could have other causes.")
            lines.append(f"Question: Will {c} occur? Answer: Yes, because {b} causes {c}.")

        elif query_type == "intervention":
            lines.append(f"We intervene to force {b} to be true (regardless of {a}).")
            lines.append(f"Question: Does this tell us anything about {a}? Answer: No, intervention on {b} breaks the causal link from {a}.")
            lines.append(f"Question: Will {c} occur? Answer: Yes, because {b} causes {c} regardless of how {b} was caused.")

        elif query_type == "counterfactual":
            lines.append(f"We observe: {a} is true, {b} is true, {c} is true.")
            lines.append(f"Question: If {a} had been false, would {c} still be true?")
            lines.append(f"Answer: Only if {b} had another cause. If {a} is the only cause of {b}, then no.")

        yield " ".join(lines)


def generate_precision_calibration(num_sequences: int = 500) -> Iterator[str]:
    """Generate tasks testing precision/confidence calibration.

    Multiple sources with varying reliability state facts.
    Model should weight by source precision.
    """
    for _ in range(num_sequences):
        person = random.choice(PEOPLE)
        true_lang = random.choice(LANGUAGES)
        false_lang = random.choice([l for l in LANGUAGES if l != true_lang])

        statements = []

        # Multiple weak sources say one thing
        for _ in range(random.randint(3, 5)):
            statements.append(f"Someone mentioned {person} uses {false_lang}.")

        # One strong source says another
        statements.append(f"{person}'s GitHub profile shows their top language is {true_lang}.")

        random.shuffle(statements)
        text = " ".join(statements)
        text += f" Question: What language does {person} primarily use? Answer: {true_lang}"
        text += f" Reasoning: One high-confidence source (GitHub profile) outweighs multiple low-confidence rumors."

        yield text


def generate_all_synthetic(
    num_belief: int = 1000,
    num_contradiction: int = 500,
    num_causal: int = 500,
    num_precision: int = 500,
    output_path: str | None = None,
) -> list[str]:
    """Generate all synthetic datasets.

    Args:
        num_belief: number of belief tracking sequences
        num_contradiction: number of contradiction sequences
        num_causal: number of causal reasoning sequences
        num_precision: number of precision calibration sequences
        output_path: if provided, save as JSONL

    Returns:
        list of all generated sequences
    """
    all_data = []

    all_data.extend(generate_belief_tracking(num_belief))
    all_data.extend(generate_contradiction_tasks(num_contradiction))
    all_data.extend(generate_causal_chains(num_causal))
    all_data.extend(generate_precision_calibration(num_precision))

    random.shuffle(all_data)

    if output_path:
        with open(output_path, 'w') as f:
            for text in all_data:
                f.write(json.dumps({"text": text}) + "\n")

    return all_data
