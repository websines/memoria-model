# Memoria

A self-evolving neural architecture with persistent cognitive state.

Hybrid transformer with structured belief region (polar form), causal relation graph, Telos goal system, and Bethe free energy objective. The model maintains beliefs, reasons causally, generates goals from surprise, and improves through experience — all internally, with no external memory.

## Quickstart

```bash
# Install
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# See model info
python -m memoria info --config small

# Run tests
python -m memoria test

# Train (single GPU)
cp .env.example .env  # fill in HF_TOKEN and WANDB_API_KEY
python -m memoria train --config small --max-steps 5000

# Train with time budget (e.g., overnight 8 hours)
python -m memoria train --config small --time-budget 28800

# Resume from checkpoint
python -m memoria train --config small --resume checkpoints/step_2000.pt

# Evaluate
python -m memoria eval checkpoints/final.pt --suite all
```

## Architecture

```
Input → [Transformer Block × 2] → [State Interface] → [Transformer × 2] → ... → Output
                                         ↕
                              Cognitive State (persistent)
                              ├── Belief Region (polar: radius=precision, angle=content)
                              ├── Relation Region (causal edges, hard idx + soft weights)
                              ├── Goal Region (Telos: lifecycle, intrinsic generation)
                              └── Meta Region (β, surprise, SPSA self-tuning)
```

**Training:** `L = L_token + α · L_fe` (next-token prediction + Bethe free energy)

**2-pass loop:** Pass 1 (think) → Pass 2 (learn: surprise → belief update → Hebbian → Telos → consolidation → meta)

See [architecture.md](architecture.md) for full spec.

## Hardware

- **Development:** 2× RTX 3090 (WSL2), overnight training runs
- **Scale-up:** B200 for 500M crossover experiment
- **Config:** `small` (125M, single 3090), `medium` (300M, 2× 3090), `large` (500M, B200)

## Success Criterion

A 500M model with experience surpasses a 10B model without it on practical tasks. The crossover curve is the paper.
