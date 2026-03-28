# Build Plan — Memoria Neural Architecture

## Hardware
- Development: 2x RTX 3090 (24GB each), WSL2 Ubuntu
- Scale-up: Rent B200 (192GB) for 500M crossover experiment
- No B200 until crossover curve is proven at 150M on 3090s

## Stack
- Python 3.11+
- PyTorch 2.x (CUDA 12)
- HuggingFace `datasets` (streaming)
- HuggingFace `tokenizers`
- wandb or tensorboard (experiment tracking)
- No other frameworks. Keep it minimal.

## Project Structure

```
memoria-research/
├── .gitignore
├── BUILD_PLAN.md              ← this file
├── architecture.md            ← full architecture spec
├── understanding.md           ← working notes & research
├── pyproject.toml             ← Python project config
│
├── memoria/                   ← the implementation
│   ├── __init__.py
│   │
│   ├── core/                  ← Phase 1: cognitive state
│   │   ├── __init__.py
│   │   ├── polar.py           # Polar representation (cartesian↔polar, radius ops)
│   │   ├── state.py           # CognitiveState (belief, relation, goal, meta regions)
│   │   ├── free_energy.py     # L_fe computation (energy, entropy, Bethe free energy)
│   │   └── kernel_rules.py    # Immutability masks, hard constraints
│   │
│   ├── interface/             ← Phase 2: state interface layer
│   │   ├── __init__.py
│   │   ├── read_path.py       # Hopfield lookup, goal modulation, belief→hidden projection
│   │   ├── write_path.py      # Hidden→state projection, precision-gated updates, matching
│   │   └── layer.py           # StateInterfaceLayer (combines read + write)
│   │
│   ├── model/                 ← Phase 3: transformer integration
│   │   ├── __init__.py
│   │   ├── transformer.py     # GPT blocks (from autoresearch/modded-nanogpt)
│   │   ├── memoria_model.py   # Full model: transformer + state interface + cognitive state
│   │   └── config.py          # Model configs (small/medium/large)
│   │
│   ├── cognition/             ← Phase 4: pass 2 engine
│   │   ├── __init__.py
│   │   ├── surprise.py        # Surprise computation, reconsolidation trigger
│   │   ├── belief_update.py   # Precision-weighted revision, slot allocation/eviction
│   │   ├── hebbian.py         # Co-activation edge strengthening, decay
│   │   ├── telos.py           # Goal lifecycle, intrinsic generation, decomposition, progress
│   │   ├── consolidation.py   # Soft merge, periodic hard cleanup, edge merging
│   │   ├── meta_learning.py   # β computation, SPSA parameter tuning
│   │   ├── causal.py          # d-separation, intervention (clamp + propagate)
│   │   └── pass2.py           # Orchestrator: runs all pass 2 steps in order
│   │
│   ├── data/                  ← Phase 5: training data
│   │   ├── __init__.py
│   │   ├── streaming.py       # HuggingFace streaming loader (FineWeb + Stack v2)
│   │   ├── synthetic.py       # Generate cognitive tasks (belief, causal, contradiction)
│   │   ├── interleave.py      # Mix datasets with configurable weights
│   │   └── tokenizer.py       # Tokenizer setup
│   │
│   ├── training/              ← Phase 5: training loop
│   │   ├── __init__.py
│   │   ├── optimizer.py       # Muon + AdamW setup (from autoresearch)
│   │   ├── schedule.py        # LR schedule, KL annealing for α(t)
│   │   ├── train.py           # Main training loop (phase 1/2/3 switching)
│   │   └── distributed.py     # 2x 3090 data parallel setup
│   │
│   └── eval/                  ← Phase 6: evaluation
│       ├── __init__.py
│       ├── perplexity.py      # Standard LM eval (must not degrade)
│       ├── belief_tracking.py # Fact update, contradiction handling
│       ├── hallucination.py   # Calibrated refusal (precision-based abstention)
│       ├── causal.py          # Interventional reasoning (CausalARC, CausalBench)
│       ├── telos_demo.py      # Self-directed goal generation from surprise
│       ├── improvement.py     # Performance vs. experience curve
│       └── crossover.py       # 500M+experience vs. 10B fresh (the big one)
│
├── scripts/
│   ├── train_small.sh         # Quick training script for 3090
│   ├── train_full.sh          # Full training on 2x 3090
│   └── eval_all.sh            # Run all evaluations
│
├── configs/
│   ├── small.yaml             # 125M config (rapid iteration)
│   ├── medium.yaml            # 300M config (serious training)
│   └── large.yaml             # 500M config (crossover experiment)
│
├── tests/
│   ├── test_polar.py          # Polar representation math
│   ├── test_state.py          # CognitiveState creation, belief ops
│   ├── test_free_energy.py    # L_fe gradient verification
│   ├── test_interface.py      # Read/write path correctness
│   ├── test_pass2.py          # Full pass 2 loop
│   └── test_telos.py          # Goal lifecycle
│
└── prototype-research/        ← Original Memoria Rust code (reference, not tracked)
```

---

## Phases

### Phase 1: Cognitive State + Free Energy (Days 1-3)

**Goal:** Prove the math works. L_fe produces meaningful gradients on polar belief representations.

**Build:**
- `memoria/core/polar.py` — polar↔cartesian conversion, radius (precision) operations, angular distance
- `memoria/core/state.py` — CognitiveState class with all four regions, dynamic allocation (radius=0 is empty)
- `memoria/core/free_energy.py` — the actual L_fe formula:
  ```
  E_f = -w_f × r_i × r_j × log(σ(agreement × temp))
  H_i = -log(r_i + ε)
  F = ΣE - ΣH
  ```
- `tests/test_polar.py`, `tests/test_state.py`, `tests/test_free_energy.py`

**Test:**
1. Create a CognitiveState with a few beliefs
2. Add two beliefs that agree (connected by strong edge) → compute F → should be low
3. Add two beliefs that disagree (connected by strong edge) → compute F → should be high
4. Backprop through F → verify gradients push disagreeing beliefs toward agreement
5. Verify precision (radius) gradients: overconfident wrong beliefs get radius-reducing gradient

**Done when:** We can create beliefs, compute free energy, backprop, and see beliefs update sensibly. Gradients flow correctly through polar representation.

**Reference code:** `prototype-research/src/aif/free_energy.rs`, `prototype-research/src/aif/messages.rs`

---

### Phase 2: State Interface Layer (Days 4-5)

**Goal:** The bridge works. Transformer hidden states can read from and write to cognitive state.

**Build:**
- `memoria/interface/read_path.py` — Hopfield-layer content-addressable lookup into beliefs, goal modulation
- `memoria/interface/write_path.py` — hidden→state projection, cosine matching, precision-gated update buffering
- `memoria/interface/layer.py` — StateInterfaceLayer combining read + write

**Test:**
1. Create CognitiveState with known beliefs
2. Feed synthetic hidden states that are similar to certain beliefs → verify read path retrieves them
3. Feed hidden states that represent new information → verify write path buffers correct candidates
4. Test goal modulation: set active goals, verify retrieval is biased toward goal-relevant beliefs
5. Test precision gating: high-precision beliefs resist updates, low-precision beliefs accept them

**Done when:** Standalone layer correctly reads relevant beliefs and buffers precision-appropriate writes.

**Reference code:** `prototype-research/src/pipeline/scoring.rs` (factor message fusion), `prototype-research/src/aif/messages.rs`

---

### Phase 3: Transformer Integration (Days 6-10)

**Goal:** The full model trains. L_token converges. L_fe doesn't blow things up.

**Build:**
- `memoria/model/transformer.py` — GPT blocks adapted from autoresearch's train.py (Muon, RoPE, QK-Norm, ReLU², value embeddings, softcap)
- `memoria/model/config.py` — model configs:
  ```
  small:  12 layers, 768 dim, ~125M params, state interface every 4 layers (3 interfaces)
  medium: 24 layers, 1024 dim, ~300M params, state interface every 4 layers (6 interfaces)
  large:  24 layers, 1280 dim, ~500M params, state interface every 4 layers (6 interfaces)
  ```
- `memoria/model/memoria_model.py` — the full model wiring transformer + interfaces + state
- `memoria/training/optimizer.py` — Muon for transformer matrices, AdamW for everything else
- `memoria/training/schedule.py` — LR schedule + KL annealing for α(t)
- `memoria/training/train.py` — basic training loop (L_token only first, then L_token + L_fe)

**Test:**
1. Phase 1 training (α=0): train on FineWeb-Edu sample, L_token only. Verify loss decreases normally. Model should behave like a standard GPT. The state interface layers participate but L_fe is off.
2. Phase 2 training (α ramps up): enable L_fe with KL annealing. Verify L_fe decreases. Verify L_token doesn't explode. Verify beliefs develop non-random structure (radii diverge from uniform, some edges strengthen).
3. Compare: model with state vs. same model without state on held-out perplexity. State model should be equal or better (not worse — state should help, not hurt).

**Done when:** Model trains stably with L_token + L_fe. Loss curves look healthy. State develops structure.

**Reference code:** autoresearch `train.py` (full GPT + Muon + training loop)

---

### Phase 4: Pass 2 Engine (Days 11-15)

**Goal:** The cognitive loop works. Beliefs update, edges strengthen, goals form.

**Build (in order):**
- `memoria/cognition/surprise.py` — compare write candidates to existing beliefs, compute per-belief surprise
- `memoria/cognition/belief_update.py` — precision-weighted revision (Kalman-like gain), reconsolidation on high surprise, slot allocation/eviction
- `memoria/cognition/hebbian.py` — co-activated beliefs strengthen edges, unused edges decay
- `memoria/cognition/telos.py` — goal lifecycle state machine, intrinsic generation from accumulated surprise, goal-directed attention weights, decomposition, stall detection, progress tracking
- `memoria/cognition/consolidation.py` — soft merge (differentiable) + periodic hard cleanup
- `memoria/cognition/meta_learning.py` — β computation from state, SPSA tuning of meta params
- `memoria/cognition/causal.py` — d-separation on relation adjacency, intervention (clamp + propagate)
- `memoria/cognition/pass2.py` — orchestrator that runs all steps in correct order
- `memoria/core/kernel_rules.py` — immutability masks applied in pass 2

**Test:**
1. Belief update: feed contradictory information, verify higher-precision source wins
2. Hebbian: co-activate beliefs repeatedly, verify edge weight increases
3. Telos: create scenario with persistent surprise, verify goal generates
4. Consolidation: fill state with similar beliefs, verify they merge
5. Causal: build A→B→C in relations, clamp B, verify A and C respond correctly
6. Integration: train model with full pass 2 enabled. Verify state evolves sensibly over training.

**Done when:** Full pass 2 loop runs after each sequence. State evolves: beliefs update, edges form, goals emerge from surprise. Model still trains stably.

**Reference code:** `prototype-research/src/dynamics/` (surprise, consolidation, meta-learning), `prototype-research/src/api/telos*.rs` (goal system), `prototype-research/src/causal/` (d-separation, intervention)

---

### Phase 5: Training at Scale (Days 16-20)

**Goal:** Full training run on 2x 3090. Produce a real model.

**Build:**
- `memoria/data/streaming.py` — HuggingFace streaming for FineWeb-Edu + Stack v2
- `memoria/data/synthetic.py` — generate cognitive task datasets (belief tracking, causal chains, contradictions, precision calibration)
- `memoria/data/interleave.py` — mix with weights (70/20/10)
- `memoria/training/distributed.py` — DataParallel across 2x 3090
- `configs/small.yaml`, `configs/medium.yaml`

**Training runs:**
1. Small config (125M), FineWeb-Edu only, 2-3 hours on single 3090. Sanity check.
2. Small config, full data mix (FineWeb + Stack + synthetic), 4-6 hours. Full cognitive training.
3. Medium config (300M), full mix, 2x 3090, 8-12 hours. Production-quality model.

**Checkpointing:** Save model + cognitive state every N steps. The state IS part of the checkpoint.

**Done when:** We have a trained model where:
- L_token is competitive with a standard GPT of same size
- L_fe has converged to stable low value
- State has meaningful structure (inspectable beliefs, sparse relations, active goals)

---

### Phase 6: Evaluation + Crossover (Days 21-28)

**Goal:** Prove it works. Show the crossover curve.

**Build:**
- `memoria/eval/perplexity.py` — standard LM eval on held-out data
- `memoria/eval/belief_tracking.py` — feed facts across sequences, introduce contradictions, query
- `memoria/eval/hallucination.py` — ask about things not told, measure refusal rate + precision correlation
- `memoria/eval/causal.py` — CausalARC + CausalBench evaluation
- `memoria/eval/telos_demo.py` — detect surprise-driven goal generation
- `memoria/eval/improvement.py` — plot performance at interaction 1, 10, 50, 100, 500, 1000
- `memoria/eval/crossover.py` — our model vs. larger baselines (300M, 1B, 3B) on domain tasks

**The experiments:**
1. **Baseline comparison:** 125M with state vs. 125M without state. Same architecture, same training data, same compute. Only difference: cognitive state on/off. Ablation.
2. **Improvement curve:** Fix the 125M model. Run 1000 interactions on a coding task sequence. Plot accuracy at each checkpoint. Show the curve rises.
3. **Scaling challenge:** 125M with state (after experience) vs. 300M without state (fresh). On domain tasks.
4. **Demo reel:** The 5 demos (hallucination refusal, belief revision, causal intervention, self-directed goals, improvement over time). Qualitative + quantitative.

**If results are positive at 125M → scale to 500M → rent B200 → run the crossover against 10B.**

---

## Key Dependencies (Python packages)

```toml
[project]
name = "memoria"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "datasets>=3.0",
    "tokenizers>=0.20",
    "transformers>=4.45",      # for tokenizer/model utilities only
    "wandb",                   # experiment tracking
    "numpy",
    "pyyaml",                  # config files
    "pytest",                  # testing
]
```

## Ground Rules

1. **Single-file prototyping first, refactor later.** Phase 1 can start as one file. Don't over-engineer the structure until the math works.
2. **Test every component standalone before integrating.** Integration bugs are the hardest to debug. Each module must work in isolation first.
3. **Reference the Rust prototype constantly.** The algorithms are proven. Port them, don't reinvent them.
4. **Log everything to wandb.** L_token, L_fe, α, β, active beliefs, active edges, active goals, surprise distribution. If we can't see it, we can't debug it.
5. **Checkpoint aggressively.** Model + cognitive state + meta params. If training crashes at hour 6, we don't want to restart from scratch.
6. **Don't optimize early.** Standard attention, standard data loading, no custom CUDA kernels. Get it working first. Optimize when it's the bottleneck.
7. **The crossover curve is the only metric that matters.** Everything else serves that goal.
