# Memoria Neural Architecture

## Overview

A hybrid transformer with a persistent structured cognitive state, trained with next-token prediction and Bethe free energy minimization. All capabilities of the Memoria runtime and Telos goal system — previously implemented as external software (Rust/CozoDB) — are realized as native tensor operations inside the model.

The model has two components:
1. **Frozen language layers** — standard transformer blocks that handle language processing
2. **Cognitive state** — a persistent structured tensor that holds the model's beliefs, causal structure, goals, and meta-cognitive parameters

The model runs a 2-pass loop:
- **Pass 1 (think):** Process input through language layers + cognitive state → output
- **Pass 2 (learn):** Compute free energy → update cognitive state → the model evolves

Current LLMs are frozen functions that process ephemeral context. This model is a self-modifying structured belief system where the computation graph itself evolves through use, governed by a single variational objective.

---

## Model Structure

```
Input tokens
    │
    ▼
┌──────────────────┐
│ Transformer Block │  ← frozen, standard attention + MLP
│ Transformer Block │
├──────────────────┤
│ State Interface   │  ← reads from / writes to cognitive state
├──────────────────┤
│ Transformer Block │
│ Transformer Block │
├──────────────────┤
│ State Interface   │
├──────────────────┤
│ Transformer Block │
│ Transformer Block │
├──────────────────┤
│ State Interface   │
├──────────────────┤
│ Transformer Block │
└──────────────────┘
    │
    ▼
Output logits
    │
    ▼
┌──────────────────────────────────────┐
│ COGNITIVE STATE (persistent tensor)  │
│                                      │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ Belief      │  │ Relation     │  │
│  │ Region      │◄─►│ Region       │  │
│  └──────┬──────┘  └──────────────┘  │
│         │                            │
│  ┌──────▼──────┐  ┌──────────────┐  │
│  │ Goal        │  │ Meta         │  │
│  │ Region      │  │ Region       │  │
│  └─────────────┘  └──────────────┘  │
│                                      │
└──────────────────────────────────────┘
```

### Frozen Language Layers

Standard transformer blocks. Attention + MLP. Handle language comprehension and generation. Pretrained, then frozen (or fine-tuned lightly during initial training). These are the "brainstem" — stable, reliable computation that doesn't change through use.

Training techniques from nanogpt speedrun:
- Muon optimizer for matrix parameters, AdamW for embeddings/scalars
- Rotary positional embeddings (RoPE)
- QK-Norm (RMSNorm on queries and keys)
- ReLU² activation
- Zero-initialized output projections
- Skip connections from embedding to every block
- Logit softcapping
- BF16 mixed precision

### State Interface Layer

The bridge between language processing and cognitive state. Inserted every K transformer blocks (e.g., every 2 blocks). Bidirectional:

**Read path** (state → transformer):
- Content-addressable lookup into belief region (Hopfield-layer style, not brute-force attention over all beliefs)
- Returns top-k relevant beliefs based on current hidden state
- Goal modulation: active goal embeddings bias retrieval (beliefs aligned with current goals are amplified)
- Relevant beliefs are projected into transformer hidden space and added to the residual stream
- Effect: what the model knows shapes how it processes language

**Write path** (transformer → state):
- Current hidden states projected into state space
- Precision-gated update: beliefs with low precision (small radius) accept large updates; beliefs with high precision (large radius) resist change
- New beliefs allocated in empty slots if observation doesn't match any existing belief
- Effect: new observations update the model's world model, proportional to confidence gap

**Goal readout:**
- Active goal embeddings included in read path attention
- Goal priorities weight how much each goal influences retrieval
- Effect: the model's goals direct its attention and reasoning

---

## Cognitive State

Single structured tensor. Persists across sequences. Fixed capacity. ~500KB at 3-bit quantization (TurboQuant). Four regions:

### Belief Region

**What it stores:** The model's world model. Each slot is a belief about some aspect of the world — an entity, a fact, a state, a pattern. The model learns what to represent.

**Representation:** Polar form (from TurboQuant math).
- **Radius** = precision (confidence). Large radius = high confidence, dominates message passing, resists updates. Small radius = uncertain, low influence, easily updated.
- **Angle** = content (what the belief is about). Concentrated in learned directions → compresses to 3-4 bits.

```
belief_region: [N_beliefs × D]    # polar vectors
               N_beliefs = configurable max capacity (default 4096, up to 131072)
               D = 256            # representation dimension

               Active count is dynamic. Slot with radius = 0 is empty.
               Allocation = set radius > 0. Deallocation = set radius to 0.
               Message passing only touches active beliefs (radius > 0).
               Cost scales with active count, not max capacity.

               Memory at 3-bit quantization:
                 4,096 slots  →  384 KB
                 65,536 slots →  6 MB
                 131,072 slots → 12 MB
```

**Why polar:** Precision weighting is free from the geometry. Dot products between beliefs are naturally precision-weighted — large-radius vectors dominate. No separate precision mechanism needed. Quantization is native (concentrated angles compress well). Radius is human-interpretable as confidence.

**Provenance:** Encoded in a subspace of the angle dimensions. The model learns to distinguish "I observed this directly" from "I inferred this weakly" through training, because L_fe penalizes overconfident inferences.

### Relation Region

**What it stores:** How beliefs connect. Causal structure, correlations, associations. The topology IS the model's causal model of the world.

**Representation:** Soft edges with continuous weights.

```
relation_region: [N_edges × (2 + K + 1)]
                 N_edges = 4096         # fixed edge budget
                 2 = source, target     # soft attention pointers into belief region
                 K = 64                 # relation representation dimension
                 1 = edge weight        # strength/confidence in [0,1]
```

**Edge semantics:**
- Source/target are soft attention distributions over belief region (not hard indices) — allows gradient flow
- Relation representation encodes the type and nature of the connection
- Edge weight encodes confidence in the relationship
- Sparsity pressure from L_fe pushes unneeded edges toward weight 0

**Causal operations:**
- **d-separation:** Graph traversal on edges with weight > threshold. Implemented as sparse matrix operations on the adjacency derived from source/target attention.
- **Intervention (do-calculus):** Clamp a belief (set radius to max, freeze angle), zero all incoming edges to that belief, propagate messages. The result at other beliefs = interventional prediction.
- **Hebbian update:** After each forward pass, co-activated beliefs (both read during same pass) strengthen their connecting edge: `w_new = w_old + η(1 - w_old)`. Saturating — prevents runaway strengthening.

### Goal Region (Telos)

**What it stores:** The model's active goals. What it's trying to understand, achieve, or investigate. Goals emerge from surprise and persist until completed, failed, or abandoned.

```
goal_region: [N_goals × (D + G)]
             N_goals = 64         # max concurrent goals
             D = 256              # goal embedding (same space as beliefs)
             G = 8                # metadata dimensions
```

**Metadata dimensions (G=8):**
- priority: [0,1] — importance, normalized across active goals
- urgency: [0,1] — time pressure, sigmoid approaching deadline
- progress: [0,1] — completion fraction
- status: encoded as continuous (0=proposed, 0.25=active, 0.5=stalled, 0.75=completed, 1.0=failed)
- depth: hierarchy level (0=strategic, 1=tactical, 2=operational, 3=task)
- surprise_accumulator: running sum of surprise in related beliefs
- created_step: when this goal was generated
- deadline: step by which this should be resolved (0 = no deadline)

**Goal lifecycle:**

```
                ┌──────────┐
                │ Proposed │ ← generated from surprise or user input
                └────┬─────┘
                     │ (activated when surprise confirms or user confirms)
                     ▼
                ┌──────────┐
         ┌──────│  Active  │──────┐
         │      └────┬─────┘      │
         │           │            │
         ▼           ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌──────────┐
    │ Stalled │ │Completed│ │  Failed  │
    └────┬────┘ └─────────┘ └──────────┘
         │           (terminal)   (terminal)
         │ (reactivated if new evidence arrives)
         ▼
    ┌──────────┐
    │  Active  │
    └──────────┘
```

**Intrinsic goal generation:**
1. Per-belief surprise accumulated in meta region, grouped by belief cluster
2. When accumulated surprise in a cluster exceeds `β × threshold`:
   - Allocate new goal slot
   - Goal embedding = compressed representation of high-surprise beliefs (learned projection)
   - Priority = normalized surprise magnitude
   - Status = proposed (or active if β > 0.5, indicating exploration mode)
3. Deduplication: skip if cosine similarity to existing non-terminal goal > 0.85

**Goal-directed attention:**
- Active goal embeddings participate in the state interface read path
- Beliefs with high cosine similarity to active goals get retrieval boost proportional to goal priority
- Effect: the model's goals shape what it pays attention to

**Goal decomposition:**
- When a goal's depth < max_depth and no progress after N steps:
  - Learned projection splits parent goal embedding into 2-4 child goal embeddings
  - Children inherit deadline, priority is split
  - Parent tracks children's aggregate progress

**Stall detection:**
- If goal progress unchanged for N steps AND urgency > 0.2 → status = stalled
- Stalled goals get attention bonus (staleness bonus) to prevent permanent neglect
- Urgency-scaled thresholds: high urgency goals stall faster

**Deadline enforcement:**
- If current_step > deadline AND status = active → status = stalled, priority boosted
- Priority follows sigmoid approaching deadline (3 steps before = maximum urgency)

### Meta Region

**What it stores:** The model's cognitive self-awareness. Computed quantities that govern all dynamics.

```
meta_region: [M]
             M = 32     # meta parameters
```

**Key meta variables:**
- **β (exploration/exploitation):** `β = H_var / (E_factor + H_var + ε)` where H_var = total entropy across beliefs (from precision/radius), E_factor = total energy across factors (from relation consistency). NOT a hyperparameter — computed from actual state every forward pass.
  - High β (beliefs are uncertain) → explore: generate more goals, larger state updates, more plasticity
  - Low β (beliefs are confident) → exploit: conservative updates, pursue existing goals

- **accumulated_surprise:** Running sum of update magnitudes from pass 2. Drives goal generation threshold.

- **consolidation_timer:** Steps since last consolidation. When exceeded → trigger belief merging.

- **learning_rate_modulation:** Scales pass 2 update magnitudes. Tuned by SPSA.

- **reconsolidation_threshold:** Surprise level that triggers full belief rewrite vs. incremental update. Tuned by SPSA.

**Meta-learning (SPSA):**
- Every N passes, perturb meta parameters slightly
- Measure free energy before and after
- Gradient-free update: adjust parameters in direction that reduced free energy
- Effect: the model tunes its own cognitive parameters to minimize free energy. Self-optimizing.

---

## Training

### Objective

```
L = L_token + α(t) · L_fe
```

**L_token:** Standard next-token cross-entropy loss. Teaches language comprehension and generation. Same as any LLM.

**L_fe:** Bethe free energy over the cognitive state. Computed from:
- **Energy (E):** Sum over relations of disagreement between connected beliefs. If two beliefs connected by a strong edge are inconsistent → high energy. Formally: `-Σ_f log p(b_i, b_j | ψ_f)` where f is a factor connecting beliefs i,j with potential ψ_f.
- **Entropy (H):** Sum over beliefs of uncertainty. High-radius (confident) beliefs contribute low entropy. Low-radius (uncertain) beliefs contribute high entropy. Formally: `Σ_i H(precision_i)`.
- **Free energy:** `F = E - H`. Minimizing F forces beliefs to be consistent (low E) while maintaining appropriate uncertainty (not overconfident without evidence).

**α(t):** KL annealing schedule. Starts at 0 (pure language learning), ramps to α_max over warmup_steps. Standard technique from VAE training. Prevents L_fe from destabilizing early training when state is random noise.

### Training Data

**Standard text corpus** (FineWeb or similar): Provides the language learning signal via L_token. Same as any LLM pretraining.

**Synthetic cognitive tasks** (interleaved): Provide signal that exercises the cognitive state.
- Belief tracking: facts about entities stated, updated, contradicted across sequences
- Causal reasoning: "A causes B. B causes C. If A is false, what about C?"
- Precision calibration: facts from sources of varying reliability
- Temporal persistence: information from sequence N queried in sequence N+K
- Interventional reasoning: "We set X to true. What happens to Y?"

The model sees both standard text and synthetic tasks during training. L_token drives both. L_fe additionally shapes the cognitive state on all inputs.

**Specific datasets:**
- [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — 10B tokens, educationally filtered (70% of training mix)
- [The Stack v2 dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) — code, 600+ languages, subset: Python/JS/Rust/Go (20% of training mix)
- Synthetic cognitive tasks generated locally (10% of training mix)

**Evaluation datasets:**
- [CausalARC](https://huggingface.co/datasets/jmaasch/causal_arc) — interventional/counterfactual reasoning from causal world models
- [CausalBench](https://huggingface.co/datasets/CCLV/CausalBench) — causal reasoning across code, math, text
- [BaRDa](https://huggingface.co/papers/2312.07527) — belief accuracy vs. reasoning ability
- Synthetic test sets for belief tracking, contradiction, precision, and persistence

**Streaming:** All large datasets streamed via HuggingFace `datasets` library. No local disk footprint. Only synthetic tasks stored locally. Interleaved with configurable weights.

### Training Recipe

From nanogpt speedrun / autoresearch (proven at 124M scale):
- Muon optimizer for transformer matrix params, AdamW for embeddings, scalars, and state interface layers
- Fixed time budget per experiment (5 minutes for rapid iteration, longer for final runs)
- BF16 mixed precision throughout
- Gradient accumulation for effective batch size ~500K tokens
- Cosine learning rate schedule with warmup + warmdown

Target: 150-300M parameters for the frozen transformer. Cognitive state adds negligible parameter count (~2M at full precision, ~250K quantized).

### Training Phases

**Phase 1 — Language foundation (α = 0):**
Standard LM training. No L_fe. Model learns language. Cognitive state exists but receives no free energy signal — it learns whatever helps L_token via gradient flow through state interface layers.

**Phase 2 — Cognitive awakening (α ramps 0 → α_max):**
L_fe gradually introduced. State begins organizing under free energy pressure. Beliefs develop precision structure. Relations develop sparsity. Interleave synthetic cognitive tasks.

**Phase 3 — Full training (α = α_max):**
Both losses active. Standard text + synthetic tasks. Model develops full cognitive capabilities. Meta-learning (SPSA) enabled to tune cognitive parameters.

---

## The 2-Pass Loop (Inference)

### Pass 1 — Think

```
1. Tokenize input
2. Forward through transformer blocks
3. At each state interface layer:
   a. READ: Content-addressable lookup into belief region
      - Hopfield-layer retrieval (not brute-force attention)
      - Top-k beliefs by relevance to current hidden state
      - Goal modulation: active goals bias retrieval
      - Project beliefs into hidden space, add to residual stream
   b. WRITE candidates: Project hidden states into state space
      - Buffer write candidates (don't commit yet)
4. Continue through remaining transformer blocks
5. Compute output logits
6. Apply logit softcapping
```

### Pass 2 — Learn

```
1. SURPRISE: Compare write candidates against existing beliefs
   - Per-belief surprise = distance between new observation and current belief
   - Weighted by observation precision (from write path confidence)

2. BELIEF UPDATE: Precision-weighted revision
   - For each write candidate matching an existing belief:
     - gain = obs_precision / (belief_precision + obs_precision)
     - If surprise < reconsolidation_threshold:
       - Incremental: belief_angle += gain × (obs_angle - belief_angle)
       - Radius: adjusted based on consistency of update
     - If surprise ≥ reconsolidation_threshold:
       - Reconsolidation: belief fully rewritten to observation
       - Radius reset to observation precision
   - For write candidates matching no existing belief:
     - Allocate new slot (evict lowest-precision belief if full)

3. HEBBIAN UPDATE: Strengthen co-activated edges
   - Beliefs read together in pass 1 strengthen connecting edges
   - w_new = w_old + η(1 - w_old) (saturating Hebb rule)
   - Edges not activated decay: w_new = w_old × (1 - decay_rate)

4. GOAL PROGRESS: Update Telos
   - For each active goal:
     - Compute relevance of updated beliefs (cosine sim to goal embedding)
     - progress += Σ (relevance_i × surprise_i) for relevant updates
     - If progress ≥ 1.0 → status = completed
   - Stall detection: no progress for N steps → stalled
   - Deadline enforcement: overdue → stalled, priority boosted

5. INTRINSIC GOALS: Generate from surprise
   - Accumulate surprise per belief cluster
   - If cluster_surprise > β × threshold:
     - Generate goal embedding from cluster centroid
     - Check dedup (skip if >0.85 similarity to existing goal)
     - Allocate goal slot

6. META UPDATE: Compute β and tune parameters
   - H_var = total entropy across belief precisions
   - E_factor = total energy across relation consistencies
   - β = H_var / (E_factor + H_var + ε)
   - Every N passes: SPSA step on meta parameters

7. CONSOLIDATION: Periodic compression
   - If consolidation_timer exceeded:
     - Cluster co-activated low-precision beliefs
     - Merge into single higher-precision abstract belief
     - Free belief slots
     - Merge corresponding relation edges

8. AUDIT: Log state diff
   - Record (step, changed_belief_indices, delta_magnitudes)
   - Kernel-rule-protected beliefs: verify no unauthorized changes
```

---

## Safety and Governance

### Kernel Rules

Hard masks on the cognitive state. Not soft constraints — tensor-level enforcement.

```
kernel_masks: {
    immutable_beliefs: [bool × N_beliefs]     # these beliefs cannot be modified
    immutable_edges: [bool × N_edges]          # these relations cannot be modified
    immutable_goals: [bool × N_goals]          # these goals cannot be abandoned
    protected_meta: [bool × M]                 # these parameters cannot be tuned
}
```

Applied as element-wise multiplication on update tensors in pass 2. If `immutable_beliefs[i] = true`, the update to belief i is zeroed. Mathematically impossible to violate — the mask is applied inside the computation graph, not as a post-hoc check.

Use cases:
- Enterprise directives: goals that cannot be abandoned (precision = max, immutable)
- Safety constraints: beliefs about ethical boundaries that cannot be overwritten
- Identity preservation: core beliefs/values that persist regardless of input

### Governance

- **Visibility masks:** Per-agent context determines which beliefs/goals are readable
- **Transition rules:** Valid goal status transitions encoded as a mask matrix
- **Namespace isolation:** Multi-agent setups partition the cognitive state per agent/team

### Audit Trail

State diffs logged after each pass 2. Stored outside the forward pass (side effect, not on the computation graph).

```
audit_entry: {
    step: int,
    beliefs_changed: [(index, old_radius, new_radius, surprise)],
    edges_changed: [(index, old_weight, new_weight)],
    goals_changed: [(index, old_status, new_status, reason)],
    meta_changed: [(param, old_value, new_value)],
    β: float,
    total_free_energy: float
}
```

---

## Efficiency

### Cognitive State Memory

At full precision (FP32):
```
Belief region:  1024 × 256 × 4 bytes  = 1.0 MB
Relation region: 4096 × 67 × 4 bytes  = 1.1 MB
Goal region:    64 × 264 × 4 bytes    = 66 KB
Meta region:    32 × 4 bytes           = 128 B
Kernel masks:                          ≈ 1 KB
Total:                                 ≈ 2.2 MB
```

At 3-bit quantization (TurboQuant, polar form):
```
Belief region:  1024 × 256 × 0.375 bytes = 96 KB
Relation region: 4096 × 67 × 0.375 bytes = 103 KB
Goal region + meta + masks               ≈ 10 KB
Total:                                    ≈ 209 KB
```

For comparison: a single transformer layer's KV cache at 2048 context length ≈ 2-4 MB. The entire cognitive state is smaller than one layer's cache.

### Computation Overhead

Per state interface layer:
- Read path: Hopfield-layer lookup, O(N_beliefs × D). With N_beliefs=1024, D=256, this is comparable to one attention head over 1024 tokens.
- Write path: Project + precision comparison. O(batch × D).
- Message passing in pass 2: O(N_edges × K) per iteration. Sparse. Runs once per sequence, not per token.

Estimated overhead: 5-15% over a standard transformer of the same depth. Dominated by the Hopfield lookups in the read path.

### Quantization

Polar representation is designed for quantization from day one:
- Belief angles cluster in concentrated directions → compress to 3-4 bits (PolarQuant)
- Belief radii (precision) need slightly higher precision → 8 bits
- Relation representations → 3-4 bits
- Edge weights → 4 bits
- Goal embeddings → 3-4 bits (same as beliefs)
- Meta parameters → FP16 (small, need precision)

### Serving

Each user/session needs its own cognitive state copy:
- At 209 KB quantized: 1M concurrent users ≈ 200 GB state storage
- State checkpoint/restore: serialize the tensor, ~209 KB per save
- Cold start: first interactions use a default-initialized state (uniform low precision, no relations, no goals). Model gracefully degrades to pure-transformer behavior. State populates over first ~100 interactions.

---

## What This Solves

### Hallucination (architectural, not post-hoc)

Current LLMs have no internal uncertainty signal. Confidence is performed, not computed. The formal proof that "hallucination is inevitable" assumes frozen weights with no uncertainty tracking.

This architecture breaks that assumption:
- Precision (radius) is a computed quantity per belief
- β measures global uncertainty from actual state
- When relevant beliefs have low precision, the model has an architectural signal to abstain
- L_fe penalizes overconfident wrong beliefs (high radius + inconsistent with evidence = high free energy)
- No external detection needed — the model knows its own confidence

### Persistent Learning (real, not scaffolding)

MiniMax M2.7's "self-evolution" modifies scaffold code around a frozen model. The model itself never changes.

This architecture modifies the model's actual cognitive state:
- Beliefs update with precision-weighted revision
- Causal structure builds in the relation region
- Skills compile as high-precision belief clusters
- Goals drive self-directed investigation
- Meta-learning tunes cognitive parameters
- Same parameters, improving performance over time

### Causal Reasoning (structural, not pattern-matched)

Current LLMs learned "A causes B" as a text pattern. They can't distinguish causation from correlation or perform interventional reasoning.

This architecture has causal structure in the relation region:
- Directed edges encode causal relationships
- d-separation is a graph property (computable from adjacency)
- Interventions: clamp a belief + zero incoming edges + propagate
- The model can answer "what if we change X?" by simulating on its own causal graph

### Self-Directed Agency (Telos)

No existing architecture has intrinsic goal generation. Models respond to prompts. They don't decide what to investigate.

This architecture generates goals from surprise:
- Persistent surprise in a domain → intrinsic goal to resolve it
- Goals shape attention (retrieval biased toward goal-relevant beliefs)
- Goals persist across sessions
- Full lifecycle: generation → pursuit → decomposition → completion/failure
- Gated by β: high uncertainty → more aggressive goal generation

---

## Lineage

This architecture synthesizes:

| Component | Origin | What we take |
|---|---|---|
| Factor graph message passing | Memoria `aif/` module, FGNN (Zhang 2020), DNBP (2021) | Belief propagation as differentiable tensor ops |
| Bethe free energy | Memoria, RxInfer.jl, variational inference theory | L_fe computation and gradient |
| Precision weighting | Memoria, Active Inference (Friston) | Now geometric via polar representation (TurboQuant) |
| Polar belief representation | TurboQuant (Google 2026) | Radius = precision, angle = content. Quantization-native |
| Telos goal system | Memoria Telos module | Full lifecycle, intrinsic generation, goal-directed attention |
| Persistent mutable state | Titans (Google 2025), TTT (Stanford 2024), Mamba | State persists and evolves across sequences |
| Self-modification in forward pass | TTT layers, SRWM (Schmidhuber 2022) | Pass 2 updates state via computed gradients |
| Multi-rate modification | Hope / Nested Learning (Google NeurIPS 2025) | Frozen core + fast cognitive state |
| Hybrid transformer + recurrence | Griffin (Google 2024), Jamba (AI21 2024) | Attention for reading, recurrence for state evolution |
| Training recipe | modded-nanogpt (Keller Jordan), autoresearch (Karpathy) | Muon, RoPE, QK-Norm, ReLU², training at 124M scale |
| Causal reasoning | Memoria causal module, Pearl's do-calculus | d-separation and intervention on relation graph |
| Surprise-driven dynamics | Memoria, Titans (surprise-based memorization) | But principled (free energy) not ad-hoc (raw gradient) |
| Hebbian associations | Memoria, fast weights (Ba & Hinton 2016) | Co-activation strengthens edges, saturating |
| Meta-learning | Memoria SPSA module | Self-tuning cognitive parameters via free energy |
| KL annealing | β-VAE literature | Bootstrapping L_fe during training |

---

## Solved Design Problems

### 1. Gradient flow through persistent state

**Problem:** State persists across sequences. Does gradient from L_token on sequence N flow back into sequence N-1? That's BPTT across sequences — unstable.

**Solution: Truncated BPTT with detach boundaries.**
- During training, state is **detached** (stop gradient) at sequence boundaries
- Within a sequence, gradients flow normally through state interface layers into the cognitive state
- Between sequences, only **pass 2** updates the state (not L_token gradients)
- This means: L_token shapes the state interface layers' projections (how to read/write). L_fe shapes the state content (what to believe). Clean separation.
- Precedent: this is exactly how Titans handles it — the memory module's weights are updated by surprise (pass 2), not by backprop across sequence boundaries

```
Sequence N:  L_token gradients → transformer + state interface weights
             L_fe gradients → cognitive state content
             [detach state]
Sequence N+1: starts with updated state, no gradient history from N
```

### 2. Pass 2 timing

**Problem:** When does pass 2 run? Per token? Per sequence? Adaptive?

**Solution: Per sequence during training, adaptive during inference.**

**Training:**
- Pass 2 runs once after each full sequence forward pass
- This is natural — you process a sequence, compute L_token + L_fe, then update state
- State updates are part of the training step but detached from next sequence's backward pass

**Inference:**
- Pass 2 runs after each complete response generation
- NOT per token (too expensive, state would thrash)
- Exception: reconsolidation can trigger mid-generation if surprise on a single token exceeds threshold (rare, high-surprise events like direct contradictions)

**The schedule:**
```
Training:   [sequence] → pass1 → loss → pass2 → [detach] → [next sequence]
Inference:  [input] → pass1 → [generate response] → pass2
            (mid-generation reconsolidation only on extreme surprise)
```

### 3. Batched training

**Problem:** Batch of B sequences processed in parallel. Each needs to read/write cognitive state. Shared or per-item?

**Solution: Per-item state copies within batch, merged after.**

- Each batch item gets a **copy** of the current state at batch start
- During forward pass, each item reads from its own copy
- Write candidates buffered per item
- After forward pass, pass 2 runs independently per item
- After pass 2, **merge** state copies back into canonical state:
  - For each belief slot: take the update with highest surprise (most informative)
  - For conflicting updates: precision-weighted average (higher precision wins)
  - For edges: union of Hebbian updates, max weight
  - For goals: union of new goals, dedup by similarity

**Memory:** B copies × 2.2MB = 2.2 × B MB. At B=64 (typical): 141 MB. Acceptable.

**Alternative for large batches:** Read-only state during forward pass. All writes buffered. Single merged pass 2 after the batch. Cheaper, slightly less fresh state per item.

```
Batch start:     canonical_state → B copies
Forward pass:    each item reads/writes its own copy
Pass 2:          B independent updates
Merge:           precision-weighted combination → new canonical_state
```

### 4. Relation region efficiency

**Problem:** Soft attention endpoints over N_beliefs is O(N_edges × N_beliefs) = expensive.

**Solution: Hard indices + soft weights + periodic restructuring.**

- Edges use **hard integer indices** (source_idx, target_idx) into belief region
- Edge weight is a soft continuous value in [0,1] — differentiable
- Relation representation K-dim is differentiable
- Structure changes (new edges, removed edges) happen in **pass 2 only**, not during forward pass

**Edge allocation in pass 2:**
- When two beliefs are co-activated and no edge exists between them: allocate from free pool
- When an edge weight decays below threshold (e.g., 0.01): deallocate, return to free pool
- Free pool tracked as a simple counter + index list

**Cost:** Now O(N_edges × K) for message passing, not O(N_edges × N_beliefs). With hard indices, message passing is just gather + transform + scatter. Standard GNN operation, very efficient.

**Trade-off:** Hard indices mean edge structure is not differentiable — you can't learn "which beliefs should be connected" via gradient. But you CAN learn it via:
- Hebbian co-activation (connect things that fire together)
- Surprise-driven allocation (connect things involved in prediction errors)
- Consolidation (merge edges when beliefs merge)
- These are all pass 2 operations, not gradient-dependent

```
relation_region: [N_edges × (2 + K + 1)]
                 source_idx: int      # hard index into belief region
                 target_idx: int      # hard index into belief region
                 relation: [K floats] # differentiable representation
                 weight: float        # differentiable, [0,1]
```

### 5. Belief matching — update vs. allocate

**Problem:** New observation arrives. Update existing belief or allocate new slot?

**Solution: Cosine similarity matching with adaptive threshold.**

```
1. Compute cosine similarity between observation angle and all existing belief angles
2. Find best match: max_sim = max(cosine_sim(obs_angle, belief_angles))
3. If max_sim > match_threshold:
   → UPDATE the matched belief (precision-weighted revision)
4. If max_sim ≤ match_threshold:
   → ALLOCATE new slot
   → If no free slots: evict belief with lowest score
     eviction_score = radius × recency × (1 - is_immutable)
     (low precision + old + not protected = evict first)
```

**match_threshold:** Starts at 0.7. Tuned by meta-learning (SPSA) as part of the meta region. Too high → everything allocates new slots (state fills fast). Too low → distinct beliefs get merged (loses granularity). The model learns the right threshold.

**Batch conflict:** Multiple observations in same sequence match the same belief. Process sequentially by position order — later observations see the already-updated belief. Natural: information arrives in order.

### 6. Contradiction handling in polar space

**Problem:** "Alice works at Acme" (belief A) vs. "Alice does NOT work at Acme" (observation B). What happens?

**Solution: Contradictions are high-energy states that L_fe resolves.**

**Detection:** Belief A and observation B are about the same entity (high cosine similarity on entity dimensions) but point in opposing directions on the predicate dimensions. The angular distance on predicate dimensions is large despite matching on entity dimensions.

**Formally:**
```
split belief vector into [entity_subspace | predicate_subspace]
entity_match = cosine_sim(A_entity, B_entity)     # high → same entity
predicate_conflict = cosine_sim(A_predicate, B_predicate)  # negative → contradiction
```

If entity_match > threshold AND predicate_conflict < -threshold → contradiction detected.

**Resolution via precision:**
```
If obs_radius > belief_radius:
   → Observation wins. Reconsolidation: belief rewritten to observation.
   → Old belief's content stored temporarily in a "shadow" dimension
     (allows "previously believed X, now believes Y" responses)

If obs_radius ≤ belief_radius:
   → Existing belief resists. Observation stored as low-precision alternative.
   → L_fe will increase (two inconsistent beliefs = high energy)
   → Over time, evidence accumulates toward one → precision grows → the other fades

If obs_radius ≈ belief_radius:
   → Genuine uncertainty. Both maintained. β increases (higher entropy).
   → Model may generate intrinsic goal: "Resolve contradiction about Alice's workplace"
```

**Key:** L_fe naturally penalizes contradictions. Two high-precision beliefs connected by a relation that says they should agree but they don't → high energy → strong gradient to resolve. The model is pressured to either update one belief or reduce precision on both.

### 7. Consolidation mechanics

**Problem:** Merging beliefs is discrete (delete N, create 1). Not differentiable.

**Solution: Soft consolidation with periodic hard cleanup.**

**Continuous (every pass 2):** Beliefs that are very similar (cosine_sim > 0.95) and both low precision → soft merge:
```
merged_angle = precision_weighted_average(angle_A, angle_B)
merged_radius = sqrt(radius_A² + radius_B²)  # combined evidence → higher precision
```
This is differentiable — it's just a weighted average + norm computation. Slot B is not deleted — it's zeroed (radius → 0), making it a free slot for future allocation.

**Periodic hard cleanup (every N sequences):** Non-differentiable. Treated like data augmentation.
- Cluster beliefs by angular similarity (k-means on angles)
- Within each cluster: merge beliefs with radius < threshold
- Merged belief: centroid angle, combined radius
- Free the slots
- Merge corresponding edges: take max weight, average relation vectors

**Edge handling on merge:**
```
When belief A absorbs belief B:
  - All edges pointing to B → redirect to A (update target_idx)
  - If A already has an edge to the same neighbor: take max weight
  - B's slot freed
```

### 8. Sequence boundaries during training

**Problem:** Beliefs from Wikipedia persist when processing code. Chaotic.

**Solution: Exponential decay at sequence boundaries + domain tags.**

```
At each sequence boundary:
  for each belief:
    radius *= decay_factor  # e.g., 0.95
```

This means:
- A belief reinforced every sequence maintains its precision (decay offset by updates)
- A belief not reinforced fades over ~20 sequences (0.95^20 ≈ 0.36)
- Very high precision beliefs (many reinforcements) take longer to fade — they're well-established knowledge
- Immutable beliefs (kernel rules) skip decay

**Domain context:** The model processes diverse text. Sequence-level domain tag (embedded as a special token) helps the state interface layer focus reads on domain-relevant beliefs. Not a hard partition — just soft attention bias.

**Training:** 50% of training sequences share a persistent state (consecutive sequences from same document/domain). 50% start with fresh state (new document). This teaches the model both to persist beliefs AND to handle cold starts.

### 9. L_fe tensor math (concrete)

**Problem:** Need actual formulas, not hand-waving.

**Solution:**

All beliefs in polar form: `b_i = r_i × û_i` where `r_i` = radius (precision), `û_i` = unit angle vector.

**Energy term (E):**
For each active edge f connecting beliefs i, j with relation vector ψ_f and weight w_f:

```
agreement_ij = cosine_sim(û_i, transform(û_j, ψ_f))
E_f = -w_f × r_i × r_j × log(σ(agreement_ij × temperature))
```

Where `transform(��_j, ψ_f)` is a learned linear transform of belief j's angle through the relation representation (what does j "look like" from i's perspective given their relationship). σ is sigmoid. Temperature is a meta parameter.

Intuition: two high-precision beliefs (large r) connected by a strong edge (large w) that disagree (low agreement) → high energy → strong gradient to fix.

**Entropy term (H):**
```
H_i = -log(r_i + ε)  # high precision → low entropy → low H
                       # low precision → high entropy → high H
```

Simpler than Shannon entropy but has the right properties: uncertain beliefs contribute high entropy, confident beliefs contribute low.

**Total free energy:**
```
F = Σ_f E_f - Σ_i H_i
  = Σ_f [-w_f × r_i × r_j × log(σ(agreement × temp))] - Σ_i [-log(r_i + ε)]
```

Minimizing F:
- Pushes connected beliefs toward agreement (reduce E)
- Pushes beliefs toward appropriate uncertainty (balance E and H)
- Prevents overconfidence: high r without supporting evidence → E doesn't decrease but H does → F goes down only if E goes down too → need actual consistency

**Gradient flow:** Fully differentiable. Gradients flow into:
- Belief angles (content) via agreement term
- Belief radii (precision) via both E and H
- Relation vectors (ψ_f) via transform
- Edge weights via w_f

**β computation:**
```
H_total = Σ_i H_i
E_total = Σ_f |E_f|
β = H_total / (E_total + H_total + ε)
```

### 10. Negation and absence

**Problem:** How to represent "X is NOT true" or "there is no connection between A and B"?

**Solution: Three mechanisms for three types of negation.**

**Active negation (belief that something is false):**
- A belief with specific content that encodes the negation
- "Alice does NOT work at Acme" is a belief in its own right, pointing in a specific angular direction
- The model learns through training that certain angular regions represent negation of other regions
- L_fe ensures negated beliefs and their positive counterparts have high energy when both are high-precision → forces resolution

**Absent connection (no known relationship):**
- No edge between two beliefs = no known relationship
- This is distinct from "known to be unrelated" (see below)
- The model can distinguish because absence of edge ≠ presence of negative edge

**Known independence (confirmed no relationship):**
- An edge with a specific relation type that encodes "independent" or "unrelated"
- Edge weight is HIGH (confident about the independence)
- This prevents future Hebbian linking: if an independence edge exists, co-activation doesn't create a new associative edge
- Useful for d-separation: known independence is evidence in causal reasoning

```
Three states:
  No edge:     "I don't know if A and B are related"  (ignorance)
  Positive edge: "A relates to B in way ψ"            (knowledge)
  Negative edge: "A and B are known to be independent" (anti-knowledge)
```

The model learns to use all three through training. L_fe handles the consistency: if an independence edge exists but the beliefs are actually correlated in data, free energy increases → model corrects.

---

## Open Risks (remaining after solutions)

1. **L_fe + L_token interaction.** Solved architecturally (detach at sequence boundaries, KL annealing). But empirical validation still needed — the specific loss landscape of this combination is untested.

2. **State drift at deployment scale.** Decay + consolidation + kernel rules handle known failure modes. Unknown failure modes (adversarial inputs, distribution shift) need empirical stress testing.

3. **Training curriculum design.** The 50/50 split (persistent sequences + fresh starts) is principled but the right ratio and task mix needs experimentation.

4. **Efficient Hopfield lookup at scale.** O(N_beliefs × D) per state interface layer per token. With N_beliefs=1024, D=256, this is 262K multiply-adds — comparable to one attention head. Should be fine but needs benchmarking on actual hardware.

5. **Consolidation quality.** Soft merge is differentiable but might produce blurry beliefs (averaged angles lose specificity). The periodic hard cleanup helps but is non-differentiable. The interaction between these two mechanisms needs tuning.
