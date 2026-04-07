# Memoria Training Architecture

## Overview

Memoria is a hybrid transformer-cognitive architecture that combines a language model backbone with a persistent, evolving cognitive state (beliefs, edges, goals). The system implements active inference: the model minimizes free energy by maintaining an internal world model that tracks entities, causal relations, and goals across arbitrarily long contexts.

Training operates in **two passes per step**:

- **Pass 1** (differentiable): Forward through transformer blocks interleaved with state interface layers. Produces logits, write candidates, and multiple loss terms. Gradients from all losses flow through interface layers and into cognitive state parameters (beliefs, edges, relations) via the optimizer.
- **Pass 2** (structural): Discrete operations that gradients cannot handle — slot allocation, edge creation via learned networks, consolidation, goal generation, sleep cycles, and planning. Runs on rank 0 only, modifying the cognitive state in-place via `.data` access.

## Training Phases

The training loop uses KL annealing across three phases:

| Phase | Steps | Alpha (α) | Active Losses | Purpose |
|-------|-------|-----------|---------------|---------|
| **Phase 1** | 0 → `phase1_steps` | 0.0 | L_token only | Language foundation — model learns to predict tokens |
| **Phase 2** | `phase1_steps` → `+ alpha_warmup_steps` | 0 → α_max | L_token + α·L_fe + α·L_aux | Cognitive awakening — state begins organizing |
| **Phase 3** | remainder | α_max | All losses at full weight | Full training — all cognitive systems active |

This prevents posterior collapse: the model must first learn language before the free energy loss can provide meaningful signal to organize the cognitive state.

## Architecture Diagram

```
Input tokens [B, T]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Token Embedding + RMSNorm + Working Memory Suffix (M tokens) │
│  + Engram Static Knowledge Injection (O(1) N-gram hash)       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Transformer Block 0 (Mamba-2 / SWA / MLA) ──────────────────┐
│  resid_lambda[i] * x + x0_lambda[i] * x0                      │
│  x = x + attn(norm(x), cos, sin)                               │
│  x = x + mlp(norm(x))        [ReLU² activation]                │
└────────────────────────────────────────────────────────────────┘
       │
      ...  (repeat for blocks 1, 2)
       │
       ▼
┌─ State Interface Layer 0 ─────────────────────────────────────┐
│                                                                 │
│  READ PATH (beliefs → hidden stream)                            │
│    query = query_proj(hidden)     [H → D × num_heads]          │
│    keys  = active_belief_angles   [N_active, D]                │
│    attn  = softmax(Q @ K^T / √d)   + goal modulation          │
│    retrieved = attn @ belief_values                              │
│    hidden += output_proj(retrieved)                              │
│                                                                 │
│  WRITE PATH (hidden stream → candidates for Pass 2)            │
│    obs = obs_proj(hidden)         [H → D]                      │
│    gate = write_gate(hidden)      [learned binary decision]    │
│    precision = precision_head(hidden) [observation confidence] │
│    → match against active beliefs (cosine similarity)           │
│    → WriteCandidate {belief_vector, matched_slot, similarity}  │
│                                                                 │
└─────────────────────── outputs: hidden', candidates, ──────────┘
                         attn_weights, retrieved, obs_vectors
       │
      ...  (transformer blocks + interface layers alternate every K blocks)
       │
       ▼
┌─ Refinement Loops (upper blocks × max_refinement_loops) ──────┐
│  For each loop iteration:                                       │
│    1. Loop-index encoding (additive signal)                     │
│    2. Re-run upper transformer blocks                           │
│    3. Lifeline anchor (prevent drift on token positions)        │
│    4. Re-query beliefs via last interface (retrieve-reason)     │
│    5. TTT gradient step on upper MLP deltas                     │
│    6. HaltingHead probe → P(halt) — stop if confident enough   │
│  Working memory suffix evolves freely (true scratchpad)         │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ LM Head ─────────────────────────────────────────────────────┐
│  RMSNorm → Linear(n_embd, vocab_size) → softcap(15.0)         │
│  logits [B, T, vocab_size] — never leaves forward() scope      │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
   Loss computation (inside forward)
```

## Loss Functions

### Pass 1 Losses (Differentiable)

All losses are computed inside `forward()` so the logits tensor `[B, T, vocab_size]` never crosses the DDP boundary.

**Scratch mode** uses Kendall/Gal uncertainty weighting (CVPR 2018) with learned `log_sigma` per loss group:

```
_uw(L, s) = L / (2·exp(2s)) + s
L_total = _uw(L_token, σ_token) + α · _uw(L_fe, σ_fe) + α · _uw(L_aux, σ_aux) + w_draft · L_draft
```

**Pretrained mode** uses fixed weights:

```
L_total = L_token + α · L_fe + α · 0.1 · L_utility + α · 0.1 · L_surprise
```

| Loss | Source | Trains | Description |
|------|--------|--------|-------------|
| **L_token** | `chunked_cross_entropy` | Transformer + interfaces | Next-token prediction. Chunked to bound memory on 151K vocab. |
| **L_fe_proxy** | `compute_differentiable_free_energy` | Read/write paths | Proxy FE: consistency between retrieved beliefs and observations, minus attention entropy. Trains interface projections. |
| **L_fe_bethe** | `compute_bethe_free_energy` | Beliefs, edges, relations | Proper Bethe free energy on the cognitive factor graph. Power Spherical entropy with (d_i−1) counting correction. |
| **L_utility** | `chunked_cross_entropy` on utility logits | Interface utility heads | Measures whether retrieved beliefs improve token prediction. |
| **L_surprise** | `TelosModule.surprise_loss` | RND networks, goal system | RND surprise (trains predictor to match target for seen beliefs) + goal status transitions (penalize stalled/failed, reward completed). |
| **L_halt** | `F.binary_cross_entropy` | RefinementProbe | Teaches the halting probe when to stop refinement loops. Teacher forcing with random oracle loop count. |
| **L_jac** | DEQ Jacobian regularization | Message passing | Ensures the factor graph fixed-point map stays contractive. Periodic (every 10 steps). |
| **L_draft** | `DFlashDraftHead.compute_draft_loss` | Draft head layers | Block diffusion draft quality. NOT alpha-gated — trains from step 0. |

### Bethe Free Energy (Factor Graph)

The core differentiable loss for the cognitive state:

```
F_B = Σ_factors U_f + Σ_beliefs (d_i − 1) · H_i
```

- **U_f** (factor energy): For each active edge, measures disagreement between connected beliefs weighted by edge weight and relation transform. High-precision beliefs connected by strong edges that disagree produce high energy → strong gradient.
- **H_i** (belief entropy): Power Spherical entropy, closed-form via digamma. Small radius = high entropy = uncertain.
- **(d_i − 1)**: Degree correction. Highly-connected beliefs get their entropy counted less (prevents overcounting). Isolated beliefs (degree 0) get negative correction → encourages exploration.
- **Telos energy**: Unfinished important goals increase free energy, pressuring the system to make progress.

### Beta (Exploration / Exploitation)

```
β = H / (|E| + H + ε)
```

High entropy (uncertain beliefs) → high β → more exploration. Low entropy (confident beliefs) → low β → more exploitation. β modulates Pass 2 operations (allocation rates, consolidation thresholds).

## Optimizer Configuration

The optimizer uses a **Muon + AdamW split** via `_CombinedOptimizer`:

- **Muon** (Newton-Schulz orthogonalization): For 2D+ matrix parameters in transformer blocks (attention, MLP weights). Constrains gradient updates to the Stiefel manifold, achieving ~35% faster convergence than AdamW.
- **AdamW**: For everything else — embeddings, scalars, interface layers, cognitive state, all auxiliary modules.

### Parameter Groups (Scratch Mode)

| # | Component | LR | Notes |
|---|-----------|-----|-------|
| 1 | Token embedding | 0.6 | High LR for embedding tables |
| 2 | LM head (unembedding) | 0.004 | Lower than embedding |
| 3 | Scalar lambdas (resid, x0) | 0.5 | Per-layer residual scaling |
| 4 | Block 1D params (biases, norms) | 0.5 | Same as scalars |
| 5 | State interface layers | 0.01 | Interface projections |
| 6 | Cognitive state (beliefs, edges, relations) | 0.0001 | Very slow — beliefs should be stable |
| 7 | Cognitive meta-parameters | 0.001 | Learned thresholds |
| 8 | Telos module | 0.01 | Goal generation, progress, transitions |
| 9 | Goal embeddings | 0.0001 | Slow like beliefs + 0.5× weight decay |
| 10 | TTT module (init_A/B, log_step_size, decay_gate) | 0.01 | Meta-learned TTT initialization |
| 11 | Edge proposal network | 0.01 | Learned edge creation |
| 12 | Edge directions (CoED) | 0.001 | 10× belief_lr — structural, not content |
| 13 | Cognitive controller (SEAL-style) | 0.001 | 0.1× interface_lr — should be stable |
| 14 | SleepGate | 0.001 | 0.1× interface_lr — consolidation stability |
| 15 | Message passing (relation transform + DEQ) | 0.01 | Factor graph inference |
| 16 | Kendall/Gal log_sigma | 0.5 | Uncertainty weighting params |
| 17 | Hypothesis generator | 0.01 | Autoresearch loop |
| 18 | DFlash draft head | 0.01 | Block diffusion speculative decoding |
| M | 2D matrix params (attention, MLP) | 0.04 | **Muon optimizer** (separate from AdamW) |

Gradient clipping (`clip_grad_norm_`) is applied only to AdamW params. Muon's Newton-Schulz orthogonalization produces unit-spectral-norm updates, making pre-clip counterproductive.

## LR and Alpha Scheduling

### LR Schedule (WSD — Warmup-Stable-Decay)

```
get_lr_multiplier(step, total_steps, config) → multiplier ∈ [final_lr_frac, 1.0]
```

Three phases:
1. **Warmup** (0 → `warmup_ratio`): Linear 0 → 1
2. **Stable** (`warmup_ratio` → `1 - warmdown_ratio`): Constant 1.0
3. **Warmdown** (final portion): Linear or cosine decay to `final_lr_frac`

Applied uniformly: `group['lr'] = group['initial_lr'] * lr_mult`

### Alpha Schedule (KL Annealing)

```
get_alpha(step, config) → α ∈ [0, alpha_max]
```

- Phase 1 (`step < phase1_steps`): α = 0
- Phase 2 (`phase1_steps` → `+ alpha_warmup_steps`): Linear ramp 0 → α_max
- Phase 3: α = α_max (constant)

### Context Length Schedule (SkyLadder)

```
get_context_length(step, total_steps, config) → ctx_len
```

Progressive context extension from `skyladder_start` (default 256) to `sequence_len` over the
first `skyladder_ratio` (default 60%) of training. Short context early = faster training (less
attention compute) + better representations (model learns local patterns first). Batches are
truncated to `ctx_len` each step.

Reference: SkyLadder (NeurIPS 2025, arxiv.org/abs/2503.15450) — up to 22% faster training
with better short- and long-context benchmarks.

Schedules:
- **linear** (default): `ctx = start + (target - start) × progress`
- **exponential**: log-linear ramp, spends more steps at shorter (cheaper) context
- **step**: doubles at equal intervals (256→512→1024→...→target)

Interacts with the three training phases:
- Phase 1 (language foundation): runs at short context → fast, cheap
- Phase 2 (cognitive awakening): context grows, beliefs learn medium-range dependencies
- Phase 3 (full training): at target context, all systems active

## Pass 2: Structural Cognitive Updates

Pass 2 runs once per step after `optimizer.step()` and `detach_state()`. All modifications use `.data` access (no gradients). Adaptive depth logic decides which operations run each step.

### Operations (in order)

| # | Operation | Always? | What it does |
|---|-----------|---------|-------------|
| 0 | **Controller actions** | Yes | CognitiveController (SEAL-style) produces structural decisions: allocate_rate, connect_rate, merge_threshold, prune_threshold, goal_rate. REINFORCE-trained. |
| 1 | **Structural cleanup** | Yes | Free zero-norm belief slots and zero-weight edges (garbage collection). |
| 2 | **Belief allocation** | If candidates | `compute_surprise_batch` → `allocate_new_beliefs`. New observations get slots; existing beliefs collect surprise stats. |
| 2b-i | **MESU variance update** | If matched beliefs | Update per-belief precision variance (observations reduce uncertainty). |
| 2b-ii | **Cascade revision** | If reconsolidated | Propagate precision decay to downstream beliefs in the causal graph. |
| 2b | **Adaptive LR update** | If updated beliefs | Per-belief LR scale (RWKV-7 inspired) + MESU variance boost. |
| 2c | **Confidence propagation** | If updated beliefs | Propagate confidence changes through source chains (MemOS). |
| 3 | **Edge proposal** | If edge_fill < 90% | Learned `EdgeProposer` evaluates co-activation + causal candidate pairs. Creates edges with learned weights and directions. |
| 4 | **Soft consolidation** | Periodic or near capacity | Merge near-identical beliefs (cosine > threshold). Level-aware: higher-level beliefs need higher similarity. |
| 4b | **Hard cleanup** | On consolidation timer | Prune low-precision beliefs. Reset timer. |
| 5 | **Goal generation** | If room for goals | `TelosModule.generate_goals` — differentiable goal synthesis from active beliefs. |
| 5b | **Autoresearch** | If goals + room | `HypothesisGenerator` synthesizes candidate beliefs in goal directions. Allocated as provisional (A1). |
| 6 | **Beta + stats** | Yes | `compute_beta`, update running statistics, anneal Telos temperature. |
| 6b | **Provisional evaluation** | Yes | Evaluate provisional beliefs past their window. Promote winners, evict losers. |
| 7 | **Belief promotion** | If updated beliefs | SDFT-inspired abstraction hierarchy. Promote beliefs that meet radius + access thresholds. |
| 8 | **Sleep cycle** | On consolidation timer | `SleepGate` scores each belief for strengthen/maintain/forget. |
| 9 | **Dream phase** | At sequence boundaries | `NeuroDream` — internal propagation via message passing to converge belief graph toward consistency. |
| 9b | **Belief shift** | After dream | Apply message passing results to shift beliefs. |
| 9c | **Two-factor sleep** | During sleep | Homeostatic precision normalization + conflict scanning. |
| 9d | **Self-verification** | During sleep | Causal consistency check, weakest-link precision reduction. |
| 9e | **Precision recalibration** | During sleep | Decay overconfident beliefs toward empirical precision. |
| 9f | **Interleaved replay** | During sleep | Cross-temporal contradiction detection. |
| 10 | **Planning** | At sequence boundaries | Preference/epistemic priors, causal rollouts, optional MCTS. |
| 11 | **SRWM update** | If SRWM exists | Self-referential fast-weight matrix for meta-parameter modulation. |
| 12 | **Structural plasticity** | At sequence boundaries | Split polysemantic beliefs, prune dead ones. |

## Data Pipeline

### Curated Multi-Tier Mix

All training modes (scratch and pretrained) use `curated_stream`, a weighted interleaving of HuggingFace streaming datasets organized by tier:

| Tier | Weight | Purpose | Examples |
|------|--------|---------|----------|
| state_essential | 15% | Tasks requiring cross-context memory | Belief tracking, theory-of-mind, fact revision |
| state_helps | 10% | Tasks that benefit from persistent state | Multi-hop reasoning, causal chains, verification |
| code | 30% | Raw code + code reasoning | StarCoder, OpenCodeReasoning |
| code_agent | 15% | Agentic coding | SWE trajectories, terminal agents, tool-use |
| reasoning | 10% | Mathematical and cross-domain reasoning | OpenMathReasoning, Bespoke-Stratos |
| tool_calling | 8% | Function calling and agentic | xlam-function-calling |
| general | 5% | Web text baseline | FineWeb-Edu |
| synthetic | 2% | Generated belief/causal tasks | Internal synthetic generator |

Design principle: **~45% of training data genuinely requires persistent cross-context memory.** This forces the cognitive state to be useful rather than decorative.

### Synthetic Data

`generate_all_synthetic()` produces four task types:
1. **Belief tracking**: Entity-attribute updates with queries requiring state tracking
2. **Contradiction tasks**: Conflicting information from sources with varying reliability
3. **Causal chains**: A→B→C reasoning with intervention queries
4. **Precision calibration**: Multiple sources stating facts with different confidence levels

### DataPrefetcher

Background thread that pre-loads batches into a queue (size 3). Handles error propagation and timeout (120s). Stream resumption on checkpoint reload uses HuggingFace `skip()` for O(1) seek past consumed data.

## DDP (Multi-GPU) Strategy

Uses HuggingFace Accelerate with `find_unused_parameters=True` (cognitive state is dynamic — the set of parameters contributing to loss changes between steps).

| Component | Sync Mechanism |
|-----------|---------------|
| Transformer + interface params | DDP gradient averaging (automatic) |
| Cognitive state (beliefs, edges, goals) | Manual broadcast from rank 0 before each forward, gather candidates after |
| TTT deltas (delta_A, delta_B) | NOT synchronized (requires_grad=False, modified via .data) — each rank adapts independently |
| Pass 2 | Runs on rank 0 only, barrier after completion |

## DFlash Speculative Decoding

Native block diffusion draft head for inference acceleration. Especially valuable because refinement loops multiply per-token cost by ~4×.

### Architecture

```
Target hidden states (tapped from layers [0, 5, 11])
  + Active belief embeddings
  → feature_proj (concat → D)
  → context [B, T_ctx + N_beliefs, D]

Mask embeddings + positional encoding
  → draft tokens [B, block_size, D]

3× DFlashDraftLayer:
  draft = draft + CrossAttention(Q=draft, KV=concat(context, draft))
  draft = draft + MLP(draft)     [2× expansion, ReLU²]

→ shared LM head → draft logits [B, block_size, vocab]
```

- **15.9M params** (~12.7% of main model)
- **Cross+self attention**: draft tokens attend to target features, beliefs, AND each other (block coherence)
- **Shared LM head**: No extra 117M vocab projection
- **Trained jointly**: Auxiliary loss from step 0 (not alpha-gated)
- **Inference**: `spec_generate()` — draft block → verify with full model (including refinement) → accept/reject → repeat

### Speculative Decoding Loop

```
while tokens_generated < max_new_tokens:
    1. Run full model on current sequence (prefill)
    2. Extract tapped hidden states + active beliefs
    3. Draft head generates block_size tokens in parallel
    4. Verify: run full model on [sequence + drafted block]
    5. Accept tokens until first mismatch with verifier
    6. Append accepted tokens, continue
```

Estimated speedup: ~3-4× at 50% acceptance rate, primarily from amortizing refinement loop cost across the accepted block.

## In-Place Test-Time Training (TTT)

TTT is the self-improvement mechanism. During BOTH training and inference:

1. Process tokens through the model
2. Compute next-token prediction loss on the chunk
3. Take a gradient step on persistent low-rank fast-weight deltas: `W_eff = W_frozen + delta_A @ delta_B`
4. Updated deltas persist across chunks and sessions

### TTT Enhancements

| Feature | Source | Description |
|---------|--------|-------------|
| Meta-learned initialization | TTT-E2E (arXiv:2512.23675) | Deltas start from learned warm point, not zero |
| Titans-style decay gate | Titans (arXiv:2501.00663) | Learned adaptive forgetting (largest Titans performance contributor) |
| Large-chunk accumulation | LaCT (arXiv:2505.23884) | Accumulate gradients over multiple chunks before stepping |
| Multi-step inner loop | DeltaProduct (arXiv:2502.10297) | N smaller steps per chunk for expressiveness |
| Surprise gating | Titans-inspired | Adaptive thresholds (mean ± 2σ) filter OOD and boring inputs |
| Loss rollback | Quality protection | Snapshot deltas before update; revert if loss increased |
| Belief TTT | Memoria-specific | Update belief vectors using write path observation errors at inference |

## Cognitive Seed (Cross-Run Transfer)

`save_cognitive_seed()` / `load_cognitive_seed()` enable transferring learned cognitive knowledge between training runs:

**Saved**: Meta-parameters, Telos weights, running statistics, high-confidence core beliefs, edges between core beliefs, edge proposal network.

**Loaded via content matching**: Belief slots are mutable storage locations, not stable semantic coordinates. EWC on raw slots is invalid. Instead, each seed belief is matched to the closest existing belief by cosine similarity. Only transferred if the seed has higher confidence than the match.

## Quantization-Aware Training (RotorQuant)

KV cache and belief storage use RotorQuant block-diagonal rotations for compression.
During training, STE (Straight-Through Estimator) injects quantization noise so the model
learns representations robust to 3-bit compression. This makes inference-time KV compression
and checkpoint belief quantization nearly lossless.

### Backend Selection

| Backend | Rotation | FMAs (d=128) | Centroids | When Used |
|---------|----------|-------------|-----------|-----------|
| **PlanarQuantMSE** | 2D Givens | 256 | Lloyd-Max (MSE-optimal) | 3-bit, `rotorquant` installed |
| **IsoQuantMSE** | 4D quaternion | 512 | Lloyd-Max | 4-bit, `rotorquant` installed |
| **PolarQuantizer** | Full d×d QR | d² | Uniform scalar | Fallback (no external deps) |

Install RotorQuant: `pip install -e ".[rotorquant]"` (optional, PolarQuantizer fallback works without it).

Reference: RotorQuant (scrya.com/rotorquant.pdf, March 2026) — beats TurboQuant on PPL (6.91 vs 7.07),
decode speed (28% faster), and prefill (5.3× faster) at same 10.3× compression.

### Where Quantization Acts

| Path | When | Mechanism | Effect |
|------|------|-----------|--------|
| **KV QAT** | Training (every forward) | `ste_quantize(k, quantizer)` in `SlidingWindowAttention.forward()` | Model learns K,V projections that survive 3-bit quantization |
| **Belief QAT** | Training (every read) | `ste_quantize(values, quantizer)` in `ReadPath.forward()` | Belief representations become quantization-robust |
| **KV cache compression** | Inference (T > window) | `QuantizedKVCache.compress()` → blockwise `decompress_slice()` | 10× KV memory reduction for long sequences |
| **Checkpoint compression** | Save/load | `QuantizedBeliefStore` in `state_dict_cognitive()` | ~10× belief tensor compression in checkpoints |

### STE Gradient Flow

```
Forward:  x → [quantize → dequantize] → x_hat  (with quantization noise)
Backward: grad_output → identity → grad_input   (straight-through)
```

The STE ensures gradients flow through as if no quantization occurred, while the forward pass
trains the model to be robust to the noise. Over training, K/V projections and belief write paths
naturally converge toward representations where information sits in directions that survive 3-bit rounding.

### Windowed MLA for Long Context

MLA layers (L in "MMMML") default to full causal attention O(T²). At long context (>128K),
this becomes prohibitive. Setting `mla_window_size > 0` gives MLA a sliding window, reducing
cost to O(T × W) while cognitive state + Mamba-2 handle beyond-window coherence.

```
mla_window_size = 0       → full causal O(T²), short context training
mla_window_size = 131072  → 128K window, enables 1M+ context
```

Four memory systems stack to maintain coherence at any context length:

| System | Range | Scaling | Role |
|--------|-------|---------|------|
| **MLA attention** | W tokens (window) | O(W) constant | Dense local context |
| **Mamba-2 state** | Unlimited | O(1) fixed | Compressed recurrent memory |
| **Cognitive state** | Unlimited | O(1) fixed slots | Persistent beliefs, edges, goals |
| **Engram** | Unlimited | O(1) hash | Static N-gram patterns |

Memory at 1M tokens (small config, 2 MLA layers, W=128K, RotorQuant 3-bit):
- MLA KV: ~38 MB (windowed + compressed)
- Mamba-2 state: ~400 KB (fixed)
- Total: **~65 MB** — same as at 10K tokens

### Blockwise Attention (Inference)

For long sequences (T > window_size), both SWA and windowed MLA use quantized blockwise attention:
1. Compress full K,V via RotorQuant/PolarQuant → ~10× smaller
2. Free full-precision copies
3. For each chunk, `decompress_slice()` only the needed window — no full-tensor materialization
4. Compute attention on the small decompressed window

Peak memory: O(T × 1 byte) quantized + O(W × D × 4 bytes) per chunk.
At 1M tokens with W=128K, D=128: ~38 MB MLA KV vs ~3 GB uncompressed.

## Checkpoint Strategy

```python
{
    'step': int,
    'model_state': dict,           # excludes frozen backbone in pretrained mode
    'cognitive_state': dict,       # beliefs, edges, goals, meta — separate from model_state
    'optimizer_state': dict,
    'samples_consumed': int,       # for data stream resumption
}
```

- Checkpoint interval: configurable (default 1000 steps)
- Cognitive seed saved at training end (for next run)
- Async push to HuggingFace Hub (background thread)
- Beliefs and edge relations compressed via RotorQuant/PolarQuant in `state_dict_cognitive(compress=True)`

## Monitoring (Weights & Biases)

### Key Metrics

| Category | Metrics |
|----------|---------|
| Loss | loss, loss_token, loss_fe, loss_draft |
| Cognitive state | active_beliefs, active_edges, active_goals, fill_ratio, mean_radius |
| Pass 2 operations | beliefs_cleaned, edges_created, soft_merges, goals_generated |
| TTT health | delta_A/B norms, decay_alpha, update_accepted |
| Controller | actions, dense_reward |
| Uncertainty | log_sigma (token, fe, aux) |
| Belief advantage | belief_advantage (with-state vs without-state loss delta) |
| Provisional beliefs | provisional_count, promoted, evicted |
| Autoresearch | hypotheses_generated, success_rate |

### Alerts

- Belief store > 95% capacity
- TTT delta norm > 10.0 (explosion)
- Loss spike > 3× smoothed average
- NaN/Inf in any loss component

## File Map

```
memoria/training/
  train.py          — Main training loop, DataPrefetcher, checkpoint, hub push
  optimizer.py      — Muon + AdamW setup, 18 parameter groups, _CombinedOptimizer
  schedule.py       — LR (WSD) + alpha (KL annealing) + context length (SkyLadder) schedules
  distributed.py    — Cognitive state broadcast/gather for multi-GPU
  cognitive_seed.py  — Cross-run belief transfer via content matching

memoria/model/
  config.py          — TransformerConfig, StateConfig, TrainingConfig, presets
  transformer.py     — Hybrid transformer (SWA/Mamba-2/MLA), YaRN RoPE, ReLU²
  memoria_model.py   — MemoriaModel: transformer + state + interfaces + DFlash
  pretrained_model.py — PretrainedMemoriaModel: frozen HF backbone + interfaces
  dflash_head.py     — DFlash block diffusion draft head for speculative decoding

memoria/core/
  state.py           — CognitiveState: beliefs, edges, goals, meta-parameters
  free_energy.py     — Bethe FE, EFE, Power Spherical entropy
  losses.py          — chunked_cross_entropy, differentiable FE proxy
  quantize.py        — RotorQuant/PolarQuant quantization, STE, KV cache, belief store
  polar.py           — Polar coordinate utilities for belief representation
  ttt.py             — In-Place TTT with Titans/LaCT/DeltaProduct enhancements
  message_passing.py — Factor graph message passing with DEQ solver

memoria/interface/
  layer.py           — StateInterfaceLayer (read + write, inserted every K blocks)
  read_path.py       — BeliefCache, ReadPath (Hopfield attention over beliefs)
  write_path.py      — WritePath, WriteCandidate, pack/unpack for distributed

memoria/cognition/
  pass2.py           — Pass 2 orchestrator (12 structural operations)
  surprise.py        — Prediction error computation
  belief_update.py   — Slot allocation, eviction
  consolidation.py   — Soft merge, hard cleanup
  hebbian.py         — Co-activation extraction for edge candidates
  meta_learning.py   — Beta computation, running statistics
  telos_module.py    — TelosModule: goal generation, progress, transitions
  sleep.py           — SleepGate + NeuroDream phase
  planning.py        — Active inference planning (preference/epistemic priors)
  autoresearch.py    — Hypothesis generation from goals
  provisional.py     — Provisional belief evaluation (promote/evict)
  cascade_revision.py — Causal cascade precision decay

memoria/data/
  curated.py         — Multi-tier dataset mix (45% state-essential)
  streaming.py       — FineWeb-Edu streaming
  synthetic.py       — Belief tracking, contradiction, causal chain generators
  tokenizer.py       — Tokenizer setup (Qwen3 BPE or pretrained model's own)
```
