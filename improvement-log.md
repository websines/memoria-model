# Memoria Improvement Log

## 2026-04-02 — Implementation Complete: Differentiable Cognitive State

### What was done (61/61 tests pass)

**Phase 0: Fixed broken tests**
- tests/test_interface.py — fixed 8 signature mismatches (StateInterfaceLayer returns 7 values, ReadPath returns 5, WritePath returns tuple)

**Phase 1: Fixed 5 critical bugs**
- BUG-1: eval/perplexity.py — added skip to held-out region instead of evaluating on training data
- BUG-2: consolidation.py — edge dedup now uses directed keys (src,tgt) not canonical (min,max), dedup runs once after all merges not per-merge
- BUG-3: pass2.py — soft_consolidation now periodic (every 10 steps), not every step
- BUG-4: train.py — detects actual sequence boundaries via EOS token instead of hardcoding True
- BUG-5: pass2.py — goal generation moved after consolidation

**Phase 2: Created infrastructure**
- NEW: core/meta_params.py — MetaParams(nn.Module) with 15 learned nn.Parameter scalars (sigmoid/softplus activations)
- NEW: core/running_stats.py — RunningStats(nn.Module) with 5 EMA buffers deriving 10 adaptive thresholds
- core/state.py — added meta_params and running_stats to CognitiveState, updated serialization
- model/config.py — added belief_lr, cognitive_meta_lr, warmdown_type, warmdown_ratio=0.2

**Phase 3: Wired all magic numbers**
- losses.py — fe_lambda now learned parameter via meta_params
- belief_update.py → stripped to allocation-only (allocate_new_beliefs), removed Kalman gain
- hebbian.py — stripped to edge creation only, uses meta_params for initial weight
- causal.py — uses meta_params for min_signal, decay_rate, initial_weight_scale, relation_scale
- telos.py — uses meta_params for relevance_threshold, progress_rate, dedup_threshold; uses running_stats for cooldown, stall thresholds
- consolidation.py — pass2 passes running_stats merge_similarity_threshold and hard_cleanup_precision_threshold
- meta_learning.py — precision_decay_factor from meta_params

**Phase 4: Made cognitive state differentiable (THE CORE CHANGE)**
- state.py — beliefs, edge_relations, edge_weights now requires_grad=True; removed .data from read-path accessors
- read_path.py:101 — removed .data gradient wall (state.beliefs[active_indices] instead of state.beliefs.data[...])
- write_path.py:229 — removed .detach() wall; detach only in pack_candidates for distributed transport
- memoria_model.py — detach_state() now actually detach_()s before pass2; passes fe_lambda to compute_differentiable_free_energy
- pretrained_model.py — same detach_state and fe_lambda changes
- message_passing.py — removed .data from belief/edge access
- optimizer.py — added "cognitive" param group (beliefs+edges, slow LR, weight decay) and "cognitive_meta" group (15 learned params)

**Phase 6: Training config**
- schedule.py — added linear decay-to-zero warmdown option
- config.py — warmdown_ratio 0.5→0.2, warmdown_type="linear"

**Phase 7: Updated tests**
- test_pass2.py — updated for structural-only pass2 (removed incremental_updates/reconsolidation assertions)
- test_model.py — added test_gradient_flow_to_beliefs (PASSES: beliefs receive gradients) and test_cognitive_meta_params_in_model

**Proper Bethe Free Energy (replaces fake proxy)**
- core/free_energy.py — added `power_spherical_entropy()`: closed-form entropy via `torch.digamma` (De Cao & Aziz 2020). Replaces the made-up `1/(r+1)` with proper information-theoretic entropy for directional beliefs. Dimension-aware (D=256 matters).
- core/free_energy.py — added `compute_bethe_free_energy()`: proper Bethe free energy with (d_i-1) counting correction per Yedidia/Freeman/Weiss. Energy from edge disagreement + telos goals. Entropy from Power Spherical. Fully differentiable — gradients flow into beliefs, edge weights, edge relations.
- core/free_energy.py — updated legacy `compute_entropy()` to use Power Spherical instead of `1/(r+1)`. Fixed beta computation: use `-H/N` (negentropy per belief) as uncertainty measure since differential entropy of continuous distributions is negative.
- model/memoria_model.py — L_fe is now `L_fe_proxy + L_fe_bethe`. Proxy trains interfaces (read/write paths). Bethe trains the world model (beliefs, edges, relations through the factor graph).
- model/pretrained_model.py — same combined loss structure.
- The edge graph now contributes to the training gradient. Previously: zero contribution from edges to loss.

**Differentiable Telos (learned goal system, replaces all hardcoded goal heuristics)**
- NEW: cognition/telos_module.py — `TelosModule(nn.Module)` with 4 learned components:
  1. RND surprise: frozen random projection + trained predictor. Prediction error = novelty. Replaces `1/radius` proxy.
  2. Goal generator: MLP(belief_summary → goal_embedding). Replaces threshold-based generation.
  3. Progress estimator: cosine attention over beliefs → MLP → progress in [0,1]. Replaces counter-based progress.
  4. Transition network: MLP(goal_features → status_logit_deltas) for Gumbel-Softmax lifecycle.
- core/state.py — goal_embeddings now `requires_grad=True`. Added `goal_status_logits` buffer [max_goals, 6] for Gumbel-Softmax status (replaces float encoding). Added TelosModule to state. Updated get_active_goals/num_active_goals to use Gumbel-Softmax.
- model/memoria_model.py — Telos runs in forward pass: RND surprise loss + progress estimation + status transitions. Added `loss_surprise` to combined loss.
- cognition/pass2.py — Rewritten to structural cleanup only (~130 lines, down from ~190). Removed all heuristic update rules. Kept: slot allocation, edge topology creation, structural cleanup (zero-norm removal), periodic consolidation, learned goal generation via TelosModule, beta computation, running stats.
- training/optimizer.py — Added Telos param group and goal_embeddings param group.
- Gumbel-Softmax temperature anneals from 1.0 → 0.1 over training via `telos.anneal_temperature()`.

**Pass 2 is now structural cleanup only. Zero heuristic update rules remain.**
- Continuous updates (beliefs, edges, relations, goal embeddings): gradient via optimizer.step()
- Discrete structural ops (slot allocation, edge creation, consolidation merges): pass2
- Goal lifecycle (proposed→active→completed): learned Gumbel-Softmax transition network
- Surprise signal: learned RND, not handcoded formula
- Progress estimation: learned MLP + attention, not counter

**Dependencies added**: torchopt>=0.7, nevergrad>=1.0

**Phase 5: Removed SPSA**
- train.py — `spsa_controller = None`, SPSA never runs. Its 3 tunable params (reconsolidation_threshold, match_threshold, goal_dedup_threshold) are now nn.Parameters in MetaParams, trained by backprop through L_fe.
- train.py — removed SPSA checkpoint save/load (old checkpoints with spsa_state gracefully ignored)
- SPSAController class kept in meta_learning.py for backward compat only, never instantiated
- nevergrad available as dependency if future discrete parameters need gradient-free optimization

### Magic number status: 0 cognitive heuristics at runtime
- 26 eliminated (gradients replace them)
- 15 learned nn.Parameter (trained by backprop via L_fe)
- 10 state-derived (computed from running statistics)
- 3 kept as init-only (gate bias init, alive check, chunk size)
- Structural constants (STATUS_ACTIVE=0.4, etc.) are semantic definitions, not tunable heuristics

---

## 2026-04-02 — Full Codebase Audit & Architecture Review

### Current Training State
- Step 1800/5000, Phase 1 (alpha=0), 2 GPUs
- 326.9M params (318.3M transformer + 8.6M interfaces)
- Loss: 3.698, Eval perplexity: 118.72 (on train split — see bug #1)
- Beliefs: 4096/4096 (maxed), Edges: 7199/16384, Goals: 3/64
- Beta: 0.774, dt: ~17.8s/step

---

## Critical Bugs

### BUG-1: Eval perplexity uses training split
- **File**: `memoria/eval/perplexity.py:36`
- **Issue**: `split="train"` hardcoded — all perplexity metrics are on training data
- **Impact**: Cannot measure generalization. All logged perplexity numbers are invalid.
- **Fix**: Change to a held-out split

### BUG-2: Edge deduplication causes catastrophic edge collapse
- **File**: `memoria/cognition/consolidation.py:170-171, 187-189`
- **Issue**: After belief merges, `_redirect_edges()` calls `_deduplicate_edges()` which uses canonical undirected keys `(min(src,tgt), max(src,tgt))`. This treats directed causal edges as undirected and collapses all edges sharing the same canonical pair. Called inside the merge loop, cascading across merges.
- **Evidence**: Edge count dropped from 7595 to 909 at step 1000. Perplexity spiked from 63.47 to 118.72 at step 1500 during recovery.
- **Fix**: Either (a) use directed keys `(src, tgt)` not canonical, (b) only dedup after all merges complete not per-merge, or (c) track redirected edges and only dedup those.

### BUG-3: Soft consolidation runs every step
- **File**: `memoria/cognition/pass2.py` (wherever `soft_consolidation()` is called)
- **Issue**: Runs on every pass2 call, not periodically. Combined with BUG-2, any step can trigger a cascade.
- **Fix**: Run on a configurable interval or when belief count exceeds threshold.

### BUG-4: `is_sequence_boundary=True` hardcoded
- **File**: `memoria/training/train.py:352`
- **Issue**: Pass2 always called with `is_sequence_boundary=True`. This triggers beta decay and timer increments every step instead of at actual sequence boundaries. Beta dropped from 1.0 to 0.766 faster than intended.
- **Fix**: Detect actual sequence boundaries from the data stream.

### BUG-5: Goals created before consolidation
- **File**: `memoria/cognition/pass2.py` (~line 133 vs ~line 153)
- **Issue**: Goal generation references beliefs that consolidation may subsequently merge/free. Goals can point to deallocated slots.
- **Fix**: Move goal generation after consolidation.

---

## Architectural Problems

### ARCH-1: 54 hardcoded magic numbers govern cognitive state dynamics
The cognitive state (beliefs, edges, goals) is updated entirely by Pass 2's hand-crafted rules. These rules contain 54 arbitrary constants with no mathematical derivation. SPSA only tunes 3 of them.

**Belief dynamics:**
| Constant | File:Line | What it controls |
|---|---|---|
| `10.0` | belief_update.py:128 | Precision growth headroom cap |
| `0.5` | belief_update.py:132 | Precision decay rate on disagreement (50% per step) |
| `0.5` | belief_update.py:127 | Surprise threshold for agree/disagree binary split |
| `0.1` | belief_update.py:130 | Observation contribution weight for precision growth |
| `100.0` | belief_update.py:135 | Max radius clamp (inconsistent with 10.0 growth cap) |
| `0.1` | belief_update.py:170 | Eviction recency weight |

**Edge dynamics:**
| Constant | File:Line | What it controls |
|---|---|---|
| `0.95` | consolidation.py:21 | Merge similarity threshold |
| `512` | consolidation.py:22 | Max beliefs to check for merging |
| `0.1` | consolidation.py:112 | Hard cleanup low-precision threshold |
| `3` | consolidation.py:137 | Access count threshold for removal |
| `0.05` | hebbian.py:26 | Hebbian learning rate |
| `0.01` | hebbian.py:27 | Hebbian decay rate |
| `0.01` | hebbian.py:95 | Edge death threshold |
| `0.1` | hebbian.py:110 | Initial Hebbian edge weight |
| `0.005` | causal.py:207 | Causal edge decay rate |
| `0.1` | causal.py:206 | Minimum causal signal strength |
| `0.1` | causal.py:277,290 | Relation vector scaling |
| `0.3` | causal.py:293 | Initial causal edge weight |
| `0.01` | causal.py:103,322 | Edge weight alive/death threshold |

**Goal system:**
| Constant | File:Line | What it controls |
|---|---|---|
| `50` | telos.py:42 | Cooldown steps between goal generations |
| `2.0` | telos.py:63 | Numerator in adaptive threshold formula |
| `3` | telos.py:90 | Max new goals scaling factor |
| `3.0` | telos.py:126 | Default goal depth |
| `0.3` | telos.py:179 | Relevance threshold for goal progress |
| `0.1` | telos.py:181 | Progress delta scaling |
| `20,50,100,200` | telos.py:220-226 | Stall detection thresholds per urgency |
| `0.5` | telos.py:276 | Goal dedup threshold default |

**Free energy / meta:**
| Constant | File:Line | What it controls |
|---|---|---|
| `0.1` | losses.py:115 | Lambda in F = E - 0.1*H (most important cognitive hyperparameter) |
| `5.0` | pass2.py:42 | Temperature for sigmoid in energy computation |
| `50` | pass2.py:41 | Consolidation interval |
| `1024` | pass2.py:32 | Max candidates safety valve |
| `10` | pass2.py:165 | Sequence boundary decay period |
| `5.0` | meta_learning.py:26 | Temperature for energy |
| `100` | meta_learning.py:69 | SPSA interval |
| `10` | meta_learning.py:70 | SPSA eval window |
| `0.01` | meta_learning.py:71 | SPSA perturbation scale |
| `0.001` | meta_learning.py:72 | SPSA step size |
| `0.995` | meta_learning.py:204 | Sequence boundary decay factor |

**Interface:**
| Constant | File:Line | What it controls |
|---|---|---|
| `-2.0` | write_path.py:123 | Write gate bias init |
| `0.05` | write_path.py:204 | Meaningful observation threshold |
| `1/(r+1)` | free_energy.py:98 | Entropy formula (arbitrary monotonic function) |

### ARCH-2: Cognitive state is entirely non-differentiable
- **The problem**: Beliefs, edges, goals, and meta are all updated by Pass 2 using hand-crafted rules. Gradients from L_token, L_fe, and L_utility flow through the interface parameters (query_proj, output_proj, obs_proj, write_gate, precision_head, etc.) but never into the state content.
- **Gradient walls**:
  - Read path: `state.beliefs.data[active_indices]` — `.data` access bypasses autograd (read_path.py:101)
  - Write path: `obs_detached = obs_flat.detach()` — explicit detach before building candidates (write_path.py:229)
- **Consequence**: The interface learns HOW to read/write, but the state doesn't learn WHAT to store. What it stores is determined by the 54 magic numbers.

### ARCH-3: Insufficient training tokens
- Current: ~655M tokens (~2 tokens/param)
- Chinchilla optimal: 6.5B tokens (20 tokens/param) — 10x current
- Modern practice: 20-65B tokens (60-200 tokens/param) — 30-100x current
- Phase 2/3 cognitive integration cannot succeed if the base transformer hasn't learned language

---

## Proposed Solution: Eliminate All 54 Magic Numbers

### Principle: Zero heuristics. Gradients decide everything learnable. State statistics decide everything structural.

### Complete Elimination Plan

**Category A — ELIMINATED (26 constants): Gradients replace them entirely.**

These constants exist because Pass 2 uses hand-crafted update rules. Once beliefs and edge weights become differentiable (in the computation graph via L_fe), these rules and their constants are deleted.

| # | Constant | File:Line | Why it dies |
|---|---|---|---|
| 1 | `10.0` headroom cap | belief_update.py:128 | Gradient on radius handles growth. Weight decay prevents blowup. |
| 2 | `0.5` disagree decay | belief_update.py:132 | L_fe gradient decreases radius when belief is wrong. |
| 3 | `0.5` surprise threshold | belief_update.py:127 | No binary split. Gradient is continuous. |
| 4 | `0.1` obs contribution | belief_update.py:130 | Gradient step size = optimizer LR for belief param group. |
| 5 | `100.0` max radius clamp | belief_update.py:135 | Weight decay on beliefs prevents unbounded growth. |
| 6 | `0.1` hard cleanup threshold | consolidation.py:112 | Beliefs with zero gradient flow are dead. No threshold. |
| 7 | `3` access count cutoff | consolidation.py:137 | Gradient magnitude replaces access counting. |
| 8 | `0.01` Hebbian edge death | hebbian.py:95 | Edge weights are differentiable → gradient drives useless ones to zero. |
| 9 | `0.01` causal edge death | causal.py:322 | Same as above. |
| 10 | `0.1` min causal signal | causal.py:206 | Edge creation gate decides, not a threshold. |
| 11 | `0.05` obs threshold | write_path.py:204 | Gate already outputs near-zero for weak obs. Threshold redundant. |
| 12 | `10` decay period | pass2.py:165 | No hardcoded period. Decay is continuous via gradient. |
| 13 | `1/(r+1)` entropy formula | free_energy.py:98 | With differentiable beliefs, use proper log-based entropy or let Pyro derive it. |
| 14 | `0.01` SPSA perturbation | meta_learning.py:71 | Replace SPSA with TorchOpt differentiable meta-optimization. |
| 15 | `0.001` SPSA step size | meta_learning.py:72 | Same — TorchOpt replaces gradient-free search. |
| 16-19 | SPSA clamp ranges (4 pairs) | meta_learning.py:60-63 | Sigmoid/softplus bound parameters naturally. No hard clamps. |
| 20-23 | `20,50,100,200` stall thresholds | telos.py:220-226 | Derive from running mean goal completion time. |
| 24 | `0.3` relevance threshold | telos.py:179 | Replace with soft weighting: use similarity as continuous weight, no binary threshold. |
| 25 | `0.01` Hebbian decay rate | hebbian.py:27 | Edge weights differentiable → optimizer weight decay handles this. |
| 26 | `0.005` causal decay rate | causal.py:207 | Same as above. |

**Category B — LEARNED nn.Parameter (15 constants): Trained by backprop through L_fe.**

These control dynamics that benefit from gradient-based tuning. Each becomes a trainable scalar or small tensor in a dedicated "cognitive meta" parameter group with its own LR.

| # | Constant | File:Line | Becomes |
|---|---|---|---|
| 27 | `0.1` lambda in F=E-λH | losses.py:115 | `nn.Parameter` — energy-entropy balance. Most important learned param. |
| 28 | `5.0` temperature (pass2) | pass2.py:42 | `nn.Parameter` — shared with read path pattern (log_temperature). |
| 29 | `5.0` temperature (meta) | meta_learning.py:26 | Same parameter as #28. |
| 30 | `0.05` Hebbian learning rate | hebbian.py:26 | `nn.Parameter` — per-edge-type learning rate. |
| 31 | `0.1` initial Hebbian weight | hebbian.py:110 | `nn.Parameter` — output of edge creation MLP. |
| 32 | `0.1` relation scaling | causal.py:277 | `nn.Parameter` — learnable relation scale per edge type. |
| 33 | `0.1` relation scaling | causal.py:290 | Same parameter as #32. |
| 34 | `0.3` initial causal weight | causal.py:293 | `nn.Parameter` — learned initial weight for new causal edges. |
| 35 | `2.0` threshold formula | telos.py:63 | `nn.Parameter` — learnable scalar in goal threshold. |
| 36 | `0.1` progress scaling | telos.py:181 | `nn.Parameter` — goal progress rate. |
| 37 | `0.995` boundary decay | meta_learning.py:204 | `nn.Parameter` — precision decay rate as trainable scalar. |
| 38 | reconsolidation_threshold | meta_learning.py | `nn.Parameter` — currently SPSA-tuned, move to optimizer. |
| 39 | match_threshold | meta_learning.py | `nn.Parameter` — same. |
| 40 | goal_dedup_threshold | meta_learning.py | `nn.Parameter` — same. |
| 41 | `0.95` merge threshold | consolidation.py:21 | Learned merge gate: small MLP on belief pair features → merge probability. |

**Category C — STATE-DERIVED (10 constants): Computed from running statistics, not hardcoded.**

These control structural/scheduling decisions. Replaced with functions of the cognitive state itself.

| # | Constant | File:Line | Derived from |
|---|---|---|---|
| 42 | `0.1` eviction recency | belief_update.py:170 | Eviction score = inverse gradient magnitude. Beliefs that matter to loss survive. |
| 43 | `512` max beliefs to check | consolidation.py:22 | `min(n_active, compute_budget)` where budget scales with step time. |
| 44 | `50` consolidation interval | pass2.py:41 | Trigger when belief_count / max_beliefs > 0.9 (capacity-driven). |
| 45 | `1024` max candidates | pass2.py:32 | Scale with `batch_size * n_interfaces`. |
| 46 | `50` goal cooldown | telos.py:42 | `f(active_goal_count, surprise_rate)` — more goals → longer cooldown. |
| 47 | `3` max goals scaling | telos.py:90 | Scale with beta: high uncertainty → more exploration → more goals. |
| 48 | `3.0` default depth | telos.py:126 | `f(goal_embedding_norm)` — complexity of the goal determines depth. |
| 49 | `0.5` goal dedup default | telos.py:276 | Already SPSA-tuned → now nn.Parameter (moved to Category B). |
| 50 | `100` SPSA interval | meta_learning.py:69 | Tune frequency based on free energy variance. High variance → more frequent. |
| 51 | `10` SPSA eval window | meta_learning.py:70 | `f(loss_stability)` — stable loss → shorter window needed. |

**Category D — KEPT AS INIT (3 constants): Initialization values only, not runtime behavior.**

| # | Constant | File:Line | Why it's fine |
|---|---|---|---|
| 52 | `-2.0` gate bias init | write_path.py:123 | Just initialization. Gate learns its own bias via gradient. |
| 53 | `0.01` edge alive check | causal.py:103 | Graph connectivity check, not a learning rule. Structural. |
| 54 | `4096` chunk size | losses.py:23 | Memory management, not a learning hyperparameter. |

**Result: 0 magic numbers governing cognitive dynamics at runtime.**

### Implementation Steps

**Phase 1: Tear down gradient walls**
1. [ ] Remove `.data` access in `read_path.py:101` — beliefs enter computation graph
2. [ ] Remove `.detach()` in `write_path.py:229` — write observations stay in graph
3. [ ] Make `state.beliefs` a `nn.Parameter` (or buffer with `requires_grad=True`)
4. [ ] Make `state.edge_weights` differentiable
5. [ ] Add beliefs + edge weights to optimizer in a "cognitive" param group with slow LR
6. [ ] Add weight decay to cognitive param group (replaces radius clamp + edge death thresholds)

**Phase 2: Replace Pass 2 heuristics with learned parameters**
7. [ ] Create `nn.ParameterDict` for Category B learned parameters (15 scalars)
8. [ ] Add to optimizer in "cognitive_meta" param group
9. [ ] Replace Kalman gain formula — belief content updates via L_fe gradient directly
10. [ ] Replace precision headroom/decay — radius updates via L_fe gradient directly
11. [ ] Replace SPSA with TorchOpt differentiable optimization for meta-parameters
12. [ ] Replace hardcoded edge thresholds with learned edge creation gate (small MLP)
13. [ ] Replace merge threshold with learned merge gate

**Phase 3: Derive structural parameters from state**
14. [ ] Implement running statistics tracker (surprise mean/std, precision mean/std, goal completion time)
15. [ ] Replace Category C constants with functions of running statistics
16. [ ] Remove SPSA controller entirely (all its targets are now either nn.Parameter or state-derived)

**Phase 4: Simplify Pass 2**
17. [ ] Pass 2 becomes: (a) discrete edge creation/destruction, (b) goal lifecycle, (c) consolidation merges
18. [ ] All continuous updates (belief content, precision, edge weights) handled by optimizer.step()
19. [ ] Delete `belief_update.py` Kalman gain code
20. [ ] Delete `meta_learning.py` SPSA code

---

## Training Configuration Improvements

### TRAIN-1: Switch LR schedule from 50% cosine warmdown to WSD 20% linear D2Z
- Current: 2% warmup, 48% constant, 50% cosine decay
- Proposed: 2% warmup, 78% constant, 20% linear decay-to-zero
- Rationale: WSD research (ICML 2025) shows longer stable phase at full LR is crucial. Reclaims ~1500 steps of effective training.

### TRAIN-2: Batch size warmup
- Current: Fixed 131K tokens/step from step 0
- Proposed: Warmup from 32K to 131K over first 10-20% of training
- Rationale: Allen AI showed up to 43% fewer wasted gradient steps. Critical when tokens are scarce.

### TRAIN-3: Increase total training tokens
- Current: ~655M tokens (5000 steps x 131K)
- Minimum viable: 6.5B tokens (Chinchilla optimal)
- Options: increase steps to 50K, or increase batch size, or both

### TRAIN-4: LR sweep
- Current values from Karpathy autoresearch (validated for 512-dim model)
- Autoresearch found improvements at higher values:
  - embedding_lr: 0.6 -> 0.9
  - matrix_lr (Muon): 0.04 -> 0.10
  - scalar_lr: 0.5 -> 1.0
- Sweep: matrix_lr in {0.04, 0.08}, embedding_lr in {0.6, 0.9}

### TRAIN-5: Data mix (not urgent, do when scaling)
- Current: 70% FineWeb-Edu / 20% Stack v2 code / 10% synthetic
- FineWeb-Edu is actually good for Memoria — educational text is explanatory, causal, structured. Produces meaningful beliefs and relations.
- When scaling to 6.5B+ tokens, switch to: **40% FineWeb-Edu + 30% DCLM-Baseline + 20% code + 10% synthetic**
- DCLM-Baseline (4T tokens): more diverse, stronger benchmarks than FineWeb-Edu alone
- Nemotron-CC (6.3T tokens): best quality per token, +5.6 MMLU over DCLM
- Priority: train longer on current data FIRST, then improve mix

---

## Complexity Risk Assessment

These components may not earn their cost at 326M parameter scale:

| Component | Risk | Notes |
|---|---|---|
| Full Telos goal lifecycle | HIGH | Decomposition, stall detection, deadlines — no evidence 326M model uses this meaningfully. Titans works with just "surprise -> memorize". |
| Do-calculus on relation graph | HIGH | Requires model to learn meaningful causal structure. Validate after Phase 1: does the graph have any causal structure, or is it noise? |
| SPSA meta-learning | MEDIUM | Noisy and slow for 3 params. If we expand to ~8 params for discrete ops, still manageable. |
| Consolidation machinery | MEDIUM | If beliefs become differentiable, gradient pressure naturally handles what Kalman+consolidation do now. May simplify significantly. |

---

## Verified: What's Theoretically Sound

| Component | Status | Source |
|---|---|---|
| Kalman gain formula | Correct (standard precision-weighted gain) | Friston AIF, Kalman 1960 |
| Muon optimizer | Exact match to reference (canonical coefficients) | Keller Jordan, PyTorch core |
| Saturating Hebbian edges | Textbook | Ba & Hinton 2016 |
| Causal edge learning (temporal precedence) | Grounded in Granger causality | Granger 1969 |
| Three-phase training (KL annealing) | Standard | Higgins et al. 2017 (beta-VAE) |
| Polar beliefs (radius=precision, angle=content) | Novel, geometrically sound | TurboQuant, Polar Embedding, NeurIPS 2024 |
| Bethe free energy as LM training objective | Novel, no precedent | Original contribution |
| Structured cognitive state as native tensor ops | Novel, no precedent | Original contribution |

---

## Related Work (for reference)

| System | Year | Relationship to Memoria |
|---|---|---|
| Titans (Google) | 2025 | Closest — surprise-driven memory, but unstructured |
| LM2 (Convergence Labs) | 2025 | LSTM-gated external memory, no graph structure |
| TTT (Stanford) | 2024 | Gradient-based state updates (what we should move toward) |
| Gated DeltaNet (NVIDIA/ICLR 2025) | 2025 | Error-correcting state updates, cited in write path |
| HOPE/Nested Learning (NeurIPS 2025) | 2025 | Multi-rate updates, theoretical grounding for fast/slow split |
| MemoryLLM (ICML 2024) | 2024 | Self-updated memory via FFN, no structure |
| NTM/DNC (DeepMind) | 2014/2016 | Ancestor — flat differentiable memory, died at scale |
| NAMMs (Sakana AI) | 2025 | Evolutionary optimization of non-differentiable memory ops |

---

## Libraries to Adopt

### Priority 1: Replace heuristics with gradient-based learning

**TorchOpt** (metaopt/torchopt, 628 stars, active Mar 2026)
- Differentiable optimizers with explicit, implicit, and zero-order modes
- Replace `SPSAController` with proper differentiable optimization through cognitive state updates
- Enables making belief update gain, reconsolidation threshold, decay rates into learnable parameters that receive gradients through free energy
- Supersedes Facebook's `higher` (unmaintained since 2022)

**Flash Linear Attention / Gated DeltaNet** (fla-org/flash-linear-attention, 4790 stars, active Apr 2026)
- Umbrella library containing DeltaNet, GLA, RetNet with Triton kernels
- DeltaNet's delta rule = differentiable key-value memory with learned alpha (decay gate) and beta (write gate)
- Maps directly to belief update: replace Kalman gain heuristic with learned gating
- GLA = differentiable key-value store with learned write/erase — most relevant for belief updates
- Production-grade Triton kernels for efficient state updates

**titans-pytorch** (lucidrains/titans-pytorch, 1935 stars, active Feb 2026)
- Surprise-as-gradient: treats surprise as gradient of associative memory loss (differentiable)
- Replace hand-crafted `compute_surprise_batch()` and Kalman gain in `belief_update.py`
- Write path becomes end-to-end differentiable instead of buffered candidates + pass 2

### Priority 2: Scale the relational graph

**PyTorch Geometric** (pyg-team/pytorch_geometric, 23628 stars, active Mar 2026)
- Already partially used in `core/message_passing.py`
- Replace manual edge management in `state.py` and `hebbian.py` with PyG dynamic graph primitives
- Edge weights with `requires_grad=True` → learn via backprop through message passing
- DynamicEdgeConv constructs graphs in feature space (replaces static edge allocation)
- Temporal Graph Networks for evolving relation graph

**torch-bp** (janapavlasek/torch-bp, active 2024)
- Gaussian BP and Stein Variational BP in PyTorch
- Replace hand-written precision-weighted message fusion in `FactorGraphMessagePassing`
- Gaussian BP variant maps to Memoria's precision-weighted beliefs

### Priority 3: Probabilistic free energy

**Pyro** (pyro-ppl/pyro, 8991 stars, active Jul 2025)
- Stochastic Variational Inference with `TraceMeanField_ELBO` = Bethe free energy minimization
- Define cognitive state as probabilistic graphical model, let autograd compute free energy gradient
- Replaces manual E and H computation in `core/free_energy.py`
- Makes entire Pass 2 differentiable via variational inference

### Priority 4: Gradient-free fallback for discrete operations

**Nevergrad** (facebookresearch/nevergrad, 4171 stars, active Mar 2026)
- Gradient-free optimization toolkit (CMA-ES, TBPSA, PSO, automatic algorithm selection)
- Direct upgrade for remaining `SPSAController` uses on discrete thresholds
- `NGIoh21` wizard auto-selects best algorithm for the problem

**EvoTorch** (nnaisense/evotorch, 1129 stars, active Mar 2026)
- Evolutionary strategies on GPU with native PyTorch tensor support
- PGPE or CMA-ES for searching space of discrete structural parameters
- GPU parallelism: evaluate hundreds of configurations simultaneously

### Reference architectures (for design guidance)

**LM2** (convergence-ai/lm2) — Cross-attention + input/forget/output gates for memory. Cleanest existing memory-transformer interface.
**MemoryLLM** (wangyu-ustc/MemoryLLM) — Self-updated memory via FFN per layer. M+ extension adds co-trained retriever.
**pymdp** (infer-actively/pymdp) — Discrete active inference reference. Conceptually aligned but discrete state space.
