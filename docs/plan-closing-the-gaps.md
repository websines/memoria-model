# Closing the Gaps: From Self-Improving AI to Autonomous Intelligence

> Compiled 2026-04-03. Research synthesis across 100+ papers (2024-2026) targeting four architectural gaps in Memoria.
>
> **Implementation log (2026-04-04):** Phase A foundations (A1-A3) and the Internal Autoresearch Loop (Section 0) are implemented and tested. 93/93 tests passing. See implementation notes inline.

## The Four Gaps

| Gap | Current State | Target |
|-----|---------------|--------|
| **1. Recursive Self-Improvement** | MetaParams has 31 learned nn.Parameters (sigmoid/softplus constrained, gradient-trained); meta vector slots [7:] reserved. Internal Autoresearch Loop operational (HypothesisGenerator + HypothesisTracker). | System invents new learning mechanisms, modifies its own update rules |
| **2. Long-Horizon Planning** | Telos tracks goals, EFE scores 1-step | Multi-step simulation over belief graph, MCTS via free energy |
| **3. Belief Robustness** | Precision weighting + consolidation + **MESU precision variance** (per-belief uncertainty with windowed posterior) + **causal cascade revision** (precision decay through downstream beliefs on revision) | Formal consistency guarantees, drift prevention, self-verification |
| **4. Autonomous Agency** | Passive responder (input -> output). Internal autoresearch generates goal-directed hypotheses as provisional beliefs. | Persistent daemon with self-directed exploration and tool use |

**Unifying Insight:** All four gaps converge on one pattern -- the **Internal Autoresearch Loop**. Inspired by Karpathy's autoresearch (github.com/karpathy/autoresearch), but internalized into the cognitive state rather than running as an external agent.

---

## 0. The Internal Autoresearch Loop (Cross-Cutting)

Karpathy's autoresearch: an external agent edits code, runs a 5-min experiment, keeps/discards via git, repeats forever. Found ~20 additive improvements in 700 experiments.

**Internalized in Memoria:**

| Autoresearch Step | Memoria Implementation |
|-------------------|----------------------|
| Read `program.md` (research direction) | Active Telos goal with highest EFE |
| Propose hypothesis | Generate tentative belief in scratch region |
| Modify `train.py` | Place tentative belief into cognitive state (marked provisional) |
| Run experiment (5 min) | Run N forward passes, measure free energy delta |
| Extract `val_bpb` | Compare FE before/after: did uncertainty decrease? |
| Keep/discard via git | Promote provisional belief to committed (if FE decreased) or evict |
| Log to `results.tsv` | Update meta-cognitive statistics (which hypothesis types succeed) |
| Repeat forever | Next Pass 2 cycle |

**Key mechanism: Tentative Belief Mode.** Add a `provisional` flag to belief slots. Provisional beliefs:
- Participate in forward passes (read path retrieves them, write path updates them)
- Do NOT increase their reinforcement count
- Are evaluated after K steps: if FE decreased AND precision increased, promote to committed
- If FE increased or precision dropped, evict and free the slot
- The meta-learning system tracks which *types* of hypotheses succeed, biasing future generation

This turns Pass 2 from reactive cleanup into **deliberate experimentation**.

### Implementation Status: DONE (2026-04-04)

All seven steps of the autoresearch loop are implemented:

| Autoresearch Step | Implementation | File |
|-------------------|----------------|------|
| Read `program.md` (research direction) | Active Telos goal with highest EFE | `cognition/telos_module.py` (existing) |
| Propose hypothesis | `HypothesisGenerator`: learned MLP (goal + belief summary + progress + β → hypothesis vector). Gated: starts mostly closed (bias=-1.0), learns to open for productive goals. | `cognition/autoresearch.py` |
| Modify `train.py` | `state.allocate_belief(..., provisional=True, current_fe=...)` | `core/state.py` |
| Run experiment (5 min) | Forward passes naturally test hypotheses — gradients from L_token + L_fe flow through provisional beliefs | (automatic) |
| Extract `val_bpb` | `evaluate_provisional_beliefs()`: compares current FE vs stored FE at allocation time, checks precision retention | `cognition/provisional.py` |
| Keep/discard via git | Promote (clear provisional flag) or evict (deallocate slot) | `cognition/provisional.py` |
| Log to `results.tsv` | `HypothesisTracker`: per-goal hypothesis_count, promoted, evicted, EMA success rate. Goals with success EMA < 0.2 stop producing hypotheses. | `cognition/autoresearch.py` |

**Tentative Belief Mode** adds 6 buffers to `CognitiveState`:
- `belief_provisional` (bool): is this belief a hypothesis under evaluation?
- `belief_provisional_step` (float): when was it allocated?
- `belief_provisional_fe` (float): global FE at allocation time
- `belief_provisional_radius` (float): initial radius for retention check
- `belief_precision_var` (float): MESU precision variance (A2, used by evaluation)
- `belief_reinforcement_count` (long): windowed posterior count (A2)

Provisional beliefs do NOT increment `access_count` when touched (no reinforcement before evaluation).

**All thresholds are learned** via MetaParams (10 new nn.Parameters):
- `provisional_eval_window`, `provisional_fe_threshold`, `provisional_precision_retention` (A1)
- `mesu_min_variance`, `mesu_variance_shrink`, `mesu_window_size`, `mesu_gain_boost` (A2)
- `cascade_decay_factor`, `cascade_max_depth`, `cascade_variance_boost` (A3)

Tests: 32 tests across `test_a1_provisional.py`, `test_a2_mesu.py`, `test_a3_cascade.py`, `test_autoresearch.py`.

---

## 1. Recursive Self-Improvement

### Problem
The system tunes its own hyperparameters (31 meta-params via gradient descent with sigmoid/softplus constraints, up from 21 after A1-A3 implementation) but can't invent new learning mechanisms. The update rules for beliefs, edges, and goals are hand-coded.

### Solution: Three Levels of Self-Modification

#### Level 1: Self-Referential Meta Region (Implementable Now)
**Source:** Self-Referential Weight Matrix (SRWM) -- arXiv:2202.05780 (Schmidhuber group)

The meta region is a flat `meta[32]` vector (β, surprise, timers, thresholds in slots [0:7], reserved in [7:]). Separately, `MetaParams` holds 21 gradient-trained nn.Parameters with sigmoid/softplus constraints. Replace the meta vector with a self-referential weight matrix:
```
W_meta += lr * outer(key, value)
```
Where key/value are computed from the cognitive state's current statistics. The matrix learns to produce update rules for the other regions.

**Concretely:** Instead of `MetaParams.hebbian_lr` being `sigmoid(scalar)`, it becomes the output of `W_meta @ state_features`. The update rule for Hebbian edge creation is now a *learned function* of the current cognitive state, not a fixed sigmoid-constrained scalar. The existing `CognitiveController` (SEAL-style, `state.controller`) already does something similar for structural decisions — this extends the pattern to all meta-parameters.

#### Level 2: Meta-Learned Update Functions (Medium Effort)
**Source:** ACL -- Metalearning Continual Learning Algorithms (arXiv:2312.00276)

Parameterize the entire Pass 2 update function as a small neural network:
- Input: (belief_vector, observation_vector, current_precision, prediction_error, edge_context)
- Output: (new_belief_vector, new_precision, should_allocate, should_merge)

Meta-train this network so that applying it as the Pass 2 update minimizes free energy over multi-step horizons. The system discovers its own belief update algorithm.

**Safety gate (critical):** SGM -- Statistical Godel Machine (arXiv:2510.10232). Before deploying any self-modified update rule, run a statistical confidence test (e-values, Hoeffding bounds). Only accept modifications certified superior at chosen confidence level. Global error budget bounds cumulative risk.

#### Level 3: Structural Self-Modification (Research Frontier)
**Sources:**
- SMGrNN (arXiv:2512.12713): Structural Plasticity Module monitors activation statistics, triggers neuron insertion/pruning via local signals
- FPE -- Expand Neurons Not Parameters (arXiv:2510.04500): Split polysemantic belief slots into children, partition weights disjointly
- DynMoE (arXiv:2405.14297): Dynamic expert count -- model decides its own capacity

**Implementation:**
1. Monitor per-belief activation statistics in a temporal window
2. Beliefs with sustained high co-activation across dissimilar contexts -> polysemantic -> split via FPE
3. Regions with sustained high surprise but low belief density -> allocate new slots
4. Dead beliefs (low activation, low precision) -> prune and reclaim
5. If fill ratio stays >95% after pruning -> expand the region (DynMoE-style)

All driven by local signals. No architectural hyperparameter needed.

#### Level 4: Learning New Learning Algorithms (Partially Exists)
**Already implemented (skill generalisation commit, 3fad55e):**
- `SleepGate`: learned neural gate deciding strengthen/maintain/forget — a learned consolidation algorithm
- `MetaParams`: 21 gradient-trained parameters — the system learns its own learning rates, thresholds, decay factors
- `CognitiveController`: SEAL-style policy over structural operations — learns *when* to allocate/merge/prune/connect
- `RunningStats`: EMA-derived adaptive thresholds — self-adjusting dynamics

**What this means:** Memoria already does Level 4 in embryonic form. The learning dynamics are themselves learned. The gap is that the *space* of possible learning algorithms is fixed (3 sleep actions, 5 controller outputs, 21 scalar params). The system tunes within a fixed algorithm space but can't escape it.

**Sources for the full version:**
- DiscoPOP (arXiv:2406.08414): LLM discovers novel loss functions via propose-evaluate-archive
- Godel Agent (arXiv:2410.04444): Self-referential agent modifies its own logic
- Darwin Godel Machine (arXiv:2505.22954): Archive-and-mutate pattern for agent evolution

When Memoria runs with a frontier backbone (Claude/GPT-level), the backbone can reason about the cognitive state's learning dynamics. The Internal Autoresearch Loop at this level:
1. Telos generates meta-research goal: "find a better consolidation schedule"
2. Backbone proposes a modification to the Pass 2 consolidation logic (as a parametric change)
3. Run N evaluation cycles in tentative mode
4. SGM safety gate certifies improvement
5. Commit the new logic

This is recursive self-improvement bounded by statistical safety.

### Key Papers
| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Self-Referential Weight Matrix | 2202.05780 | Meta region upgrade |
| ACL: Metalearning CL Algorithms | 2312.00276 | Learn update functions |
| Statistical Godel Machine | 2510.10232 | Safety gate for self-modification |
| SMGrNN: Self-Motivated Growing NN | 2512.12713 | Structural plasticity via local signals |
| FPE: Expand Neurons Not Parameters | 2510.04500 | Split polysemantic beliefs |
| DynMoE | 2405.14297 | Dynamic capacity expansion |
| DiscoPOP | 2406.08414 | LLM-discovered loss functions |
| Godel Agent | 2410.04444 | Self-referential agent |
| Darwin Godel Machine | 2505.22954 | Archive-and-mutate evolution |
| OPEN: Learned Optimization | 2407.07082 | Full update function learning |
| Mixture-of-Recursions | 2507.10524 | Learned per-token recursion depth |

---

## 2. Long-Horizon Planning

### Problem
Telos tracks goals and EFE scores candidate actions, but it's a 1-step lookahead. No multi-step simulation, no tree search, no "if I do X, then Y happens, then Z becomes possible."

### Solution: Planning as Inference on the Belief Graph

#### Foundation: EFE Planning = Variational Inference
**Source:** Nuijten et al. -- arXiv:2504.14898 (THE key paper)

Planning via EFE minimization is mathematically equivalent to minimizing variational free energy on a factor graph augmented with:
- **Preference priors** (from Telos goals -- desired future states)
- **Epistemic priors** (from belief uncertainty -- where information gain is highest)

This means planning falls out of the same framework Memoria already uses for belief updating. No separate planning module needed -- just extend the factor graph.

**Current state:** `compute_expected_free_energy()` in `core/free_energy.py` already computes pragmatic (goal alignment via cosine sim × priority), epistemic (Power Spherical entropy of active beliefs), and risk (prediction-observation disagreement) with learned weights `efe_epistemic_weight` and `efe_risk_weight` (softplus-constrained nn.Parameters in MetaParams). But this is a **global scalar** — it doesn't plan sequences of actions. The extension needed: evaluate EFE for *hypothetical future belief states* produced by rollout through causal edges.

#### Implementation: Message Passing for Planning
**Source:** Nuijten et al. -- arXiv:2508.02197

**Current state:** `FactorGraphMessagePassing` in `core/message_passing.py` uses implicit fixed-point solving (DEQ with Anderson acceleration, TorchDEQ) for precision-weighted BP on the factor graph. Spectral norm on `relation_transform` guarantees convergence. CoED edge directions and `apply_belief_shift()` handle directional message flow. Currently used only for belief convergence in dream phase (`sleep.py:run_dream_phase`), not planning.

Extend `FactorGraphMessagePassing` to carry epistemic value alongside precision-weighted beliefs:
1. Add Telos goals as preference factor nodes in the relation graph
2. On "future" segments (planned actions), subtract mutual information term from VFE -> yields EFE
3. Run standard message passing -- beliefs converge to plans that minimize EFE
4. The converged belief state IS the plan

#### Multi-Step Rollout via Causal Graph Dynamics
Use the causal relation graph as a learned dynamics model:
```
current_beliefs -> propagate through causal edges -> predicted future beliefs
```

Score each rollout by:
- **Pragmatic value**: cosine distance from predicted beliefs to Telos goal embeddings
- **Epistemic value**: entropy reduction in predicted belief precision
- **Risk**: prediction error magnitude

#### Hierarchical Planning via Telos Decomposition
**Source:** Dynamic Planning in Hierarchical Active Inference -- arXiv:2402.11658

Map Telos hierarchy to temporal planning hierarchy:
- **Top-level Telos** (depth 0): plan at coarse timescale (100s of steps)
- **Sub-teloi** (depth 1-2): plan at medium timescale (10s of steps)
- **Leaf teloi** (depth 3): plan at action timescale (1-step)

Top-down messages carry goal constraints. Bottom-up messages carry state estimates. Each level runs message passing at its own temporal resolution.

#### MCTS over EFE at Decision Points
**Source:** Deep AIF for Long Horizons -- arXiv:2505.19867

For critical decisions (high uncertainty, multiple competing goals):
1. Break planning horizon into segments
2. At segment boundaries: MCTS with EFE evaluation at leaf nodes
3. Within segments: gradient-based EFE minimization (single gradient step replaces exhaustive search)
4. Select the segment sequence that minimizes total EFE

#### Amortized Planning (Performance)
**Source:** Amortized Planning with Transformers -- arXiv:2402.04494

For routine decisions: train a small network to predict the result of multi-step message passing. Amortizes O(N * iterations) message passing into O(1) forward pass. Reserve full MCTS for high-uncertainty decisions.

#### Graph Distance as Fast Surprise Proxy
**Source:** arXiv:2512.01878

Shortest-path distance in the causal graph approximates surprise. For fast approximate planning, use graph distance instead of full free energy computation. Only escalate to full FE when graph distance exceeds threshold.

### Key Papers
| Paper | arXiv | Relevance |
|-------|-------|-----------|
| EFE Planning as Variational Inference | 2504.14898 | Theoretical foundation |
| Message Passing EFE Minimization | 2508.02197 | Implementation blueprint |
| Reframing EFE: Four Formulations | 2402.14460 | Mathematical toolkit |
| Deep AIF for Long Horizons | 2505.19867 | MCTS + segmented EFE |
| Dynamic Hierarchical AIF | 2402.11658 | Temporal hierarchy |
| Amortized Planning with Transformers | 2402.04494 | O(1) planning |
| Graph Distance as Surprise | 2512.01878 | Fast proxy |
| Searchformer | 2402.14083 | Search dynamics as tokens |
| Monte Carlo Tree Diffusion | 2502.07202 | MCTS + diffusion hybrid |
| Multi-Token Prediction | 2404.19737 | Forces internal planning |
| Dreamer-v3 | 2301.04104 | RSSM world model |
| CompACT: Planning in 8 Tokens | 2603.05438 | Compressed planning states |

---

## 3. Belief Robustness

### Problem
The cognitive state could drift, accumulate errors, develop false beliefs that reinforce each other. Repeated retrieval of a false belief increases its reinforcement count -> increases precision -> makes it harder to correct. Runaway positive feedback.

### Solution: Five Defense Layers

#### Layer 1: MESU -- Uncertainty-Scaled Learning Rates (Highest Priority) -- IMPLEMENTED

**Source:** Bayesian Metaplasticity from Synaptic Uncertainty -- arXiv:2312.10153 (Nature Communications 2025)
**Additional source:** Palimpsa (arXiv:2602.09075) -- MESU applied to attention states (closest existing work to Memoria's belief store)

**The single most relevant algorithm.** Each belief slot now has a precision *variance* (`belief_precision_var`) alongside its radius:
- High variance = uncertain = high learning rate (belief can shift easily)
- Low variance = confident = low learning rate (belief resists change)
- **Windowed posterior**: `belief_reinforcement_count` caps how much variance can shrink. Beyond the learned window size, the variance floor rises.

**Implementation (2026-04-04):**

In `cognition/surprise.py` — MESU-modulated Kalman gain:
```python
# Base gain (standard Kalman)
gain_raw = obs_radii / (existing_radii + obs_radii)
# MESU modulation: high variance amplifies gain
precision_var = state.belief_precision_var[matched_slots]
gain_boost = state.meta_params.mesu_gain_boost  # learned, (0, inf)
mesu_factor = (1.0 + precision_var * gain_boost).clamp(max=3.0)
gain = (gain_raw * mesu_factor).clamp(max=1.0)
```

In `cognition/pass2.py` — variance update per observation:
```python
new_var = var * (1.0 - gain^2 * shrink_rate)  # shrink_rate is learned
# Windowed posterior: if reinforcement_count > window_size, floor rises
effective_floor = min_var * (1.0 + max(0, count - window) / window)
belief_precision_var[idx] = max(new_var, effective_floor)
```

In `cognition/consolidation.py` — merge variances via harmonic mean:
```python
combined_var = (var_i * var_j) / (var_i + var_j)  # two estimates reduce uncertainty
```

All parameters learned (MetaParams): `mesu_min_variance`, `mesu_variance_shrink`, `mesu_window_size`, `mesu_gain_boost`.
Tests: `tests/test_a2_mesu.py` (7 tests).

#### Layer 2: Causal Cascade Revision (AGM-Darwiche-Pearl) -- IMPLEMENTED

**Source:** Machine Learning as Iterated Belief Change -- arXiv:2506.13157
**Additional sources:** Reactive message passing (arXiv:2603.20927), BP functoriality (arXiv:2503.15705)

When a belief is revised (contradicted by high-precision evidence), propagate precision decay through downstream beliefs in the causal graph.

**Implementation (2026-04-04) — `cognition/cascade_revision.py`:**

BFS-based cascade from each reconsolidated belief through causal edges:
1. Identify all beliefs reachable from the revised belief via causal edges (BFS frontier)
2. For each downstream belief at depth d: precision *= (1 - decay_factor^d), variance += variance_boost * decay_factor^d
3. Visited set prevents cycles. Immutable beliefs are skipped. Max depth is learned.

Two effects per downstream belief:
- **Precision decay**: radius shrinks (downstream becomes less confident)
- **Variance increase**: MESU variance grows (downstream becomes more plastic, ready to re-learn)

Triggered automatically in pass2 when any belief has `should_reconsolidate=True` from surprise computation.

All parameters learned (MetaParams): `cascade_decay_factor`, `cascade_max_depth`, `cascade_variance_boost`.
Tests: `tests/test_a3_cascade.py` (7 tests).

#### Layer 3: Two-Factor Consolidation During Sleep
**Source:** Two-Factor Synaptic Consolidation -- PNAS 2025

**Current state:** `SleepGate` in `cognition/sleep.py` is a 3-action gate (strengthen/maintain/forget) with learned MLP (`belief_dim+6 → 128 → 64 → 3`). It scores each belief using metadata (radius, access_count, level, age, n_edges, mean_edge_weight) and applies multiplicative scaling: strengthen_factor (1.0-1.2) or forget_factor (0.5-1.0). Beliefs forgotten below `hard_cleanup_precision_threshold` get deallocated. Dream phase runs message passing via DEQ (Anderson acceleration) to find BP fixed point, then shifts beliefs toward converged messages.

**Missing:** no homeostatic total-precision normalization, no cross-temporal replay, no conflict-aware scanning.

Treat radius and angle as two multiplicative factors. During sleep cycle:
1. **Strengthen**: beliefs with diverse evidence sources get radius boost
2. **Prune**: beliefs with radius < threshold after N sleep cycles get evicted
3. **Homeostatic scaling**: normalize total precision budget so it doesn't inflate unboundedly
4. **Interleaved replay** (from SCoRe, 2025): mix recent high-surprise beliefs with randomly sampled old high-precision beliefs. Run message passing between them to catch cross-temporal contradictions.

#### Layer 4: Self-Verification Pass
**Source:** InternalInspector (arXiv:2406.12053, EMNLP 2024) + SleepGate (arXiv:2603.14517)

During consolidation, run a consistency check:
1. For each high-precision belief, use causal graph to identify beliefs it should predict
2. Run message passing from the belief through edges
3. If predicted beliefs diverge from stored beliefs by > threshold: **inconsistency detected**
4. Reduce precision of the weakest link in the chain
5. **Conflict-aware supersession**: compare recent beliefs against old ones via angular cosine. Beliefs with similarity > 0.85 that carry contradictory content trigger supersession tagging.

#### Layer 5: Empirical Precision Recalibration
**Source:** Epistemic Uncertainty Collapse (arXiv:2409.02628) + calibration literature

Track prediction accuracy per belief:
- Each time a belief's prediction is confirmed: +1 to confirmed_count
- Each time contradicted: +1 to contradicted_count
- **Empirical precision** = confirmed / (confirmed + contradicted)
- If stored radius >> empirical precision: decay radius toward empirical
- Run this recalibration during sleep cycle

This prevents the #1 failure mode: a belief that *feels* confident but is *actually* wrong.

### Key Papers
| Paper | arXiv | Relevance |
|-------|-------|-----------|
| MESU: Bayesian Metaplasticity | 2312.10153 | Uncertainty-scaled learning rates |
| Iterated Belief Change (Darwiche-Pearl) | 2506.13157 | Cascade revision |
| Two-Factor Synaptic Consolidation | PNAS 2025 | Multiplicative radius/angle updates |
| SleepGate | 2603.14517 | Conflict detection, retention scoring |
| Interleaved Replay (SCoRe) | 2025 | Cross-temporal contradiction detection |
| InternalInspector I2 | 2406.12053 | Internal state verification |
| Epistemic Uncertainty Collapse | 2409.02628 | Calibration in large models |
| Bayesian Predictive Coding | 2503.24016 | Conjugate prior belief updates |
| Factor Graph BP (ICML 2024) | 2311.14649 | Gaussian BP for continual learning |
| AGM-style Belief Change for ML | 2025 | Formal revision theory |
| Stability Through Representational Drift | PNAS 2025 | Drift finds robust configs |
| KG Inconsistency Detection | 2502.19023 | Axiom-checking passes |

---

## 4. Autonomous Agency

### Problem
The system is a passive responder. It processes input, updates state, produces output. It can't decide to go gather information, can't run between conversations, can't pursue goals autonomously.

### Solution: The Daemon Loop + EFE-Driven Action Selection

#### The Daemon Loop
```
loop {
    // PERCEIVE
    input = await_event()  // user message, tool result, timer, or curiosity trigger
    perception = forward_pass(input, cognitive_state)
    
    // UPDATE BELIEFS
    cognitive_state = pass2_update(perception, cognitive_state)
    
    // PLAN
    goals = telos_evaluate(cognitive_state)  // rank by EFE
    if needs_deep_planning(goals):
        plan = mcts_plan(cognitive_state, goals, depth=N)
    else:
        plan = greedy_efe(cognitive_state, goals)
    
    // ACT
    action = plan.best_action  // could be: respond, tool_call, search, wait, explore
    result = execute(action)
    
    // OBSERVE
    prediction_error = compare(result, cognitive_state.predictions)
    if prediction_error > anomaly_threshold:
        cognitive_state = emergency_reconsolidate(prediction_error)
        continue
    
    // LEARN
    maybe_crystallize_skill(action, result, cognitive_state)
    
    // EXPLORE (if no pressing goals)
    if max_epistemic_value(cognitive_state) < curiosity_threshold:
        telos_generate_exploration_goals(cognitive_state)
    
    // SLEEP (periodic)
    if should_sleep(cognitive_state):
        cognitive_state = sleep_consolidation(cognitive_state)
}
```

#### EFE-Driven Action Selection
**Source:** CDE (arXiv:2509.09675), IMAGINE (arXiv:2505.17621)

Treat every possible action (respond, search, tool_call, wait, explore) as a candidate in the POMDP. Score each by Expected Free Energy:
- **Pragmatic value**: does this action advance active Telos goals?
- **Epistemic value**: does this action reduce uncertainty in beliefs?
- **Risk**: what's the prediction error if this action fails?

The agent naturally:
- Gathers information when uncertain (epistemic dominates)
- Executes plans when confident (pragmatic dominates)
- Avoids risky actions when cognitive state is fragile (risk dominates)
- Explores when no goals are pressing (curiosity = high epistemic value, low pragmatic)

#### Curiosity as Intrinsic Motivation
**Sources:** CDE (arXiv:2509.09675), CD-RLHF (arXiv:2501.11463)

Two curiosity signals, both derived from existing Memoria infrastructure:
1. **Actor-side**: perplexity over generated output = novel territory = explore
2. **Critic-side**: variance of EFE estimates across candidate actions = uncertain about what to do = investigate

When both signals are low: the system is confident and productive -> exploit.
When either is high: the system is uncertain -> explore (generate Telos exploration goals).

#### Autonomous Information Gathering
**Source:** ALAS (arXiv:2508.15805), Deep Active Learning Survey (arXiv:2405.00334)

When the daemon loop identifies high-uncertainty belief regions:
1. Generate a Telos goal: "reduce uncertainty about X"
2. EFE action selection naturally picks information-gathering actions (search, tool use)
3. New information flows through tell() -> belief update -> precision increases
4. Goal completes when uncertainty drops below threshold

No special "information gathering module" -- it emerges from EFE minimization.

#### Skill Crystallization + Transfer
**Sources:** SkillRL (arXiv:2602.08234), AutoSkill (arXiv:2603.01145), DUSDi (arXiv:2410.11251)

When the daemon loop detects recurring successful action patterns:
1. **Detect**: 3+ similar episodes with positive outcomes
2. **Extract**: identify the common action sequence
3. **Disentangle** (DUSDi): ensure each skill component affects one state factor
4. **Store**: crystallize as tensor pattern in cognitive state
5. **Compose**: skills combine via vector operations (not string concatenation)
6. **Evolve**: free energy gradient on success/failure refines skill parameters
7. **Transfer**: canonical tensor format enables sharing between Memoria instances

#### Tool Use via MCP
Memoria doesn't reinvent tool connectivity. It uses MCP (Model Context Protocol) as the interface layer. Tools are actions in the POMDP, selected by EFE like any other action. New tools discovered via MCP get added to the action space dynamically. Over time, the system learns which tools reduce free energy in which contexts.

### Key Papers
| Paper | arXiv | Relevance |
|-------|-------|-----------|
| CDE: Curiosity-Driven Exploration | 2509.09675 | Actor+critic curiosity |
| IMAGINE: Intrinsic Motivation | 2505.17621 | Error-conditioned exploration |
| CD-RLHF | 2501.11463 | Curiosity in alignment |
| ALAS: Autonomous Learning Agent | 2508.15805 | Self-directed curriculum |
| ELL: Lifelong Learning Framework | 2508.19005 | Experience-driven learning |
| Self-Evolving Agents Survey | 2507.21046 | Non-parametric updates validated |
| SkillRL | 2602.08234 | Dynamic co-evolving skills |
| AutoSkill | 2603.01145 | Skill self-evolution |
| DUSDi: Disentangled Skills | 2410.11251 | Composable skill components |
| Deep AIF Long-Horizon | 2505.19867 | MCTS + EFE for planning |
| MASC: Self-Correction | 2510.14319 | Metacognitive error detection |
| MAP: Modular Planner | Nature Comms 2025 | Brain-inspired agent modules |
| From Pixels to Planning (Friston) | 2407.20292 | Scale-free active inference |
| Autonomous Deep Agent | 2502.07056 | HTDAG + autonomous tool creation |

---

## 5. Implementation Priority

### Phase A: Foundations (Implement First)
These are prerequisites for everything else.

| # | What | Why | Effort | Status |
|---|------|-----|--------|--------|
| A1 | **Tentative Belief Mode** | Enables internal autoresearch loop. Add `provisional` flag to belief slots. | Small | **DONE** (2026-04-04) `cognition/provisional.py`, 9 tests |
| A2 | **MESU precision variance** | Prevents runaway false beliefs. Extend radius to (mean, variance). | Medium | **DONE** (2026-04-04) `cognition/surprise.py` + `core/state.py`, 7 tests |
| A3 | **Causal cascade revision** | When a belief changes, downstream beliefs must update. | Small | **DONE** (2026-04-04) `cognition/cascade_revision.py`, 7 tests |
| A0 | **Internal Autoresearch Loop** | Cross-cutting: goal-directed hypothesis generation + evaluation cycle. | Medium | **DONE** (2026-04-04) `cognition/autoresearch.py`, 9 tests |
| A4 | **SGM safety gate** | Required before any self-modification. Statistical confidence tests. | Medium | NOT STARTED — prerequisite for Phase C only |

### Phase B: Planning (Unlocks Agency)
| # | What | Why | Effort |
|---|------|-----|--------|
| B1 | **Augment factor graph with preference/epistemic priors** | Planning = VFE minimization on augmented graph | Medium |
| B2 | **Multi-step rollout via causal edges** | Simulate future belief states | Medium |
| B3 | **MCTS at decision points** | Deep planning when uncertain | Large |
| B4 | **Hierarchical Telos planning** | Coarse-to-fine temporal decomposition | Medium |

### Phase C: Self-Improvement (Unlocks Recursion)
| # | What | Why | Effort |
|---|------|-----|--------|
| C1 | **SRWM meta region** | Meta region produces update rules, not just thresholds | Medium |
| C2 | **Meta-learned Pass 2 update function** | System discovers its own belief update algorithm | Large |
| C3 | **Structural plasticity** (SMGrNN/FPE) | Split polysemantic beliefs, grow capacity | Medium |
| C4 | **Learned recursion depth** (MoR) | Replace fixed 2-pass with learned N-pass | Medium |

### Phase D: Full Agency (Unlocks Autonomy)
| # | What | Why | Effort |
|---|------|-----|--------|
| D1 | **Daemon loop** | Persistent process with event-driven perception | Medium |
| D2 | **EFE action selection** over tool/search/respond/wait | Natural action selection | Medium |
| D3 | **Curiosity-driven Telos generation** | Self-directed exploration | Small |
| D4 | **Skill crystallization + disentanglement** | Composable procedural memory | Large |

### Phase E: Robustness Hardening (Ongoing)
| # | What | Why | Effort |
|---|------|-----|--------|
| E1 | **Two-factor sleep consolidation** | Homeostatic precision + pruning | Medium |
| E2 | **Self-verification pass** | Catch internal contradictions | Medium |
| E3 | **Empirical precision recalibration** | Prevent confident-but-wrong | Small |
| E4 | **Interleaved replay** | Cross-temporal contradiction detection | Medium |

---

## 6. The Convergence Thesis

The literature (2024-2026) converges on eight findings that validate Memoria's direction:

1. **Persistent agents outperform session-based ones** across every benchmark (Self-Evolving Agents Survey)
2. **MCTS over EFE is the right planning algorithm** -- multiple independent groups converge on this
3. **Curiosity signals during training produce measurable gains** (CDE, IMAGINE, CD-RLHF)
4. **Skills must co-evolve with the agent**, not be static libraries (SkillRL, AutoSkill)
5. **Non-parametric updates (cognitive state, not weight fine-tuning) are the future** (Self-Evolving Agents Survey explicitly)
6. **Multi-timescale world models are key** (VERSES/Friston, Dreamer, MTRSSM)
7. **Self-correction via prediction error is natural** under active inference (MASC, MAP)
8. **The autoresearch loop works** and is "shockingly effective" even with crude keep/discard (Karpathy)

**No existing system combines all of these.** They have either persistent state OR intrinsic motivation OR skill libraries OR active inference -- never the full stack. Memoria's architecture is positioned to be the first system that unifies all eight under a single variational free energy objective.

The path from "Memoria as a model" to "Memoria as an autonomous self-improving agent":
1. Internal Autoresearch Loop (tentative beliefs + FE evaluation)
2. Planning as inference (extend factor graph with preference/epistemic priors)
3. Bounded recursive self-improvement (SRWM meta region + SGM safety gate)
4. Daemon loop with EFE action selection
5. Robustness hardening (MESU + cascade revision + self-verification)

Each step builds on the last. Each is implementable with the existing architecture. None requires a fundamental redesign.

---

## Full Paper Index

### Recursive Self-Improvement
- Self-Referential Weight Matrix (arXiv:2202.05780)
- ACL: Metalearning CL Algorithms (arXiv:2312.00276)
- Statistical Godel Machine (arXiv:2510.10232)
- SMGrNN (arXiv:2512.12713)
- FPE: Expand Neurons Not Parameters (arXiv:2510.04500)
- DynMoE (arXiv:2405.14297)
- DiscoPOP (arXiv:2406.08414)
- Godel Agent (arXiv:2410.04444)
- Darwin Godel Machine (arXiv:2505.22954)
- OPEN: Learned Optimization (arXiv:2407.07082)
- Mixture-of-Recursions (arXiv:2507.10524)
- Meta-Learning Biologically Plausible Plasticity Rules (Nature Comms 2023)
- Mesa-Optimization in Transformers (arXiv:2405.16845)
- REFINE: Reinforced Fast Weights (arXiv:2602.16704)
- Sparse Growing Transformer (arXiv:2603.23998)
- SensLI: Sensitivity-Based Layer Insertion (arXiv:2311.15995)

### Long-Horizon Planning
- EFE Planning as Variational Inference (arXiv:2504.14898)
- Message Passing EFE Minimization (arXiv:2508.02197)
- Reframing EFE (arXiv:2402.14460)
- Deep AIF for Long Horizons (arXiv:2505.19867)
- Dynamic Hierarchical AIF (arXiv:2402.11658)
- Amortized Planning with Transformers (arXiv:2402.04494)
- Graph Distance as Surprise (arXiv:2512.01878)
- Searchformer (arXiv:2402.14083)
- Monte Carlo Tree Diffusion (arXiv:2502.07202)
- Multi-Token Prediction (arXiv:2404.19737)
- Dreamer-v3 (Nature 2025)
- CompACT (arXiv:2603.05438)
- UniZero (arXiv:2406.10667)
- RAP: Reasoning via Planning (arXiv:2305.14992)
- Semformer (arXiv:2409.11143)
- Bethe FE Deformation Invariance (arXiv:2510.05380)

### Belief Robustness
- MESU: Bayesian Metaplasticity (arXiv:2312.10153)
- Iterated Belief Change (arXiv:2506.13157)
- Two-Factor Synaptic Consolidation (PNAS 2025)
- SleepGate (arXiv:2603.14517)
- Interleaved Replay / SCoRe (2025)
- InternalInspector I2 (arXiv:2406.12053)
- Epistemic Uncertainty Collapse (arXiv:2409.02628)
- Bayesian Predictive Coding (arXiv:2503.24016)
- Factor Graph BP (arXiv:2311.14649)
- AGM-style Belief Change (IJAR 2025)
- Stability via Representational Drift (PNAS 2025)
- KG Inconsistency Survey (arXiv:2502.19023)
- EWC for KG Continual Learning (arXiv:2512.01890)
- Sleep Replay with Optimal Stopping (AAAI 2025)
- Language Models Need Sleep (OpenReview)

### Autonomous Agency
- CDE: Curiosity-Driven Exploration (arXiv:2509.09675)
- IMAGINE: Intrinsic Motivation (arXiv:2505.17621)
- CD-RLHF (arXiv:2501.11463)
- ALAS: Autonomous Learning Agent (arXiv:2508.15805)
- ELL: Lifelong Learning Framework (arXiv:2508.19005)
- Self-Evolving Agents Survey (arXiv:2507.21046)
- SkillRL (arXiv:2602.08234)
- AutoSkill (arXiv:2603.01145)
- DUSDi (arXiv:2410.11251)
- MAP: Modular Planner (Nature Comms 2025)
- MASC: Self-Correction (arXiv:2510.14319)
- From Pixels to Planning / Friston (arXiv:2407.20292)
- Autonomous Deep Agent (arXiv:2502.07056)
- AgentRL (arXiv:2510.04206)
- AFlow (arXiv:2410.10762)
- Voyager (arXiv:2305.16291)

### Autoresearch
- Karpathy autoresearch (github.com/karpathy/autoresearch)
- AI Scientist v2 (Sakana AI, Nature)
- Nested Learning (NeurIPS 2025)
