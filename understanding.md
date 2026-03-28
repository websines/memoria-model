# Understanding — Memoria → Next Architecture

## March 28, 2026

### What Memoria IS (established)

Memoria is a research prototype written in Rust (~31k lines) that proved out the principles of a self-evolving memory runtime for AI agents. Key validated ideas:

- **Factor graph computation**: The `aif/` module implements Bethe free energy over factor graphs with precision-weighted message passing. This is the core computational model.
- **Free energy as unified objective**: Every operation — storage, retrieval, forgetting, consolidation — minimizes the same variational free energy. No ad-hoc heuristics.
- **Precision weighting**: All signals converted to precision-weighted messages, fused optimally. Provenance determines precision (direct=1.0 → inferred=0.3). Kalman-like gain for belief updates.
- **Surprise-driven dynamics**: No fixed timers. Reconsolidation, causal attribution, reflection — all triggered by surprise accumulation crossing adaptive thresholds.
- **Beta (exploration/exploitation)**: Computed from the ratio of entropy to energy in current beliefs. NOT a hyperparameter. Governs everything — consolidation aggressiveness, goal generation, skill selection, meta-learning step sizes.
- **Telos goal system**: Full lifecycle goals with intrinsic generation from surprise hotspots, attention scoring, deadline enforcement, stall detection, decomposition. Goals are gated by beta and free energy — no hardcoded thresholds.
- **Causal reasoning**: Structural causal graphs with Bayesian edge accumulation, Bayes-Ball d-separation, Pearl's do-calculus.
- **Meta-learning**: SPSA self-tuning of hyperparameters via free energy minimization.
- **Hebbian associations**: Co-accessed representations strengthen links (saturating Hebb's rule).

CozoDB was the substrate (relational + graph + vector), but it was the testbed, not the point.

### What Memoria is NOT

Memoria is not a product to ship. It is not infrastructure to bolt onto existing LLMs. It is not a RAG system. It is not a database to query.

Memoria is **research that validated a set of principles**. The principles are the output, not the code.

### The actual thesis (emerged during conversation)

**The next stage post-LLMs is a model with persistent mutable internal state, governed by Active Inference principles, where the factor graph IS a native layer inside the transformer.**

Key realizations in order:

1. **Current LLMs are fundamentally limited** — stateless, no goals, no real learning at inference, no causal reasoning, no calibrated uncertainty. These are architectural gaps, not scale gaps. More parameters/data/context won't fix them.

2. **Memoria+Telos baked INTO the model** would be the next jump. Not as external scaffolding but as native architecture. A model that maintains beliefs, generates goals from surprise, reasons causally, and updates itself — all internally.

3. **Self-modification of an active region** — not the entire model, but a designated mutable section. Frozen core (stable, pretrained) + active state (mutable, self-modifying). Like biological brains: brainstem is stable, neocortex is plastic.

4. **The 2-pass architecture**:
   - Pass 1 (inference): Input flows through frozen layers + mutable state → output
   - Pass 2 (adaptation): Free energy computed → gradients update ONLY the mutable state

5. **No external database** — this is critical. An LLM doesn't have a grammar database but learned grammar. Similarly, this model doesn't have a Memoria database — it learns to be its own Memoria. If you keep an external store, you've just rebuilt today's RAG pattern with extra steps.

6. **Factor graphs as the internal representation** — this is the key architectural insight. The mutable state IS a factor graph encoded as tensors:
   - Variable nodes = beliefs (with uncertainty)
   - Factor nodes = relationships/constraints
   - Edges = causal structure (sparse, directed)
   - Message passing = the forward computation through the layer
   - Free energy = the native loss function (Bethe free energy, defined directly on factor graphs)
   - This preserves causal reasoning BY CONSTRUCTION (d-separation, do-calculus are graph properties)

7. **Memoria already has the factor graph implementation** — the `aif/` module IS the working proof. The math is validated. The path forward is translating those operations from "Rust code on CozoDB records" to "differentiable tensor operations as a transformer layer."

### The architecture (current understanding)

```
Standard transformer with:
  - Frozen layers (pretrained, language/reasoning)
  - FactorGraphState layers (interleaved, mutable, persistent)
    - Variable beliefs tensor  [N_vars x D]  — mutable
    - Factor potentials tensor [N_factors x K] — mutable
    - Edge structure           [sparse]        — mutable
    - Forward = message passing (differentiable)
    - Loss = Bethe free energy (native to the structure)
    - Backward = update mutable state only

Training:
  - L_token: next-token prediction (teaches language)
  - L_fe: free energy minimization over factor graph state (teaches belief calibration, surprise sensitivity, causal structure, goal-directed behavior)

Everything else (precision weighting, Hebbian association, Telos-like goals, consolidation, forgetting) EMERGES from training under these objectives, not engineered.
```

### What Memoria's role becomes

Memoria was the experiment. The `aif/` module's factor graph math becomes the blueprint for the `FactorGraphStateLayer`. The principles (free energy, precision weighting, surprise, beta, Telos) become the training objective and architectural bias. The Rust code is the reference implementation that proves the algorithms work before porting to differentiable tensor ops.

### Open questions

- Dynamic graph size — graphs need to grow/shrink. Variable-size tensors are hardware-unfriendly. Hierarchical subgraph summarization?
- Training curriculum — need data where causal structure matters and changes over time. Standard text corpora may not provide enough signal.
- Capacity — how many variable nodes before message passing per layer becomes too expensive?
- Stability — self-modifying state over thousands of interactions without drift or collapse.
- Interpretability — can we inspect the factor graph to understand what the model believes?
- How does context window interact with the factor graph state? Context = working memory, factor graph = long-term? Or does the factor graph reshape how context is processed?

### Comparison: autoresearch (Karpathy)

Looked at karpathy/autoresearch (59k stars). Completely opposite philosophy: minimal infrastructure, trust the foundation model, one file to edit, one metric, autonomous experimentation loop. Works for narrow, single-metric optimization. But for open-ended tasks needing persistent knowledge, causal reasoning, and self-directed goals — that's where the Memoria-derived architecture would dominate.

Autoresearch is an agent modifying code. The vision here is a model modifying itself.

### Critical connection: Mamba's selective scan IS factor graph message passing

The parallel scan in Mamba computes `h_t = A_t * h_{t-1} + B_t * x_t`. This is mathematically equivalent to belief propagation on a linear chain factor graph:
- Each timestep t = variable node with state h_t
- Each transition (t-1, t) = factor node encoding the recurrence
- The parallel scan's upward/downward sweep = forward-backward message passing

Kalman filtering IS message passing on a linear-Gaussian factor graph. So SSM recurrence is already factor graph inference — just on a trivial (linear chain) graph topology.

**Implication**: We're not proposing something alien to the field. We're proposing to generalize what SSMs already do (linear chain factor graph) to arbitrary graph topologies with explicit causal structure and precision weighting. The math is the same family. The generalization is adding structure.

---

## Prior Art & Related Work (compiled March 28, 2026)

Everything below is organized by relevance to the thesis: **a transformer with mutable factor graph state layers, trained with free energy minimization, where Memoria's principles emerge internally.**

---

### TIER 1: Directly aligned — mutable internal state at inference time

#### Google Titans — "Learning to Memorize at Test Time" (2025)
- **Authors**: Ali Behrouz, Peilin Zhong (Google Research)
- **Paper**: [arxiv.org/abs/2501.00663](https://arxiv.org/abs/2501.00663)
- **Blog**: [research.google/blog/titans-miras-helping-ai-have-long-term-memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- **What it does**: Neural long-term memory module where the memory IS a deep neural network. Uses **surprise** (gradient magnitude) to decide what to memorize. Adaptive forget gate manages capacity. Persists across sequence boundaries.
- **Key insight**: "Momentary surprise" (current input) + "past surprise" (recent context flow) together determine what gets stored permanently. This is very close to Memoria's surprise-driven dynamics.
- **Relevance**: **HIGHEST**. Titans proves that surprise-driven mutable internal state works at scale. The gap: Titans uses raw gradient surprise. Memoria uses precision-weighted free energy — a richer, more principled signal. The thesis is essentially "Titans but with Memoria's factor graph math instead of raw gradients."
- **Performance**: Outperforms GPT-4 on BABILong, scales to 2M+ token contexts, fewer parameters.

#### MIRAS Framework — Unifying sequence models as associative memory (2025)
- **Authors**: Google Research (same team as Titans)
- **Paper**: Companion to Titans
- **Blog**: [research.google/blog/titans-miras-helping-ai-have-long-term-memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- **What it does**: Theoretical framework proving that most modern sequence models (Transformers, Mamba, RetNet, DeltaNet) are all doing the same thing: **online optimization over associative memory**. Unifies the landscape.
- **Relevance**: MIRAS says the field is converging on "models that update internal memory during inference." The question is what objective governs the update. Memoria's answer: free energy minimization over factor graphs.

#### TTT Layers — "Learning to (Learn at Test Time)" (2024)
- **Authors**: Yu Sun et al. (Stanford)
- **Paper**: [arxiv.org/abs/2407.04620](https://arxiv.org/abs/2407.04620)
- **Code**: [github.com/test-time-training/ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax)
- **What it does**: Hidden state IS a model (linear or MLP). Update rule = gradient descent on self-supervised loss during forward pass. Two variants: TTT-Linear (hidden state = linear model), TTT-MLP (hidden state = 2-layer MLP).
- **Key insight**: Unlike Mamba, TTT can keep reducing perplexity with more context (>16k tokens). The hidden state has unbounded expressiveness because it's a model, not a fixed-size vector.
- **Relevance**: **HIGH**. TTT proves differentiable self-modification during forward pass works. Limitation: uses self-supervised reconstruction loss, resets per sequence, no causal structure, no goals. The factor graph layer would be TTT with structure — same principle (gradient update on internal state) but with a graph topology that preserves causal reasoning.

#### Google Hope + Nested Learning (NeurIPS 2025)
- **Authors**: Ali Behrouz et al. (Google Research — same team)
- **Paper**: [arxiv.org/abs/2512.24695](https://arxiv.org/abs/2512.24695)
- **Blog**: [research.google/blog/introducing-nested-learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- **What it does**: Treats an ML model as a system of nested optimization problems. Introduces **Continuum Memory System (CMS)** — memory as a spectrum of MLP blocks updating at different frequency rates (short/medium/long range). Hope architecture: self-modifying recurrent network that treats its own weights as writable memory.
- **Key insight**: "Memory is a continuum, not a binary." Different parts of the model update at different rates. This maps directly to frozen core (slow/never) + active state (fast/every interaction).
- **Relevance**: **HIGH**. Nested Learning provides the theoretical justification for multi-rate self-modification. Hope demonstrates it works. The factor graph layer would be one more level in this continuum — updating at the rate governed by free energy.

#### Schmidhuber — Self-Referential Weight Matrix (ICML 2022)
- **Authors**: Kazuki Irie, Imanol Schlag, Robert Csordas, Jurgen Schmidhuber
- **Paper**: [arxiv.org/abs/2202.05780](https://arxiv.org/abs/2202.05780)
- **Code**: [github.com/IDSIA/modern-srwm](https://github.com/IDSIA/modern-srwm)
- **What it does**: A weight matrix that learns to modify itself using outer products and delta update rules. The matrix is both the parameters and the state — it reads its own weights, computes an update, and writes back.
- **Key insight**: True self-reference — the network's weights are simultaneously its computation substrate and its modifiable state. This is the purest form of "the model modifies itself."
- **Relevance**: **HIGH**. SRWM is the theoretical ancestor of what we're proposing. Limitation: operates on flat weight matrices, no graph structure, no principled objective like free energy. But proves the concept of self-referential modification.

---

### TIER 2: Key building blocks — factor graphs, message passing, memory architectures

#### Factor Graph Neural Networks (NeurIPS 2020, JMLR 2024)
- **Authors**: Zhen Zhang, Fan Wu, Wee Sun Lee
- **Paper**: [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/61c66a2f4e6e10dc9c16ddf9d19745d6-Paper.pdf) / [JMLR 2024](https://jmlr.org/papers/volume24/21-0434/21-0434.pdf)
- **What it does**: Proves that Max-Product Belief Propagation can be exactly parameterized by a Factor Graph Neural Network. Bridges classical probabilistic inference and neural network learning.
- **Relevance**: **CRITICAL**. This is the mathematical proof that factor graph message passing CAN be a neural network layer. The forward pass of an FGNN IS belief propagation. This validates that Memoria's `aif/` module can be ported to a differentiable layer.

#### Neural Enhanced Belief Propagation on Factor Graphs (2020)
- **Authors**: Victor Garcia Satorras, Max Welling
- **Paper**: [arxiv.org/abs/2003.01998](https://arxiv.org/abs/2003.01998)
- **What it does**: Runs a Factor Graph GNN jointly with belief propagation. The GNN receives BP messages and outputs corrected versions. Hybrid model combining learned and classical inference.
- **Relevance**: Shows how to combine neural computation with principled message passing. The correction mechanism maps to how frozen transformer layers could interact with factor graph state layers.

#### Differentiable Nonparametric Belief Propagation (2021)
- **Paper**: [arxiv.org/abs/2101.05948](https://arxiv.org/abs/2101.05948)
- **What it does**: Makes belief propagation fully differentiable with nonparametric (particle-based) beliefs. Replaces hand-crafted factors with differentiable neural networks. End-to-end learnable.
- **Relevance**: Proves factor graph inference can be end-to-end differentiable. The factor potentials are learned, not engineered — exactly what we want for the active state layers.

#### Belief Propagation Neural Networks (NeurIPS 2020)
- **Paper**: [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/07217414eb3fbe24d4e5b6cafb91ca18-Paper.pdf)
- **What it does**: Learns modifications to standard BP message passing so outputs are closer to ground truth. Retains BP's convergence properties while improving accuracy.
- **Relevance**: Shows BP can be "corrected" by learned components — relevant for how the frozen transformer layers might refine the factor graph state.

#### Differentiable Cluster Graph Neural Networks (2024)
- **Paper**: [arxiv.org/abs/2405.16185](https://arxiv.org/abs/2405.16185)
- **What it does**: Incorporates clustering bias into message passing using cluster-nodes. Closed-form optimization steps that ARE message passing on a bipartite graph.
- **Relevance**: Hierarchical graph structure — maps to Memoria's hierarchical message passing (bottom-up/top-down). Could inform how factor graph layers handle variable graph sizes.

#### Modern Hopfield Networks — "Hopfield Networks is All You Need" (2020)
- **Authors**: Hubert Ramsauer, Bernhard Schafl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, Markus Holzleitner, Sepp Hochreiter et al.
- **Paper**: [arxiv.org/abs/2008.02217](https://arxiv.org/abs/2008.02217)
- **Code**: [github.com/ml-jku/hopfield-layers](https://github.com/ml-jku/hopfield-layers)
- **What it does**: Shows that transformer attention IS the update rule of a modern Hopfield network. Exponential storage capacity. Three types of fixed points correspond to different attention behaviors.
- **Relevance**: Foundational connection — attention = associative memory retrieval. Factor graph message passing is a more structured version of the same principle. Hopfield layers store patterns; factor graph layers store structured beliefs with causal relations.

#### Gaussian Belief Propagation (interactive tutorial)
- **Resource**: [gaussianbp.github.io](https://gaussianbp.github.io/)
- **What it does**: Interactive visualization and implementation of Gaussian BP on factor graphs. Clean reference for implementing precision-weighted message passing as tensor operations.
- **Relevance**: Practical reference for porting Memoria's precision-weighted messages to differentiable tensor ops. Gaussian BP is especially relevant because precision weighting is native to Gaussian message passing.

---

### TIER 3: Persistent state architectures — different approaches, same goal

#### Mamba — Selective State Spaces (2023)
- **Authors**: Albert Gu, Tri Dao
- **Paper**: [arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
- **Code**: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- **What it does**: SSM with input-dependent parameters — selectively propagates or forgets information. Linear-time, constant-space (no KV cache). 5x throughput of transformers.
- **Limitation per MIRAS**: Mamba can't keep improving with more context past ~16k tokens (unlike TTT). Fixed-size state vector is a capacity bottleneck.
- **Relevance**: Proves selective state updates at linear cost. But the state is a flat vector — no structure, no causal reasoning. Factor graph state would be the structured equivalent.

#### Mamba-2 — State Space Duality (2024)
- **Authors**: Tri Dao, Albert Gu
- **Paper**: [arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060)
- **Blog**: [tridao.me/blog/2024/mamba2-part1-model](https://tridao.me/blog/2024/mamba2-part1-model/)
- **What it does**: Shows that SSMs and attention are dual — connected through structured matrix theory. Enables hardware-efficient implementations that match or exceed Mamba-1.
- **Relevance**: The duality insight supports the thesis that there's a unified computation underlying both attention and state-space models. Factor graph message passing may be a third perspective on the same underlying operation.

#### xLSTM — Extended LSTM (NeurIPS 2024 Spotlight)
- **Authors**: Maximilian Beck, Sepp Hochreiter et al.
- **Paper**: [arxiv.org/abs/2405.04517](https://arxiv.org/abs/2405.04517)
- **What it does**: Two variants: sLSTM (scalar memory with new mixing) and mLSTM (fully parallelizable, **matrix memory** with covariance update rule). Exponential gating for better gradient flow.
- **Key insight**: mLSTM's matrix memory with covariance updates is an associative memory — closer to a factor than a flat state vector. The covariance update rule is related to precision estimation.
- **Relevance**: mLSTM shows matrix-valued persistent state works. Covariance updates ≈ precision tracking. Could inform how factor potentials are updated.

#### RWKV — RNN with Transformer Performance (v7 "Goose", 2024)
- **Authors**: Bo Peng (BlinkDL) et al.
- **Paper**: RWKV-5/6 at COLM 2024
- **Code**: [github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- **What it does**: Pure RNN, no attention, transformer-level performance. v5 introduced matrix-valued states (64x64). Infinite context length, constant memory. Now at v7.
- **Relevance**: Proves persistent recurrent state can compete with transformers. Matrix states in RWKV-5+ show that richer internal state representations help. But no principled update objective — factor graph free energy would fill that gap.

#### Differentiable Neural Computer (Nature, 2016)
- **Authors**: Alex Graves et al. (DeepMind)
- **Paper**: Nature 538, 471-476
- **Code**: [github.com/google-deepmind/dnc](https://github.com/google-deepmind/dnc)
- **What it does**: Neural network with external differentiable memory. Read/write heads, content-based addressing, temporal linking. First serious "memory augmented neural network."
- **Relevance**: Historical ancestor. DNC's key ideas (content addressing, temporal links, allocation) map to factor graph operations (variable lookup, causal edges, node creation). But DNC memory is external; we want internal.

#### Memorizing Transformers (ICLR 2022)
- **Authors**: Yuhuai Wu, Markus Rabe et al. (Google)
- **Paper**: [arxiv.org/abs/2203.08913](https://arxiv.org/abs/2203.08913)
- **What it does**: kNN lookup into non-differentiable external (key, value) memory. Scales to 262K tokens. Simple but effective.
- **Relevance**: Shows persistent memory helps. But it's external and non-differentiable — exactly the pattern we're moving away from. Useful as a baseline to beat.

#### Fast Weights (NeurIPS 2016)
- **Authors**: Jimmy Ba, Geoffrey Hinton et al.
- **Paper**: [arxiv.org/abs/1610.06258](https://arxiv.org/abs/1610.06258)
- **Code**: [github.com/GokuMohandas/fast-weights](https://github.com/GokuMohandas/fast-weights)
- **What it does**: Introduces weights that change faster than standard weights but slower than activations. Store temporary memories of recent past. Neurally plausible attention mechanism.
- **Key insight**: Three timescales — slow weights (long-term), fast weights (short-term memory), activations (instant). This maps directly to: frozen layers (slow) + factor graph state (fast) + context (activations).
- **Relevance**: Theoretical foundation. Hinton's intuition in 2016 was exactly right. Factor graph state layers are "fast weights" with structure.

---

### TIER 4: Active Inference & Free Energy — the theoretical foundation

#### Active Inference & Free Energy Principle (Friston, ongoing)
- **Key paper**: "Generalised free energy and active inference" — [Biological Cybernetics, 2019](https://link.springer.com/article/10.1007/s00422-019-00805-w)
- **Overview**: [Neural Computation survey, 2024](https://direct.mit.edu/neco/article/36/5/963/119791/An-Overview-of-the-Free-Energy-Principle-and)
- **What it does**: Theoretical framework where all perception, learning, and action minimize variational free energy. ELBO (evidence lower bound) used in VAEs is equivalent to negative free energy.
- **Relevance**: THE theoretical foundation. Memoria implements this. The thesis is to make it architectural.

#### Canonical Neural Networks Perform Active Inference (2022)
- **Paper**: [Nature Communications Biology](https://www.nature.com/articles/s42003-021-02994-2)
- **What it does**: Shows that standard neural network operations (forward pass, backprop) can be interpreted as active inference under certain conditions.
- **Relevance**: Suggests that adding free energy as an explicit loss might not fight the architecture — it might align with what gradient-based learning is already approximately doing.

#### Experimental Validation of Free Energy Principle with In Vitro Neural Networks (2023)
- **Paper**: [Nature Communications](https://www.nature.com/articles/s41467-023-40141-z)
- **What it does**: Experimentally validates that real biological neural networks minimize free energy. Not just theory — measured in vitro.
- **Relevance**: The strongest evidence that free energy minimization is the right objective for neural computation. If biological neurons do it, artificial ones should too.

#### pymdp — Active Inference in Python
- **Code**: [github.com/infer-actively/pymdp](https://github.com/infer-actively/pymdp)
- **Paper**: [arxiv.org/abs/2201.03904](https://arxiv.org/abs/2201.03904)
- **What it does**: Active inference for discrete-state POMDPs. Factorized generative models with automatic message passing. NumPy-based, PyTorch/JAX integration planned.
- **Relevance**: Reference implementation for Active Inference math. Different domain (discrete POMDPs vs. continuous neural state) but the factor graph message passing algorithms are the same.

#### Predictive Coding Networks (2024-2025)
- **Survey**: [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S089360802501041X)
- **Benchmark**: [VERSES AI blog](https://www.verses.ai/research-blog/benchmarking-predictive-coding-networks-made-simple)
- **What it does**: Alternative to backpropagation based on free energy minimization. Bidirectional error propagation. Energy-based training.
- **Limitation**: Scalability bottleneck — works well with 5-7 layers, degrades in deep networks. Gradient explosion/vanishing.
- **Relevance**: Cautionary tale. Pure predictive coding doesn't scale yet. The hybrid approach (frozen transformer trained with backprop + factor graph layers trained with free energy) avoids this — you get the scalability of transformers AND the principled dynamics of free energy.

---

### TIER 5: Adaptive architectures

#### Liquid Neural Networks — MIT (2020-2026)
- **Authors**: Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus (MIT CSAIL)
- **Paper**: [arxiv.org/abs/2006.04439](https://arxiv.org/abs/2006.04439) (original) / [Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00556-7) (CfC)
- **Company**: Liquid AI — shipped LFM-7B (Jan 2025), LFM2.5 (Jan 2026)
- **What it does**: Continuous-time neural networks where parameters adjust based on input even after training. Uses ODE dynamics. CfC variant finds closed-form solution, eliminating ODE solver overhead.
- **Key insight**: The model's parameters are functions of its input — it literally rewires itself at inference time.
- **Relevance**: Proves adaptive inference-time computation works commercially. Liquid AI shipped real products. The continuous-time dynamics are related to Memoria's activation decay (`a(t) = exp(-(t-t_last)/τ)`). NOTE: Memoria already uses liquid neural networks for adaptive routing.

---

### Synthesis: How these all connect to the thesis

```
THE CONVERGENCE:

Titans (2025)        → Surprise-driven mutable internal state WORKS
Hope/Nested (2025)   → Multi-rate self-modification WORKS
TTT (2024)           → Gradient-based internal state update WORKS
SRWM (2022)          → True self-referential modification WORKS
FGNN (2020)          → Factor graph message passing AS a neural layer WORKS
DNBP (2021)          → Differentiable factor graphs WORK
Mamba-2 (2024)       → SSM/attention duality — unified computation exists
xLSTM (2024)         → Matrix state with covariance ≈ precision tracking
Hopfield (2020)      → Attention IS associative memory retrieval
MIRAS (2025)         → All sequence models ARE online memory optimization
FEP validation (2023)→ Biological neurons minimize free energy
Liquid NN (2020-26)  → Adaptive inference-time computation SHIPS commercially

WHAT'S MISSING (the gap):
None of these use free energy as the self-modification objective.
None of these have structured factor graph internal state.
None of these have intrinsic goal generation (Telos).
None of these have principled causal reasoning in the state.

Memoria validated all four of these in software.
The thesis: make them architectural.
```

---

### Additional papers from deep research (Tiers 1-5 supplement)

#### Infini-attention — Efficient Infinite Context Transformers (Google, 2024)
- **Authors**: Tsendsuren Munkhdalai, Manaal Faruqui, Siddharth Gopal
- **Paper**: [arxiv.org/abs/2404.07143](https://arxiv.org/abs/2404.07143)
- Compressive memory in attention using delta rule. Each attention head maintains a persistent compressive memory matrix updated with new KV pairs. Unbounded input length, bounded memory. **Directly relevant**: delta rule updates on persistent state = a simple form of belief updating.

#### GatedDeltaNet — Gated Linear Attention with Delta Rule (NVIDIA, 2024)
- **Authors**: Songlin Yang, Jan Kautz, Ali Hatamizadeh
- **Paper**: [arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464)
- Precise overwriting of specific entries in persistent KV state matrix. More fine-grained memory management than simple additive accumulation. **Relevant**: delta rule = error-correcting update, related to precision-weighted belief revision.

#### Griffin — Gated Linear Recurrences + Local Attention (Google DeepMind, 2024)
- **Authors**: Soham De, Samuel Smith, et al.
- **Paper**: [arxiv.org/abs/2402.19427](https://arxiv.org/abs/2402.19427)
- Hybrid: gated linear recurrence (RG-LRU) provides persistent memory, local attention handles fine-grained retrieval. Google's practical take on hybrid persistent-memory at scale.

#### "Transformers Learn In-Context by Gradient Descent" (2023)
- **Authors**: Johannes von Oswald et al.
- **Paper**: [arxiv.org/abs/2212.07677](https://arxiv.org/abs/2212.07677)
- **Key insight**: Transformer in-context learning IS mathematically equivalent to gradient descent on an implicit inner model. Attention layers implement one step of GD on a linear regression problem. This means: transformers are already doing implicit self-modification. Making it explicit (with factor graph structure and free energy objective) is the natural next step.

#### "Linear Transformers Are Secretly Fast Weight Programmers" (Schmidhuber group, 2021)
- **Authors**: Imanol Schlag, Kazuki Irie, Jurgen Schmidhuber
- **Paper**: [arxiv.org/abs/2102.11174](https://arxiv.org/abs/2102.11174)
- Linear attention = fast weight programmer writing to associative memory via outer products. Connects Mamba-2 SSD duality to Schmidhuber's self-modification vision. The persistent state in linear attention/SSMs IS a fast weight matrix being continuously reprogrammed.

#### Recurrent Memory Transformer (RMT) — Persistent memory tokens (2022)
- **Authors**: Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev
- **Paper**: [arxiv.org/abs/2207.06881](https://arxiv.org/abs/2207.06881)
- Special memory tokens persist across segment boundaries. Model reads/writes to them via attention. Demonstrated 1M+ token processing. Simple but effective.

#### DeltaNet — Conditional State Space Models with Selective State Transitions (2024)
- **Authors**: Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim
- **Paper**: [arxiv.org/abs/2310.18020](https://arxiv.org/abs/2310.18020)
- Error-correcting delta rule for state updates — more stable than simple accumulation. The state IS a fast weight matrix updated at each step. Connects to fast weight programmer tradition.

#### HiPPO — Recurrent Memory with Optimal Polynomial Projections (NeurIPS 2020)
- **Authors**: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Re
- **Paper**: [arxiv.org/abs/2008.07669](https://arxiv.org/abs/2008.07669)
- Theoretical foundation for ALL modern SSMs. Proves SSM states are provably optimal compressors of sequential history using polynomial projections. **Why this matters**: establishes that fixed-size recurrent state CAN faithfully represent long histories — the capacity question has a mathematical answer.

---

### Implementation-critical codebases

| Repo | What | Why it matters |
|------|------|----------------|
| [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl) | Reactive message passing on factor graphs with Bethe free energy (Julia) | Most complete implementation of real-time Active Inference on factor graphs. Reference for porting Memoria's `aif/` to differentiable tensor ops |
| [brentyi/dfgo](https://github.com/brentyi/dfgo) | Differentiable factor graph optimization in PyTorch | Factor graph solves as differentiable layers. Direct precedent for factor graph state layers |
| [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) | TTT layers in PyTorch | Reference for gradient-based internal state updates during forward pass |
| [IDSIA/modern-srwm](https://github.com/IDSIA/modern-srwm) | Self-Referential Weight Matrices | Self-modification implementation reference |
| [sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) | Efficient linear attention / GLA / DeltaNet / HGRN2 | Hardware-efficient persistent state update kernels |
| [infer-actively/pymdp](https://github.com/infer-actively/pymdp) | Active Inference for POMDPs | Factor graph message passing algorithms reference |
| [VersesTech/cavi-cmn](https://github.com/VersesTech/cavi-cmn) | VERSES AI (Friston's group) — differentiable message passing | Active Inference + neural architecture from FEP originators |
| [ml-jku/hopfield-layers](https://github.com/ml-jku/hopfield-layers) | Modern Hopfield network layers for PyTorch | Associative memory as plug-in layers |
| [NX-AI/xlstm](https://github.com/NX-AI/xlstm) | Official xLSTM | Matrix memory with covariance updates |
| [google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma) | Griffin (RG-LRU) | Google's production hybrid persistent-memory architecture |

---

### Key theoretical connections (from Active Inference deep research)

1. **Predictive coding approximates backprop** (Millidge et al., 2022) — Free energy minimization in predictive coding converges to standard backpropagation at equilibrium. This means adding free energy as a loss doesn't fight the architecture — it aligns with what gradient-based learning already approximately does.

2. **Attention IS precision weighting** (multiple groups) — Transformer attention scores function as precisions that gate message passing. This is the same operation as precision-weighted message passing in Active Inference. The architectures are already doing the same thing — we're just making it explicit and structured.

3. **Next-token prediction IS free energy minimization** (Friston, 2023) — Autoregressive LLM training can be interpreted as variational free energy minimization. The standard training objective is already a special case of what we're proposing — we're generalizing it to include structured internal state.

4. **ELBO = negative free energy** — The evidence lower bound used to train VAEs is mathematically equivalent to negative variational free energy. The deep learning field has been doing free energy minimization since VAEs — just not calling it that, and not applying it to mutable internal state.

---

## How this differs from current LLMs (the hard differentiation)

### The wall in every current LLM

Every LLM has a hard boundary:
- **Weights** (what the model IS) — frozen at training time, never change during use
- **Context** (what the model SEES) — ephemeral, gone when the window ends

Everything built today (RAG, tools, agents, memory systems) stuffs things into context. The model itself never changes. Same model on conversation 1000 as conversation 1.

### What's structurally impossible today (not hard — impossible)

1. **Principled belief revision**: Model has no beliefs — just weights. RAG retrieves contradictory facts but the model has no mechanism to resolve them beyond positional/recency heuristics.

2. **Causal interventional reasoning**: "A causes B" is a string pattern in weights, not a graph edge. Can't do do-calculus because there's no causal graph. Tools can build an external graph but the model reads a description of reasoning, it doesn't reason over the graph.

3. **Calibrated uncertainty**: No internal variable representing uncertainty. "I'm 80% confident" is pattern-matched from training distribution, not computed from belief state.

4. **Compounding competence**: Weights frozen → model can't get better through use. In-context learning vanishes when context ends. Fine-tuning requires offline human curation.

5. **Provenance-aware computation**: A fact in weights has no source. Model can't distinguish training knowledge from user-provided information in a way that affects computation (not just prompt text).

### What our model does

The factor graph state is INSIDE the computation graph. Not retrieved into context. Part of how the model thinks, not what it's told.

- **Has beliefs, not just weights**: Variable nodes with precision, provenance, update history. Belief revision happens as tensor operations in the forward pass.
- **Reasons causally, not correlationally**: Graph edges ARE causal structure. d-separation is topological. Intervening (clamping a node) produces different computation than observing.
- **Computes its own uncertainty**: β = H_var / (E_factor + H_var). Real number from actual state. Drives behavior (exploration vs. exploitation) without prompting.
- **Changes through use**: Mutable state layers mean future inputs flow through different computation paths. The model that learned "A→B→C" processes new inputs differently at the tensor level.
- **Behaviors emerge from training**: Precision weighting, surprise-driven updates, causal structure — none engineered. All consequences of L_token + L_fe. Ablation: remove L_fe → behaviors disappear.

### The one-sentence version

> Current LLMs are frozen functions that process ephemeral context. This model is a self-modifying structured belief system where the computation graph itself evolves through use, governed by a single variational objective.

### Proof-of-concept principle

Every benchmark task must be designed so that:
1. Standard Transformer CANNOT solve it (not "solves worse" — cannot)
2. Transformer + RAG CANNOT solve it (kills "just add tools" objection)
3. Flat-state model (Mamba) CANNOT solve it (proves structure matters, not just persistence)
4. Our model CAN solve it, and ablations show WHY

Causal interventional reasoning is the strongest candidate — formally provable that without causal graph structure, no correlation-based reasoning can correctly answer do-calculus queries.

---

## Real holes (March 28, 2026 — honest audit)

These aren't hypothetical risks. These are actual unsolved problems in the current vision.

### Hole 1: "Belief" in tensor space is undefined

In Memoria: belief = "Alice works at Acme, confidence 0.8, provenance: direct." Explicit, typed, inspectable.

In factor graph state layer: variable node = vector of floats. We ASSUME the model learns to use these as beliefs. But it might just use them as generic state — like Mamba uses its hidden state for whatever helps prediction. If so, calling them "beliefs" is branding, not architecture.

**Status: UNSOLVED.** Need to either:
- (a) Engineer explicit structure (each node = mean vector + precision scalar + provenance tag) and show SOME properties emerge, or
- (b) Show purely learned representations develop measurably belief-like properties (testable but risky)

Option (a) is honest. Option (b) is the stronger claim but may not work for v1.

### Hole 2: Fixed topology = just a GNN

If graph topology is fixed (same edges always), "factor graph state layer" is just a GNN layer with persistent node states and an extra loss term. That's a real contribution but calling it a "factor graph" oversells it. For genuine causal structure, edges must appear/disappear/rewire. That means dynamic topology. GPUs hate dynamic sparse ops.

**Status: UNSOLVED.** Options:
- (a) Fixed topology for v1, honest about limitations. "Structured persistent state" not "emergent causal graphs"
- (b) Fixed-capacity edge budget with learned allocation — edges are soft (continuous weight) not hard (present/absent). Soft attention over a fixed adjacency budget. GPU-friendly. Topological change = edge weights going to ~0 or ~1.
- (c) Periodic restructuring — run message passing on fixed topology, periodically (every N steps) prune/grow edges based on accumulated statistics. Amortizes the dynamic cost.

Option (b) is probably the practical answer — continuous relaxation of discrete structure.

### Hole 3: L_fe bootstrap / chicken-and-egg

At step 1 of training, graph state is random noise. Bethe free energy over random state is meaningless. L_fe provides no useful gradient. But state becomes meaningful via... L_fe gradients.

**Status: PARTIALLY SOLVED.** Likely answer:
- Early training: L_fe weight α starts near 0, L_token dominates. Model learns language first.
- α warms up on schedule. As state develops some structure from L_token backprop alone, L_fe gradually kicks in to refine it.
- Precedent: VAE training does exactly this (KL annealing). β-VAE starts with β=0 (pure reconstruction) and ramps β up.
- So: `α(t) = min(t / warmup_steps, α_max)`. Standard technique.

### Hole 4: L_fe in tensor space — the actual math

We keep saying "Bethe free energy over factor graph state" but haven't written the formula for tensors.

**Status: NEEDS DEFINITION.** Working sketch:
- Each variable node i has: mean μ_i ∈ R^d, log-precision λ_i ∈ R (learned, not fixed)
- Each factor f connecting variables (i,j) has: potential ψ_f ∈ R^k (learned)
- Energy term: E = -Σ_f log p(μ_i, μ_j | ψ_f) — how well connected beliefs agree
- Entropy term: H = Σ_i H(λ_i) — Shannon entropy from precision (high precision = low entropy = confident)
- Free energy: F = E - H
- Minimizing F forces: beliefs to be consistent (low E) while maintaining appropriate uncertainty (not overconfident without evidence)

This is basically variational message passing on a factor graph where factors are learned neural potentials. RxInfer.jl does exactly this. The math is known — just needs to be written as PyTorch tensor ops.

### Hole 5: Emergence is a risky claim

Claiming precision weighting, surprise sensitivity, causal structure, AND goal-directed behavior all emerge from L_token + L_fe is too many claims at once. If ANY of them don't emerge, the whole narrative collapses.

**Status: NEEDS SCOPING.** Realistic plan for v1:
- ENGINEER: explicit precision dimension on variable nodes (like Memoria's confidence field). Not emergent — designed.
- ENGINEER: explicit edge structure with learned weights. Not emergent — designed with learned parameters.
- SHOW EMERGES: surprise-driven update magnitudes (measure state change per token, correlate with prediction error)
- SHOW EMERGES: precision-like behavior (ablate precision dimension → performance drops on contradiction tasks)
- DEFER: goal-directed behavior (Telos). Too ambitious for v1. Save for follow-up.
- DEFER: full causal reasoning with interventions. Show correlational structure emerges; interventional reasoning is v2.

### Hole 6: "Just a GNN with persistent state + extra loss"

Skeptical reviewer framing. Need clear answer for why factor graph framing is essential.

**Status: PARTIALLY SOLVED.** The honest differentiators:
- GNN layers don't have a principled update objective for their state. We have free energy — it's not just a regularizer, it's the natural objective for belief states on factor graphs (Bethe approximation, well-understood theory).
- GNN layers don't distinguish variable nodes from factor nodes. Factor graphs have typed bipartite structure — variables hold beliefs, factors encode relationships. This isn't cosmetic — it determines what can be learned.
- GNN layers don't have precision weighting. Our variable nodes have explicit precision that gates influence — this is native to factor graph message passing, not standard GNN aggregation.
- Persistence across sequences is not standard GNN behavior.

But we should be honest: this IS a GNN variant. A specific, principled, persistent GNN with variational loss. The novelty is the combination + the specific design choices informed by Active Inference, not a fundamentally new computation primitive.

### Hole 7: Causal reasoning from text is still correlation

Model learns "A causes B" from text containing the STRING "A causes B." Translating text patterns into graph edges isn't fundamentally different from storing the pattern in weights — it's still learned from correlational data.

**Status: REAL HOLE.** Genuine causal reasoning needs interventional data. Options:
- (a) Synthetic training data with explicit interventions: "We set X=true. Y changed. Z did not." This exists in causal reasoning benchmarks.
- (b) The graph structure enables TESTING causal claims (clamping + propagation) even if the claims originate from text. A model that can test its causal beliefs is better than one that can't, even if both learn from text.
- (c) Be honest: v1 shows structural belief tracking, not full causal reasoning. Causal interventions are a demonstrated CAPABILITY of the architecture, validated on synthetic data, not a claim about learning causation from natural language.

Option (c) is the right framing for the paper.

### Summary: what v1 actually proves (honest version)

> "We introduce persistent factor graph state layers with variational free energy loss. We show:
> 1. Structured persistent state outperforms flat persistent state on belief tracking tasks
> 2. Explicit precision dimensions develop useful confidence-tracking behavior
> 3. State update magnitudes correlate with surprise (emergent, not engineered)
> 4. The architecture supports causal interventional queries when given appropriate training data
> 5. Standard language modeling performance is maintained
>
> Dynamic structure learning, emergent causal reasoning from natural language, and goal-directed behavior remain open problems that the architecture is designed to support in future work."

This is honest, publishable, and leaves room for the bigger vision without overclaiming.

---

## Breaking the loop (March 28, 2026 — late session)

We kept oscillating: grand vision → practical Mamba mod → lost vision → expand → repeat.

The root cause: trying to design architecture + experiment + paper simultaneously.

### The one question that matters first

> Does structured state + free energy objective behave measurably differently than flat state?

### How to answer it

NOT a language model. NOT 150M parameters. A tiny synthetic experiment:

- Small world: ~20 entities, typed properties, causal relationships
- Observation stream: facts arrive over time, some reliable (precision 0.9), some weak (precision 0.3)
- Contradictions: entity properties change, model must update correctly
- Causal structure: A→B→C relationships, testable via intervention queries
- Ground truth: we control everything, so we can measure exactly

Tiny model. Runs in minutes. Three conditions:
1. No persistent state (baseline)
2. Flat persistent state (Mamba-like diagonal)
3. Structured persistent state + L_fe (our claim)

If 3 > 2 > 1 on belief tracking, contradiction handling, and interventional queries → principle validated → scale to language.
If 3 ≈ 2 → structure doesn't matter, rethink.
If 3 < 2 → L_fe hurts, rethink.

### What this resolves before committing to architecture

- Do we need factor graph or is Mamba-mod enough?
- Does L_fe actually help or is it just a regularizer?
- Does explicit precision give measurable benefit?
- Can interventional reasoning work in this representation?

Build small, learn fast, then commit.

---

## TurboQuant math as design principle (March 28, 2026)

Google's TurboQuant (research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) isn't just a compression technique. Its core math maps directly onto our belief representation.

### PolarQuant = our belief representation

PolarQuant decomposes vectors into polar coordinates: radius (magnitude) + angle (direction). Angles are concentrated and predictable → compress to 3 bits.

This IS our precision + belief content:
- **Radius = precision.** How confident the belief is. Large radius = high confidence.
- **Angle = belief content.** What the belief is about. The direction in representation space.

**Design decision: represent beliefs in polar form natively.** Don't have a separate "belief vector + precision scalar." The precision IS the radius. One representation.

Consequences:
- High precision beliefs (large radius) dominate dot products → precision weighting is FREE from the geometry. No separate mechanism needed.
- Low precision beliefs (small radius) are naturally noisy/low-influence → correct behavior by construction.
- Quantization is native: concentrated angles compress to 3-4 bits. The architecture is designed to be quantized from day one.
- Interpretability: radius is human-readable as confidence.

### QJL = precision-weighted message passing

QJL uses 1-bit projections to estimate dot products between high-precision query and low-precision stored data. The estimator "strategically balances" the precision asymmetry.

This IS factor message fusion. High-precision belief sends message to low-precision belief → the estimator accounts for the precision difference. Memoria's `aif/` does this with explicit precision weights. QJL does it as a mathematical primitive for dot products.

### Two-stage decomposition = belief + uncertainty

```
TurboQuant                    Our architecture
─────────────                 ────────────────
PolarQuant (main signal)  →   Belief content (angle) + precision (radius)
QJL (residual correction) →   Uncertainty / residual error on beliefs
Concentrated angles       →   World model has structure (beliefs cluster)
Random rotation           →   Decorrelation (independence in graph structure)
```

### Implications for state design

Old design (before this insight):
```
Belief: [D-dim vector] + [1 precision scalar] + [1 provenance tag]
```

New design:
```
Belief: [D-dim polar vector where radius = precision, angle = content]
         + [1 provenance tag if needed, or encode in angle subspace]
```

Simpler. The geometry does the precision weighting. Message passing between beliefs = dot products in polar space = naturally precision-weighted. Factor potentials can use QJL-style asymmetric estimation for efficient, precision-aware computation.

Memory: a belief region of 1024 beliefs × 256 dims at 3-bit quantization = ~96KB. With full Telos goal region + relation region + meta region, the entire cognitive state fits in <500KB. Negligible.

---

## Generative world modeling — resolved (March 28, 2026)

We don't need a separate simulation engine. The generative world model is already in the architecture:

1. **Static causal inference:** Message passing with clamped nodes (single step, fast)
2. **Temporal simulation:** Transformer generates outcomes conditioned on belief state (multi-step, autoregressive)
3. **Planning / action selection:** EFE computed over simulated outcomes (select minimum expected free energy action)

Key insight: simulated beliefs get LOWER precision (smaller radius) than observed beliefs. Confidence degrades with simulation depth. The model knows its predictions get less reliable over time. When reality contradicts a simulation, the observation (high precision) overwrites the prediction (low precision) via normal belief revision. Self-correcting.

This is Active Inference's "planning as inference" — realized natively because the model IS an LLM that can generate, AND has a calibrated belief state to generate FROM.

## HRM + graph traversal algorithms (March 28, 2026)

Paper: "Hierarchical Reasoning Model" (arxiv.org/abs/2506.21734). 27M params, two recurrent modules at different timescales (slow abstract + fast detailed), solves Sudoku and optimal maze pathfinding with 1000 training samples. No CoT, no pretraining.

### Why it matters

1. **Multi-timescale recurrence validates our design.** HRM's high-level/low-level split maps to our meta+goals (slow) vs. beliefs+relations (fast). The strong inductive bias from hierarchical structure makes 27M params do what larger models can't — suggests our structured state could similarly amplify small model reasoning.

2. **Graph algorithms ARE message passing.** HRM learned optimal pathfinding (a graph algorithm) through recurrence. Our relation region already does message passing. Graph algorithms are just specific message-passing patterns:

   - Reachability → K rounds propagation
   - Shortest path → min-distance relaxation
   - d-separation (Bayes-Ball) → blocked-path detection
   - PageRank → importance distribution iteration
   - Topological sort → directional propagation
   - Cycle detection → message-returns-to-origin
   - Connected components → flood-fill

   All differentiable. All expressible as tensor ops on the relation region adjacency.

3. **Hierarchical message passing schedule.** Instead of flat message passing, do:
   - Low-level pass: fine-grained between individual beliefs (fast, local)
   - High-level pass: abstract between belief clusters (slow, global)
   - This is Memoria's bottom-up/top-down propagation from `aif/`, validated by HRM.

### Design decisions
- Engineered graph ops (d-separation, intervention) where we know the exact algorithm
- Learned graph reasoning (HRM-style) where training discovers the traversal pattern
- Both coexist: engineered ops handle known algorithms, learned recurrence handles novel reasoning

## Build progress (March 28, 2026)

### Phase 1: Core — COMPLETE
- `memoria/core/polar.py` — polar representation (to_polar, to_cartesian, angular_distance, precision_weighted_average)
- `memoria/core/state.py` — CognitiveState with all 4 regions, dynamic allocation, kernel rules, checkpoint/restore
- `memoria/core/free_energy.py` — L_fe: compute_energy (relation agreement), compute_entropy (belief uncertainty), compute_telos_energy (goal drive), compute_free_energy (full Bethe), β computation
- `tests/test_polar.py` — 9 tests covering roundtrip, distance, similarity, weighted average, active detection
- `tests/test_state.py` — 12 tests covering allocation, deallocation, kernel rules, edges, checkpointing
- `tests/test_free_energy.py` — 7 tests covering agreement/disagreement energy, precision amplification, entropy, β, gradient flow
- `REFERENCES.md` — every module mapped to specific repos/papers to reference

### Phase 2: Interface — COMPLETE
- `memoria/interface/read_path.py` — Hopfield-style content-addressable lookup (softmax attention over beliefs). Multi-head. Top-k sparse retrieval. Goal modulation via attention bias. Zero-init output projection (residual friendly).
- `memoria/interface/write_path.py` — Hidden→belief projection + learned precision estimator (Softplus head). Cosine matching against existing beliefs (adaptive threshold from meta). Buffers WriteCandidates for pass 2 (doesn't modify state in forward pass).
- `memoria/interface/layer.py` — StateInterfaceLayer combining read + write. Pre-norm, residual connection on read, returns candidates for pass 2.
- `tests/test_interface.py` — 8 tests: empty state graceful degradation, belief retrieval, goal modulation effect, write matching, precision estimation, gradient flow, shape invariance

### Phase 3: Transformer Integration — COMPLETE
- `memoria/model/config.py` — MemoriaConfig with presets: small (125M), medium (300M), large (500M). Training config with KL annealing schedule, Muon/AdamW params.
- `memoria/model/transformer.py` — GPT backbone ported from autoresearch: RoPE, QK-Norm, ReLU², per-layer residual scalars, logit softcapping. Exposes forward_blocks() for interleaving.
- `memoria/model/memoria_model.py` — Full model: transformer blocks interleaved with StateInterfaceLayers at configurable positions. compute_loss() does L_token + α·L_fe. detach_state() for sequence boundaries.
- `tests/test_model.py` — 10 tests: instantiation, forward pass, L_token loss, L_token+L_fe loss, backward gradients, interface positions, state detach, empty state degradation, parameter counting, candidate production

### Phase 4: Pass 2 (Cognition) — COMPLETE
- `memoria/cognition/surprise.py` — precision-weighted surprise from Memoria's formula: surprise = pred_error × obs_precision. Kalman-like gain. Reconsolidation trigger.
- `memoria/cognition/belief_update.py` — incremental (angle shift by gain) vs. reconsolidation (full rewrite). Radius adjusts: consistent updates increase precision, contradictions decrease it. Eviction of weakest belief when full.
- `memoria/cognition/hebbian.py` — saturating Hebb rule: w += η(1-w). Co-activation creates edges, inactive edges decay and prune below 0.01. From Ba & Hinton Fast Weights.
- `memoria/cognition/telos.py` — full lifecycle: intrinsic generation from surprise hotspots (gated by β), progress tracking (relevance × surprise), stall detection (urgency-scaled thresholds), deadline enforcement. Status machine: empty→proposed→active→stalled→completed/failed.
- `memoria/cognition/consolidation.py` — soft merge (cosine_sim > 0.95, precision-weighted average, combined radius = sqrt(r²+r²)). Periodic hard cleanup (remove below threshold). Edge redirection on merge.
- `memoria/cognition/meta_learning.py` — β = H/(|E|+H+ε) computed from state. SPSA tuning of reconsolidation_threshold and match_threshold. Sequence boundary decay (×0.95 on radii).
- `memoria/cognition/causal.py` — d-separation via BFS with blocking. Interventions: clamp belief, zero incoming edges, propagate outward. Returns simulated downstream beliefs.
- `memoria/cognition/pass2.py` — orchestrator: surprise → belief update → Hebbian → goal progress → intrinsic goals → stall detection → meta → consolidation. Full stats returned.
- `memoria/core/kernel_rules.py` — mark immutable, verify integrity against snapshots.
- `tests/test_pass2.py` — 10 tests: empty, new beliefs, update, reconsolidation, Hebbian, intrinsic goals, goal progress, consolidation, kernel rules, full cycle.

### Phase 5: Data + Training — COMPLETE
- `memoria/data/tokenizer.py` — GPT-2 tokenizer setup
- `memoria/data/streaming.py` — HuggingFace streaming for FineWeb-Edu (10BT) + Stack v2 dedup. Document packing, no disk footprint.
- `memoria/data/synthetic.py` — 4 task generators: belief tracking, contradiction handling, causal chains, precision calibration. ~2500 sequences per generation.
- `memoria/data/interleave.py` — weighted mixing (70/20/10), random source selection, auto-restart on exhaustion
- `memoria/training/optimizer.py` — AdamW with per-group LRs (embedding, unembedding, scalar, matrix, interface). Muon TODO.
- `memoria/training/schedule.py` — LR: warmup→constant→cosine warmdown. α: KL annealing (0→α_max over phase 1→2→3).
- `memoria/training/distributed.py` — device setup, DataParallel placeholder for 2x 3090
- `memoria/training/train.py` — full training loop: 3-phase (language→cognitive awakening→full), gradient accumulation, pass 2 after each step, wandb logging, checkpointing (model + cognitive state + optimizer)
- `scripts/train_small.sh` — quick train script

### Phase 6: Evaluation — COMPLETE
- `memoria/eval/perplexity.py` — standard LM perplexity on held-out FineWeb (must not degrade)
- `memoria/eval/belief_tracking.py` — fact tracking + contradiction resolution accuracy
- `memoria/eval/hallucination.py` — calibrated refusal: known vs unknown confidence separation, β comparison
- `memoria/eval/causal.py` — d-separation accuracy + intervention propagation on relation graph
- `memoria/eval/telos_demo.py` — intrinsic goal generation from surprising stream (3-phase: consistent→contradictory→novel domain)
- `memoria/eval/improvement.py` — THE hero figure: accuracy vs interactions curve. Should rise for our model, flat for baseline.
- `memoria/eval/crossover.py` — small+experience vs large+fresh comparison. Finds crossover point.
- `scripts/eval_all.sh` — run all evals on a checkpoint

### ALL PHASES COMPLETE.

---

## 10 design problems solved (March 28, 2026 — late session)

All solutions written into architecture.md. Summary:

1. **Gradient flow** → Truncated BPTT. Detach state at sequence boundaries. L_token shapes interface weights, L_fe shapes state content. Same as Titans.
2. **Pass 2 timing** → Per sequence during training. Per response during inference. Mid-generation reconsolidation only on extreme surprise.
3. **Batched training** → Per-item state copies, precision-weighted merge after. 64 batch × 2.2MB = 141MB. Acceptable.
4. **Relation efficiency** → Hard indices + soft weights. O(N_edges × K) not O(N_edges × N_beliefs). Gather-transform-scatter. Structure changes in pass 2 only.
5. **Belief matching** → Cosine similarity with adaptive threshold (meta-learned via SPSA). Above threshold = update, below = allocate.
6. **Contradiction** → High-energy state in L_fe. Resolution via precision: higher precision wins. Equal precision → both maintained → β increases → intrinsic goal to resolve.
7. **Consolidation** → Soft merge (differentiable, continuous) + periodic hard cleanup (non-differentiable, like data augmentation). Merged radius = sqrt(r_A² + r_B²).
8. **Sequence boundaries** → Exponential radius decay (×0.95). Reinforced beliefs maintain precision. Unreinforced fade in ~20 sequences. 50/50 persistent/fresh training split.
9. **L_fe math** → `E_f = -w_f �� r_i × r_j × log(σ(agreement × temp))`, `H_i = -log(r_i + ε)`, `F = ΣE - ΣH`. Fully differentiable. Gradients into angles, radii, relations, and weights.
10. **Negation** → Three states: no edge (ignorance), positive edge (knowledge), negative edge (anti-knowledge/known independence). Model learns all three through training.

## The success criterion (March 28, 2026)

**A 500M model with experience surpasses a 10B model without it.**

This is the north star. Not benchmarks. Not perplexity. The crossover curve.

### Why the theory supports this

A 10B model's advantage over 500M is: more knowledge in weights, deeper reasoning, richer representations.

Our 500M + cognitive state compensates:
- Knowledge gap → persistent beliefs accumulate domain-specific knowledge. Depth beats breadth on domain tasks.
- Reasoning gap → Telos decomposes hard problems into solvable subproblems. Iterative reasoning over cognitive state extends effective depth through multiple passes. HRM showed 27M matching larger models through hierarchical structure.
- Representation gap → belief region IS an additional representation space for experienced domains.

### Where crossover happens (80-90% of practical coding tasks)
- Context/memory bottleneck tasks (most coding)
- Domain-specific pattern recognition
- Codebase understanding and navigation
- Debugging with historical context
- Calibrated uncertainty (knows its limits vs. hallucinates)
- Anything requiring multi-session persistence

### Where it doesn't (10-20% edge cases)
- Novel algorithmic reasoning never encountered
- Single-pass extreme reasoning depth
- Tasks completely outside experience domain
- General knowledge breadth

### The path
1. Does architecture work at all? (150M with state vs. without)
2. Does experience improve performance? (plot performance vs. interactions)
3. Does it beat larger models? (500M+state vs. 1B, 2B, 5B without)
4. The crossover: 500M+experience vs. 10B fresh on real coding tasks

### Implications if crossover exists
- Current paradigm: want better model → train bigger (costs billions)
- New paradigm: want better model → use it more (costs time)
- Scaling laws become the old paradigm
- Small model + experience > big model for practical tasks
- Decentralized AI (anyone can grow a model, not just big labs)

### Belief capacity is the new context length
Context = what you see right now (ephemeral). Beliefs = what you know (persistent, structured, precise).
- Context: linear cost, unstructured, bigger = slower, no learning
- Beliefs: sublinear cost (sparse active set), structured, bigger = same speed, compounds with experience
- 128K context ≈ one long document. 65K beliefs ≈ a career of experience.

---

## Course correction: Memoria+Telos IS the v1 spec (March 28, 2026)

Previous sections kept deferring features to "v2." Wrong. Memoria+Telos defines what the model must do. ALL features. The PROCESS (engineered vs. emergent, Mamba vs. custom, etc.) is flexible. The capability set is not.

### Full feature mapping: Memoria+Telos → Neural Architecture

The state tensor has explicit structured regions. This is a cognitive architecture implemented as tensor operations. Some things are engineered. Some emerge. Both are fine.

```
State Tensor Layout:
┌─────────────────────────────────────────────────┐
│ BELIEF REGION          [N_beliefs × (D + 1 + 1)]│
│  - D-dim representation per belief              │
│  - 1 precision scalar per belief                │
│  - 1 provenance tag per belief                  │
├─────────────────────────────────────────────────┤
│ RELATION REGION        [N_edges × (2 + K + 1)]  │
│  - 2 endpoint indices (soft attention over      │
│    belief region)                                │
│  - K-dim relation representation                │
│  - 1 edge weight (strength/confidence)          │
├─────────────────────────────────────────────────┤
│ GOAL REGION            [N_goals × (D + G)]      │
│  - D-dim goal embedding                         │
│  - G-dim goal metadata:                         │
│    priority, urgency, progress, status,         │
│    depth, surprise_accumulator, created_step    │
├─────────────────────────────────────────────────┤
│ META REGION            [M]                       │
│  - β (exploration/exploitation)                 │
│  - accumulated_surprise                         │
│  - learning rate modulation                     │
│  - consolidation threshold                      │
└─────────────────────────────────────────────────┘
```

### Feature-by-feature mapping

#### MEMORIA CORE

| # | Feature | Neural implementation | Engineered or emergent |
|---|---------|----------------------|----------------------|
| 1 | Factor graph / Bethe free energy | Message passing between belief region and relation region. L_fe computed over this structure. | Engineered structure, learned content |
| 2 | Precision weighting | Explicit precision scalar per belief. Gates incoming messages: high precision = resistant to update, low precision = easily changed. Modulates outgoing messages: high precision beliefs influence neighbors more. | Engineered mechanism |
| 3 | Surprise-driven dynamics | Per-belief surprise = magnitude of update from pass 2. Accumulated in meta region. Triggers reconsolidation when threshold exceeded. | Engineered trigger, learned sensitivity |
| 4 | β (explore/exploit) | Computed from meta region: β = total_entropy / (total_energy + total_entropy). NOT a hyperparameter. Computed from actual state every forward pass. | Engineered formula |
| 5 | Causal reasoning / d-separation | Relation region encodes directed edges. Sparsity pattern = graph topology. d-separation computed via graph ops on the relation region. Interventions = clamping belief + zeroing incoming edges + propagating. | Engineered graph ops |
| 6 | Hebbian associations | After each forward pass, co-activated beliefs strengthen their connecting edge in relation region: w_new = w_old + η(1 - w_old). Saturating. | Engineered update rule |
| 7 | Reconsolidation | When surprise on a belief exceeds threshold (modulated by β): full rewrite of belief representation instead of incremental update. Hard gate: surprise > threshold → overwrite. | Engineered threshold |
| 8 | Compression / consolidation | Periodic operation (every N steps): cluster co-activated beliefs, merge into single higher-precision abstract belief. Frees slots in belief region. | Engineered periodic op |
| 9 | Meta-learning | Meta region parameters (learning rates, thresholds, consolidation frequency) updated via SPSA on free energy. The model tunes its own cognitive parameters. | Engineered SPSA, learned parameters |
| 10 | Kernel rules | Hard masks on state: certain beliefs marked immutable (precision = ∞, update gate = 0). Certain edges fixed. Enforced as tensor masks, not soft constraints. | Fully engineered |
| 11 | Governance | Visibility masks per agent context. Transition rules on goal status encoded as valid-transition matrix. | Engineered |
| 12 | Audit trail | State diff logged after each pass 2. Checkpoint stack of (step, changed_indices, old_values, new_values). Not in forward pass — side effect. | Engineered logging |
| 13 | Multi-agent / namespaces | Per-agent state partition. Agent context selects which state slice is active. Isolation via masking. | Engineered partitioning |
| 14 | Secondary indexes | Learned hash over belief representations for O(1) lookup by content. Small auxiliary structure. | Engineered |

#### TELOS

| # | Feature | Neural implementation | Engineered or emergent |
|---|---------|----------------------|----------------------|
| 15 | Explicit goal representation | Goal region of state tensor. Each slot = goal embedding + metadata (priority, urgency, progress, status, depth). | Engineered structure |
| 16 | Intrinsic goal generation | When accumulated surprise in a state region exceeds β-adjusted threshold → allocate new goal slot. Goal embedding = compressed representation of high-surprise beliefs. Priority = normalized surprise magnitude. | Engineered trigger, learned content |
| 17 | Goal-directed attention | Active goal embeddings modulate the C_t readout: beliefs similar to active goals get amplified in output. Dot product between goal embeddings and belief representations → attention boost. | Engineered modulation |
| 18 | Goal lifecycle | Status dimension in goal metadata. Valid transitions encoded as mask. Proposed→Active (when claimed or surprise confirms), Active→Stalled (no progress for N steps), Active→Completed (progress ≥ 1.0), etc. | Engineered state machine |
| 19 | Goal decomposition | When a goal's depth < max: spawn child goals in empty slots. Child embedding = learned projection of parent embedding into subgoal space. Children inherit deadline, namespace. | Engineered spawning, learned projection |
| 20 | Progress tracking | Progress scalar updated when beliefs relevant to goal change. Relevance = cosine similarity between goal embedding and updated belief. Progress += Δ_relevant_beliefs. | Engineered tracking |
| 21 | Stall detection | If goal progress doesn't change for N steps AND goal.urgency > threshold → status = Stalled. Stalled goals get attention boost (staleness bonus from Memoria). | Engineered detection |
| 22 | Deadline enforcement | If current_step > goal.deadline AND status = Active → status = Stalled. Priority boosted as deadline approaches (sigmoid). | Engineered |
| 23 | Multi-agent goals | Goal region partitioned per namespace. Claiming = writing agent_id to goal slot. Delegation = spawning child in another agent's partition. | Engineered |
| 24 | Enterprise directives | Goals with precision = ∞, kernel-rule protection. Cannot be modified or deleted. Priority fixed. | Engineered (same as kernel rules) |

### What's engineered vs. emergent — and why that's OK

Most features are ENGINEERED mechanisms. The LEARNED parts are:
- What beliefs to form (content of belief region)
- What relationships to encode (content of relation region)
- What goals to generate (content of goal embedding)
- How to decompose goals (learned projection)
- How to modulate attention based on goals (learned weights)
- The meta-learning parameters (learned via SPSA)

This is a COGNITIVE ARCHITECTURE, not a pure neural network. That's the honest framing. Like ACT-R or SOAR but implemented as differentiable tensor operations trained end-to-end with L_token + L_fe.

The model doesn't discover the cognitive mechanisms — we engineer them from Memoria's validated principles. The model discovers what to DO with them.

### The 2-pass loop with full features

**Pass 1 (forward/inference):**
1. Input → frozen transformer layers
2. At state layers: message passing between belief and relation regions
3. Goal-directed attention modulates readout (C_t shaped by active goals)
4. Output = next token prediction

**Pass 2 (adaptation):**
1. Compute prediction error → surprise per belief
2. Precision-weighted belief updates (high precision = small update, low = large)
3. Hebbian strengthening of co-activated edges
4. Reconsolidation on beliefs exceeding surprise threshold
5. Update accumulated surprise in meta region
6. Compute β from current state
7. Check goal progress, stall detection, deadline enforcement
8. If surprise accumulator exceeds β-adjusted threshold → generate intrinsic goals
9. If consolidation timer fires → compress/merge beliefs
10. Meta-learning step on cognitive parameters (every N passes)
11. Log state diff to audit trail

**Periodic (every N sequences):**
- Consolidation: cluster and merge beliefs
- Goal decomposition: break down high-level goals
- Structure pruning: remove weak edges in relation region

### What this actually is

It's Memoria+Telos, implemented as a structured state tensor inside a neural network, where:
- The frozen transformer layers are the "language engine" (perception/generation)
- The structured state is the "cognitive engine" (beliefs, goals, reasoning)
- Free energy is the unified objective
- The 2-pass loop is the perception-action cycle

The process is different (tensors instead of CozoDB, gradient updates instead of explicit writes). The capabilities are identical.
