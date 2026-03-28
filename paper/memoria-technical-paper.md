# Memoria: A Self-Evolving Memory Runtime for Autonomous Agents via Active Inference

**Abstract** — We present *Memoria*, a persistent memory runtime for autonomous AI agents grounded in the Active Inference Framework (AIF). Unlike existing agent memory systems that rely on ad hoc heuristics for storage, retrieval, and forgetting, Memoria provides a principled, unified computational substrate where all memory dynamics — surprise detection, belief revision, goal generation, skill crystallization, and causal reasoning — emerge from free energy minimization. The system implements precision-weighted belief updates, Bethe free energy computation over a unified graph-relational-vector store, intrinsic motivation via surprise-driven goal generation, structural causal models with interventional calculus, and meta-learning for continuous self-optimization. We describe the architecture, formalize the mathematical framework, and demonstrate how each subsystem contributes to a coherent, self-improving agent memory that moves beyond passive recall toward active epistemic agency.

---

## 1. Introduction

The rapid advancement of large language model (LLM) agents has exposed a fundamental limitation: **agents do not learn from experience**. Each conversation begins from a blank slate. Each tool-use session re-discovers strategies that worked before. Each multi-step task loses context when the session ends. This is not merely an engineering inconvenience — it is a *theoretical* gap. An agent without persistent, self-organizing memory cannot form beliefs, track goals, detect contradictions, or improve its procedures over time.

Existing approaches to agent memory fall into three categories:

1. **Retrieval-Augmented Generation (RAG)**: Vector stores paired with embedding-based retrieval. These systems index documents but do not *learn* — they have no concept of surprise, no mechanism for belief revision, and no capacity for goal persistence.

2. **Key-Value Memory**: Simple structured storage (e.g., user preferences, entity facts). These systems are brittle, require manual schema design, and cannot represent uncertainty or evolve autonomously.

3. **Episodic Buffers**: Conversation history caches with summarization. These systems compress but do not abstract, and lack any formal framework for when to forget, what to consolidate, or how to prioritize.

None of these approaches address the core challenge: **how should an agent's memory organize, maintain, and evolve itself?** We argue that this question has a principled answer in the Active Inference Framework (AIF) [1, 2], which casts all cognition — perception, action, learning, and planning — as free energy minimization on a generative model.

**Memoria** implements this vision as a production-grade Rust library comprising ~31,000 lines of code across 86 source modules. Its key contributions are:

1. **AIF-native memory architecture**: Every memory operation — storage, retrieval, forgetting, consolidation — is unified under a single variational objective (Section 3).

2. **Precision-weighted factor graph scoring**: Memory retrieval fuses multiple heterogeneous signals (embedding similarity, temporal activation, Hebbian co-access, graph centrality, goal alignment) via precision-weighted message passing, yielding principled relevance scores rather than ad hoc heuristics (Section 4).

3. **Intrinsic goal generation**: The system autonomously creates goals from surprise hotspots, with generation thresholds that emerge from the exploration-exploitation balance parameter β, closing the perception-action loop (Section 5).

4. **Structural causal reasoning**: A Bayesian causal graph with NOTEARS structure learning [3], Bayes-Ball d-separation [4], and Pearl's do-calculus [5] enables the agent to reason about *why* outcomes occurred, not just *what* happened (Section 6).

5. **Procedural memory with Expected Free Energy (EFE) selection**: Skills crystallize from successful episodes and are selected for future tasks using an EFE criterion that balances pragmatic value, epistemic gain, and goal alignment (Section 7).

6. **Self-tuning dynamics**: A meta-learning loop using Simultaneous Perturbation Stochastic Approximation (SPSA) [6] continuously adapts the system's own hyperparameters to minimize free energy (Section 8).

The remainder of this paper formalizes these contributions and describes their implementation within a unified architecture.

---

## 2. Architecture Overview

### 2.1 Unified Storage Substrate

Memoria is built on an embedded database engine that unifies three data models in a single process:

- **Relational**: Typed relations with columnar storage for structured data (memories, facts, entities, goals, skills, audit logs).
- **Graph**: First-class edges with typed relationships (causal links, Hebbian co-activations, chunk hierarchies, source provenance chains).
- **Vector**: HNSW (Hierarchical Navigable Small World) [7] indexes for approximate nearest-neighbor search on high-dimensional embeddings.

This unification eliminates the impedance mismatch that arises when agent systems stitch together separate vector databases, relational stores, and graph databases. All queries — semantic similarity, graph traversal, relational joins, and temporal range scans — execute within a single transactional context.

**Bi-temporal versioning.** Every relation carries a Validity dimension, enabling time-travel queries. When a memory's content or confidence is updated, the previous version remains accessible at its historical timestamp. This provides a complete audit trail and supports counterfactual reasoning ("what did the agent believe at time *t*?").

### 2.2 Service Architecture

Memoria defines four pluggable service traits that abstract external capabilities:

| Service | Interface | Purpose |
|---------|-----------|---------|
| **Embedder** | `embed(text) → Vec<f32>` | Semantic vector representations |
| **NER** | `extract(text) → Vec<Entity>` | Named entity recognition |
| **Reranker** | `rerank(query, docs) → Vec<(doc, score)>` | Cross-encoder relevance refinement |
| **LLM** | `complete(messages, max_tokens) → Response` | Natural language reasoning |

Each trait admits multiple implementations — API clients (OpenAI-compatible, Anthropic), local ONNX models, or mock services for testing. The LLM service is invoked at seven integration points: goal decomposition, goal detection, relation verification, episodic reflection, causal attribution, intrinsic goal summarization, and skill improvement. All other operations (scoring, belief updates, surprise computation, decay) are pure computation — no LLM in the hot path.

### 2.3 Background Task Queue

A persistent, priority-ordered task queue with optimistic locking dispatches 25+ asynchronous task types. Tasks are enqueued by user-facing operations (e.g., `tell()` enqueues NER linking and surprise computation) and by the periodic `tick()` scheduler (e.g., compression, reflection, meta-learning). The queue is backed by the same unified store, ensuring transactional consistency between foreground mutations and background processing.

### 2.4 Core API

The public interface centers on four verbs:

- **`tell(text, context)`**: Ingest knowledge — chunks, embeds, extracts entities, computes surprise against existing beliefs, and enqueues downstream tasks.
- **`ask(query, context)`**: Retrieve knowledge — performs HNSW search, enriches candidates with activation, Hebbian, PageRank, and telos signals, fuses via precision-weighted messages, and optionally reranks.
- **`prime(context)`**: Pre-task context injection — returns EFE-ranked skills, pre-activated memories, β, active goals, predictions, and regime stability.
- **`feedback(outcome, context)`**: Post-task learning — updates free energy, β, skill performance, and triggers consolidation if surprise exceeds threshold.

---

## 3. Active Inference as the Unifying Principle

### 3.1 Background

The Active Inference Framework [1, 2] posits that adaptive systems minimize *variational free energy* — a tractable upper bound on the negative log-evidence of a generative model. For an agent maintaining beliefs *q(s)* about hidden states *s* given observations *o*:

$$F = \underbrace{D_{KL}[q(s) \| p(s)]}_{\text{complexity}} - \underbrace{\mathbb{E}_{q}[\ln p(o|s)]}_{\text{accuracy}}$$

Minimizing *F* with respect to *q* yields perception (belief updating); minimizing *F* with respect to actions yields planning. This dual minimization provides a single objective for all cognitive processes.

### 3.2 Bethe Free Energy over the Memory Store

We adapt the Bethe free energy approximation [8] to the agent's entire knowledge store. Let the store contain *N* facts, each with confidence $c_i \in [0,1]$ and reinforcement count $r_i$. We define:

**Precision** of each belief:
$$\pi_i = c_i \cdot \ln(\max(r_i, 1) + 1)$$

This captures two intuitions: confidence encodes subjective certainty, while reinforcement count provides frequentist support. A belief stated once with high confidence (e.g., user assertion) and a belief confirmed many times with moderate confidence (e.g., repeated extraction) can have equivalent precision.

**Factor energy** (model fit):
$$E_{\text{factor}} = \sum_{i=1}^{N} -\ln(\pi_i + \epsilon)$$

where $\epsilon = 10^{-10}$ prevents singularity. Low-precision beliefs contribute large energy terms — they represent uncertainty the model has not resolved.

**Variable entropy** (belief uncertainty):
$$H_{\text{var}} = \sum_{i=1}^{N} H(c_i) = \sum_{i=1}^{N} [-c_i \ln c_i - (1-c_i)\ln(1-c_i)]$$

**Bethe free energy**:
$$F = E_{\text{factor}} - H_{\text{var}}$$

This quantity decreases as the agent's beliefs become more precise and more certain. Every memory operation that changes a confidence or adds a fact alters *F*, providing a global objective that all dynamics seek to minimize.

### 3.3 The Exploration-Exploitation Parameter β

From the free energy decomposition, we derive a scalar parameter that governs the system's exploration-exploitation balance:

$$\beta = \frac{H_{\text{var}}}{E_{\text{factor}} + H_{\text{var}} + \epsilon} \in [0, 1]$$

When entropy dominates (many uncertain beliefs), $\beta \to 1$ — the system is in *exploration mode*, favoring actions that reduce uncertainty. When factor energy dominates (many imprecise beliefs but low entropy, i.e., confidently wrong), $\beta \to 0$ — the system is in *exploitation mode*, favoring actions that leverage existing knowledge.

$\beta$ is not a hyperparameter — it is *computed* from the current state of the memory store. It governs:

- **Intrinsic goal generation** thresholds (Section 5)
- **Skill selection** balance between pragmatic and epistemic value (Section 7)
- **Meta-learning** step sizes (Section 8)

### 3.4 Goal Contributions to Free Energy

Active telos (goals) contribute to factor energy as *preference* terms:

$$E_{\text{telos}} = \sum_{g \in \text{active}} -\ln(\pi_g^{\text{pref}}) \cdot d_g$$

where $\pi_g^{\text{pref}} = p_g \cdot c_g$ is the preference precision (priority × confidence) and $d_g = 1 - \text{progress}_g$ is the goal distance. Unmet high-priority goals increase free energy, motivating the agent to pursue them.

### 3.5 Prediction Error Contributions

The system maintains forward predictions (Section 8.3). Average prediction error $\bar{e}$ contributes:

$$E_{\text{pred}} = -\ln(1 - \bar{e})$$

Poor predictions increase free energy, triggering the meta-learning and reflection subsystems to improve the model.

---

## 4. Precision-Weighted Memory Retrieval

### 4.1 Multi-Signal Candidate Enrichment

When the agent issues a query, retrieval proceeds in stages:

1. **HNSW search**: The query embedding is compared against the memory HNSW index, returning candidates with L2 distance $d_i$. We convert to similarity: $s_i = 1 - d_i$.

2. **Activation decay**: Each memory has a temporal activation based on last access time:
$$a_i(t) = \exp\left(-\frac{t - t_{\text{last\_access}}}{\tau_{\text{activation}}}\right)$$

3. **Hebbian co-access weight**: Memories that are frequently retrieved together develop associative bonds:
$$w_{ij}^{(t+1)} = w_{ij}^{(t)} + \eta(1 - w_{ij}^{(t)})$$
where $\eta = 0.1$ is the learning rate. This saturating rule ensures weights approach but never exceed 1.0.

4. **Graph centrality**: PageRank scores computed over the entity-memory-edge graph provide a structural importance signal.

5. **Telos alignment**: Cosine similarity between the candidate memory and active goal embeddings, weighted by goal priority and remaining distance:
$$\text{telos\_boost}_i = \max_{g \in \text{active}} \left[\text{sim}(\mathbf{m}_i, \mathbf{g}) \cdot p_g \cdot (1 - \text{progress}_g)\right]$$

6. **Belief precision**: As defined in Section 3.2:
$$\pi_i = c_i \cdot \ln(\max(r_i, 1) + 1)$$

### 4.2 Factor Message Fusion

Each signal is represented as a *factor message* $(\mu_k, \pi_k)$ with value $\mu_k$ (in log domain) and precision $\pi_k$. The posterior score for each candidate is computed via precision-weighted averaging:

$$\text{score}_i = \frac{\sum_{k} \pi_k \cdot \mu_k^{(i)}}{\sum_{k} \pi_k}$$

This is the optimal estimator under Gaussian factor assumptions and naturally handles heterogeneous signal scales. High-precision signals (e.g., strong embedding similarity) dominate; low-precision signals (e.g., weak Hebbian weight) contribute proportionally less.

**Hard blocks**: Any factor message with $\mu = -\infty$ (e.g., a kernel rule denying access) immediately forces the candidate's score to $-\infty$, implementing unbypassable constraints within the scoring pipeline.

### 4.3 Hierarchical Message Passing

For hierarchically chunked documents (sentence → paragraph → section → document), we perform bidirectional message passing:

**Bottom-up**: Sentence-level scores propagate upward, with each parent receiving $\max(\text{child scores})$ (at-least-one-match semantics).

**Top-down**: Document-level relevance propagates downward with decay factor $\alpha = 0.9$ per level:
$$\text{child\_score} = \alpha \cdot \text{parent\_score}$$

**Combined score**: $\text{final} = \text{bottom\_up} \times \text{top\_down}$, retaining only nodes with both signals.

---

## 5. Intrinsic Goal Generation

### 5.1 Surprise Computation

When new knowledge is ingested via `tell()`, the system computes surprise against existing beliefs. For a new observation $o$ contradicting existing fact $f$:

$$\text{surprise}(o, f) = \underbrace{1}_{\text{prediction error}} \times \underbrace{c_o \cdot w_{\text{prov}(o)}}_{\text{observation precision}}$$

where $w_{\text{prov}}$ maps provenance to weight: *direct* = 1.0, *user\_stated* = 0.95, *agent\_reported* = 0.8, *extracted* = 0.6, *inferred* = 0.3.

A Kalman-like gain determines whether the new observation should trigger belief revision:

$$\kappa = \frac{\pi_{\text{obs}}}{\pi_{\text{belief}} + \pi_{\text{obs}}}$$

When $\kappa > 0.3$, reconsolidation is triggered: the existing fact's confidence shifts toward the observation:

$$c_f^{\text{new}} = (1 - \kappa) \cdot c_f + \kappa \cdot c_o$$

When surprise exceeds 2.0, the system additionally enqueues a causal attribution task (Section 6).

### 5.2 Surprise Hotspot Aggregation

Unresolved surprise entries are aggregated by variable (entity or topic). A *hotspot* forms when $\geq 3$ unresolved entries cluster around a common variable.

### 5.3 Adaptive Generation Thresholds

The decision to generate intrinsic goals is governed by β:

$$\text{threshold} = \bar{S} \cdot (1 + \beta)$$

where $\bar{S}$ is the mean unresolved surprise. The number of goals generated is bounded:

$$n_{\text{max}} = \lceil \beta \cdot 5 \rceil, \quad n_{\text{max}} \in [1, 5]$$

Generated goals receive `Intrinsic` provenance with initial confidence 0.3. When $\beta > 0.5$ (exploration-dominant regime), goals are automatically activated; otherwise, they remain in `Proposed` status awaiting human or agent approval.

**Duplicate suppression**: Before creating a new intrinsic goal, the system checks embedding similarity against all existing goals. If $\text{sim} \geq 0.85$, the goal is suppressed to prevent redundant exploration.

This mechanism closes the AIF loop: the agent predicts → observes mismatch → accumulates surprise → generates goals to investigate the surprise → explores → learns → reduces free energy.

---

## 6. Structural Causal Reasoning

### 6.1 Bayesian Causal Graph

Memoria maintains a persistent directed graph where nodes represent entities/variables and edges represent causal relationships. Edges are accumulated via Bayesian updates from three sources:

1. **Counterfactual attribution** (Section 6.4)
2. **LLM-proposed edges** during reflection (initial confidence 0.3)
3. **NOTEARS structure learning** (Section 6.2)

For each observation of edge $(u, v)$ with likelihood $\ell$:

$$\text{strength}^{(t+1)} = \frac{\text{strength}^{(t)} \cdot n^{(t)} + \ell}{n^{(t)} + 1}$$

$$n^{(t+1)} = n^{(t)} + 1$$

$$c^{(t+1)} = \min\left(0.95, \; c^{(t)} + (1 - c^{(t)}) \cdot 0.1\right)$$

This monotonic confidence growth, capped at 0.95, ensures that even highly reinforced edges retain some epistemic humility.

### 6.2 NOTEARS Structure Learning

For periodic validation and discovery of causal structure, we implement the NOTEARS algorithm [3], which formulates DAG learning as a continuous optimization:

$$\min_W \quad \frac{1}{2n}\|X - XW\|_F^2 + \lambda\|W\|_1$$
$$\text{s.t.} \quad h(W) = \text{tr}(e^{W \circ W}) - d = 0$$

where $W \in \mathbb{R}^{d \times d}$ is the weighted adjacency matrix, $X \in \mathbb{R}^{n \times d}$ is the observation matrix, $\lambda = 0.1$ is the L1 sparsity penalty, $\circ$ denotes element-wise multiplication, and $h(W) = 0$ enforces the DAG constraint via the matrix exponential characterization [3].

We solve this via the augmented Lagrangian method:

$$\mathcal{L}(W, \alpha, \rho) = \frac{1}{2n}\|X - XW\|_F^2 + \lambda\|W\|_1 + \alpha \cdot h(W) + \frac{\rho}{2}h(W)^2$$

The inner loop applies gradient descent with L1 proximal steps (soft thresholding), followed by outer updates:

$$\alpha \leftarrow \alpha + \rho \cdot h(W), \quad \rho \leftarrow \min(10^{16}, \; 10\rho)$$

Convergence is declared when $h(W) < 10^{-8}$. Discovered edges with $|w_{ij}| > 0.3$ are merged into the Bayesian causal graph, adjusting existing edge confidences.

### 6.3 d-Separation via Bayes-Ball

To determine conditional independence — crucial for efficient attribution — we implement the Bayes-Ball algorithm [4]. The algorithm propagates "balls" through the graph tracking arrival direction (from-parent vs. from-child), correctly handling:

- **Chains** ($A \to B \to C$): blocked iff $B$ is conditioned
- **Forks** ($A \leftarrow B \to C$): blocked iff $B$ is conditioned
- **Colliders** ($A \to B \leftarrow C$): blocked *unless* $B$ or a descendant of $B$ is in the conditioning set

Pre-computing the ancestor set of the conditioning set $Z$ enables efficient collider activation checks.

### 6.4 Counterfactual Attribution

When a task fails, the system performs counterfactual attribution to identify which recalled memories may have contributed:

1. **Baseline**: Compute success rate among semantically similar tasks ($\text{sim} > 0.5$, top 50 neighbors).

2. **For each memory used in the failed task**:
   - Check d-separation: skip if conditionally independent of outcome given other memories
   - Compute counterfactual rate: success rate among similar tasks that did *not* use this memory
   - Compute observational impact: $\Delta_{\text{obs}} = \text{counterfactual\_rate} - \text{baseline\_rate}$

3. **Interventional analysis** via do-calculus: apply $\text{do}(\text{memory} = 0)$ and forward-propagate through the causal graph:
   $$v_{\text{new}} = v_{\text{orig}} \cdot (1 - w) + v_{\text{intervention}} \cdot w$$
   where $w$ is the causal edge weight.

   Interventional impact: $\Delta_{\text{int}} = \text{mean}(|v_{\text{orig}} - v_{\text{new}}|)$ across affected nodes.

4. **Blended impact**:
$$\Delta_{\text{final}} = 0.4 \cdot \Delta_{\text{obs}} + 0.6 \cdot \Delta_{\text{int}}$$

The 60/40 weighting favors interventional evidence (which accounts for confounders) over observational evidence (which may reflect selection bias).

---

## 7. Procedural Memory and Skill Selection

### 7.1 Skill Lifecycle

Skills represent *procedural* knowledge — learned sequences of steps that achieve goals. They follow a lifecycle:

| Phase | Provenance | Description |
|-------|-----------|-------------|
| **Bootstrap** | Manual | Seeded from documentation or expert knowledge |
| **Crystallize** | Experience | Extracted from successful episode patterns |
| **Specialize** | Adaptation | Variant created for specific context |
| **Generalize** | Abstraction | Common structure extracted from multiple variants |

Each skill maintains performance statistics: success rate, average duration, usage count, and last-used timestamp. A version counter increments with each improvement cycle.

### 7.2 EFE-Based Skill Selection

When the agent faces a new task, skills are ranked by Expected Free Energy (EFE) [9]:

$$\text{EFE}(s) = \underbrace{\pi_s \cdot r_s}_{\text{pragmatic}} + \beta \cdot \underbrace{(H_s + \nu_s)}_{\text{epistemic}} + \gamma \cdot \underbrace{\text{telos}(s)}_{\text{goal alignment}}$$

**Pragmatic value** favors skills with high precision and track record:
$$\pi_s = c_s \cdot \ln(v_s + 1)$$
$$r_s = \begin{cases} \text{empirical success rate} & \text{if usage count} > 0 \\ 0.5 & \text{otherwise (uninformative prior)} \end{cases}$$

**Epistemic value** favors skills whose outcomes are uncertain (high information gain potential):
$$H_s = -r_s \ln r_s - (1 - r_s)\ln(1 - r_s) \quad \text{(Shannon entropy)}$$
$$\nu_s = \frac{1}{\text{usage\_count} + 1} \quad \text{(novelty)}$$

**Goal alignment** favors skills relevant to active objectives:
$$\text{telos}(s) = \max_{g \in \text{active}} \left[\text{sim}(\mathbf{s}, \mathbf{g}) \cdot p_g \cdot (1 - \text{progress}_g)\right]$$

The balance between these terms is governed by $\beta$ (exploration-exploitation, computed from free energy) and $\gamma$ (goal-directedness, tuned by meta-learning). This ensures that when the agent is uncertain ($\beta$ high), it prefers novel or untested skills; when confident ($\beta$ low), it prefers proven procedures; and always, it prefers skills aligned with its current goals.

---

## 8. Self-Tuning Dynamics

### 8.1 Temporal Confidence Decay

All beliefs decay over time, with provenance-dependent time constants:

$$c_i(t) = c_i^{(0)} \cdot \exp\left(-\frac{t - t_0}{\tau_i}\right)$$

| Provenance | $\tau$ |
|------------|--------|
| Direct | 365 days |
| User-stated | 180 days |
| Extracted | 90 days |
| Inferred | 30 days |

Reinforcement extends the effective time constant:
$$\tau_{\text{eff}} = \tau \cdot (1 + \ln(\max(r_i, 1)))$$

This models the psychological finding that frequently accessed memories persist longer [10], while also respecting that inferred knowledge should decay faster than directly observed knowledge.

### 8.2 Memory Compression

As the store grows, raw memories are promoted through compression levels:

$$\text{Raw} \xrightarrow{\text{cluster + summarize}} \text{Episode Summary} \xrightarrow{\text{pattern extract}} \text{Abstraction} \xrightarrow{\text{meta-abstract}} \text{Core Belief}$$

Clustering uses a Union-Find algorithm over co-activation edges (minimum cluster size: 3). The LLM generates summaries for each cluster, which are embedded and stored at the next level. Source memories are marked as compressed and become eligible for garbage collection.

Abstractions that receive sufficient evidential support ($\text{evidence} \geq \text{promotion\_threshold}$) receive a confidence boost:

$$c^{\text{new}} = c + (1 - c) \cdot 0.1 \cdot (\text{evidence} - \text{threshold} + 1)$$

capped at 0.99, preventing any belief from reaching absolute certainty.

### 8.3 Prediction Generation

Memoria maintains forward predictions using four complementary generators:

1. **PPM-C** (Prediction by Partial Matching) [11]: Variable-order Markov model over task sequences, predicting the next task type from context history.

2. **ETS** (Exponential Trend Smoothing) [12]: Extrapolates telos progress trajectories, detecting stalls when observed progress falls below the 95% confidence interval.

3. **BOCPD** (Bayesian Online Changepoint Detection) [13]: Monitors the surprise stream for regime shifts, using a Normal-Gamma prior. A changepoint is detected when $P(\text{run\_length} < 5) > 0.5$.

4. **Episodic pattern** (LLM-assisted): Generates high-level predictions from episode summaries and active goals (invoked every 5th cycle due to cost).

Predictions carry confidence scores and expiration timestamps. Expired predictions are resolved as hits (embedding similarity $\geq 0.6$ to a matching observation) or misses. Misses contribute to surprise:

$$\Delta S_{\text{miss}} = c_{\text{prediction}}$$

connecting prediction failure directly to the free energy computation.

### 8.4 Meta-Learning via SPSA

The system continuously tunes its own hyperparameters to minimize free energy using a two-phase approach:

**Phase 1: Cold Start (Bayesian Exploration)**. The system collects $(θ, F(θ))$ observations over a budget of initial ticks, then selects the parameter configuration yielding lowest free energy.

**Phase 2: Online Tracking (SPSA)** [6]. The Simultaneous Perturbation Stochastic Approximation algorithm estimates gradients using only two function evaluations per iteration:

$$\hat{g}_k = \frac{F(\theta_k + c_k \Delta_k) - F(\theta_k - c_k \Delta_k)}{2c_k} \Delta_k$$

where $\Delta_k$ is a Bernoulli $\pm 1$ perturbation vector, and gain sequences follow:

$$a_k = \frac{0.1}{(k+1)^{0.602}}, \quad c_k = \frac{0.05}{(k+1)^{0.101}}$$

The parameter update is:
$$\theta_{k+1} = \theta_k - a_k \hat{g}_k$$

**Tuned parameters** include: consolidation surprise threshold, compression memory threshold, telos goal-directedness weight $\gamma$, activation decay time constant, minimum cluster size, maximum retrieval distance, and abstraction promotion threshold. All are bounded to safe ranges with configurable step sizes.

Updated configurations are applied via lock-free hot-swap (using atomic reference counting), ensuring that background dynamics and foreground queries immediately observe new parameters without restart.

### 8.5 Learned Embedding Projection

To adapt generic embeddings to the agent's specific domain, Memoria trains a linear projection matrix $W \in \mathbb{R}^{d \times d}$ using triplet loss:

$$\mathcal{L} = \frac{1}{N}\sum_{(q,p,n)} \max\left(0, \; d_{\cos}(Wq, Wp) - d_{\cos}(Wq, Wn) + m\right)$$

where $d_{\cos}(a, b) = 1 - \frac{a \cdot b}{\|a\|\|b\|}$ and margin $m = 0.3$.

Triplets are mined from task outcomes: positive pairs come from successful retrievals, negative pairs from failed ones. The matrix is trained via SGD with manual backpropagation through the cosine distance, and applied to all query embeddings at inference time. The projection loss contributes to free energy:

$$E_{\text{proj}} = -\ln\left(\max(1 - \mathcal{L}_{\text{last}}, \epsilon)\right)$$

ensuring that poor projection quality drives the system toward further learning.

---

## 9. Episodic Reflection and Reconsolidation

### 9.1 Surprise-Triggered Reflection

When accumulated unresolved surprise exceeds a configurable threshold (itself tuned by meta-learning), the system initiates a reflection cycle:

1. **Episode prioritization**: Unreflected episodes are ranked by telos alignment:
$$\text{priority}(e) = \max_{g \in \text{active}} \left[\text{sim}(\mathbf{e}, \mathbf{g}) \cdot p_g\right]$$

2. **Pattern extraction**: The LLM analyzes prioritized episode summaries to extract:
   - Semantic facts (S-P-O triples)
   - Entity updates
   - Abstract patterns (promoted to compression level 2)
   - Causal hypotheses (edges with `LlmProposed` mechanism, initial confidence 0.3)

3. **Surprise resolution**: All surprise entries are marked as resolved, resetting the accumulator.

### 9.2 Memory Reconsolidation

When a specific memory has high surprise and high Kalman gain (indicating the belief should shift), the system rewrites the memory content:

1. Identify entities linked to the memory
2. Find contradicting facts for those entities
3. The LLM generates updated content incorporating newer information
4. The updated content is re-embedded and stored as a new temporal version
5. The previous version remains accessible via time-travel queries

This mirrors the neuroscientific process of *memory reconsolidation* [14], where recalled memories enter a labile state and are re-stored with modifications.

---

## 10. Goal System (Telos)

### 10.1 Hierarchical Goals

Goals (termed *telos*, from the Greek for "purpose") are persistent, decomposable objectives that shape the agent's perception and action:

| Depth | Level | Example |
|-------|-------|---------|
| 0 | North Star | "Build a production-ready API" |
| 1 | Strategic | "Implement authentication system" |
| 2 | Tactical | "Set up OAuth2 provider integration" |
| 3 | Operational | "Write token refresh logic" |
| 4 | Task | "Add refresh endpoint to router" |

Goals are decomposed via LLM-driven planning, with each parent spawning 2-5 children at depth + 1.

### 10.2 Attention Allocation

Active goals compete for attention via a formula-driven scoring mechanism:

$$\text{score}(g) = p_g \cdot (1 + u_{\text{eff}}) \cdot \rho(t) + \sigma(t)$$

**Deadline urgency** uses a sigmoid centered 3 days before the deadline:
$$u_{\text{deadline}} = \sigma\left(\frac{t - (d - 3\text{d})}{1\text{d}}\right)$$

**Effective urgency**: $u_{\text{eff}} = \max(u_g, u_{\text{deadline}})$

**Recency penalty** (prevents fixation):
$$\rho(t) = 1 - \exp\left(-\frac{t - t_{\text{last\_attended}}}{24\text{h}}\right)$$

**Staleness bonus** (rescues stuck goals):
$$\sigma(t) = 0.3 \cdot \min\left(\frac{t - t_{\text{stalled}}}{1\text{d}}, 1\right)$$

### 10.3 Progress Estimation

Progress is estimated by blending two signals:

- **Criteria-based**: Fraction of success criteria met ($w = 0.6$)
- **Children-based**: Priority-weighted average of child progress ($w = 0.4$)

$$\text{progress} = 0.6 \cdot \frac{|\text{criteria\_met}|}{|\text{criteria}|} + 0.4 \cdot \frac{\sum_c p_c \cdot \text{progress}_c}{\sum_c p_c}$$

### 10.4 Multi-Agent Coordination

In multi-agent deployments, the telos system supports:

- **Team visibility**: All active goals visible within a namespace
- **Goal claiming**: Agents claim ownership of unclaimed goals; `Proposed` goals automatically activate upon claim
- **Delegation**: Creating subtelos assigned to specific agents
- **Conflict detection**: Goals with embedding similarity $\geq 0.9$ are flagged as potential duplicates
- **Enterprise directives**: External policies converted to depth 0-1 telos with pinned confidence (no decay)

---

## 11. Cross-Agent Trust and Access Control

### 11.1 Scope Grants

Access is governed by pattern-based grants mapping (agent\_pattern × namespace\_pattern) to permission sets. Grants support wildcards and carry optional expiration timestamps.

### 11.2 Trust Scoring

When an agent accesses memories from another agent's namespace, trust is computed as a weighted composite:

$$\text{trust} = 0.4 \cdot f_{\text{grant}} + 0.3 \cdot f_{\text{prov}} + 0.3 \cdot f_{\text{acc}}$$

where:
- $f_{\text{grant}} = 0.9$ if a direct scope grant exists, $0.3$ otherwise
- $f_{\text{prov}} = \text{mean}(w_{\text{prov}(m)})$ for all memories $m$ in the namespace
- $f_{\text{acc}} = \text{success\_count} / \text{total\_tasks}$ in the namespace

### 11.3 Kernel Rules

Unbypassable rules enforced at the store level (below the API layer) implement hard constraints:

- **Deny**: Prevent specific operations on matched patterns
- **Require**: Mandate fields or conditions before mutations
- **Protect**: Prevent modification of critical memories
- **Redact**: Strip sensitive fields from query results

Kernel rules are encoded as factor messages with $\mu = -\infty$, ensuring they override all other scoring signals (Section 4.2).

---

## 12. Plan-Execute-Summarize Cognitive Loop

The complete cognitive loop integrates all subsystems:

**Plan**: Select relevant skills via EFE (Section 7.2), retrieve context via `prime()`, and build a task plan with LLM assistance using skill lineage (ancestry up to 5 generations) as context.

**Execute**: The agent executes the plan using selected skills, recording observations via `tell()`.

**Summarize**: A five-point abductive reflection analyzes the outcome:

1. Plan-outcome alignment (which steps succeeded/failed)
2. Divergence analysis (why deviations occurred)
3. Causal hypotheses (2-3 ranked by plausibility)
4. Counterfactual analysis (highest-leverage intervention point)
5. Actionable advice (specific to next similar task)

The summary is embedded and stored as a `TaskOutcome`, feeding into future skill selection, Hebbian weight updates, and projection training.

---

## 13. Implementation

Memoria is implemented in approximately 31,000 lines of Rust, organized into 8 major subsystems: `aif` (Active Inference), `api` (public interface), `causal` (structural causal models), `dynamics` (background processes), `pipeline` (scoring and chunking), `queue` (task management), `skills` (procedural memory), and `store` (database abstraction).

The system is validated by 72+ tests across 16 integration test suites and 20+ unit test modules. Integration tests exercise the full pipeline with real embedding models, named entity recognizers, rerankers, and language models. Test fixtures include 100+ Wikipedia passages (SQuAD), 10 factual contradiction pairs, and 5 task pattern sequences for skill crystallization.

Key implementation decisions:

- **Lock-free concurrency**: Hot cache (DashMap), config hot-swap (ArcSwap), and shared services (Arc) enable safe concurrent access without mutex contention.
- **Log-domain arithmetic**: All scoring computations use $\ln(\cdot)$ to prevent underflow in probability products.
- **Optimistic locking**: The task queue uses atomic CAS (compare-and-swap) on task status to prevent duplicate processing.
- **Zero unsafe code**: No `unsafe` blocks in the library crate.

---

## 14. Discussion

### 14.1 Relation to Prior Work

**Memory-augmented LLM agents.** Systems such as MemGPT [15], Mem0, and Zep provide vector-indexed memory stores with heuristic summarization. These systems lack a unifying objective function — decisions about what to store, when to forget, and how to score relevance are based on hand-tuned rules rather than principled optimization. Memoria's free energy objective provides a single criterion that governs all these decisions.

**Active Inference in AI.** Prior work has applied AIF to robotics [16] and simple navigation agents [17]. To our knowledge, Memoria is the first system to apply AIF comprehensively to an LLM agent's memory system, including the full loop from surprise detection through intrinsic goal generation to skill crystallization.

**Causal reasoning in agents.** Systems like Causal-Agent [18] use causal models for planning but do not *learn* causal structure from experience. Memoria combines online Bayesian edge accumulation with periodic NOTEARS validation and counterfactual attribution, enabling the causal model to evolve as the agent gathers evidence.

**Meta-learning.** SPSA has been applied to hyperparameter optimization in various contexts [6], but its application to continuous self-tuning of a memory system's dynamics parameters — where the objective (free energy) is itself a function of the memory state — is, to our knowledge, novel.

### 14.2 Limitations

**LLM dependency for qualitative reasoning.** Seven integration points require LLM calls, which are slow and expensive. The system degrades if the LLM is unavailable, though all quantitative operations (scoring, surprise, decay, meta-learning) continue without LLM access.

**Embedding quality.** The system assumes that embedding similarity correlates with semantic relevance. When this assumption fails (e.g., for domain-specific terminology), the learned projection (Section 8.5) provides partial mitigation, but cold-start quality depends on the base embedder.

**Scale.** The current implementation has been tested with stores up to ~10,000 memories. Performance characteristics at 100K+ memories — particularly for HNSW search, PageRank computation, and NOTEARS structure learning — require further investigation.

### 14.3 Future Directions

**Federated memory.** Extending the multi-agent scope system to support cross-organization knowledge sharing with differential privacy guarantees.

**Neuromorphic dynamics.** Replacing the discrete tick-based scheduler with event-driven dynamics that fire in response to state changes, more closely mirroring biological neural circuits.

**Hierarchical active inference.** Extending the single-level Bethe approximation to a hierarchical generative model with multiple timescales, enabling long-horizon planning grounded in the same free energy objective.

---

## 15. Conclusion

We have presented Memoria, a self-evolving memory runtime that grounds all agent memory operations in the Active Inference Framework. By computing Bethe free energy over the agent's knowledge store, deriving exploration-exploitation balance from the entropy-energy ratio, generating intrinsic goals from surprise hotspots, and selecting skills via Expected Free Energy, Memoria provides a *principled* alternative to the ad hoc heuristics that characterize existing agent memory systems.

The system demonstrates that AIF is not merely a theoretical framework but a practical architectural principle that can unify diverse memory operations — from temporal decay to causal reasoning to meta-learning — under a single variational objective. We believe this approach represents a significant step toward agents that genuinely learn from experience, rather than merely storing and retrieving information.

---

## References

[1] K. Friston, "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, vol. 11, no. 2, pp. 127-138, 2010.

[2] T. Parr, G. Pezzulo, and K. J. Friston, *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press, 2022.

[3] X. Zheng, B. Aragam, P. Ravikumar, and E. P. Xing, "DAGs with NO TEARS: Continuous optimization for structure learning," *Advances in Neural Information Processing Systems*, vol. 31, 2018.

[4] R. D. Shachter, "Bayes-Ball: Rational pastime (for determining irrelevance and requisite information in belief networks and influence diagrams)," *Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence*, pp. 480-487, 1998.

[5] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press, 2009.

[6] J. C. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation," *IEEE Transactions on Automatic Control*, vol. 37, no. 3, pp. 332-341, 1992.

[7] Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, pp. 824-836, 2020.

[8] J. S. Yedidia, W. T. Freeman, and Y. Weiss, "Constructing free-energy approximations and generalized belief propagation algorithms," *IEEE Transactions on Information Theory*, vol. 51, no. 7, pp. 2282-2312, 2005.

[9] K. J. Friston, L. Da Costa, D. Hafner, C. Hesp, and T. Parr, "Sophisticated inference," *Neural Computation*, vol. 33, no. 3, pp. 713-763, 2021.

[10] H. Ebbinghaus, *Memory: A Contribution to Experimental Psychology*. Teachers College, Columbia University, 1913.

[11] J. G. Cleary and I. H. Witten, "Data compression using adaptive coding and partial string matching," *IEEE Transactions on Communications*, vol. 32, no. 4, pp. 396-402, 1984.

[12] R. J. Hyndman and Y. Khandakar, "Automatic time series forecasting: the forecast package for R," *Journal of Statistical Software*, vol. 27, no. 3, pp. 1-22, 2008.

[13] R. P. Adams and D. J. C. MacKay, "Bayesian online changepoint detection," *arXiv preprint arXiv:0710.3742*, 2007.

[14] K. Nader, G. E. Schafe, and J. E. LeDoux, "Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval," *Nature*, vol. 406, no. 6797, pp. 722-726, 2000.

[15] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, "MemGPT: Towards LLMs as operating systems," *arXiv preprint arXiv:2310.08560*, 2023.

[16] G. Oliver, P. Lanillos, and G. Cheng, "An empirical study of active inference on a humanoid robot," *IEEE Transactions on Cognitive and Developmental Systems*, vol. 14, no. 2, pp. 462-471, 2022.

[17] A. Fountas, N. Sajid, P. A. M. Mediano, and K. Friston, "Deep active inference agents using Monte-Carlo methods," *Advances in Neural Information Processing Systems*, vol. 33, 2020.

[18] A. Kıcıman, R. Ness, A. Sharma, and C. Tan, "Causal reasoning and large language models: Opening a new frontier for causality," *arXiv preprint arXiv:2305.00050*, 2023.
