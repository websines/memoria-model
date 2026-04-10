# Memoria Training Architecture

## Thesis

Every AI system today has amnesia. Every conversation resets to zero. Every "memory" system is the same trick: stuff retrieved text into the prompt. The model itself never changes, never learns, never adapts.

Memoria's thesis: a model with the right *architecture* — persistent memory, causal graphs, active inference, self-improvement — can punch above its weight class against models 10-100× larger. The scaling law has a massive architectural constant that nobody is optimizing.

```
Industry bet:    capability ∝ N^0.076    (more params = more capability)
Memoria's bet:   capability ∝ N^0.076 × f(architecture, state, experience)
```

Where `f()` might be worth 1000× of raw parameters.

## Novel Contributions

### Log-Linear DeltaProduct (H layers)

The backbone recurrent layer is a novel composition of two research contributions that has not been implemented before:

- **DeltaProduct** (NeurIPS 2025): Multi-Householder error-correcting state updates. Each token triggers n_h=3 steps of online gradient descent on the recurrent state via Householder reflections. The state actively *learns* from prediction errors at every token.
- **Log-Linear Attention** (ICLR 2026): Fenwick tree hierarchical state management. Instead of one fixed-size state, maintains ~log₂(T) states at different temporal resolutions. Recent tokens get high-res individual states, distant tokens are compressed into coarser levels.

**Nobody has combined these.** The Log-Linear paper (Guo, Yang, Dao, Kim) tested their framework only with Mamba-2 and Gated DeltaNet (single Householder step, n_h=1). DeltaProduct (Siems et al.) used only flat single-state recurrence. We compose them: each level of the Fenwick tree gets error-corrected via 3 Householder reflections. The result is hierarchical error-correcting recurrence — a recurrent layer that both *learns* at every token AND maintains multi-scale temporal context.

**Implementation approach**: We use FLA's optimized Triton kernel (`chunk_gated_delta_product`) for the intra-chunk computation, and manage the inter-chunk Fenwick tree hierarchy in PyTorch. The two modifications are orthogonal — DeltaProduct's Householder interleaving affects intra-chunk token processing, the Fenwick tree affects inter-chunk state propagation. No custom Triton kernel needed.

### Learning Continuum

The architecture creates a continuous learning hierarchy with no gaps between timescales:

| Timescale | Mechanism | What it learns | State type |
|-----------|-----------|---------------|------------|
| **Every token** | DeltaProduct Householder steps | Fast error correction (entity changed, fact updated, pronoun resolved) | KV memory matrix per Fenwick level |
| **Every few tokens** | Fenwick tree level merges | Automatic temporal consolidation (recent detail → coarser long-term) | Hierarchical state promotion |
| **Every chunk** | TTT gradient steps | Deep structural adaptation (domain patterns, user style) | MLP weight deltas |
| **Every step** | Pass 2 (12 operations) | Discrete structure (new belief, new edge, goal transitions) | Cognitive state slots |

Previous architectures have gaps: Mamba-2 compresses passively between TTT steps (no learning). Transformers have no within-sequence learning at all. DeltaProduct fills the token-level gap, and the Fenwick tree adds the automatic consolidation level that mirrors what Pass 2's sleep/consolidation does for beliefs.

### Five Memory Systems

| System | Mechanism | Range | Scaling | Timescale |
|--------|-----------|-------|---------|-----------|
| **MLA + DSA** | Belief-conditioned sparse attention | Full context (sparse) | O(T×K) | Immediate |
| **DeltaProduct state** | Error-correcting KV matrix | Unlimited | O(1) per level | Continuous |
| **Fenwick hierarchy** | Log-growing temporal levels | Unlimited | O(log T) | Multi-scale |
| **Cognitive state** | Explicit beliefs + edges + goals | Unlimited | O(1) fixed slots | Persistent (cross-session) |
| **Engram** | Hash-based N-gram lookup (token mode) | Unlimited | O(1) | Static |

Each system handles something the others cannot. MLA+DSA gives exact sparse recall across the full context, guided by cognitive state beliefs. DeltaProduct error-corrects within-sequence state. The Fenwick hierarchy provides multi-scale temporal context. Cognitive state stores discrete facts that survive across sessions and directs MLA's sparse attention. Engram handles static patterns at zero cost. No single system is redundant.

**BLT mode:** When `blt_enabled=True`, EngramCache is reconfigured for byte IDs (vocab=260, hidden_dim=local_dim=384) and hashes byte N-grams ("th", "the", "ing") instead of token N-grams. Its output is injected into `byte_hidden` (the encoder's skip connection) so both the encoder's DeltaProduct layers AND the decoder benefit. This stacks with the ByteEncoder's learned causal N-gram conv — EngramCache provides O(1) hash-table patterns, the conv provides learned local context.

### Self-Modifying Architecture

Memoria is a self-modifying architecture: the system changes its own compute graph — the number of nodes, the topology of edges, the optimization objective, and the parameters governing those modifications — at runtime. This is not parameter tuning within a fixed graph. The graph itself grows, splits, prunes, and rewires based on what the system learns.

**11 self-modification mechanisms, organized in a recursive hierarchy:**

| Layer | Mechanism | What changes at runtime | Module |
|-------|-----------|------------------------|--------|
| 0 | Belief allocation/eviction | Number of active nodes in factor graph | `belief_update.py`, `state.py` |
| 1 | Edge proposal/pruning | Causal graph topology (which beliefs connect) | `edge_proposal.py`, `state.py` |
| 2 | Structural plasticity | Belief splitting (polysemantic → children) and pruning | `structural_plasticity.py` |
| 3 | Provisional beliefs + autoresearch | Hypothesis beliefs generated, tested, promoted or killed; PARL fair round-robin allocation | `autoresearch.py`, `provisional.py` |
| 4 | SRWM | Meta-parameters become state-dependent via fast-weight matrix | `srwm.py` |
| 5 | Cognitive controller | PARL staged reward: r_perf + r_parallel + r_finish (annealed). Learned policy over rates of layers 0-4 | `cognitive_controller.py` |
| 6 | SGM safety gate | Statistical validation of modifications (e-value testing) | `safety_gate.py` |
| 7 | Telos goal generation | Creates new goals from surprise → changes what the system optimizes for | `telos_module.py` |
| 8 | Adaptive depth | Variable computation depth per belief (ACT halting) + per-position refinement routing (MoR) | `adaptive_depth.py`, `memoria_model.py:RefinementRouter` |
| 9 | Message passing + dream | Belief positions shift via loopy BP on the causal graph | `message_passing.py` |
| 10 | Parallel goal pursuit | Per-head goal routing in read path (MoH-style); different heads pursue different goals simultaneously | `read_path.py:GoalRouter` |
| 11 | BLT byte encoding | Learned byte→patch compression replaces fixed tokenizer; N-gram conv replaces EngramCache hash tables | `blt.py:ByteEncoder` |

**The recursive autoresearch loop.** These mechanisms don't operate independently — they form a closed loop of self-directed architectural modification:

```
Telos generates goals from surprising beliefs
  → Autoresearch generates hypothesis beliefs in goal directions
    → Hypotheses participate in the forward pass as provisional beliefs
      → Free energy measures their impact on prediction quality
        → Promote if FE improved + precision held; evict otherwise
          → Promoted beliefs change the cognitive state
            → Changed state triggers new edge proposals (graph rewires)
              → Changed state modulates SRWM fast-weight matrix
                → SRWM modulates the parameters governing hypothesis generation
                  → Back to step 1 with different operating point
```

This is genuine recursive self-modification: the system modifies how it modifies itself. The SRWM changes the thresholds used by structural plasticity, the cognitive controller changes the rates at which plasticity operates, the SGM safety gate validates everything with sequential hypothesis testing, and Telos creates the goals that drive the entire loop. Each pass through the loop operates at a different point in parameter space because the previous pass changed the SRWM state.

**Relationship to Godel machines.** The SGM safety gate implements the key requirement from Schmidhuber's Godel machine: modifications are only deployed when there is statistical evidence (e-values, Vovk & Wang) that they improve the system. The difference: Godel machines are theoretical (require proof search over all possible self-modifications). Memoria's loop is gradient-driven and empirically validated — the autoresearch loop is a learned, differentiable approximation of proof search over the belief space.

### Huber-Robust Belief Matching (MIRAS/YAAD)

All belief matching losses use Huber loss instead of raw cosine disagreement. The Huber transition point (`huber_delta`) is a learned MetaParam (softplus, init 0.5).

- **Below delta**: quadratic penalty — precise gradients for small belief-observation disagreements
- **Above delta**: linear penalty — caps gradient contribution from outlier matches

**Why this matters for self-modification.** The write gate fires on ~12% of tokens. Over 500B training tokens, that's ~60B match operations. If 1% are spurious matches (cosine outlier sensitivity), that's 600M bad belief updates poisoning the persistent cognitive state. Beliefs are persistent — a bad update at token 50M is still there at token 500B unless consolidation catches it. Huber matching is cheap insurance (one MetaParam, zero architectural changes) that keeps the cognitive state clean over long training runs. The cleaner the state, the better the recursive self-modification loop works — every mechanism in the hierarchy depends on belief quality.

Applied in `compute_differentiable_free_energy` (L_fe_proxy energy term) and `compute_expected_free_energy` (EFE risk term). Reference: MIRAS/YAAD (arXiv:2504.13173); Huber (1964).

### PARL-Style Internal Parallel Goal Pursuit

Multiple active goals are pursued simultaneously through three coordinated mechanisms:

**1. Multi-head goal routing (GoalRouter).** Each read path head is softly assigned to a different active goal via Gumbel-Softmax routing. With H=4 heads and G=3 goals, head 0 retrieves beliefs relevant to goal A, head 1 retrieves for goal B, head 2 for goal C, and head 3 gets shared (uniform). The router learns when to specialize heads vs. when to share them. Zero-init output → starts with uniform routing (backward compatible), learns to specialize.

**2. Batched autoresearch with fair allocation.** Hypothesis generation is already batched across goals (one forward pass through `HypothesisGenerator`). Allocation is now fair round-robin: available slots are distributed proportionally across viable goals weighted by hypothesis success EMA, instead of first-come-first-served which starves later goals.

**3. PARL staged reward shaping.** The cognitive controller receives three reward components:
- `r_perf`: belief_advantage + state deltas (always active)
- `r_parallel`: rewards diverse goal pursuit (normalized entropy over per-goal hypothesis counts)
- `r_finish`: penalizes abandoned goals (prevents spurious parallelism from gaming r_parallel)

`r_parallel` and `r_finish` are multiplied by `(1 - training_progress)` — they anneal to zero over training. This teaches the controller that parallelism is genuinely better for performance, then removes the crutch. All weights are learned MetaParams.

**Anti-serial-collapse mechanism.** Without PARL, the controller naturally collapses to serial goal pursuit because investing in one goal reduces reward variance. The staged reward breaks this equilibrium during early training (when r_parallel > 0), and by the time it anneals out, the policy has learned the structural advantages of parallel pursuit.

State encoding extended to 10 features (was 8): +goal_diversity (normalized entropy), +goal_completion_rate.

Reference: PARL (Kimi K2.5, arXiv:2602.02276) — staged reward, serial collapse prevention
Reference: MoH (arXiv:2410.11842) — mixture-of-head attention routing
Reference: MOORE (arXiv:2311.11385) — orthogonal expert specialization
Reference: D3PO (arXiv:2602.07764) — provable diversity regularization
Reference: GCR-PPO (arXiv:2509.14816) — per-objective gradient decomposition

### Belief-Conditioned Sparse Attention (DSA)

MLA layers use a learned Lightning Indexer for sparse global attention, extended with belief conditioning — a novel contribution. The cognitive state directly influences which tokens the transformer attends to, making attention goal-directed via active inference. Documented in detail in the MLA section below.

Key optimizations:
- **Chunk-based scoring**: O(T×C×H) memory instead of O(T²×H). Never materializes the full score matrix.
- **Triton fused kernel**: `_fused_chunk_score_kernel` fuses dot product + ReLU + weighted sum + belief bias + causal mask into one `@triton.jit` kernel per chunk. Automatic PyTorch fallback on CPU.
- **Causal masking in sparse path**: Explicit attention mask prevents future-token leakage at early positions.
- **KL alignment only in dense path**: Avoids self-referential training signal at long context.

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
┌─ Transformer Block 0 (DeltaProduct / Log-Linear GDN / MLA) ──┐
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

## Backbone Components

### Layer Types ("HHHHL" Pattern)

The `window_pattern` string defines per-layer architecture. Default "HHHHL" = 4 Log-Linear DeltaProduct₃ + 1 MLA per 5-layer cycle.

| Type | Class | Scaling | When to use |
|------|-------|---------|-------------|
| **H** (Log-Linear DeltaProduct) | `LogLinearDeltaProductBlock` | O(T log T) | **Default — error-correcting recurrence + Fenwick tree hierarchy (novel)** |
| **D** (DeltaProduct) | `DeltaProductBlock` | O(T) linear | Flat error-correcting recurrence (fast training fallback) |
| **E** (Log-Linear GDN) | `LogLinearGDNBlock` | O(T log T) | Hierarchical GDN with fused kernel (fast + hierarchical fallback) |
| **S** (Sliding Window) | `SlidingWindowAttention` | O(T×W) | Legacy — local attention with RotorQuant KV compression |
| **L** (MLA Global) | `MLACausalSelfAttention` | O(T²) or O(T×W) | Periodic dense attention with latent KV compression |

Available patterns:
| Pattern | Description | Training Speed | Expressivity |
|---------|-------------|---------------|-------------|
| **HHHHL** | 4 Log-Linear DeltaProduct₃ + 1 MLA | Moderate (sequential chunks) | **Maximum** |
| DDDEL | 3 DeltaProduct₃ + 1 Log-Linear GDN + 1 MLA | Fast (all fused kernels) | High |
| DDDML | 3 DeltaProduct₃ + 1 MLA | Fastest | Good |

### GatedDeltaProduct₃ (D layers)

Wraps `fla.layers.GatedDeltaProduct` — error-correcting recurrence via products of Householder reflections with gated forgetting:

```
For each token t, apply n_h=3 Householder reflection steps:
  H_{t,j} = (I - β_{t,j} · k_{t,j} · k_{t,j}ᵀ) · H_{t,j-1} + β_{t,j} · k_{t,j} · v_{t,j}ᵀ
With forget gate:
  H_t = g_t · H_{t,3}
Output:
  o_t = H_t · q_t
```

Each step computes "what did I predict at key k vs what I see as value v" and corrects the state to reduce that error. This is online gradient descent on an associative recall loss — the recurrent layer *learns* at every token, not just compresses.

**Eigenvalue range [-1,1] is critical.** With β ∈ [0, 2], eigenvalues of each Householder reflection span [-1, 1]. This enables the state to represent permutations, negation, and cyclic state transitions — operations that real-valued diagonal SSMs (Mamba-2) fundamentally cannot express. The DeltaProduct paper shows that models restricted to [0, 1] eigenvalues completely fail at state-tracking benchmarks regardless of depth or n_h. This is THE architectural reason DeltaProduct succeeds at entity tracking where Mamba-2 fails.

**Three Householder reflections (n_h=3)** can approximate any orthogonal state transition matrix (Cartan-Dieudonné theorem). This means the recurrent layer can represent any rotation, reflection, or permutation of its internal state — the maximum expressivity achievable with O(1) inference state.

- `num_householder=3`: Householder reflections per token (the expressivity knob)
- `head_dim=128`: key dimension per head
- `expand_v=2`: value dimension = head_dim × 2
- `allow_neg_eigval=True`: enables [-1,1] eigenvalue range (REQUIRED)
- `conv_size=4`: short causal convolution (captures immediate context)
- `use_forget_gate=True`: scalar forget gate g_t ∈ [0,1]
- **State**: d_k × d_v matrix per head (key-value memory, not a diagonal vector)
- **No KV cache**: recurrent state is fixed size regardless of sequence length
- **No RoPE needed**: sequence ordering implicit in recurrence
- **Spectral norm ≤ 1**: guaranteed by Householder construction, ensures stability at arbitrary sequence lengths (critical for SkyLadder progressive context extension)
- **Length extrapolation**: minimal degradation extrapolating from training length to 4×+ (DeltaProduct paper, Figures 8, 16-18)
- Drop-in replacement for attention: `[B, T, D] → [B, T, D]`

**Learning hierarchy with TTT**: DeltaProduct handles fast, shallow error corrections at every token (entity state changed, fact updated, pronoun resolved). TTT handles deeper structural adaptations at chunk boundaries (domain adaptation, user-specific patterns). Pass 2 handles discrete operations (new belief, new edge) that no gradient can express. This creates a learning continuum across timescales with no gap between token-level and chunk-level learning.

Reference: DeltaProduct (NeurIPS 2025, arXiv:2502.10297)
Reference: Gated Delta Networks (ICLR 2025, arXiv:2412.06464)
Library: `flash-linear-attention` (fla-org/flash-linear-attention)

### Log-Linear GDN (E layers)

Wraps `hattention.HGatedDeltaNet` — Gated DeltaNet enhanced with a logarithmically growing hierarchical state via Fenwick tree decomposition:

```
Standard GDN: 1 fixed-size state S_t per head
Log-Linear GDN: ~log₂(T) states {S_t⁰, S_t¹, ..., S_tᴸ} per head

Each level covers an exponentially larger chunk of history:
  Level 0: current token (size 1)
  Level 1: last 1 token
  Level 2: last 2 tokens
  Level 3: last 4 tokens
  ...
  Level L: last 2^(L-1) tokens

Output: o_t = Σ_l λ_t^(l) · q_tᵀ · S_t^(l)
  where λ_t^(l) are data-dependent level weights (learned)
```

At 200K tokens: log₂(200K) ≈ 17 levels. Recent tokens get individual high-resolution states, distant tokens are compressed into coarser levels. The data-dependent λ weights let the model learn how much to trust each temporal scale.

**Why a separate layer type (not applied to DeltaProduct)**: Log-Linear is an orthogonal enhancement that can theoretically "lift" any linear attention mechanism. However, Log-Linear DeltaProduct (combining Fenwick tree with multi-Householder products) has not been implemented — it would require a custom Triton kernel merging `chunk_gated_delta_product` with `chunk_h_gated_delta_rule`. Using Log-Linear GDN (n_h=1) as a dedicated E layer type gives us the hierarchical memory benefit now, while DeltaProduct₃ layers handle the error-correction role. Future work: merge them into a single Log-Linear DeltaProduct layer.

**Inductive bias**: The Fenwick tree compresses distant tokens more aggressively and keeps recent tokens at high resolution. This complements MLA (exact local attention) and cognitive state (exact distant recall via beliefs). The E layer provides a middle ground — approximate multi-scale context that neither the D layers (fixed-size state, no temporal hierarchy) nor the L layers (exact but windowed) offer.

- O(T log T) compute, O(log T) state per head
- Surpasses FlashAttention-2 throughput at sequences > 8K
- GDN base with single-step delta rule (n_h=1) + gated forgetting
- Data-dependent level scales via learned projection + per-head learnable L parameter
- No KV cache — hierarchical compressed state instead

Reference: Log-Linear Attention (ICLR 2026, arXiv:2506.04761)
Reference: Gated Delta Networks (ICLR 2025, arXiv:2412.06464)
Library: `hattention` (HanGuo97/log-linear-attention) + `flash-linear-attention`

### MLA (L layers) — Multi-Head Latent Attention

DeepSeek V3-style compressed attention with decoupled RoPE:

```
x → c_kv_compress → latent [B, T, d_latent]     (position-invariant, cacheable)
latent → k_up → K_nope [B, T, n_kv, head_dim - d_rope]   (content keys)
x → c_k_rope → K_rope [B, T, n_kv, d_rope]               (positional keys, small)
K = cat(K_rope_with_RoPE, K_nope)                          (full key)
latent → v_up → V [B, T, n_kv, head_dim]                  (values)
```

- `mla_latent_dim=192`: KV compression bottleneck (vs full 768-dim)
- `mla_rope_dim=64`: only 64 dims carry positional encoding
- `mla_window_size=0`: full causal O(T²); set >0 for windowed O(T×W) at long context
- Windowed MLA uses RotorQuant KV compression + blockwise attention

#### DSA: Belief-Conditioned Sparse Attention (Lightning Indexer)

When `dsa_enabled=True`, MLA layers use a learned **Lightning Indexer** for sparse global attention instead of windowed or full causal. Based on DeepSeek-V3.2 DSA (arXiv:2512.02556), extended with belief conditioning — a novel contribution.

```
                    ┌─ Lightning Indexer ─────────────────────────┐
hidden [B,T,D] ──→ │ q_I = q_proj(hidden)  [B, T, H_I, d_I]    │
                    │ k_I = k_proj(hidden)  [B, T, H_I, d_I]    │
                    │ k_I = RotorQuant_STE(k_I)  ← 3-bit QAT    │
                    │                                              │
                    │ score(t,s) = Σ_j w(t,j)·ReLU(q_I·k_I)     │
                    │            + λ · max_b sim(k_I, belief_b)   │
                    │                  ↑ belief conditioning       │
                    │                                              │
                    │ indices = top_k(scores)  [B, T, K]          │
                    └──────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─ Sparse MLA Attention ──────────────────────┐
                    │ k_sparse, v_sparse = gather(K, V, indices)  │
                    │ output = softmax(Q · k_sparse / √d) · v     │
                    └──────────────────────────────────────────────┘
```

**Score function** (per indexer head j):
```
I(t, s) = Σ_j w(t,j) · ReLU( q_I(t,j) · k_I(s,j) )
         + λ_belief · max_b cosine_sim(k_I(s), belief_proj(b))
```

- `dsa_index_dim=32`: indexer projection dimension (tiny — cheap scoring)
- `dsa_index_heads=4`: parallel scoring channels (each head can specialize)
- `dsa_top_k=2048`: tokens selected per query at inference (out of 1M)
- `dsa_top_k_ratio=0.25`: fraction selected during training (indexer always has to choose)
- `dsa_index_bits=3`: RotorQuant bits for indexer key compression (STE QAT)
- `dsa_belief_lambda=0.1`: belief conditioning strength

**Belief conditioning** (novel): Active beliefs from the cognitive state are projected into indexer space. Their max-similarity to each token's indexer key adds an additive bias. This means MLA preferentially attends to tokens relevant to current beliefs — attention becomes goal-directed via active inference. No existing architecture does this because no other architecture has a cognitive state to condition on.

**Three attention modes** (selected per-layer based on config):

| Mode | When | Complexity | Coverage |
|------|------|-----------|----------|
| Full causal | `dsa_enabled=False, mla_window_size=0` | O(T²) | Full, dense |
| Windowed | `dsa_enabled=False, mla_window_size>0` | O(T×W) | Dense within window |
| **DSA sparse** | `dsa_enabled=True` | O(T×C×H) scoring + O(T×K) attention | **Sparse global** |

**Chunk-based scoring**: The indexer never materializes a T×T score matrix. Keys are processed in chunks of C=256: each chunk produces a `[B, T, H, C]` score tensor, reduced across heads to `[B, T, C]`, then merged with a running top-k buffer via `torch.topk` on `[B, T, k+C]`. Memory stays at O(T×max(C,k)×H) regardless of context length. At T=1M, C=256, H=4: peak scoring memory is ~4MB per batch vs ~4TB for the naive T² approach.

**Triton fused scoring kernel**: When Triton is available (CUDA training), `_fused_chunk_score_kernel` fuses the per-chunk dot product + ReLU + head-weighted sum + belief bias + causal mask into a single `@triton.jit` kernel. One program per (b, t) query per chunk; accumulates in float32 regardless of input dtype (bf16 safe). `HAS_BB` is `tl.constexpr` — dead-code-eliminated when no beliefs. The top-k merge remains in PyTorch (`torch.topk` uses radix selection which is hard to beat). On CPU or without Triton, `_topk_pytorch` provides an identical pure-PyTorch fallback. Dispatch is automatic via `_triton_available and hidden.is_cuda`.

**KL alignment loss**: During phase 1 (short context, dense attention is cheap), the indexer is trained via KL divergence against the dense MLA attention distribution. This teaches it "which tokens would full attention focus on?" At long context (sparse path), KL is skipped — computing the dense target would defeat the purpose of sparse attention, and training the indexer against its own selection is self-referential. The phase 1 KL weight (default 1.0) reduces to maintenance level (0.1) in phase 2+3.

**Causal masking in sparse path**: When gathered keys include future positions (early query positions where t < k), an explicit attention mask zeros out those entries. This prevents information leakage from future tokens even though the indexer's causal mask already biases against selecting them.

**RotorQuant on indexer keys**: Same STE QAT pipeline as KV cache, weights, and beliefs. At d_I=32, RotorQuant's Givens rotations (3-bit: 16 pairs) or quaternion rotations (4-bit: 8 groups) fit perfectly. The model learns indexer representations that survive 3-bit compression.

Memory overhead at 1M tokens:
- Indexer keys (3-bit RotorQuant): 1M × 32 × 0.375 bytes = **12 MB**
- Scoring peak (chunk-based): ~4 MB per batch element
- vs current windowed KV: ~38 MB (saved, since DSA replaces windowed path)

Reference: DeepSeek-V3.2 DSA (arXiv:2512.02556), NSA (arXiv:2502.11089, ACL 2025)

### Engram Cache (O(1) Static Knowledge)

Hash-based N-gram lookup inspired by DeepSeek Engram (arXiv:2601.07372):

1. **Tokenizer compression**: NFKC + lowercase + accent strip ("Apple"/"apple"/"APPLE" → same ID, ~23% vocab reduction)
2. **N-gram hashing**: multiplicative-XOR hash for 2-grams and 3-grams across `n_heads` hash tables
3. **Retrieval**: `hash % table_size` → embedding lookup (O(1) per token)
4. **Context-aware gating**: `gate = sigmoid(sqrt_sign(hidden · retrieved / √d))` — suppresses hash collisions

Injected once at layer 0, before any transformer block. Handles static patterns (function signatures, import idioms, common phrases) so beliefs can focus on dynamic/experiential memory.

### Working Memory Suffix

`M` learnable tokens (default 8) appended to the hidden stream:

- Real tokens cannot attend to WM (causally after them)
- WM can attend to all real tokens
- During refinement loops: real tokens anchored by lifeline, WM evolves freely
- Acts as a vector scratchpad — richer than text-based chain-of-thought
- Initialized `randn(...) × 0.02`, trained by gradient descent

### RoPE Configuration

- `rope_base=500000`: High base frequency for native long context (Llama 3 style)
- `max_position=204800`: RoPE extrapolation ceiling (~200K)
- `rope_scaling="none"`: Native (scratch) or `"yarn"` (pretrained extension)
- Applied partially in MLA: only first `d_rope` dims of Q get rotated; K_nope is position-invariant

## Cognitive State Structure

The cognitive state is a persistent, slot-based world model with four regions:

### Beliefs (World Model)

- `beliefs`: `[max_beliefs, D]` — nn.Parameter, trained by optimizer
- **Polar representation**: radius = precision (confidence), angle = content direction
- Empty slots: radius ≈ 0 (available for allocation)
- Dot products between beliefs are naturally precision-weighted (large radius dominates)
- Typical: 16K-65K slots, 256-dim

### Edges (Causal/Associative Relations)

- `edge_src/tgt`: `[max_edges]` — source/target belief indices
- `edge_relations`: `[max_edges, K]` — learned relation semantics (K=64)
- `edge_weights`: `[max_edges]` — edge strength
- `edge_causal_obs`: `[max_edges]` — causal observation count (0 = Hebbian/associative)
- `edge_direction`: `[max_edges]` — CoED learned causal direction angle
- Typical: 65K-262K edges

### Goals (Telos — Emergent Objectives)

- `goal_embeddings`: `[max_goals, D]` — goal direction vectors
- `goal_metadata`: `[max_goals, 8]` — priority, urgency, progress, status, depth, surprise, created, deadline
- `goal_status_logits`: `[max_goals, 6]` — Gumbel-Softmax over {empty, proposed, active, stalled, completed, failed}
- Typical: 64-512 goals

### Meta Region

- `meta`: `[32]` — learned meta-parameters (β, accumulated_surprise, thresholds, etc.)
- `meta_params`: `MetaParams` — 72 learned constants replacing hardcoded hyperparameters throughout the system (sigmoid/softplus constrained, initialized to original values)

### Belief Metadata Buffers

Per-belief tracking (all `[max_beliefs]`):

| Buffer | Type | Purpose |
|--------|------|---------|
| `belief_level` | int | Abstraction level L0-L3 (SDFT hierarchy) |
| `belief_last_accessed` | int | Step of last retrieval |
| `belief_access_count` | int | Total retrieval count |
| `belief_prev_surprise` | float | Last surprise score |
| `belief_lr_scale` | float | Per-belief adaptive LR (RWKV-7 inspired) |
| `belief_provisional` | bool | Provisional (hypothesis, not yet confirmed) |
| `belief_provisional_step` | int | Step when provisionally allocated |
| `belief_provisional_fe` | float | Free energy at allocation time |
| `belief_provisional_radius` | float | Radius at allocation time |
| `belief_precision_var` | float | MESU precision variance |
| `belief_reinforcement_count` | int | Times reinforced by observations |
| `belief_confirmed_count` | int | Confirmations |
| `belief_contradicted_count` | int | Contradictions |
| `belief_sources` | int | Source belief index (provenance chain) |
| `belief_source_type` | int | How it was created (observation, consolidation, hypothesis) |
| `belief_created_step` | int | Training step when allocated |
| `immutable_beliefs` | bool | Protected from merge/prune (core knowledge) |

### Belief Hierarchy (SDFT-Inspired Abstraction)

Beliefs progress through four levels based on evidence and access:

| Level | Name | How achieved | Properties |
|-------|------|-------------|------------|
| **L0** | Observation | Direct allocation from write path | Ephemeral, specific, can be pruned |
| **L1** | Consolidated | Merged from multiple similar L0 beliefs | Combined radius = √(r_A² + r_B²) |
| **L2** | Abstract | Survived many sleep cycles, high access count | Stable patterns |
| **L3** | Core | Promotion threshold met, marked immutable | Protected from merge/prune/split |

**Scarcity drives generalization**: limited slots force consolidation, consolidation forces abstraction, abstraction IS generalization. Remove the slot limit and the hierarchy collapses.

## State Interface (Read + Write Paths)

### Read Path (Beliefs → Hidden Stream)

Hopfield-style content-addressable retrieval with goal modulation:

1. **Query projection**: `hidden [B, T, H] → query [B, T, num_heads, D]`
2. **Goal modulation**: PARL per-head routing (`GoalRouter`) assigns each head to a different active goal via Gumbel-Softmax. Legacy: unified `goal_gate` bias across all heads.
3. **Depth-conditioned temperature**: early interfaces retrieve broadly, late interfaces focus
4. **Top-k sparse attention**: only k beliefs per position (32-64), avoids O(N_active) cost
5. **Post-retrieval convolution**: depthwise 1D conv adds coherence between positions (Engram ShortConv)
6. **Read gate**: per-position sigmoid — "does this position even need beliefs?"
7. **Utility head**: auxiliary logits from beliefs alone (trains "are beliefs useful for prediction?")
8. **Belief QAT**: STE quantize→dequantize during training for compression robustness

### Write Path (Hidden Stream → Candidates)

Precision-gated observation buffering:

1. **Observation projection**: `hidden [B, T, H] → obs [B, T, D]`
2. **Write gate**: learned sigmoid, biased closed (-2.0 init → ~12% open) — prevents state flooding
3. **Precision head**: softplus — learned observation confidence
4. **Batch matching**: one matmul `[N_obs, N_active]` cosine similarity against all active beliefs
5. **WriteCandidate**: `{belief_vector, matched_slot, similarity, source_position, source_layer}`
6. **No commitment during forward**: candidates buffered for Pass 2 (slot allocation is non-differentiable)

### Surprise Computation (Pass 2)

For each write candidate, compute prediction error:

```
surprise = angular_distance(observation, existing_belief) × observation_precision
gain = obs_radius / (obs_radius + belief_radius)              # Kalman-like
gain *= (1 + precision_var × boost).clamp(max=3.0)            # MESU modulation
should_reconsolidate = (gain > reconsolidation_threshold)     # full rewrite?
```

High-variance beliefs (uncertain) amplify gain → more willing to change. Low-variance beliefs (confident) dampen gain → resist updates. Matches Titans surprise-driven memorization.

## Loss Functions

### Pass 1 Losses (Differentiable)

All losses are computed inside `forward()` so the logits tensor `[B, T, vocab_size]` never crosses the DDP boundary.

**Scratch mode** uses Kendall/Gal uncertainty weighting (CVPR 2018) with learned `log_sigma` per loss group:

```
_uw(L, s) = L / (2·exp(2s)) + s
L_total = _uw(L_token, σ_token) + α · _uw(L_fe, σ_fe) + α · _uw(L_aux, σ_aux) + w_draft · L_draft + w_dsa_kl · L_dsa_kl
```

**Pretrained mode** uses fixed weights:

```
L_total = L_token + α · L_fe + α · 0.1 · L_utility + α · 0.1 · L_surprise
```

| Loss | Source | Trains | Description |
|------|--------|--------|-------------|
| **L_token** | `chunked_cross_entropy` | Transformer + interfaces | Next-token prediction. Chunked to bound memory on 151K vocab. |
| **L_fe_proxy** | `compute_differentiable_free_energy` | Read/write paths | Proxy FE: Huber loss on cosine disagreement between retrieved beliefs and observations, minus attention entropy. Huber delta is a learned MetaParam (softplus, init 0.5) — quadratic for small errors, linear for outliers. Prevents spurious matches from poisoning persistent beliefs over 500B tokens. Reference: MIRAS/YAAD (arXiv:2504.13173). |
| **L_fe_bethe** | `compute_bethe_free_energy` | Beliefs, edges, relations | Proper Bethe free energy on the cognitive factor graph. Power Spherical entropy with (d_i−1) counting correction. |
| **L_utility** | `chunked_cross_entropy` on utility logits | Interface utility heads | Measures whether retrieved beliefs improve token prediction. |
| **L_surprise** | `TelosModule.surprise_loss` | RND networks, goal system | RND surprise (trains predictor to match target for seen beliefs) + goal status transitions (penalize stalled/failed, reward completed). |
| **L_halt** | `F.binary_cross_entropy_with_logits` | RefinementProbe | Teaches the halting probe when to stop refinement loops. Teacher forcing with random oracle loop count. |
| **L_ponder** | `ponder_cost × mean(gate)` | RefinementRouter | Per-position penalty for continuing refinement. Ponder cost is a learned MetaParam. Encourages router to halt positions early. Reference: PonderNet (arXiv:2107.05407). |
| **L_jac** | DEQ Jacobian regularization | Message passing | Ensures the factor graph fixed-point map stays contractive. Periodic (every 10 steps). |
| **L_draft** | `DFlashDraftHead.compute_draft_loss` | Draft head + KV injection | Streak-distilled draft quality: position-weighted CE (w_i = λ^i) + expected streak bonus. NOT alpha-gated — trains from step 0. λ and streak_weight are learned MetaParams. |
| **L_dsa_kl** | `LightningIndexer.compute_kl_loss` | Indexer projections | KL alignment: trains indexer to predict dense attention distribution. NOT alpha-gated. Full weight phase 1, reduced (0.1) after. Only computed in dense path (short context). |

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
| 18a | BLT byte encoder | 0.01 | Byte embedding + N-gram conv + LocalDeltaProduct + strided pool |
| 18b | BLT byte decoder | 0.01 | Down projection + LocalDeltaProduct + 4 multi-byte heads |
| 19 | DFlash draft head (+ KV injection + OPUT) | 0.01 | Block diffusion with KV injection, streak distillation, adaptive block, OPUT self-correction. Attention 4-bit, MLP 3-bit, injection 3-bit RotorQuant. |
| 20 | Refinement router (MoR adaptive) | 0.01 | Per-position routing in refinement loops |
| — | GoalRouter (PARL per-head routing) | 0.01 | Included in group 5 (interface params) |
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

### Cognitive Subsystems (Pass 2 Detail)

#### Telos — Intrinsic Motivation

Goal system driven by RND (Random Network Distillation) surprise:

1. **Surprise detection**: Frozen random projection (target) vs trained predictor. High prediction error = novel belief. The predictor improves on seen beliefs, so novel beliefs stay surprising.
2. **Goal generation**: Most surprising beliefs become seeds → `goal_generator` network transforms them into goal direction vectors. Goals point toward regions of belief space the model doesn't understand yet.
3. **Goal modulation**: Active goals bias read path attention via `goal_gate` — retrieval becomes goal-directed, not just query-driven.
4. **Progress tracking**: `estimate_progress` measures how close beliefs are to the goal region. Progress 0→1 as understanding builds.
5. **Status transitions**: Gumbel-Softmax over {proposed→active→completed/stalled/failed} via learned `transition_net`.
6. **Telos energy**: Unfinished important goals increase free energy → gradient pressure to make progress.

#### Autoresearch — Hypothesis Generation

Internalized Karpathy autoresearch loop:

1. **HypothesisGenerator**: active goals → candidate beliefs (hypotheses) in goal direction
2. **Generate gate**: learned, biased closed (-1.0) — prevents flooding state with bad hypotheses
3. **Provisional allocation**: hypotheses enter as L0 beliefs with `provisional=True`
4. **Evaluation**: after learned window (MetaParam), check: did free energy improve? Did precision hold?
5. **HypothesisTracker**: per-goal success EMA — skip goals that never produce useful hypotheses

#### Cognitive Controller (SEAL-Style)

Learned RL policy for structural decisions:

- **State**: 8 features (mean/std radii, fill ratio, edge density, β, surprise, goal density, belief_advantage_ema)
- **Actions**: 5 continuous via Beta distribution — allocate_rate, merge_threshold, prune_threshold, connect_rate, goal_rate
- **Training**: PPO clipped surrogate + adaptive entropy (SAC-style) + value baseline
- **Reward**: `belief_advantage` (does cognitive state improve token prediction?) + dense state-delta signals

#### Sleep / Dream Cycle

**SleepGate** (learned scoring):
- Input: belief vector + 6 metadata features (radius, access_count, level, age, n_edges, mean_edge_weight)
- Output: softmax over {strengthen, maintain, forget}
- Strengthen: radius × (1 + factor), Forget: radius × (1 - factor)
- Factors adaptive: strengthen more when room to grow, forget more when full

**NeuroDream** (offline message passing):
- Runs message passing WITHOUT token input — pure internal reflection
- Beliefs shift toward incoming messages weighted by relative precision
- Only directions update (radii preserved) — dreaming reorganizes, doesn't create confidence
- Convergence checked via shift magnitude; iterations adaptive based on edge density

**Full sleep pipeline**: SleepGate → NeuroDream → belief shift → two-factor sleep (homeostatic normalization) → self-verification (causal consistency) → precision recalibration → interleaved replay (cross-temporal contradiction detection)

#### Message Passing (Factor Graph)

Loopy belief propagation on the edge graph with DEQ (Deep Equilibrium) solver:

```
For each edge (src → tgt):
  bias = relation_transform(edge_relation)        # learned relation semantics
  target' = normalize(target_angle + bias)         # transformed target
  msg_precision = weight × radius × cos(direction) # CoED directional encoding
  message = msg_precision × target'                # precision-weighted message

Aggregate per belief: weighted mean of incoming messages
```

**DEQ solver** (Anderson acceleration): finds fixed point in ~10 iterations with constant memory. Spectral norm on relation_transform guarantees contractivity → unique fixed point exists.

**Belief shift**: `shift_rate = β/D × msg_precision / (msg_precision + belief_radius)` — bounded, precision-relative.

#### Planning (B1-B4)

Four tiers of active inference planning:

- **B1**: Inject preference (goal) and epistemic (uncertainty) priors into message passing
- **B2**: Causal rollout — simulate K steps forward through edges, compute Expected Free Energy
- **B3**: MCTS when β > 0.5 and multiple goals — UCB exploration with EFE rollout policy
- **B4**: Hierarchical planning — group goals by depth, plan at different horizons

All parameters learned MetaParams. Planning is internal (no action execution) — it biases next-step structural decisions.

#### Cascade Revision

When a belief is revised (high surprise → reconsolidation), BFS-propagate precision decay through causal edges:

```
For each downstream belief at depth d:
  radius *= (1 - decay_factor^d)             # exponential decay
  precision_var += variance_boost × decay_factor^d  # rightfully uncertain
```

Prevents orphaned high-precision children of corrected parents. Only causal edges participate (not associative). Immutable beliefs skip cascade.

#### Structural Plasticity

- **Split**: polysemantic beliefs (high activation entropy) → two children at angle ± perturbation, each with radius/√2 (energy conservation)
- **Prune**: low frequency + low radius + learned prune_net decision → deallocate
- **Growth pressure**: fill_ratio × (1 + surprise) — tracks capacity need

#### Edge Proposal

Learned `EdgeProposer` evaluates candidate pairs from co-activation (Hebbian) + causal observation counts. Creates edges with learned weights and CoED direction angles. Runs when edge_fill < 90%.

## Refinement Loops (Latent Chain-of-Thought)

Unlike text-based CoT, refinement loops reason in latent space:

```
For loop_i in range(max_loops):
  1. Loop-index encoding (additive signal, fraction of max loops)
  2. Re-run upper transformer blocks on current hidden states
  3. Lifeline anchor: real tokens += gate × original_hidden (prevent drift)
     Working memory suffix: evolves freely (true scratchpad)
  3b. Predictive refinement (loop > 0): contractive scaling + per-position routing
  4. Re-query beliefs via last interface (error-gated: skip if delta < threshold)
  5. TTT gradient step on upper MLP deltas (self-improvement within reasoning)
  6. HaltingHead probe: P(halt) — stop early if confident enough
```

- **Training**: teacher forcing with random oracle loop count; halt probe learns from random targets
- **Inference**: halt when P(halt) > threshold or max loops reached
- **Cost**: ~3-4× per token (amortized by DFlash speculative decoding ~7-8× + predictive refinement)
- **Advantage over text CoT**: no output tokens wasted, vector representations richer than language, TTT adapts weights during reasoning, different beliefs retrieved each loop as understanding shifts

### Predictive Refinement (MoR + SCORE + Error-Gated Retrieval)

Three complementary mechanisms make later refinement loops cheaper and more focused, with all thresholds as learned MetaParams:

**1. SCORE-style contractive scaling**: step size `dt = (1 - contraction_rate)^l` shrinks geometrically per loop. Two-Scale Latent Dynamics (arXiv:2509.23314) empirically confirms that later iterations in recurrent-depth transformers produce smaller, increasingly orthogonal updates — the contraction rate formalizes this.

**2. MoR-style per-position routing**: a lightweight `RefinementRouter` (hidden_dim+1 → bottleneck → 1, sigmoid) examines the delta vector (what this loop changed) and produces a per-position gate ∈ (0, 1). Positions where the gate is near zero have effectively converged — their contribution from this loop is suppressed. The router sees the delta *content* (not just magnitude), so it can learn which kinds of changes matter (e.g., belief-retrieval-induced changes vs noise).

**3. Error-gated retrieval**: the belief re-query step (Step 4) is the most expensive operation in the loop. With predictive refinement, the per-position delta norms determine whether enough positions changed meaningfully to justify re-querying beliefs. When fewer positions have significant deltas than the learned retrieval threshold, the re-query is skipped entirely.

```
Loop 0: full computation (establishes the "prediction")
Loop 1+: delta = blocks(x) - x_pre
          dt = (1 - contraction_rate)^loop_i          ← learned MetaParam
          gate = router(delta, loop_fraction)          ← per-position [B, S, 1]
          x = x_pre + dt * gate * delta                ← gated update
          if active_frac < retrieval_threshold:         ← learned MetaParam
              skip belief re-query (error-gated)
```

**Ponder loss**: the mean gate value across loops is penalized by a learned ponder cost (MetaParam, softplus). This encourages the router to halt positions early when possible, discovering the optimal compute/quality tradeoff rather than using a hardcoded budget.

**Interaction with existing mechanisms**:
- The HaltingHead decides *when to stop the loop entirely* (global). The router decides *which positions update within each loop* (per-position). They're complementary.
- The lifeline anchor applies *before* the router gate, so token positions are still anchored even when the router partially suppresses their delta.
- Working memory positions are routed like any other position — the router learns to keep WM open when it's actively being written and close it when stable.

**MetaParams** (all learned, no hardcoded thresholds):

| Parameter | Activation | Init | Controls |
|-----------|-----------|------|----------|
| `refinement_contraction_rate` | sigmoid | 0.2 | SCORE step-size decay per loop |
| `refinement_retrieval_threshold` | sigmoid | 0.1 | Min delta fraction for belief re-query |
| `refinement_ponder_cost` | softplus | 0.5 | Regularization penalty for continuing |
| `parl_parallel_reward_weight` | softplus | 0.5 | Peak weight for parallelism reward (PARL) |
| `parl_finish_reward_weight` | softplus | 0.5 | Peak weight for goal-finish penalty (PARL) |
| `parl_goal_diversity_threshold` | sigmoid | 0.3 | Min normalized entropy for diverse pursuit |
| `dflash_streak_decay` | sigmoid | 0.85 | Position weight decay λ for streak distillation |
| `dflash_streak_weight` | softplus | 0.1 | Weight on expected streak length bonus |
| `dflash_entropy_threshold` | softplus | 2.0 | Entropy cutoff (nats) for adaptive block sizing |
| `oput_self_correct_weight` | softplus | 0.5 | Weight on OPUT on-policy self-correction loss |

**References**:
- MoR: Mixture-of-Recursions (NeurIPS 2025, arXiv:2507.10524) — per-token adaptive recursion depth
- SCORE: Contractive Recurrent Depth (arXiv:2603.10544) — ODE-inspired step-size control
- PonderNet (arXiv:2107.05407) — learned halting with geometric prior
- PARL (Kimi K2.5, arXiv:2602.02276) — staged reward, serial collapse prevention
- MoH (arXiv:2410.11842) — mixture-of-head attention routing
- Two-Scale Latent Dynamics (NeurIPS 2025, arXiv:2509.23314) — geometric evidence for contractive refinement
- DeltaLLM (arXiv:2507.19608) — temporal sparsity via delta thresholding

## Five Memory Systems

Memoria maintains coherence at any context length through five complementary memory systems:

| System | Range | Scaling | Timescale | What it stores |
|--------|-------|---------|-----------|---------------|
| **MLA + DSA** | Full context (sparse K tokens) | O(T×K) | Immediate | Exact sparse recall, belief-guided token selection |
| **DeltaProduct state** | Unlimited | O(1) fixed | Continuous | Error-corrected recurrent memory, entity tracking |
| **Log-Linear GDN state** | Unlimited | O(log T) growing | Multi-scale | Hierarchical temporal context, recent=high-res, distant=compressed |
| **Cognitive state** | Unlimited | O(1) fixed slots | Persistent | Facts, relations, goals — survives across sessions + guides DSA |
| **Engram** | Unlimited | O(1) hash | Static | Common N-gram patterns, signatures, idioms |

Memory at 1M tokens (small config, 2 MLA layers, DSA with RotorQuant 3-bit):
- DSA indexer keys: ~12 MB (1M × 32 × 3-bit RotorQuant)
- DeltaProduct state: ~2 MB per layer (fixed d_k × d_v matrix)
- Log-Linear GDN state: ~0.6 MB per layer × log₂(1M) ≈ 20 levels
- Cognitive state: ~26 MB (65K beliefs × 256 dims)
- **Total: ~54 MB** — lighter than windowed MLA (~80 MB) with global reach

No single system covers everything. MLA+DSA provides exact sparse recall across the entire context, directed by beliefs. DeltaProduct error-corrects within-sequence state. Log-Linear GDN maintains multi-scale temporal hierarchy. Cognitive state stores discrete facts retrievable by content AND guides DSA's sparse attention toward relevant tokens. Engram handles static patterns with zero distance penalty.

## Model Configurations

### Presets

| Config | Params | Layers | Pattern | Beliefs | Target Hardware |
|--------|--------|--------|---------|---------|----------------|
| `small_config()` | ~245M (+117M embed) | 12 | HHHHL | 16K | Single 3090 |
| `medium_config()` | ~456M | 24 | HHHHL | 32K | 2× 3090 |
| `large_config()` | ~694M | 24 | HHHHL | 65K | B200 / multi-GPU |
| `lfm2_config()` | 350M frozen + 15M adapters | 16 | LFM2 hybrid | 16K | Single 3090 |
| `qwen_config()` | 2B frozen + 25M adapters | 24 | Qwen3.5 hybrid | 32K | Single 24GB GPU |

### Pretrained Backbone Mode

Freeze HF backbone, train only state interface layers (~15-25M params):
- LFM2.5-350M: 28T tokens of data exposure for free, hybrid conv+attention
- Qwen3.5-2B: 36T tokens, strongest language at this scale

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

## BLT — Byte Latent Transformer (Tokenizer-Free I/O)

Replaces the 151K-token embedding (117M params) + LM head (117M params) with a byte-level encoder/decoder (3.6M params total). The global DeltaProduct backbone operates on patches (compressed byte groups), while lightweight local layers handle byte↔patch conversion.

**Why this is critical:**
- LM head was **71% of per-token inference bandwidth** (233 MB FP16)
- Softmax bottleneck: 768-dim cannot represent 151K output distributions
- Gradient bottleneck: 99.5% of gradient signal lost through 151K LM head
- BLT eliminates all three: 260 byte classes, 768 >> 260 (overcomplete)

### Architecture

```
Raw bytes [B, T_bytes]
  → ByteEncoder:
    byte_embed(260 → local_dim=384)
    + causal N-gram conv (8-byte receptive field, learned)
    + 2× LocalDeltaProduct (O(T), 1 Householder, head_dim=64)
    + strided Conv1d pool (stride=patch_size=6) → patches
    + Linear(local_dim → n_embd=768)
  → patches [B, P, 768]
  → byte_hidden [B, T_bytes, 384] (skip connection for decoder)
    + EngramCache injection (hash-table byte N-grams, O(1) lookup)

      ↓ (P = T_bytes / 6, ~4-6× shorter than bytes)

  Global DeltaProduct Backbone (unchanged, operates on patches)
    12-48 layers × HHHHL pattern
    State interfaces every 4 layers (read/write beliefs at patch level)
    Refinement loops on upper layers
    TTT gradient steps (patch-level targets: last byte per patch)

      ↓

  → ByteDecoder:
    Linear(n_embd → local_dim)
    + repeat_interleave to byte positions
    + intra-patch positional encoding
    + gated skip connection from encoder byte_hidden
    + 2× LocalDeltaProduct (O(T))
    → 4 multi-byte prediction heads (each 384 → 260, ~100K params)
  → byte_logits [B, T_bytes, 260]
```

### Key Properties

| Property | Token-level (old) | BLT (new) |
|----------|------------------|-----------|
| Vocab / output classes | 151,936 | **260** (256 bytes + 4 special) |
| Embedding params | 117M | **100K** (260 × 384) |
| LM head params | 117M | **100K** per head (× 4 heads = 400K) |
| Active model params (small) | 395M | **146M** (62% smaller) |
| Per-token BW (LM head) | 233 MB (FP16) / 58 MB (4-bit) | **0.4 MB** |
| Softmax bottleneck | Yes (768 << 151K) | **None** (768 >> 260) |
| Gradient pass-through | 0.5% | **100%** (overcomplete) |
| Tokenizer required | Yes (Qwen3 BPE) | **None** (raw bytes) |

### Local DeltaProduct Layers

Same GatedDeltaProduct kernel as global backbone but lighter:
- 1 Householder reflection (vs 3 in global) — byte patterns are simpler
- local_dim=384 (vs n_embd=768) — bytes need less capacity
- O(T) on byte sequences — handles 4-6× longer sequences naturally
- Falls back to causal conv + gated MLP on CPU (no FLA Triton)

### Multi-Byte Prediction Heads

4 independent prediction heads on the decoder output:
- Head 0: predicts next byte (standard autoregressive)
- Head k: predicts byte k+1 ahead (for multi-step DFlash speculation)
- Each head: Linear(384 → 260) = 100K params
- Total: 400K params (was 117M shared LM head)

### Scaling

| Model size | LM head % of backbone (old) | BLT I/O % of backbone |
|------------|---------------------------|----------------------|
| 400M | 235% | 7.3% |
| 2B | 44% | 0.5% |
| 7B | 17% | 0.1% |
| 30B | 5% | 0.03% |

BLT I/O cost vanishes at scale. The freed params (92M at 400M, ~900M at 7B) can be reinvested as additional backbone layers.

### Inference Throughput (RTX 3090, small_config)

| Config | Text bytes/s | vs Vanilla AR |
|--------|-------------|---------------|
| Vanilla AR (151K, FP16 head) | 5,600/s | 1.0× |
| DFlash + 4-bit head (best token-level) | 88,500/s | 15.9× |
| **BLT + DFlash** | **182,000/s** | **32.7×** |

Reference: BLT (Meta, arXiv:2412.09871) — byte latent transformer, tested at 400M-8B
Reference: MambaByte (arXiv:2401.13660) — byte-level SSM, proved SSM+bytes works
Reference: EvaByte (HKU NLP, 2025) — linear attention + bytes + multi-byte heads
Reference: Bolmo (Allen AI, arXiv:2512.15586) — mLSTM byte encoder at 1B-7B
Reference: ByteFlow (arXiv:2603.03583) — learned byte segmentation
Reference: LM head gradient bottleneck (arXiv:2603.10145) — 95-99% gradient loss

## DFlash Speculative Decoding

Native block diffusion draft head for inference acceleration. Especially valuable because refinement loops multiply per-token cost by ~4×. Four improvements over baseline:

1. **KV injection**: Per-layer K/V projections for tapped target features (not concatenated as context tokens). Each draft layer independently interprets target representations. Injection projections are 3-bit RotorQuant eligible.
2. **Streak distillation**: Position-weighted CE (w_i = λ^i, λ learned) + expected streak length bonus. Optimizes consecutive acceptance, not per-token accuracy.
3. **Adaptive block size**: Draft up to `max_block_size` (32) tokens, but verify only the confident prefix. Entropy-based cutoff via learned MetaParam threshold.
4. **OPUT on-policy self-correction** (DMax arXiv:2604.08302): During training, the draft head samples from its own predictions, constructs SPD hybrid embeddings (confidence-weighted interpolation between predicted token embedding and mask embedding), and runs a second forward pass. Trains the draft head to recover from its own errors within a block. Training-only — zero inference cost. The OPUT noise-robustness also enables RotorQuant on all draft head weights (previously excluded as "sensitive").

**Gradient flow into backbone**: Tapped target features are NOT detached — L_draft gradient flows through tapped hidden states at layers [0, 5, 11] into the backbone. This provides an implicit multi-token prediction (MTP) signal: the backbone learns to produce hidden states that help the draft head predict k tokens ahead consecutively. Unlike dedicated MTP heads, this signal is streak-optimized, multi-layer, and belief-conditioned.

**Draft head RotorQuant**: OPUT noise-robustness training makes the draft head tolerant to quantization noise. All draft layer weights are now quantized: attention projections at 4-bit (IsoQuantMSE), MLP at 3-bit (PlanarQuantMSE), KV injection at 3-bit, feature_proj at 4-bit. Reduces draft head bandwidth from ~32 MB to ~6 MB per round.

### Architecture

```
Target hidden states (tapped from layers [0, 5, 11])
  → feature_proj (concat → D)
  → KV injection signal [B, T_tap, D]

Active belief embeddings → context [B, N_beliefs, D]  (or empty)

Mask embeddings + positional encoding (up to max_block_size=32)
  → draft tokens [B, draft_length, D]

3× DFlashDraftLayer:
  injection → per-layer norm_inject → k_inject, v_inject (3-bit RotorQuant)
  K = [k_inject(injection), k_proj(concat(beliefs, draft))]
  V = [v_inject(injection), v_proj(concat(beliefs, draft))]
  draft = draft + CrossAttention(Q=draft, KV=above)
  draft = draft + MLP(draft)     [2× expansion, ReLU²]

→ shared LM head → draft logits [B, draft_length, vocab]
```

- **Cross+self attention with KV injection**: draft tokens attend to beliefs + each other via standard KV, AND to target features via injected KV pairs
- **Shared LM head**: No extra 117M vocab projection
- **Trained jointly**: Streak-distilled loss from step 0 (not alpha-gated)
- **Inference**: `spec_generate()` — adaptive draft → verify confident prefix → accept/reject → repeat

### Speculative Decoding Loop (Optimized)

Verify-as-prefill reuse: the verify step from round N provides logits + tap hiddens
for round N+1. Only ONE full model forward per round (not two).

```
# Initial prefill (once)
result = forward(generated)
cached_logits, cached_taps = result['logits'], result['dflash_tapped']

while tokens_generated < max_new_tokens:
    1. Get guaranteed token from cached_logits[:, -1, :]
    2. Extract KV injection from cached_taps (no recomputation)
    3. Draft head generates max_block_size tokens in parallel (cheap)
    4. Entropy-based adaptive cutoff → verify only confident prefix
    5. Verify: forward(candidate_seq) → verify_logits, verify_taps
    6. Accept tokens until first mismatch with verifier
    7. cached_logits, cached_taps = verify_logits, verify_taps  ← REUSE
```

### Bandwidth Analysis (RTX 3090)

| Component | Size | % of per-token BW |
|-----------|------|-------------------|
| Backbone (4/3-bit RotorQuant) | 37.8 MB | 24.9% |
| Refinement (3 loops × 4 blocks) | 37.8 MB | 24.9% |
| Interface layers (FP16) | 18.1 MB | 11.9% |
| LM head (4-bit RotorQuant) | 58.4 MB | 38.4% |
| **Total per AR token** | **152.1 MB** | |

DFlash round cost (verify + draft): 185.3 MB → ~9 accepted tokens

| Configuration | Tok/s | Speedup |
|---------------|-------|---------|
| Vanilla AR | ~3,000 | 1.0x |
| DFlash (all optimizations) | ~22,000 | 7.4x |

Key optimizations and their contributions:
- KV injection: +21% acceptance rate (tighter verifier alignment)
- Streak distillation: +38% (consecutive accuracy optimization)
- Adaptive block size: +90% (draft many, verify few in hard regions)
- Verify-as-prefill reuse: 2x (eliminates redundant forward pass per round)
- LM head 4-bit QAT: 2x (233 MB → 58 MB, 71% bandwidth savings)

Reference: DFlash (arXiv:2602.06036) — KV injection, block diffusion
Reference: SpecDiff-2 (arXiv:2511.00606) — streak distillation
Reference: FailFast (arXiv:2512.20573) — adaptive speculation length
Reference: DEER (arXiv:2512.15176) — single-step diffusion drafting
Reference: DMax (Chen et al. — arXiv:2604.08302) — OPUT on-policy training + SPD hybrid embeddings

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

## Quantization-Aware Training (RotorQuant + CAGE)

Four quantization paths, all using RotorQuant block-diagonal rotations:

1. **Activation QAT** (KV cache, beliefs): 3-bit STE noise during training → lossless runtime compression
2. **Weight QAT** (transformer backbone + LM head): 4-bit (attention/DeltaProduct/LM head) or 3-bit (MLP) STE noise → deployable at low-bit with near-zero quality loss
3. **DFlash KV injection QAT**: 3-bit for k_inject/v_inject projections (draft accuracy < verifier — aggressive quantization safe)
4. **CAGE correction**: Post-optimizer step nudges weights toward quantization grid points

LM head quantization is critical in token mode: at 151K vocab × 768 dim, the FP16 LM head is 233 MB — 71% of per-token bandwidth. At 4-bit it drops to 58 MB. With BLT enabled, the LM head is replaced by byte decoder heads (260 classes, 0.4 MB total) — the bottleneck disappears entirely.

### Backend Selection

| Backend | Rotation | FMAs (d=128) | Centroids | When Used |
|---------|----------|-------------|-----------|-----------|
| **PlanarQuantMSE** | 2D Givens | 256 | Lloyd-Max (MSE-optimal) | 3-bit (KV, beliefs, MLP weights), `rotorquant` installed |
| **IsoQuantMSE** | 4D quaternion | 512 | Lloyd-Max | 4-bit (attention/DeltaProduct weights), `rotorquant` installed |
| **PolarQuantizer** | Full d×d QR | d² | Uniform scalar | Fallback (no external deps) |

Install RotorQuant: `pip install -e ".[rotorquant]"` (optional, PolarQuantizer fallback works without it).

Reference: RotorQuant (scrya.com/rotorquant.pdf, March 2026) — beats TurboQuant on PPL (6.91 vs 7.07),
decode speed (28% faster), and prefill (5.3× faster) at same 10.3× compression.

### Where Quantization Acts

| Path | When | Bits | Mechanism | Effect |
|------|------|------|-----------|--------|
| **KV QAT** | Training (every forward) | 3 | `ste_quantize(k, quantizer)` in attention layers | K,V projections survive 3-bit compression |
| **Belief QAT** | Training (every read) | 3 | `ste_quantize(values, quantizer)` in `ReadPath` | Belief representations compress losslessly |
| **Weight QAT** | Training (every forward) | 4/3 | `WeightQuantLinear` wraps all backbone `nn.Linear` | Weight matrices deployable at 4-bit |
| **DSA Indexer QAT** | Training (every MLA forward) | 3 | `ste_quantize(k_I, quantizer)` in `LightningIndexer` | Indexer keys survive 3-bit for global token scoring |
| **CAGE correction** | Training (after optimizer.step) | — | `cage_step()` nudges weights toward grid | Weights converge to quantization-friendly distributions |
| **KV cache compression** | Inference (T > window) | 3 | `QuantizedKVCache.compress()` → `decompress_slice()` | 10× KV memory reduction |
| **Checkpoint compression** | Save/load | 3 | `QuantizedBeliefStore` in `state_dict_cognitive()` | ~10× belief tensor compression |

### Weight QAT: Which Layers

Quantized (via `WeightQuantLinear` wrapper):
- H-block projections: q/k/v/b/a/g/o/l_proj (4-bit, `IsoQuantMSE`)
- MLA projections: c_q, c_kv_compress, k_up, v_up, c_k_rope, c_proj (4-bit)
- MLP: c_fc, c_proj (3-bit, `PlanarQuantMSE` — MLPs tolerate aggressive quantization)

NOT quantized (kept bf16/fp32):
- Token embeddings, LM head (lookup tables, need precision for rare tokens)
- State interface layers (bridge cognitive state and hidden stream)
- Cognitive state (beliefs, edges, goals — knowledge store)
- Telos, SleepGate, EdgeProposer, Controller (small, sensitive)
- DFlash mask/pos embeddings (tiny, anchor for SPD interpolation — draft layer weights ARE quantized via OPUT robustness)
- Short convolutions in DeltaProduct (tiny, affect recurrent state quality)

### CAGE Schedule (Phase-Aligned)

CAGE is silent during phase 1 (model learns language freely), ramps during phase 2
(cognitive awakening), and runs at full strength in phase 3:

```
Phase 1 (0 → phase1_steps):      λ = 0              (silent)
Phase 2 (phase1_steps → +phase2): λ ramps 0 → λ_base (nudging begins)
Phase 3 (remainder):              λ = λ_base          (full correction)
```

The CAGE correction is optimizer-agnostic (works with both Muon and AdamW):
```
e = weight - Q(weight)           # quantization error
weight -= lr × λ × e             # push toward nearest grid point
```

Reference: CAGE (arXiv 2510.18784, IST-DASLab 2025) — halves quantization error vs STE alone.

### STE Gradient Flow

```
Forward:  x → [quantize → dequantize] → x_hat  (with quantization noise)
Backward: grad_output → identity → grad_input   (straight-through)
```

The STE ensures gradients flow through as if no quantization occurred, while the forward pass
trains the model to be robust to the noise. Over training, weight matrices, K/V projections,
and belief write paths naturally converge toward representations where information sits in
directions that survive low-bit rounding.

### DSA Sparse MLA for Long Context

MLA layers default to full causal attention O(T²). At long context (>128K), three modes are available:

```
dsa_enabled = True          → sparse global O(T×K), belief-conditioned (RECOMMENDED)
mla_window_size = 131072    → 128K window O(T×W) (legacy fallback, no DSA)
mla_window_size = 0         → full causal O(T²), short context only
```

With DSA enabled, MLA layers use the Lightning Indexer to select the top-K most relevant tokens from the ENTIRE context, then run full attention only on those tokens. The indexer is belief-conditioned: active beliefs bias token selection toward evidence relevant to current cognitive state. This creates an active inference loop where beliefs direct attention, attention surfaces evidence, and evidence revises beliefs.

Five memory systems stack to maintain coherence at any context length:

| System | Range | Scaling | Role |
|--------|-------|---------|------|
| **MLA + DSA** | Full context (sparse) | O(T×K) | Exact sparse recall, belief-guided |
| **DeltaProduct state** | Unlimited | O(1) fixed | Error-corrected recurrent memory |
| **Log-Linear GDN state** | Unlimited | O(log T) growing | Hierarchical multi-scale context |
| **Cognitive state** | Unlimited | O(1) fixed slots | Persistent beliefs + DSA guidance |
| **Engram** | Unlimited | O(1) hash | Static N-gram patterns |

Memory at 1M tokens (small config, 2 MLA layers, DSA, RotorQuant 3-bit):
- DSA indexer keys: ~12 MB (3-bit RotorQuant, full context)
- DeltaProduct state: ~2 MB per layer (fixed)
- Log-Linear GDN state: ~12 MB (log₂(1M) ≈ 20 levels)
- Total: **~54 MB** — lighter than windowed MLA with global coverage

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

## Inference

### Generation

Standard autoregressive generation with cognitive state active:

1. **Prefill**: Process full prompt through model. Beliefs form, Engram activates, DeltaProduct + Log-Linear GDN states build.
2. **Decode**: Generate tokens one-by-one. Each token triggers read path (retrieve beliefs), write path (buffer observations), TTT (adapt weights).
3. **Between chunks**: Pass 2 runs — allocate beliefs, create edges, update goals. Dream phase if at sequence boundary.
4. **Cognitive state persists**: After generation ends, beliefs/edges/goals remain for next conversation.

### Speculative Decoding (DFlash)

DFlash draft head amortizes refinement loop cost. Optimized loop reuses verify as next prefill (1 forward per round, not 2):

```
while tokens_generated < max_new_tokens:
  1. Run full model on current sequence (with refinement loops)
  2. Draft head generates block_size tokens in parallel (cheap: 15.9M params)
  3. Verify: run full model on [sequence + drafted block]
  4. Accept tokens until first mismatch
  5. Append accepted, continue
```

~7-8× speedup with KV injection + streak distillation + adaptive block + verify-as-prefill reuse. Estimated ~22K tok/s on RTX 3090 (with 4-bit LM head QAT).

### Session Persistence

```python
# End of session: save cognitive state
state_dict = model.state.state_dict_cognitive(compress=True)  # RotorQuant compressed
torch.save(state_dict, "user_state.pt")

# Start of next session: restore
model.state.load_state_cognitive(torch.load("user_state.pt"))
# Model remembers everything from previous sessions
```

## File Map

```
memoria/training/
  train.py          — Main training loop, DataPrefetcher, checkpoint, hub push
  optimizer.py      — Muon + AdamW setup, 18 parameter groups, _CombinedOptimizer
  schedule.py       — LR (WSD) + alpha (KL annealing) + context length (SkyLadder) schedules
  distributed.py    — Cognitive state broadcast/gather for multi-GPU
  cognitive_seed.py  — Cross-run belief transfer via content matching

memoria/model/
  config.py              — TransformerConfig, StateConfig, TrainingConfig, presets
  transformer.py         — Hybrid transformer (H/D/E/MLA dispatch), YaRN RoPE, ReLU²
  deltaproduct_layers.py — DeltaProductBlock (D), LogLinearDeltaProductBlock (H), LogLinearGDNBlock (E)
  fenwick_state.py       — FenwickStateTree: O(log T) hierarchical state for Log-Linear layers
  memoria_model.py       — MemoriaModel: transformer + state + interfaces + BLT + DFlash
  blt.py                 — BLT byte encoder/decoder (tokenizer-free byte-level I/O)
  pretrained_model.py    — PretrainedMemoriaModel: frozen HF backbone + interfaces
  dflash_head.py         — DFlash block diffusion draft head for speculative decoding

memoria/core/
  state.py           — CognitiveState: beliefs, edges, goals, meta-parameters
  free_energy.py     — Bethe FE, EFE (Huber-robust risk term), Power Spherical entropy
  losses.py          — chunked_cross_entropy, differentiable FE proxy (Huber-robust belief matching)
  quantize.py        — RotorQuant/PolarQuant quantization, STE, KV cache, belief store, DSA Lightning Indexer + Triton fused scoring kernel
  polar.py           — Polar coordinate utilities for belief representation
  ttt.py             — In-Place TTT with Titans/LaCT/DeltaProduct enhancements
  message_passing.py — Factor graph message passing with DEQ solver

memoria/interface/
  layer.py           — StateInterfaceLayer (read + write, inserted every K blocks)
  read_path.py       — BeliefCache, ReadPath (Hopfield attention over beliefs)
  write_path.py      — WritePath, WriteCandidate, pack/unpack for distributed

memoria/core/
  meta_params.py     — 72 learned MetaParams replacing hardcoded constants (sigmoid/softplus constrained, incl. huber_delta, PARL, DFlash streak/entropy)

memoria/cognition/
  pass2.py           — Pass 2 orchestrator (12 structural operations)
  surprise.py        — Prediction error × precision, MESU-modulated Kalman gain
  belief_update.py   — Slot allocation, eviction
  consolidation.py   — Soft merge (cosine > threshold), hard cleanup, L0→L1 promotion
  hebbian.py         — Co-activation extraction for edge candidates
  meta_learning.py   — Beta computation, running statistics
  telos_module.py    — TelosModule: RND surprise, goal generation, status transitions
  sleep.py           — SleepGate (learned strengthen/maintain/forget) + NeuroDream (offline BP)
  planning.py        — B1-B4: preference/epistemic priors, causal rollout, MCTS, hierarchical
  autoresearch.py    — HypothesisGenerator + HypothesisTracker (goal→hypothesis→evaluate)
  provisional.py     — Provisional belief evaluation (learned window, FE + precision criteria)
  cascade_revision.py — BFS precision decay through causal edges after reconsolidation
  cognitive_controller.py — SEAL-style RL policy (PPO + SAC entropy), 5 continuous actions
  structural_plasticity.py — Polysemantic splitting (activation entropy), dead belief pruning

memoria/data/
  curated.py         — Multi-tier dataset mix (45% state-essential)
  streaming.py       — FineWeb-Edu streaming
  synthetic.py       — Belief tracking, contradiction, causal chain generators
  tokenizer.py       — Tokenizer setup (Qwen3 BPE or pretrained model's own)
```
