# Replacing Loopy BP with Learned Single-Pass Inference

> Research survey compiled 2026-04-04. Focused on practical approaches for Memoria's factor graph (4096 beliefs, 16384 edges, continuous polar-valued, precision-weighted, directional via CoED).

## Current System

```
FactorGraphMessagePassing._single_pass(state) -> {messages, precisions, agreement}
  - Messages: precision-weighted, directional (CoED cos/sin scaling)
  - Aggregation: scatter_add + normalize by total precision
  - Loopy BP: damped iteration with learned alpha in (0,1) and learned iteration count
  - Runs ONCE per training step (not inner-loop)
  - File: memoria/core/message_passing.py
```

---

## 1. GNNs as Learned Message Passing (replacing BP iterations with layers)

### 1a. Belief Propagation Neural Networks (BPNN)
**Paper:** Kuck & Chakraborty, NeurIPS 2020
**Key idea:** BPNN is a strict generalization of BP. Each BPNN layer modifies the standard BP message update with learned neural corrections while preserving BP's permutation equivariance. The network learns to find better fixed points faster.

**Convergence guarantees:** BPNN-D (the constrained variant) is guaranteed to converge on tree-structured factor graphs and returns exact partition functions. On loopy graphs, it computes a lower bound whenever the Bethe approximation from BP fixed points is a provable lower bound. It converges within L iterations on trees of height L.

**Relevance to Memoria:** Your `_single_pass` already computes BP-like messages. A BPNN layer would wrap each `_single_pass` with a learned residual correction. The correction network takes the raw messages as input and outputs an additive or multiplicative correction, preserving the precision-weighted structure.

**Limitation:** Still iterative -- just converges faster (fewer iterations needed). Does not eliminate the loop, only reduces it from ~10 to ~2-3 iterations.

### 1b. Factor Graph Neural Networks (FGNN)
**Paper:** Zhang, Wu, Lee -- NeurIPS 2020, JMLR 2023
**Key idea:** Derives an efficient approximate Sum-Product BP algorithm for higher-order PGMs, then neuralizes it into an FGNN by allowing richer message update rules. Two module types: Variable-to-Factor (VF) and Factor-to-Variable (FV), stacked into layers.

**Critical insight:** FGNN can exactly parameterize Max-Product BP. With the right aggregation operators, one architecture represents both Max-Product and Sum-Product BP.

**Implementation:** PyTorch, available at github.com/zzhang1987/Factor-Graph-Neural-Network. Uses bipartite graph structure matching your variable-node/factor-node decomposition.

**Relevance:** Your factor graph is already bipartite (beliefs = variable nodes, edges with relations = factor nodes). FGNN's VF/FV modules map directly to your forward/reverse message flows. A 2-layer FGNN could replace ~5 BP iterations.

### 1c. Neural Enhanced Belief Propagation (NEBP)
**Paper:** Satorras & Welling, AISTATS 2021
**Key idea:** Runs a GNN on the factor graph conjointly with BP. At every iteration, the GNN receives BP messages as input and outputs corrected messages. The GNN acts as a learned correction to standard BP.

**Advantage:** Handles both model misspecification (approximate factor graph) and loopy structure. The GNN learns to compensate for both simultaneously.

**Code:** Available from the authors. Tested on LDPC error correction.

### 1d. Theory-Guided MPNN (TMPNN)
**Paper:** Cui et al., AISTATS 2024
**Key idea:** Instead of black-box neural messages, uses an analytically derived message function from a generalized Bethe free energy with a learnable variational assumption. The message function has semantically meaningful parameters (not opaque weights).

**Advantage over V-MPNN:** More data-efficient, generalizes to unseen graph structures. Trained with algorithmic supervision (no exact inference labels needed).

**Direct relevance:** Your system already minimizes Bethe free energy (`core/free_energy.py`). TMPNN's generalized Bethe energy with learnable variational parameters would extend your existing framework naturally.

---

## 2. Deep Equilibrium Models (DEQ) -- Fixed-Point Without Iteration

### 2a. Core DEQ Concept
**Paper:** Bai, Kolter, Koltun -- NeurIPS 2019
**Key idea:** Instead of unrolling N iterations of a weight-tied network, directly solve for the fixed point z* where z* = f(z*) using root-finding (Anderson acceleration or Broyden's method). Backprop through the fixed point using implicit differentiation -- no need to store intermediate activations.

**Memory:** Constant memory regardless of effective "depth" (iteration count). This is critical for your 4096-node graph where storing intermediate messages per iteration is expensive.

**Backward pass:** Uses the implicit function theorem: dL/dx = -dL/dz* (df/dz*)^{-1} df/dx. Computed via a second fixed-point solve (or Neumann series approximation).

### 2b. DEQ for Message Passing (The Recommended Approach)

**Concrete proposal for Memoria:**

```python
# Pseudocode: DEQ wrapper around your existing _single_pass

import torchdeq

class DEQMessagePassing(nn.Module):
    def __init__(self, belief_dim, relation_dim):
        super().__init__()
        self.bp_layer = FactorGraphMessagePassing(belief_dim, relation_dim)
        self.deq = torchdeq.get_deq(
            f_solver='anderson',    # Anderson acceleration
            b_solver='broyden',     # Broyden for backward
            f_max_iter=30,          # max forward iterations
            b_max_iter=30,          # max backward iterations  
            f_tol=1e-5,             # convergence tolerance
            stop_mode='rel',        # relative residual stopping
        )
    
    def forward(self, state):
        # Initial guess: single pass
        z0 = self.bp_layer._single_pass(state)
        
        # Define the fixed-point function
        def f(messages):
            # Inject messages back into state temporarily, re-run BP
            return self.bp_layer._single_pass_with_messages(state, messages)
        
        # Solve for fixed point implicitly
        z_star = self.deq(f, z0)
        return z_star
```

**Why this is the best fit:**
1. Your BP loop is already a fixed-point iteration (damped messages converge to a fixed point)
2. DEQ finds the same fixed point but via Anderson acceleration (1.5-8x faster per IGNN-Solver results)
3. Backward pass uses implicit differentiation -- constant memory regardless of "iteration count"
4. Your learned damping parameter becomes unnecessary (the solver handles convergence)
5. TorchDEQ handles all the tricky parts: Jacobian regularization, phantom gradients for stability

### 2c. TorchDEQ Library
**Paper:** Geng, NeurIPS 2023 Workshop
**Repo:** github.com/locuslab/torchdeq
**Features:**
- Solvers: `anderson`, `broyden`, `fixed_point_iter`, `simple_fixed_point_iter`
- Backward: implicit differentiation (IFT), 1-step grad, phantom grad
- Jacobian regularization: `jac_loss_weight` parameter stabilizes training
- DEQ Zoo includes IGNN (implicit graph neural network) -- direct precedent
- Spectral radius estimation during validation

**Practical tips from TorchDEQ docs:**
- Apply BatchNorm/LayerNorm before and after the DEQ layer
- Use Jacobian regularization (jac_loss_weight ~0.01) for training stability
- Start with `fixed_point_iter` for debugging, switch to `anderson` for speed
- Monitor spectral radius of the Jacobian during validation (should be < 1)

### 2d. Implicit Graph Neural Networks (IGNN)
**Paper:** Gu & Chang, NeurIPS 2020
**Key idea:** Node representations are fixed points of an equilibrium equation. Uses Perron-Frobenius theory for well-posedness guarantees -- the spectral radius of |W| must be bounded by the inverse spectral radius of the adjacency matrix.

**Well-posedness condition:** lambda_pf(|W|) < lambda_pf(A)^{-1}

**Relevance:** Your `relation_transform` weight matrix plays the role of W. The condition says: keep `relation_transform`'s spectral norm bounded. This is easy to enforce with spectral normalization on that single Linear layer.

**IGNN-Solver (2024):** Accelerates IGNN inference 1.5-8x using learned Anderson acceleration. A tiny GNN learns the acceleration parameters, treating the fixed-point iterations as a graph-dependent temporal process. On graphs up to ogbn-products (~2.4M nodes), so your 4096-node graph is well within range.

### 2e. Monotone Operator Viewpoint
**Paper:** Baker et al., ICML 2023
**Key idea:** Recharacterizes IGNN well-posedness using monotone operator theory. This allows a more expressive parameterization (no Perron-Frobenius spectral radius constraint). Uses Cayley transform for orthogonal parameterization that stabilizes long-range dependency learning.

Uses Anderson-accelerated Douglas-Rachford splitting to solve for the fixed point efficiently.

### 2f. Subhomogeneous DEQ
**Paper:** Sittoni & Tudisco, ICML 2024
**Key idea:** Uses nonlinear Perron-Frobenius theory for weaker well-posedness assumptions. Allows general weight matrices (no spectral norm constraint) as long as activation functions are subhomogeneous and a final normalization layer is added.

**Code:** github.com/COMPiLELab/SubDEQ

**Advantage:** Your polar normalization (beliefs are radius * direction) is inherently a normalization layer. This makes the subhomogeneous framework natural -- your `F.normalize(...)` calls already provide the structural guarantee.

---

## 3. State-Space Models on Graphs

### 3a. Graph Mamba (GMN)
**Paper:** Behrouz & Hashemi, KDD 2024
**Key idea:** Adapts Mamba's selective SSM to graph-structured data. Since SSMs are sequential (require ordered input) but graphs are unordered, GMN addresses this with:
1. Neighborhood tokenization: convert local subgraphs to token sequences
2. Token ordering: define a traversal order (BFS, DFS, or learned)
3. Bidirectional selective SSM: scan forward and backward along the ordering

**Input-dependent dynamics:** Like Mamba, the B, C, Delta matrices are functions of the input at each position. This means the "transition" through the graph adapts to the content of each node.

**Performance:** Competitive with Graph Transformers at much lower cost. Captures long-range dependencies better than standard message-passing GNNs.

**Relevance to Memoria:** The ordering problem is the main challenge. Your beliefs don't have a natural sequence order. However, you could order them by:
- Precision (radius) -- high-confidence beliefs first
- Recency (belief_last_accessed)
- Topological sort of the edge graph
- Learned ordering (as in Graph Mamba)

**Limitation for your case:** Graph Mamba replaces the entire GNN architecture. It doesn't naturally map to factor-graph BP because it loses the variable/factor bipartite structure. The SSM processes a linearized sequence of nodes, not the factor graph topology directly.

### 3b. Graph-Mamba (Different Paper)
**Paper:** Wang et al., arXiv 2024
**Key idea:** Integrates Mamba block with input-dependent node selection mechanism for long-range context modeling. Uses graph-aware, node-wise selective state-space updates.

**STG-Mamba variant:** Integrates Kalman-style GNN fusion for spatiotemporal graph forecasting. The Kalman fusion is relevant to your precision-weighted updates.

### 3c. Assessment for Memoria
Graph SSMs are **not the right fit** for replacing BP on your factor graph. Reasons:
1. They destroy the bipartite factor graph structure
2. The node ordering problem adds complexity without clear benefit
3. Your graph is static within a training step (no temporal dimension to exploit)
4. The precision-weighted, directional message semantics don't map to SSM state transitions

Graph SSMs are better suited for node classification on large graphs with long-range dependencies, not for inference on factor graphs.

---

## 4. Amortized Inference (Single-Pass Without Any Iteration)

### 4a. Amortized Region Approximation
**Paper:** Lin, 2023
**Key idea:** Optimizes the Bethe/Kikuchi energy using a neural network that maps the factor graph structure directly to approximate marginals in a single forward pass. No message passing at all.

**How it works:** Train a GNN to take the factor graph as input and output the Bethe free energy minimizer directly. The GNN is trained on many factor graph instances so it "amortizes" the cost of iterative BP across a training distribution.

**Relevance:** This is the most aggressive option. Instead of iterating BP or using DEQ to find the fixed point, you train a separate network to predict what the BP fixed point would be. One forward pass, done.

**Risk:** May not generalize well as your factor graph evolves (beliefs are added/removed, edges change). The amortized network sees a distribution of graphs during training, but your cognitive state is a single evolving graph.

### 4b. Learning in Deep Factor Graphs with Gaussian BP
**Paper:** Nabarro, Van Der Wilk, Davison -- ICML 2024
**Key idea:** Treats all quantities (inputs, outputs, parameters, activations) as random variables in a Gaussian factor graph. Training and prediction are both inference problems. Uses Gaussian BP for both, with inherently local updates.

**Critical insight:** "Training and prediction are essentially the same computation" -- just different observed nodes in the same factor graph.

**Relevance:** Your precision-weighted messages are already a form of Gaussian BP (radius = precision, message = precision * direction). This paper validates the approach and shows it scales to deep networks with continual learning.

**Code:** github.com/sethnabarro/gbp_learning

### 4c. V-MPNN / TMPNN Progression
**V-MPNN (UAI 2022):** Learnable variational distribution parameterized by a neural network, minimizing a neural-augmented free energy. Does not require exact inference results for training.

**TMPNN (AISTATS 2024):** Replaces black-box neural messages with analytically derived message function from generalized Bethe free energy. Semantically meaningful parameters. Better generalization, more data-efficient.

---

## 5. Convergence and What's Lost

### 5a. What Iterative BP Guarantees
- On trees: exact marginals, convergence in O(diameter) steps
- On loopy graphs: converges to a stationary point of the Bethe free energy (if it converges)
- Damped BP: provably converges under certain spectral conditions (Mooij & Kappen 2005)
- Your learned damping ensures stability at the cost of convergence speed

### 5b. What Single-Pass / DEQ Preserves
- **DEQ:** Finds the same fixed point as iterative BP (it's the same equation, different solver). Convergence is guaranteed if the mapping is contractive (spectral radius < 1). The implicit differentiation backward pass is exact (not an approximation).
- **BPNN-D:** Provably maintains BP's convergence properties with learned corrections. Lower bound on partition function preserved.
- **Amortized:** No guarantee of finding the BP fixed point. The network approximates it. Quality depends on training distribution.

### 5c. What's Lost with Each Approach

| Approach | Variational Properties | Convergence Guarantee | Practical Risk |
|----------|----------------------|----------------------|----------------|
| DEQ + Anderson | Fully preserved | Yes (if contractive) | Solver may not converge in allotted iterations |
| IGNN | Preserved (fixed point = BP fixed point) | Yes (Perron-Frobenius) | Spectral norm constraint limits expressivity |
| SubhomDEQ | Preserved | Yes (weaker assumptions) | Normalization layer required |
| BPNN-D | Partially preserved (lower bound) | Yes on trees, approx on loops | Learned corrections may overfit |
| FGNN | Not guaranteed | Not guaranteed | Black-box message updates |
| Amortized | Not guaranteed | N/A (single pass) | May not generalize to novel graph states |
| Graph Mamba | Not applicable | N/A | Destroys factor graph structure |

### 5d. Oversmoothing / Oversquashing Tradeoff
If you replace BP iterations with GNN layers:
- **Oversmoothing:** Node features converge to indistinguishable vectors with too many layers. Your 4096-node graph with ~4 edges per node is sparse enough that this kicks in around 5-8 layers.
- **Oversquashing:** Information from distant nodes gets compressed through bottlenecks. With diameter ~10 in a sparse graph, you need ~10 layers to propagate information fully.
- **Sweet spot:** 2-4 layers with residual connections. DEQ avoids both problems by finding the fixed point directly.

---

## 6. Practical Implementation Plan

### Recommended Approach: DEQ Wrapper (Minimal Change, Maximum Benefit)

**Why DEQ is the best fit for Memoria:**
1. **Drop-in replacement:** Your `_single_pass` becomes the fixed-point function `f`. The DEQ solver replaces the for-loop in `forward()`.
2. **Same fixed point:** The DEQ finds the same fixed point as your current damped BP, just faster and with constant memory.
3. **Your polar normalization is the well-posedness guarantee:** Beliefs are `radius * direction` with normalized direction. This satisfies the subhomogeneous DEQ requirements.
4. **Spectral norm on `relation_transform`:** One line change ensures the IGNN well-posedness condition.
5. **Your learned damping/iterations become unnecessary:** The Anderson solver handles convergence adaptively.
6. **TorchDEQ is mature:** Well-tested library with good defaults.

### Implementation Steps

**Step 1: Install TorchDEQ**
```bash
pip install torchdeq
```

**Step 2: Add spectral normalization to relation_transform**
In `message_passing.py`:
```python
from torch.nn.utils import spectral_norm
self.relation_transform = spectral_norm(nn.Linear(relation_dim, belief_dim, bias=False))
```

**Step 3: Refactor `_single_pass` to accept/return message tensors**
Currently `_single_pass` reads beliefs from `state` and returns messages. For DEQ, it needs to accept previous messages as input (to define the fixed-point map).

```python
def _single_pass_deq(self, state, prev_messages, prev_precisions):
    """Fixed-point function: z -> f(z)
    
    Takes previous messages/precisions, updates beliefs accordingly,
    runs one BP pass, returns new messages/precisions.
    """
    # Temporarily blend prev_messages into beliefs (with damping)
    # Run the standard BP computation
    # Return new messages, precisions as a flat tensor
    ...
```

**Step 4: Wrap with TorchDEQ**
```python
import torchdeq

class DEQFactorGraphMP(nn.Module):
    def __init__(self, belief_dim, relation_dim):
        super().__init__()
        self.bp = FactorGraphMessagePassing(belief_dim, relation_dim)
        self.deq = torchdeq.get_deq(
            f_solver='anderson',
            b_solver='fixed_point_iter',
            f_max_iter=30,
            b_max_iter=25,
            f_tol=1e-5,
            b_tol=1e-5,
            stop_mode='rel',
        )
        # Jacobian regularization for training stability
        self.jac_reg = 0.01
    
    def forward(self, state):
        if not state.edge_active.any():
            return self._empty_result(state)
        
        # Initial guess from one BP pass
        init = self.bp._single_pass(state)
        z0 = torch.cat([init['messages'].flatten(), init['precisions']])
        
        # Fixed-point function
        def f(z):
            messages = z[:self.bp.belief_dim * state.config.max_beliefs].reshape(
                state.config.max_beliefs, self.bp.belief_dim)
            precisions = z[self.bp.belief_dim * state.config.max_beliefs:]
            result = self.bp._single_pass_with_prev(state, messages, precisions)
            return torch.cat([result['messages'].flatten(), result['precisions']])
        
        # Solve
        z_star, info = self.deq(f, z0)
        
        # Unpack
        messages = z_star[:self.bp.belief_dim * state.config.max_beliefs].reshape(
            state.config.max_beliefs, self.bp.belief_dim)
        precisions = z_star[self.bp.belief_dim * state.config.max_beliefs:]
        
        # Agreement from final state
        final = self.bp._single_pass(state)
        
        return {
            'messages': messages,
            'precisions': precisions,
            'agreement': final['agreement'],
            'solver_steps': info.get('nstep', -1),
        }
```

**Step 5: Add Jacobian regularization to the training loss**
```python
# In training loop, after computing DEQ output:
if hasattr(deq_result, 'jac_loss'):
    total_loss += 0.01 * deq_result['jac_loss']
```

**Step 6: Monitor convergence**
```python
# Log solver steps and residual norm
wandb.log({
    'deq/solver_steps': result['solver_steps'],
    'deq/spectral_radius': torchdeq.spectral_radius(f, z_star),
})
```

### Scale Considerations for 4096 Nodes, 16384 Edges

**Memory per iteration of current system:**
- Messages: 4096 x 256 = 1M floats (4 MB at fp32)
- Precisions: 4096 floats (16 KB)
- Agreement: 16384 floats (64 KB)
- Per iteration: ~4 MB. With 5 iterations: ~20 MB (storing intermediate messages for grad)

**Memory with DEQ:**
- Forward: Only stores z0 and z* (~8 MB total, constant regardless of iteration count)
- Backward: One additional fixed-point solve (~8 MB)
- Total: ~16 MB constant vs ~20 MB scaling with iterations

**Compute:**
- Anderson acceleration typically converges in 5-15 iterations for well-conditioned problems
- Each Anderson step costs ~1.1x one BP pass (small overhead for history management)
- Net effect: similar or fewer iterations, but constant memory and exact gradients

### What to Remove After DEQ Integration
- `_raw_damping` parameter (solver handles convergence)
- `_raw_iterations` parameter (solver determines when to stop)
- `damping` property
- `effective_iterations` property
- The for-loop in `forward()`

---

## 7. Alternative: Adaptive Depth Message Passing (ADMP-GNN)

If DEQ feels too heavy, a lighter alternative is per-node early exit:

**ADMP-GNN (CIKM 2025):** Each node independently decides when to stop receiving messages. Nodes with simple local structure exit after 1-2 layers; nodes in complex neighborhoods continue for more layers.

**Early-Exit GNN (2025):** Uses Gumbel-Softmax exit heads at each layer for differentiable per-node/graph-level halting.

**Relevance:** Your beliefs vary in how much they need updating. High-precision core beliefs (level 3) may need zero message-passing iterations, while newly-created beliefs (level 0) may need several. Per-node adaptive depth could reduce average compute significantly.

**Implementation:** Attach a learned exit gate to each belief that reads the residual between old and new messages. If the residual is below a threshold (learned), stop updating that belief.

---

## 8. Libraries and Tools

| Library | Purpose | URL | Status |
|---------|---------|-----|--------|
| TorchDEQ | DEQ solver, implicit differentiation | github.com/locuslab/torchdeq | Mature, well-documented |
| SubDEQ | Subhomogeneous DEQ (weaker constraints) | github.com/COMPiLELab/SubDEQ | Research code |
| torch-bp | Gaussian BP in PyTorch | github.com/janapavlasek/torch-bp | Small, focused |
| gbp_learning | Learning in factor graphs via GBP | github.com/sethnabarro/gbp_learning | ICML 2024 code |
| FGNN | Factor graph neural network | github.com/zzhang1987/Factor-Graph-Neural-Network | PyTorch |
| PyG | MessagePassing base class | pytorch-geometric.readthedocs.io | Already in your deps |
| locuslab/deq | Original DEQ code | github.com/locuslab/deq | NeurIPS 2019 |
| torchdyn | Neural ODEs / dynamical systems | github.com/DiffEqML/torchdyn | Alternative to TorchDEQ |

---

## 9. Decision Matrix

| Criterion | DEQ Wrapper | FGNN (2-layer) | Amortized GNN | Graph Mamba | ADMP-GNN |
|-----------|-------------|----------------|---------------|-------------|----------|
| Change to codebase | Minimal | Moderate | Large | Large | Moderate |
| Same fixed point as BP | Yes | No | No | No | Approx |
| Convergence guarantee | Yes (contractive) | No | No | No | Per-node |
| Memory scaling | Constant | Linear in layers | Constant | Linear | Sub-linear |
| Preserves precision semantics | Yes | Partially | Trainable | No | Yes |
| Preserves CoED directions | Yes | Needs adaptation | Trainable | No | Yes |
| Training stability | Good (with Jac reg) | Good | Good | Good | Good |
| Implementation effort | ~1 day | ~3 days | ~1 week | ~1 week | ~2 days |

---

## 10. Recommendation

**Primary: DEQ wrapper around existing `_single_pass`** using TorchDEQ with Anderson acceleration. This:
- Finds the same fixed point as your current loopy BP
- Eliminates the learned damping/iteration parameters
- Provides constant memory usage
- Has convergence guarantees (add spectral norm to `relation_transform`)
- Requires minimal code changes (~100 lines)
- TorchDEQ is well-tested with graph neural network examples (IGNN in their DEQ Zoo)

**Secondary (future): Per-node adaptive depth** via ADMP-GNN-style exit gates. This would let level-3 beliefs skip message passing entirely while level-0 beliefs get full propagation. Combine with DEQ for the "difficult" nodes.

**Do not pursue:** Graph Mamba for this specific task. The factor graph structure and precision-weighted semantics don't map to SSM state transitions. SSMs are better for sequence modeling, not factor graph inference.

---

## Sources

- [Belief Propagation Neural Networks (NeurIPS 2020)](https://arxiv.org/abs/2007.00295)
- [Factor Graph Neural Networks (JMLR 2023)](https://jmlr.org/papers/volume24/21-0434/21-0434.pdf)
- [Neural Enhanced Belief Propagation on Factor Graphs (AISTATS 2021)](https://arxiv.org/abs/2003.01998)
- [Theory-guided Message Passing Neural Network (AISTATS 2024)](https://proceedings.mlr.press/v238/cui24a/cui24a.pdf)
- [Deep Equilibrium Models (NeurIPS 2019)](https://arxiv.org/abs/1909.01377)
- [TorchDEQ Library (2023)](https://github.com/locuslab/torchdeq)
- [Implicit Graph Neural Networks (NeurIPS 2020)](https://arxiv.org/abs/2009.06211)
- [IGNN-Solver: Accelerating Implicit GNNs (2024)](https://arxiv.org/abs/2410.08524)
- [Implicit GNNs: A Monotone Operator Viewpoint (ICML 2023)](https://proceedings.mlr.press/v202/baker23a.html)
- [Subhomogeneous Deep Equilibrium Models (ICML 2024)](https://proceedings.mlr.press/v235/sittoni24a.html)
- [Graph Mamba: Learning on Graphs with SSMs (KDD 2024)](https://arxiv.org/abs/2402.08678)
- [Learning in Deep Factor Graphs with Gaussian BP (ICML 2024)](https://arxiv.org/abs/2311.14649)
- [V-MPNN: Variational Message Passing Neural Network (UAI 2022)](https://proceedings.mlr.press/v180/cui22a.html)
- [Monotone Operator Theory-Inspired Message Passing (AISTATS 2024)](https://proceedings.mlr.press/v238/baker24a.html)
- [Adaptive Depth Message Passing GNN (CIKM 2025)](https://arxiv.org/abs/2509.01170)
- [Early-Exit Graph Neural Networks (2025)](https://arxiv.org/html/2505.18088)
- [Anderson Acceleration for Fixed-Point Iterations](https://epubs.siam.org/doi/10.1137/10078356X)
- [Gaussian Belief Propagation Tutorial](https://gaussianbp.github.io/)
- [torch-bp: BP in PyTorch](https://github.com/janapavlasek/torch-bp)
- [PyG Factor Graph Discussion](https://github.com/pyg-team/pytorch_geometric/issues/2012)
- [Adaptive Message Passing Framework (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/93b7e2780c4f6599837fdd3718c51fad-Paper-Conference.pdf)
- [Implicit vs Unfolded Graph Neural Networks (JMLR 2025)](https://www.jmlr.org/papers/volume26/22-0459/22-0459.pdf)
- [Fast Deep Belief Propagation (2025)](https://www.mdpi.com/2227-7390/13/20/3349)
- [Positive Concave Deep Equilibrium Models (2024)](https://arxiv.org/html/2402.04029v2)
