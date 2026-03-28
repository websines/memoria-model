# Reference Implementations

Each module should reference existing implementations before writing from scratch.
Don't reinvent what's already battle-tested.

## Phase 1: Core (polar, state, free_energy)

| Module | Primary Reference | What to take |
|--------|------------------|--------------|
| `core/polar.py` | [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | PolarQuant math for radius/angle decomposition |
| `core/state.py` | Own design | — |
| `core/free_energy.py` | [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl) | Bethe free energy computation on factor graphs. See `src/objectives/bethe_free_energy.jl` for the canonical formulation |
| `core/free_energy.py` | [gaussianbp.github.io](https://gaussianbp.github.io/) | Gaussian BP tutorial — precision-weighted message passing as tensor ops. Clean reference. |
| `core/free_energy.py` | `prototype-research/src/aif/free_energy.rs` | Our own validated implementation (ported) |

## Phase 2: Interface (read/write paths)

| Module | Primary Reference | What to take |
|--------|------------------|--------------|
| `interface/read_path.py` | [hopfield-layers](https://github.com/ml-jku/hopfield-layers) | Content-addressable lookup. Use their `HopfieldLayer` or `HopfieldPooling` as the retrieval mechanism into belief region. Exponential storage capacity, proven. |
| `interface/write_path.py` | `prototype-research/src/aif/belief_update.rs` | Kalman-like gain formula, precision-gated updates |
| `interface/write_path.py` | [DeltaNet](https://arxiv.org/abs/2310.18020) / [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) | Delta rule for error-correcting state updates. More stable than simple additive. See `fla/ops/delta_rule/` |
| `interface/layer.py` | [Griffin/RecurrentGemma](https://github.com/google-deepmind/recurrentgemma) | How Google wires recurrent state + local attention in a hybrid architecture |

## Phase 3: Transformer Integration

| Module | Primary Reference | What to take |
|--------|------------------|--------------|
| `model/transformer.py` | [autoresearch train.py](https://github.com/karpathy/autoresearch) | Full GPT implementation: Muon optimizer, RoPE, QK-Norm, ReLU², value embeddings, softcap. We already read this file. |
| `model/transformer.py` | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Training speed tricks. Reference for optimized training loop. |
| `model/memoria_model.py` | [TTT layers](https://github.com/test-time-training/ttt-lm-pytorch) | How to wire mutable internal state into a transformer forward pass. Pass 2 gradient update mechanics. |
| `model/memoria_model.py` | [Griffin](https://github.com/google-deepmind/recurrentgemma) | Hybrid architecture pattern: which layers are attention, which are recurrent/state. |

## Phase 4: Pass 2 / Cognition

| Module | Primary Reference | What to take |
|--------|------------------|--------------|
| `cognition/surprise.py` | `prototype-research/src/dynamics/surprise.rs` | Surprise = prediction error × observation precision |
| `cognition/belief_update.py` | `prototype-research/src/aif/belief_update.rs` | Kalman-like gain, reconsolidation threshold |
| `cognition/belief_update.py` | [Titans](https://arxiv.org/abs/2501.00663) | Surprise-based memorization. They use gradient magnitude as surprise — we use precision-weighted prediction error. Compare approaches. |
| `cognition/hebbian.py` | `prototype-research/src/pipeline/hebbian.rs` | Saturating Hebb rule: w_new = w_old + η(1 - w_old) |
| `cognition/hebbian.py` | [Fast Weights (Ba & Hinton)](https://arxiv.org/abs/1610.06258) | Outer-product Hebbian updates. The original fast weights paper. |
| `cognition/telos.py` | `prototype-research/src/api/telos*.rs` | Full goal lifecycle, intrinsic generation, decomposition, progress, stall detection |
| `cognition/consolidation.py` | `prototype-research/src/dynamics/compression.rs` | Belief merging, co-activation clustering |
| `cognition/meta_learning.py` | `prototype-research/src/dynamics/meta_learning.rs` | SPSA: simultaneous perturbation stochastic approximation for self-tuning |
| `cognition/causal.py` | `prototype-research/src/causal/bayes_ball.rs` | d-separation via Bayes-Ball algorithm |
| `cognition/causal.py` | `prototype-research/src/causal/do_operator.rs` | Pearl's do-calculus: clamp + propagate |
| `cognition/causal.py` | [dfgo](https://github.com/brentyi/dfgo) | Differentiable factor graph optimization in PyTorch. How they handle graph ops as differentiable layers. |
| `cognition/pass2.py` | [TTT layers](https://github.com/test-time-training/ttt-lm-pytorch) | Mechanics of updating internal state during/after forward pass |

## Phase 5: Data

| Module | Primary Reference | What to take |
|--------|------------------|--------------|
| `data/streaming.py` | [HuggingFace datasets streaming docs](https://huggingface.co/docs/datasets/stream) | Streaming API for FineWeb-Edu and Stack v2 |
| `data/synthetic.py` | [CausalARC](https://huggingface.co/datasets/jmaasch/causal_arc) | Structure of interventional/counterfactual reasoning tasks |
| `data/synthetic.py` | [BaRDa](https://huggingface.co/papers/2312.07527) | Belief accuracy tasks with true/false entailments |

## Phase 6: Eval

| Module | Primary Reference | What to take |
|--------|------------------|--------------|
| `eval/causal.py` | [CausalBench](https://huggingface.co/datasets/CCLV/CausalBench) | Causal reasoning evaluation across domains |
| `eval/hallucination.py` | [Semantic Entropy](https://arxiv.org/abs/2406.15927) | Baseline comparison: their entropy-based detection vs. our precision-based architectural solution |

## Key Libraries to Install

```
pip install hopfield-layers      # Hopfield attention for read path
```

Check if flash-linear-attention has a pip package, otherwise clone:
```
git clone https://github.com/sustcsonglin/flash-linear-attention
```

## Papers to Keep Open While Coding

1. **Factor Graph Neural Networks** (Zhang 2020) — FGNN proves message passing = neural layer
2. **Differentiable Nonparametric Belief Propagation** (2021) — differentiable factor potentials
3. **Belief Propagation Neural Networks** (2020) — learned corrections to BP messages
4. **Titans** (Behrouz 2025) — surprise-driven memory, closest existing architecture
5. **TTT layers** (Sun 2024) — gradient-based internal state update mechanics
6. **HRM** (2025) — hierarchical multi-timescale recurrence at 27M params
