# Memoria

A self-evolving memory runtime for AI agents, written in Rust.

Memoria gives agents a persistent, adaptive memory layer that learns from usage.
Memories are stored as dense vectors with typed metadata, scored by multiple signals
(embedding similarity, temporal activation, Hebbian association), and ranked by
weights that are learned online from agent feedback.

## Features

- **Adaptive retrieval** — Four built-in scorers (embedding, activation, Hebbian, BM25) with weights learned via REINFORCE
- **Cross-encoder reranking** — Optional reranker refines top results via joint (query, doc) scoring
- **Pluggable embeddings** — API embedder (OpenAI-compatible) or local ONNX (FastEmbed) — no feature flag needed for API mode
- **Pluggable rerankers** — API reranker (Jina/Cohere-compatible) or local ONNX (FastEmbed)
- **BM25 term matching** — Okapi BM25 inverted index for keyword-aware retrieval alongside dense vectors
- **Semantic chunking** — Built-in text chunker splits long documents before embedding
- **Hebbian associations** — Co-accessed memories strengthen links automatically
- **Exponential activation decay** — Per-memory time constants with kind-specific configuration
- **LSH indexing** — Locality-sensitive hashing for approximate nearest-neighbor search
- **Lifecycle hooks** — Before/after hooks for every operation (remember, recall, forget, patch, feedback, tick)
- **Pluggable extensions** — Custom scorers, hooks, secondary indexes, and kind configs
- **Batch operations** — Atomic multi-operation batches with per-op error isolation
- **Online learning** — Scorer weight updates, co-activation tracking, skill crystallization
- **Persistence** — CRC32-protected WAL + point-in-time snapshots (optional)
- **Kernel rules** — Unbypassable store-level enforcement: deny kinds, require fields, protect from deletion, redact on read — sealed at build time, enforced inside `MemoryStore` on every code path
- **Governance** — Visibility/existence/transition rules with conditional enforcement and agent-scoped transitions (optional)
- **Memory pinning** — Exempt critical memories from garbage collection
- **Optimistic concurrency** — Version-based conflict detection on patches
- **Hard TTL / expiration** — Deterministic expiry with `expires_at` for GDPR compliance
- **Bulk purge** — Atomic deletion by kind/field/namespace filter (right-to-be-forgotten)
- **Audit event stream** — Structured event logging via `MemoriaEvent` + `EventSink` + optional persistent audit log (optional)
- **Persistent audit trail** — SHA-256 hash-chained, append-only audit log with independent verification and querying (optional, feature = `persistence` + `audit`)
- **Memory namespaces** — Multi-tenant isolation with namespace-scoped queries and visibility rules
- **Scope enforcer** — Unbypassable, identity-aware namespace isolation via `AgentContext` (agent/user/team/org hierarchy)
- **Adaptive routing** — Liquid neural network router for multi-agent setups (optional)
- **Zero unsafe, zero unwrap** — All floats NaN-safe, all errors propagated

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
memoria = "0.1"
```

```rust
use memoria::{Memoria, Memory, Query, AgentId, AgentContext};

let memoria = Memoria::builder()
    .new(128)  // 128-dimensional embeddings
    .build()
    .unwrap();

// Store
let ctx = AgentContext::new(AgentId::new("my-agent"));
let memory = Memory::new("chat.message", vec![0.0_f32; 128]);
let id = memoria.remember(memory, &ctx).unwrap();

// Retrieve
let query = Query::new().with_embedding(vec![0.0_f32; 128]);
let results = memoria.recall(&query, &ctx).unwrap();

// Learn from feedback
if let Some(top) = results.first() {
    memoria.feedback(top.memory.id, &ctx, true);
}
```

## Feature Flags

No features are enabled by default. The core library compiles in ~2s with ~100 crates.

| Flag | What it enables |
|------|-----------------|
| `embeddings` | Local ONNX embedder + reranker via fastembed (~350 crates, ~130-260 MB RAM) |
| `routing` | Liquid neural network adaptive router |
| `governance` | Visibility/existence/transition rule hooks |
| `persistence` | WAL + snapshot durability |
| `observability` | `tracing` + `metrics` integration |
| `audit` | Structured audit event stream + persistent hash-chained audit log |
| `full` | All of the above |

Without `embeddings`, use `ApiEmbedderConfig` / `ApiRerankerConfig` to connect to any OpenAI-compatible embedding and Jina/Cohere-compatible reranking endpoint.

```toml
memoria = { version = "0.1", features = ["persistence", "governance"] }
```

## Architecture

```
Agent
  |
  v
Memoria (runtime)
  |-- MemoryStore (DashMap + LSH index + BM25 inverted index + kind index + kernel rules)
  |-- RetrievalEngine (4-scorer ranking pipeline + optional cross-encoder reranking)
  |-- Embedder (API [OpenAI-compat] or local ONNX [FastEmbed])
  |-- Reranker (API [Jina/Cohere-compat] or local ONNX)
  |-- ActivationManager (exponential decay per memory)
  |-- HebbianLayer (co-access association learning)
  |-- ScorerWeightLearner (REINFORCE policy gradient)
  |-- CoActivationTracker (abstraction emergence)
  |-- TextChunker (semantic text chunking)
  |-- Hooks (before/after lifecycle interceptors)
  |-- SecondaryIndexes (field-based O(1) lookup)
  |-- ScopeEnforcer (multi-tenant namespace access control)
  '-- KindRegistry (per-kind behavior config)
```

### Extension Points

| Trait | Purpose | Example |
|-------|---------|---------|
| `Scorer` | Custom scoring signal | Recency, domain relevance, user preference |
| `Hook` | Lifecycle interception | Rate limiting, audit logging, access control |
| `SecondaryIndex` | Fast field lookup | Index by author, tag, status |
| `KindConfig` | Per-kind behavior | Different decay rates, importance, cascade rules |
| `Embedder` | Text → vector embedding | `ApiEmbedder`, `FastEmbedder`, custom |
| `Reranker` | Cross-encoder reranking | `ApiReranker`, `FastEmbedReranker`, custom |

## Built-in Hook Packs

| Pack | Feature gate | Purpose |
|------|-------------|---------|
| `KernelRules` | — | Unbypassable store-level enforcement (sealed at build time) |
| `GovernanceHooks` | `governance` | Visibility and transition rules |
| `ObservabilityHooks` | — | Atomic operation counters |
| `SkillCrystallizer` | — | Episode-to-skill emergence |
| `TrajectoryTracker` | — | Per-agent causal chain tracking |
| `AntiPatternGuard` | — | Suppress known-bad retrieval paths |

## API Reference

See [api_spec.md](docs/api_spec.md) for the complete API specification with type signatures, method documentation, and usage examples.

## Requirements

- Rust 1.75+
- No runtime dependencies beyond what's in `Cargo.toml`

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
