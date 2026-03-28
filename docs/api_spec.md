# Memoria API Specification

> A self-evolving memory runtime for AI agents.
> Version 0.1.0 · Rust 1.75+ · License: MIT OR Apache-2.0

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Feature Flags](#2-feature-flags)
3. [Core Types](#3-core-types)
4. [Configuration](#4-configuration)
5. [Builder API](#5-builder-api)
6. [Memoria Runtime API](#6-memoria-runtime-api)
7. [Extension Traits](#7-extension-traits)
8. [Built-in Hook Packs](#8-built-in-hook-packs)
9. [Persistence](#9-persistence)
10. [Audit Event Stream](#10-audit-event-stream-feature--audit)
11. [Kernel Rules](#11-kernel-rules--unbypassable-store-level-enforcement)
12. [Error Handling](#12-error-handling)
13. [Scope Enforcer](#13-scope-enforcer--multi-tenant-access-control)

---

## 1. Quick Start

```rust
use memoria::{MemoriaBuilder, Memory, Query, AgentId, AgentContext};

// Build a runtime with 128-dimensional embeddings
let memoria = MemoriaBuilder::new(128)
    .build()?;

// Store a memory
let ctx = AgentContext::new(AgentId::new("my-agent"));
let memory = Memory::new("chat.message", vec![0.0_f32; 128])
    .with_field("author", serde_json::json!("alice"));
let id = memoria.remember(memory, &ctx)?;

// Retrieve by similarity
let query = Query::new()
    .with_embedding(vec![0.0_f32; 128])
    .with_kind("chat.message")
    .with_limit(5);
let results = memoria.recall(&query, &ctx)?;

for scored in &results {
    println!("{}: {:.3} ({})", scored.memory.id, scored.score, scored.memory.kind);
}

// Provide feedback (drives online learning)
if let Some(top) = results.first() {
    memoria.feedback(top.memory.id, &ctx, true);
}
```

---

## 2. Feature Flags

No features are enabled by default. The core library (multi-signal retrieval, BM25, hooks, API embedder/reranker) compiles in ~2s with ~100 crates.

| Flag            | Default | Dependencies            | What it enables                        |
|-----------------|---------|-------------------------|----------------------------------------|
| `embeddings`    | off     | `fastembed`, ONNX runtime | Local ONNX embedder + reranker (~350 crates, ~130-260 MB RAM) |
| `routing`       | off     | —                       | Liquid neural network adaptive router  |
| `governance`    | off     | `sha2`, `hex`           | Visibility/existence/transition rules  |
| `persistence`   | off     | `memmap2`, `crc32fast`  | WAL + V2 snapshot durability — full restart survival |
| `observability` | off     | `tracing`, `metrics`    | Structured logging + metric counters   |
| `audit`         | off     | `sha2`                  | Structured audit event stream + persistent hash-chained audit log |
| `full`          | off     | all of the above        | Everything enabled                     |

Without `embeddings`, use `ApiEmbedderConfig` and `ApiRerankerConfig` to connect to any OpenAI-compatible embedding endpoint and Jina/Cohere-compatible reranking endpoint.

```toml
[dependencies]
memoria = { version = "0.1", features = ["persistence", "governance"] }
```

---

## 3. Core Types

### 3.1 `MemoryId`

Time-ordered, globally unique identifier wrapping UUID v7 + 16-bit partition hint.

```rust
pub struct MemoryId { /* uuid: Uuid, partition: u16 */ }

impl MemoryId {
    pub fn new() -> Self;                                    // partition 0, fresh v7
    pub fn with_partition(partition: u16) -> Self;           // custom partition
    pub fn from_uuid(uuid: Uuid) -> Self;                   // wrap existing UUID
    pub fn from_uuid_and_partition(uuid: Uuid, partition: u16) -> Self;
    pub fn uuid(&self) -> Uuid;
    pub fn partition(&self) -> u16;
}
```

**Traits:** `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, `Ord`, `PartialOrd`, `Serialize`, `Deserialize`, `Display`, `Default`

Lexicographic ordering matches creation-time ordering (UUID v7 property).

### 3.2 `Memory`

The core data unit. Every memory has a kind, an embedding vector, arbitrary JSON fields, and typed edges.

```rust
pub struct Memory {
    pub id: MemoryId,
    pub kind: String,                    // dot-separated, e.g. "chat.message"
    pub embedding: Vec<f32>,             // dense vector, must match store dim
    pub fields: Map<String, Value>,      // arbitrary JSON metadata
    pub edges: Vec<Edge>,                // typed relationships
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub pinned: bool,                    // default: false — if true, immune to GC
    pub version: u64,                    // default: 0 — incremented on every patch
    pub expires_at: Option<SystemTime>,  // hard TTL — memory removed on expiry
    pub namespace: Option<String>,       // multi-tenant isolation
}

impl Memory {
    pub fn new(kind: impl Into<String>, embedding: Vec<f32>) -> Self;
    pub fn with_id(self, id: MemoryId) -> Self;
    pub fn with_field(self, key: impl Into<String>, value: Value) -> Self;
    pub fn with_edge(self, edge: Edge) -> Self;
    pub fn pinned(self, pinned: bool) -> Self;
    pub fn expires_at(self, t: SystemTime) -> Self;
    pub fn with_ttl(self, duration: Duration) -> Self;
    pub fn with_namespace(self, ns: impl Into<String>) -> Self;
    pub fn validate(&self, expected_dim: usize) -> MemoriaResult<()>;
}
```

All new fields use `#[serde(default)]` for backward-compatible deserialization.

### 3.3 `Edge`

Directed, typed relationship between two memories.

```rust
pub struct Edge {
    pub kind: String,       // e.g. "calls", "part_of", "replied_to"
    pub target: MemoryId,
}

impl Edge {
    pub fn new(kind: impl Into<String>, target: MemoryId) -> Self;
}
```

### 3.4 `AgentId`

Opaque string identifier for an agent interacting with the runtime.

```rust
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(id: impl Into<String>) -> Self;
    pub fn as_str(&self) -> &str;
}
```

**Traits:** `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Display`, `Serialize`, `Deserialize`, `From<&str>`, `From<String>`

### 3.5 `AgentContext`

Hierarchical identity context for multi-tenant access control. Every public API method requires an `&AgentContext`.

```rust
pub struct AgentContext {
    pub agent: AgentId,
    pub user: Option<String>,
    pub team: Option<String>,
    pub org: Option<String>,
}

impl AgentContext {
    pub fn new(agent: AgentId) -> Self;
    pub fn with_user(self, user: impl Into<String>) -> Self;
    pub fn with_team(self, team: impl Into<String>) -> Self;
    pub fn with_org(self, org: impl Into<String>) -> Self;
    pub fn system() -> Self;       // internal-only context for GC, WAL replay
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> MemoriaResult<Self>;
}
```

**Traits:** `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Serialize`, `Deserialize`

### 3.6 `Query`

Fluent builder for retrieval queries.

```rust
pub struct Query {
    pub kind: Option<String>,
    pub field_filters: Vec<FieldFilter>,
    pub embedding: Option<Vec<f32>>,
    pub text: Option<String>,            // raw query text — enables BM25 scoring + reranking
    pub context: Vec<MemoryId>,          // recent IDs for Hebbian boost
    pub hints: Vec<QueryHint>,
    pub limit: usize,                    // default: 10
    pub namespace: Option<String>,       // filter by namespace (None = all)
}

impl Query {
    pub fn new() -> Self;
    pub fn with_kind(self, kind: impl Into<String>) -> Self;
    pub fn with_embedding(self, embedding: Vec<f32>) -> Self;
    pub fn with_text(self, text: impl Into<String>) -> Self;
    pub fn with_filter(self, filter: FieldFilter) -> Self;
    pub fn with_context(self, id: MemoryId) -> Self;
    pub fn with_context_list(self, ids: Vec<MemoryId>) -> Self;
    pub fn with_limit(self, limit: usize) -> Self;
    pub fn with_namespace(self, ns: impl Into<String>) -> Self;
    pub fn add_hint(self, hint: QueryHint) -> Self;
    pub fn is_vector_query(&self) -> bool;
}
```

### 3.7 `FieldFilter` & `FilterOp`

Hard predicates applied before scoring.

```rust
pub struct FieldFilter {
    pub field: String,
    pub op: FilterOp,
}

impl FieldFilter {
    pub fn matches(&self, fields: &Map<String, Value>) -> bool;
}

pub enum FilterOp {
    Eq(Value),           // exact JSON equality
    Ne(Value),           // not equal
    Gt(f64),             // numeric >
    Gte(f64),            // numeric >=
    Lt(f64),             // numeric <
    Lte(f64),            // numeric <=
    Contains(String),    // substring match (string fields)
    Exists,              // field is present (any value)
}
```

### 3.8 `QueryHint`

Per-memory scoring nudges applied after retrieval.

```rust
pub enum QueryHint {
    Suppress(MemoryId),  // strongly downweight
    Boost(MemoryId),     // strongly upweight
    Warn(MemoryId),      // attach warning if this memory appears
}
```

### 3.9 `MemoryPatch` & `EmbeddingUpdate`

Partial update descriptor for an existing memory.

```rust
pub struct MemoryPatch {
    pub set_fields: Option<Map<String, Value>>,
    pub remove_fields: Option<Vec<String>>,
    pub add_edges: Option<Vec<Edge>>,
    pub remove_edges: Option<Vec<MemoryId>>,
    pub set_kind: Option<String>,
    pub embedding_update: Option<EmbeddingUpdate>,
    pub set_pinned: Option<bool>,
    pub expect_version: Option<u64>,              // optimistic concurrency check
    pub set_expires_at: Option<Option<SystemTime>>, // Some(None) clears TTL
    pub set_namespace: Option<String>,
}

impl MemoryPatch {
    pub fn is_empty(&self) -> bool;
    pub fn set_field(self, key: impl Into<String>, value: Value) -> Self;
    pub fn remove_field(self, key: impl Into<String>) -> Self;
    pub fn add_edge(self, edge: Edge) -> Self;
    pub fn remove_edge(self, target: MemoryId) -> Self;
    pub fn with_kind(self, kind: impl Into<String>) -> Self;
    pub fn with_embedding_update(self, update: EmbeddingUpdate) -> Self;
    pub fn with_pinned(self, pinned: bool) -> Self;
    pub fn expect_version(self, version: u64) -> Self;
    pub fn with_expires_at(self, t: SystemTime) -> Self;
    pub fn clear_expires_at(self) -> Self;
    pub fn with_namespace(self, ns: impl Into<String>) -> Self;
}

pub enum EmbeddingUpdate {
    Replace(Vec<f32>),                           // overwrite entirely
    Blend { embedding: Vec<f32>, alpha: f32 },   // EMA: new = α·new + (1-α)·current
    Delta(Vec<f32>),                             // additive: current += delta
}
```

### 3.10 `ScoredMemory`

Retrieval result with scoring breakdown.

```rust
pub struct ScoredMemory {
    pub memory: Memory,
    pub score: f32,
    pub breakdown: Vec<(String, f32)>,  // (scorer_name, weighted_contribution)
}
```

### 3.11 Batch Types

```rust
pub enum MemoryOp {
    Remember(Memory),
    Forget(MemoryId),
    Patch(MemoryId, MemoryPatch),
    Link(MemoryId, MemoryId, String),    // (from, to, edge_kind)
    Unlink(MemoryId, MemoryId),          // (from, to)
}

pub struct BatchResult {
    pub succeeded: usize,
    pub failed: Vec<(usize, MemoriaError)>,  // (op_index, error)
}

pub struct MemoriaStats {
    pub memory_count: usize,
    pub hebbian_edges: usize,
    pub scorer_count: usize,
    pub hook_count: usize,
    pub index_count: usize,
}
```

---

## 4. Configuration

All config structs derive `Serialize`/`Deserialize` for JSON/TOML loading. Fields with defaults are annotated `#[serde(default)]`.

### 4.1 `MemoriaConfig`

Top-level config. Call `validate()` before use.

```rust
pub struct MemoriaConfig {
    pub store: StoreConfig,
    pub retrieval: RetrievalConfig,
    pub learning: LearningConfig,
    pub hooks: HookConfig,
    pub chunking: ChunkingConfig,

    /// API embedder config — used when no custom Embedder is provided
    /// and the `embeddings` feature is not enabled.
    pub embeddings_api: Option<ApiEmbedderConfig>,

    /// API reranker config — used when `enable_reranking` is true
    /// and the `embeddings` feature is not enabled.
    pub reranker_api: Option<ApiRerankerConfig>,

    #[cfg(feature = "embeddings")]
    pub embeddings: EmbeddingsConfig,

    #[cfg(feature = "persistence")]
    pub persistence: PersistenceConfig,

    #[cfg(feature = "routing")]
    pub routing: RoutingConfig,
}

impl MemoriaConfig {
    pub fn validate(&self) -> MemoriaResult<()>;
}
```

### 4.2 `StoreConfig`

```rust
pub struct StoreConfig {
    pub embedding_dim: usize,         // REQUIRED, no default — must be > 0
    pub capacity: usize,              // default: 100_000
    pub gc_threshold: f32,            // default: 0.01, range [0.0, 1.0]
    pub enable_lsh: bool,             // default: true
    pub lsh_num_tables: usize,        // default: 6 — more tables = higher recall
    pub lsh_num_bits: usize,          // default: 12 — bits per hash table
    pub hnsw_connectivity: usize,     // default: 0 (disabled) — M parameter for HNSW
    pub hnsw_expansion_add: usize,    // default: 0 — efConstruction for HNSW
    pub hnsw_expansion_search: usize, // default: 0 — ef for HNSW search
}
```

### 4.3 `RetrievalConfig`

```rust
pub struct RetrievalConfig {
    pub default_limit: usize,       // default: 10, must be > 0
    pub max_candidates: usize,      // default: 1024, must be >= default_limit
    pub spread_on_recall: bool,     // default: false — auto-spread from query.context
    pub spread_depth: usize,        // default: 2 — hops for auto-spread
    pub spread_damping: f32,        // default: 0.5 — energy decay per hop
    pub enable_reranking: bool,     // default: false — cross-encoder reranking after scoring
}
```

### 4.4 `LearningConfig`

Online learning hyperparameters (see spec §14 for equations).

| Field                        | Default | Range          | Description                          |
|------------------------------|---------|----------------|--------------------------------------|
| `weight_lr`                  | 0.02    | [0, 1]         | Per-memory importance learning rate  |
| `weight_min`                 | 0.01    | [0, 1]         | Floor on importance weight           |
| `weight_max`                 | 10.0    | (0, ∞)         | Ceiling on importance weight         |
| `baseline_decay`             | 0.99    | [0, 1]         | EMA decay for REINFORCE baseline     |
| `hebbian_lr`                 | 0.1     | [0, 1]         | Hebbian co-activation learning rate  |
| `hebbian_decay_rate`         | 0.01    | [0, 1]         | L2-style Hebbian weight decay        |
| `hebbian_temporal_window_secs` | 10.0  | (0, ∞)         | Co-access window (seconds)           |
| `hebbian_min_weight`         | 0.01    | [0, 1]         | Below this, Hebbian links are pruned |

### 4.5 `HookConfig`

```rust
pub struct HookConfig {
    pub strict_mode: bool,   // default: false — if true, hook errors abort the operation
}
```

### 4.6 `PersistenceConfig` (feature = `persistence`)

```rust
pub struct PersistenceConfig {
    pub wal_path: PathBuf,
    pub snapshot_path: PathBuf,
    pub sync_policy: SyncPolicy,            // default: EveryNEntries(100)
    pub auto_snapshot_interval: u64,        // default: 10_000 — WAL entries between auto-snapshots
}

pub enum SyncPolicy {
    Always,                       // flush after every WAL entry
    EveryNEntries(usize),         // flush every N entries (default N=100)
    Interval(Duration),           // flush on wall-clock interval
}
```

### 4.7 `RoutingConfig` (feature = `routing`)

```rust
pub struct RoutingConfig {
    pub hidden_size: usize,       // default: 64
    pub n_agents: usize,          // default: 8
    pub exploration_rate: f32,    // default: 0.1, range [0, 1]
    pub tau_min: f32,             // default: 0.1, must be > 0
    pub tau_max: f32,             // default: 5.0, must be >= tau_min
}
```

### 4.8 `ApiEmbedderConfig`

Configuration for an OpenAI-compatible embedding endpoint. Always available (no feature flag required).

```rust
pub struct ApiEmbedderConfig {
    pub base_url: String,           // e.g. "http://localhost:8080"
    pub api_key: Option<String>,    // optional bearer token
    pub model: String,              // e.g. "text-embedding-nomic-embed-text-v2"
    pub dim: usize,                 // embedding dimension (e.g. 768)
}
```

Sends `POST {base_url}/v1/embeddings` with `{ "model": "...", "input": ["text1", "text2"] }`. Expects `{ "data": [{ "embedding": [...], "index": 0 }] }`.

### 4.9 `ApiRerankerConfig`

Configuration for a Jina/Cohere-compatible reranking endpoint. Always available (no feature flag required).

```rust
pub struct ApiRerankerConfig {
    pub base_url: String,           // e.g. "http://localhost:8080"
    pub api_key: Option<String>,    // optional bearer token
    pub model: String,              // e.g. "jina-reranker-v3"
}
```

Sends `POST {base_url}/v1/rerank` with `{ "model": "...", "query": "...", "documents": ["doc1", "doc2"] }`. Expects `{ "results": [{ "index": 0, "relevance_score": 0.9 }] }`.

---

## 5. Builder API

```rust
let memoria = Memoria::builder()      // returns MemoriaBuilder with sentinel dim=0
    .new(128)                          // OR use Memoria::builder().config(cfg)
    .add_scorer(my_scorer)             // register custom scorers
    .add_hook(my_hook)                 // register lifecycle hooks
    .add_index(FieldIndex::new("author"))  // register secondary indexes
    .register_kind("chat.message", KindOpts {
        importance: 0.8,
        tau: 600.0,
        cascade: false,
    })
    .build()?;                         // validates config, wires all subsystems
```

### `MemoriaBuilder` methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(embedding_dim: usize) -> Self` | Create builder with dimension |
| `config` | `fn config(self, cfg: MemoriaConfig) -> Self` | Replace entire config |
| `persistence_paths` | `fn persistence_paths(self, wal: impl Into<PathBuf>, snap: impl Into<PathBuf>) -> Self` | Set WAL + snapshot paths (feature = `persistence`) |
| `embedder` | `fn embedder(self, e: impl Embedder + 'static) -> Self` | Override the auto-configured embedder |
| `add_scorer` | `fn add_scorer(self, s: impl Scorer + 'static) -> Self` | Register custom scorer |
| `add_hook` | `fn add_hook(self, h: impl Hook + 'static) -> Self` | Register lifecycle hook |
| `add_index` | `fn add_index(self, i: impl SecondaryIndex + 'static) -> Self` | Register secondary index |
| `register_kind` | `fn register_kind(self, kind: &str, kc: impl KindConfig + 'static) -> Self` | Kind-specific config |
| `kernel_rules` | `fn kernel_rules(self, rules: Arc<KernelRules>) -> Self` | Set unbypassable store-level enforcement rules |
| `grant` | `fn grant(self, grant: ScopeGrant) -> Self` | Register a scope enforcement grant |
| `audit_path` | `fn audit_path(self, path: impl Into<PathBuf>) -> Self` | Enable persistent hash-chained audit log (feature = `persistence` + `audit`) |
| `with_event_sink` | `fn with_event_sink(self, sink: EventSink) -> Self` | Register audit event sink (feature = `audit`) |
| `build` | `fn build(self) -> MemoriaResult<Memoria>` | Validate, restore from WAL/snapshot if present, and construct |

**Built-in scorers** (registered automatically by `build()`):
1. **EmbeddingScorer** — cosine similarity between query and candidate embeddings
2. **ActivationScorer** — current activation level (exponential decay since last access)
3. **HebbianScorer** — association strength with context memories
4. **BM25Scorer** — Okapi BM25 term-frequency scoring (requires `query.text`)

**Embedder initialization priority** (in `build()`):
1. User-provided via `.embedder()` — always wins
2. `FastEmbedder` — auto-constructed when `embeddings` feature is enabled
3. `ApiEmbedder` — constructed from `config.embeddings_api` when set
4. `None` — text convenience methods (`remember_text`, etc.) return `FeatureDisabled` error

**Reranker initialization priority** (when `enable_reranking` is true):
1. `FastEmbedReranker` — when `embeddings` feature is enabled
2. `ApiReranker` — constructed from `config.reranker_api` when set
3. `None` — reranking step is skipped

---

## 6. Memoria Runtime API

### 6.1 Store & Retrieve

#### `remember` — Store a memory

```rust
pub fn remember(&self, memory: Memory, ctx: &AgentContext) -> MemoriaResult<MemoryId>
```

**Pipeline:** before_remember hooks → validate → lookup kind τ → insert store → create activation → notify indexes → on_remember hooks

**Errors:** `Blocked`, `InvalidEmbeddingDim`, `InvalidParameter`

#### `recall` — Retrieve by query

```rust
pub fn recall(&self, query: &Query, ctx: &AgentContext) -> MemoriaResult<Vec<ScoredMemory>>
```

**Pipeline:** before_recall hooks → gather candidates (kind + ANN + indexes) → spread activation from context (if `spread_on_recall`) → score & rank (4 scorers) → adaptive router reweight (if routing) → field filters → cross-encoder reranking (if enabled and query has text) → on_recall hooks → record Hebbian co-access → record co-activation

**Errors:** `Blocked`

#### `get` — Retrieve by ID

```rust
pub fn get(&self, id: MemoryId) -> MemoriaResult<Option<Memory>>
```

Returns `Ok(None)` if the memory does not exist. Never fails.

#### `forget` — Remove a memory

```rust
pub fn forget(&self, id: MemoryId, ctx: &AgentContext) -> MemoriaResult<Option<Memory>>
```

Cleans up activation, Hebbian associations, co-activation tracking, and indexes.

**Errors:** `Blocked`

### 6.2 Incremental Updates

#### `patch` — Partial update

```rust
pub fn patch(&self, id: MemoryId, patch: MemoryPatch, ctx: &AgentContext) -> MemoriaResult<bool>
```

Returns `true` if found and patched, `false` if not found. If `expect_version` is set on the patch and doesn't match the memory's current version, returns `VersionConflict`.

**Patch application order:** version check → set_kind → remove_fields → set_fields → remove_edges → add_edges → embedding_update → set_pinned → set_expires_at → set_namespace → increment version → update timestamp

**Errors:** `Blocked`, `InvalidEmbeddingDim`, `VersionConflict`

#### `reinforce` — Blend embedding

```rust
pub fn reinforce(&self, id: MemoryId, new_embedding: &[f32], alpha: f32) -> MemoriaResult<bool>
```

Shorthand for `patch()` with `EmbeddingUpdate::Blend`. Formula: `result = α · new + (1 - α) · current`

#### `link` — Create typed edge

```rust
pub fn link(&self, from: MemoryId, to: MemoryId, kind: &str) -> MemoriaResult<()>
```

**Errors:** `MemoryNotFound` if `from` doesn't exist, `Blocked`

#### `unlink` — Remove edge

```rust
pub fn unlink(&self, from: MemoryId, to: MemoryId) -> MemoriaResult<bool>
```

### 6.3 Graph Traversal

#### `subgraph` — BFS graph traversal

```rust
pub fn subgraph(
    &self,
    root: MemoryId,
    depth: usize,
    opts: SubgraphOpts,
) -> MemoriaResult<Subgraph>
```

Returns a connected subgraph rooted at `root`, traversing up to `depth` hops through edges. Follows forward edges (source → target) and/or reverse edges (target → source) depending on `opts`.

```rust
pub struct SubgraphOpts {
    pub edge_kinds: Option<Vec<String>>,  // filter by edge kind (None = all)
    pub forward: bool,                     // follow forward edges (default: true)
    pub reverse: bool,                     // follow reverse edges (default: true)
    pub max_nodes: usize,                  // hard cap on nodes (default: 100)
}

pub struct Subgraph {
    pub root: MemoryId,
    pub memories: HashMap<MemoryId, Memory>,
    pub edges: Vec<(MemoryId, Edge)>,       // (source_id, edge)
    pub actual_depth: usize,                // deepest hop reached
}

impl Subgraph {
    pub fn node_count(&self) -> usize;
    pub fn edge_count(&self) -> usize;
}
```

**Errors:** `MemoryNotFound` if `root` doesn't exist

#### `spread_activation` — Energy propagation through edges

```rust
pub fn spread_activation(
    &self,
    seed: MemoryId,
    depth: usize,
    damping: f32,
) -> usize
```

Propagates activation energy from `seed` through explicit edges (forward + reverse). At each hop, energy decays by `damping` and is scaled by the Hebbian association weight between nodes. Returns the count of memories that received a boost.

**Short-circuits when:** seed energy < 1e-6, damping ≤ 0, or depth = 0.

### 6.4 Batch Operations

```rust
pub fn batch(&self, ops: Vec<MemoryOp>, ctx: &AgentContext) -> MemoriaResult<BatchResult>
```

Each operation runs independently. Failures are collected without aborting the batch.

```rust
let result = memoria.batch(vec![
    MemoryOp::Remember(memory_a),
    MemoryOp::Remember(memory_b),
    MemoryOp::Link(id_a, id_b, "related".into()),
    MemoryOp::Forget(old_id),
])?;
println!("{} succeeded, {} failed", result.succeeded, result.failed.len());
```

### 6.5 Learning

#### `accessed` — Record access

```rust
pub fn accessed(&self, id: MemoryId, ctx: &AgentContext)
```

Fire-and-forget. Spikes activation and records Hebbian co-access.

#### `feedback` — Record usefulness

```rust
pub fn feedback(&self, id: MemoryId, ctx: &AgentContext, useful: bool)
```

Updates scorer weights via REINFORCE, spikes activation, records Hebbian access, runs on_feedback hooks. Fire-and-forget.

### 6.6 Maintenance

#### `tick` — Advance time

```rust
pub fn tick(&self, dt_seconds: f32)
```

Decays activations, Hebbian edges, and co-activation counts. Runs on_tick hooks. Triggers auto-snapshot when WAL entries exceed `auto_snapshot_interval`. Call periodically (e.g. every second or every logical turn).

#### `purge` — Bulk delete by filter

```rust
pub fn purge(&self, filter: &PurgeFilter, ctx: &AgentContext) -> MemoriaResult<usize>
```

Atomically removes all memories matching the filter. Returns the count of removed memories. Runs `before_forget` hooks for each candidate. Writes a single `Purge` WAL entry for crash safety.

```rust
pub struct PurgeFilter {
    pub kind: Option<String>,
    pub field_filters: Vec<FieldFilter>,
    pub include_edges: bool,       // also remove memories with edges to purged ones
    pub namespace: Option<String>,
}

impl PurgeFilter {
    pub fn new() -> Self;
    pub fn with_kind(self, kind: impl Into<String>) -> Self;
    pub fn with_filter(self, filter: FieldFilter) -> Self;
    pub fn with_include_edges(self, include: bool) -> Self;
    pub fn with_namespace(self, ns: impl Into<String>) -> Self;
}
```

**Errors:** `Blocked` (per-memory, from hooks)

#### `gc` — Garbage collect

```rust
pub fn gc(&self)
```

Two-phase garbage collection:
1. **Phase 1 — Hard expiry:** Removes memories where `now > expires_at`. Ignores pinning — expired memories are always removed.
2. **Phase 2 — Activation decay:** Removes memories whose activation has decayed below `gc_threshold`. Respects pinning — pinned memories are never removed by activation decay.

### 6.7 Post-Build Extension

These require `&mut self`:

```rust
pub fn add_scorer(&mut self, scorer: impl Scorer + 'static);
pub fn add_hook(&mut self, hook: impl Hook + 'static);
pub fn add_index(&mut self, index: impl SecondaryIndex + 'static);
```

This requires only `&self` (uses interior mutability):

```rust
pub fn register_kind(&self, kind: &str, config: impl KindConfig + 'static);
```

### 6.8 Introspection

```rust
pub fn len(&self) -> usize;
pub fn is_empty(&self) -> bool;
pub fn stats(&self) -> MemoriaStats;
```

---

## 7. Extension Traits

Memoria exposes four extension traits. Implement any combination to customize behavior.

### 7.1 `Scorer` — Custom scoring signal

```rust
pub trait Scorer: Send + Sync {
    fn name(&self) -> &str;
    fn score(&self, query: &Query, candidate: &Memory, ctx: &ScorerContext) -> f32;
}

pub struct ScorerContext<'a> {
    pub agent: &'a AgentId,
}
```

Scorer weights are learned automatically from feedback via REINFORCE. Return a raw signal; the runtime normalizes and weights it.

```rust
struct RecencyScorer;
impl Scorer for RecencyScorer {
    fn name(&self) -> &str { "recency" }
    fn score(&self, _query: &Query, candidate: &Memory, _ctx: &ScorerContext) -> f32 {
        let age = candidate.created_at.elapsed().unwrap_or_default().as_secs_f32();
        1.0 / (1.0 + age / 3600.0)  // half-life of 1 hour
    }
}
```

### 7.2 `Hook` — Lifecycle hooks

```rust
pub trait Hook: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> i32 { 0 }  // higher = runs first

    // Before-hooks (can block or modify)
    fn before_remember(&self, memory: &mut Memory, ctx: &BeforeHookContext) -> HookAction;
    fn before_recall(&self, query: &mut Query, ctx: &BeforeHookContext) -> HookAction;
    fn before_forget(&self, id: MemoryId, ctx: &BeforeHookContext) -> HookAction;
    fn before_patch(&self, id: MemoryId, patch: &mut MemoryPatch, ctx: &BeforeHookContext) -> HookAction;

    // After-hooks (observe and react)
    fn on_remember(&self, memory: &Memory, ctx: &AfterHookContext);
    fn on_recall(&self, query: &Query, results: &mut Vec<ScoredMemory>, ctx: &AfterHookContext);
    fn on_forget(&self, id: MemoryId, ctx: &AfterHookContext);
    fn on_patch(&self, id: MemoryId, patch: &MemoryPatch, ctx: &AfterHookContext);
    fn on_feedback(&self, id: MemoryId, useful: bool, ctx: &AfterHookContext);
    fn on_tick(&self, dt: f32, ctx: &AfterHookContext);
}

pub enum HookAction {
    Continue,
    Block { reason: String },
}

pub struct BeforeHookContext<'a> {
    pub agent: &'a AgentId,
    pub agent_context: &'a AgentContext,
    pub current_memory: Option<&'a Memory>,
}
pub struct AfterHookContext<'a> {
    pub agent: &'a AgentId,
    pub agent_context: &'a AgentContext,
}
```

**Semantics:**
- Before-hooks see **proposed** state. First `Block` wins — remaining hooks are skipped.
- After-hooks see **committed** state. All run, no short-circuiting.
- All hook methods have default no-op implementations — override only what you need.

### 7.3 `SecondaryIndex` — Custom indexing

```rust
pub trait SecondaryIndex: Send + Sync {
    fn name(&self) -> &str;
    fn on_insert(&self, id: MemoryId, memory: &Memory);
    fn on_remove(&self, id: MemoryId, memory: &Memory);
    fn on_update(&self, id: MemoryId, old: &Memory, new: &Memory);
    fn lookup(&self, key: &str) -> Vec<MemoryId>;
}
```

**Built-in:** `FieldIndex` — indexes a single JSON field for O(1) lookup.

```rust
let index = FieldIndex::new("author");
// After registration, queries with FieldFilter on "author" will use this index
```

### 7.4 `KindConfig` — Per-kind behavior

```rust
pub trait KindConfig: Send + Sync {
    fn importance(&self) -> f32 { 0.5 }           // base importance weight
    fn default_tau(&self) -> f32 { 300.0 }         // activation decay time constant (seconds)
    fn cascade_on_forget(&self) -> bool { false }  // remove dependents on forget
    fn pinned(&self) -> bool { false }             // auto-pin memories of this kind
    fn max_ttl_secs(&self) -> Option<u64> { None } // hard TTL cap for this kind
}
```

**Built-in:** `KindOpts` — simple struct implementing `KindConfig`.

```rust
let opts = KindOpts {
    importance: 0.9,
    tau: 600.0,           // slower decay (10 min half-life)
    cascade: true,        // cascade deletion
    pinned: false,        // auto-pin memories of this kind
    max_ttl_secs: None,   // hard TTL cap (e.g. Some(86400) for 1 day)
};
memoria.register_kind("chat.message", opts);
```

### 7.5 `Embedder` — Text embedding

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> MemoriaResult<Vec<Vec<f32>>>;
    fn embed_one(&self, text: &str) -> MemoriaResult<Vec<f32>>;
}
```

**Built-in implementations:**
- `ApiEmbedder` — OpenAI-compatible HTTP client (always available)
- `FastEmbedder` — Local ONNX inference via fastembed (feature = `embeddings`)

### 7.6 `Reranker` — Cross-encoder reranking

```rust
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, documents: &[&str], limit: usize) -> MemoriaResult<Vec<usize>>;
}
```

Returns up to `limit` indices into the original `documents` slice, ordered by descending relevance.

**Built-in implementations:**
- `ApiReranker` — Jina/Cohere-compatible HTTP client (always available)
- `FastEmbedReranker` — Local ONNX inference (feature = `embeddings`)

---

## 8. Built-in Hook Packs

Located in `memoria::hooks::*`. Register via `add_hook()`.

### 8.1 `GovernanceHooks` (feature = `governance`)

```rust
use memoria::hooks::governance::GovernanceHooks;
```

Enforces visibility rules, existence checks, and state transition constraints.

- **ExistenceRule** — Block `remember()` if required fields are missing. Supports conditional enforcement via `when_field` (a `FieldFilter` that must match for the rule to apply).
- **VisibilityRule** — Hide or redact memories during `recall()` based on agent role, memory kind, and/or namespace.
- **TransitionRule** — Enforce valid state transitions with optional agent-scoped restrictions via `AllowedTransition::with_agents()`.

```rust
pub struct ExistenceRule {
    pub name: String,
    pub memory_kind: String,
    pub required_fields: Vec<String>,
    pub deny_reason: Option<String>,
    pub when_field: Option<FieldFilter>,  // conditional enforcement
}

pub struct VisibilityRule {
    pub name: String,
    pub agent_role: Option<String>,
    pub memory_kind: Option<String>,
    pub memory_namespace: Option<String>,  // namespace-scoped visibility
    pub visibility: Visibility,
}

pub struct AllowedTransition {
    pub from: String,
    pub to: String,
    pub allowed_agents: Option<Vec<String>>,  // substring match on agent ID
}

impl AllowedTransition {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self;
    pub fn with_agents(from: impl Into<String>, to: impl Into<String>, agents: Vec<String>) -> Self;
}
```

### 8.2 `ObservabilityHooks`

```rust
use memoria::hooks::observability::ObservabilityHooks;
```

Atomic operation counters. Tracks remember/recall/forget/patch/feedback counts via `AtomicU64`. Call snapshot methods to read current counts.

### 8.3 `SkillCrystallizer`

```rust
use memoria::hooks::skills::{SkillCrystallizer, SkillConfig};

let crystallizer = SkillCrystallizer::new(SkillConfig {
    min_episodes: 3,
    similarity_threshold: 0.7,   // base threshold — adapted per-seed by local density
    density_k: 5,                // k-NN for local density estimation
    proficiency_decay_rate: 0.001,
    transfer_ratio: 0.3,
});
```

Detects repeated successful episodes and crystallizes them into reusable "skill" memories using **density-adaptive clustering**:

1. Precomputes pairwise cosine similarity for all successful episodes
2. Estimates local density per episode (mean similarity to `density_k` nearest neighbours)
3. Sorts seeds by density descending (densest-first — **order-independent**)
4. Per-seed adaptive threshold: `base + 0.15 * (local_density - global_mean)`, clamped to `[0.5, 0.95]`
5. Proficiency blends cluster size with **cohesion** (average pairwise similarity): `0.6 * size_factor + 0.4 * cohesion`

Hooks into `on_feedback` to track episode outcomes.

### 8.4 `TrajectoryTracker`

```rust
use memoria::hooks::trajectory::TrajectoryTracker;
```

Per-agent step tracking via `DashMap<AgentId, Vec<Step>>`. Records the full causal chain of recalls, actions, and outcomes for each agent task.

### 8.5 `AntiPatternGuard`

```rust
use memoria::hooks::anti_patterns::AntiPatternGuard;
```

Learns from failures and suppresses known-bad retrieval paths. Implements `before_recall` to inject `QueryHint::Suppress` for memories associated with past failures.

---

## 9. Persistence (feature = `persistence`)

Memoria survives process restarts. On `build()`, it automatically restores all state from the snapshot and WAL. No manual restore calls needed.

### 9.1 Startup Recovery

When `build()` finds existing persistence files, it runs:

1. **Snapshot restore** — Load the V2 snapshot file (if it exists). Restores: memories, activation states, Hebbian co-access edges, and REINFORCE scorer weights.
2. **WAL replay** — Read WAL entries with sequence numbers newer than the snapshot. Apply inserts, removes, patches, links, activation updates, and scorer weight changes.
3. **Open WAL** — Open the WAL for new writes.

Recovery modes:
- **Snapshot + WAL delta** — Normal case. Snapshot provides the bulk, WAL fills the gap.
- **WAL-only** — No snapshot file exists (e.g. first run, or crash before first snapshot). Full replay from sequence 0.
- **Snapshot-only** — WAL is empty or unreadable after the snapshot point.

### 9.2 What persists

| State | Persisted via | Restored via |
|-------|---------------|-------------|
| Memories (content, embeddings, fields, edges) | WAL `Insert`/`Remove`/`Patch`/`Link`/`Unlink` + Snapshot V2 | Snapshot restore + WAL replay |
| Activation states (level, tau, access count) | WAL `Activation` + Snapshot V2 | Snapshot restore + WAL replay |
| Hebbian co-access edges (weight, reinforcement count) | Snapshot V2 | Snapshot restore |
| Scorer weights (REINFORCE-learned) | WAL `Batch` + Snapshot V2 | Snapshot restore + WAL replay |

### 9.3 Write-Ahead Log

CRC32-protected, append-only log with configurable sync policies. Each frame:

```
┌──────────┬──────────┬───────────┬──────────┬───────┐
│ EntryType│ Sequence │ Timestamp │ PayloadLen│ CRC32 │
│  1 byte  │  8 bytes │  8 bytes  │  4 bytes  │ 4 bytes│
├──────────┴──────────┴───────────┴──────────┴───────┤
│                    Payload                          │
└────────────────────────────────────────────────────┘
```

Entry types: `Insert(0x01)`, `Remove(0x02)`, `Patch(0x03)`, `Activation(0x04)`, `Hebbian(0x05)`, `Link(0x06)`, `Unlink(0x07)`, `Checkpoint(0x08)`, `Batch/ScorerWeights(0x09)`, `Purge(0x0A)`, `AuditDeny(0x0B)`, `Recall(0x0C)`, `Feedback(0x0D)`, `ScopeDeny(0x0E)`

### 9.4 Snapshot V2

```
┌───────────────────────────────────────────────┐
│ Magic "MMEM"  │ Version=2  │ EmbeddingDim     │
│ RecordCount   │ WalSequence                    │
├───────────────────────────────────────────────┤
│ Memory records (length-prefixed JSON)          │
├───────────────────────────────────────────────┤
│ ActivationCount + [MemoryId + ActivationDto]   │
│ HebbianCount + [IdA + IdB + HebbianEdgeDto]    │
│ ScorerWeightsDto (length-prefixed JSON)        │
├───────────────────────────────────────────────┤
│ Trailing CRC32                                 │
└───────────────────────────────────────────────┘
```

V1 snapshots (memories only) are still readable — the reader detects the version and returns empty activation/Hebbian/scorer data for V1 files.

### 9.5 Auto-Snapshot

When `tick()` is called, Memoria checks if the WAL sequence has advanced beyond `auto_snapshot_interval` (default: 10,000 entries) since the last snapshot. If so, it automatically takes a new snapshot and writes a WAL checkpoint marker.

### 9.6 `snapshot()` — Manual snapshot

```rust
pub fn snapshot(&self) -> MemoriaResult<()>
```

Saves a V2 snapshot containing all memories, activation states, Hebbian edges, and scorer weights. The snapshot records the current WAL sequence so that future WAL replay only processes newer entries.

### 9.7 Serialization

JSON-based encode/decode with fixed 18-byte MemoryId binary format (16 bytes UUID + 2 bytes partition). DTOs for non-serializable types:

- `ActivationStateDto` — Converts `Instant` to elapsed seconds for serialization
- `HebbianEdgeDto` — Converts `Instant` fields to elapsed seconds
- `ScorerWeightsDto` — Flat snapshot of scorer names, weights, and REINFORCE baseline

---

## 10. Audit Event Stream (feature = `audit`)

When the `audit` feature is enabled, every core operation emits a structured `MemoriaEvent` to a registered `EventSink`.

```rust
use memoria::audit::{MemoriaEvent, EventSink};

pub enum MemoriaEvent {
    Remembered { id: MemoryId, kind: String, agent_context: AgentContext, timestamp: SystemTime },
    Recalled { query_kind: Option<String>, result_count: usize, agent_context: AgentContext, timestamp: SystemTime },
    Forgotten { id: MemoryId, agent_context: AgentContext, timestamp: SystemTime },
    Patched { id: MemoryId, agent_context: AgentContext, new_version: u64, timestamp: SystemTime },
    Purged { count: usize, agent_context: AgentContext, timestamp: SystemTime },
    GcCompleted { removed_count: usize, timestamp: SystemTime },
    Blocked { operation: String, hook: String, reason: String, agent_context: AgentContext, timestamp: SystemTime },
}

pub type EventSink = std::sync::mpsc::Sender<MemoriaEvent>;
```

Register via `MemoriaBuilder::with_event_sink(sink)`. Events are sent fire-and-forget (`let _ = sink.send(...)`) — a dropped receiver does not block operations.

### 10.2 Persistent Audit Log (feature = `persistence` + `audit`)

When both `persistence` and `audit` features are enabled, Memoria can write a SHA-256 hash-chained, append-only audit log that is never truncated (unlike the recovery WAL).

```rust
let memoria = MemoriaBuilder::new(768)
    .audit_path("data/audit.bin")
    .build()?;
```

Every mutation, recall, feedback, and scope denial is recorded. The log is self-describing and independently verifiable.

#### Frame format

```
┌──────────┬──────────┬───────────┬───────────┬──────────┬───────┬─────────┬─────────┐
│  CRC32   │  Length  │ PrevHash  │ EntryType │ Sequence │  TS   │ CtxLen  │ Context │
│  4 bytes │ 4 bytes │ 32 bytes  │  1 byte   │ 8 bytes  │8 bytes│ 2 bytes │ M bytes │
├──────────┴──────────┴───────────┴───────────┴──────────┴───────┴─────────┴─────────┤
│                               Payload (N bytes)                                     │
└────────────────────────────────────────────────────────────────────────────────────┘
```

#### AuditEntry & AuditReader

```rust
pub struct AuditEntry {
    pub entry_type: WalEntryType,
    pub sequence: u64,
    pub timestamp: u64,
    pub prev_hash: [u8; 32],
    pub agent_context: AgentContext,
    pub payload: Vec<u8>,
}

pub struct AuditReader { /* ... */ }
impl Iterator for AuditReader {
    type Item = MemoriaResult<AuditEntry>;
}
```

#### Verification

```rust
use memoria::persistence::audit_log::audit_verify;

// Walks the audit log and recomputes the SHA-256 hash chain.
// Returns Ok(()) if the chain is intact, or the first mismatch.
audit_verify("data/audit.bin")?;
```

#### Querying

```rust
use memoria::persistence::audit_log::{AuditFilter, audit_query};

let entries = audit_query("data/audit.bin", &AuditFilter {
    agent: Some("agent-1".into()),
    entry_types: None,
    since: None,
    until: None,
    memory_id: None,
    limit: 100,
    ..Default::default()
})?;
```

#### Real-time streaming

```rust
pub trait AuditSink: Send + Sync {
    fn on_event(&self, entry: &AuditEntry);
}
```

---

## 11. Kernel Rules — Unbypassable Store-Level Enforcement

Unlike the hook-based `GovernanceHooks` (which can be bypassed by internal code paths like WAL replay, GC, or snapshot restore), kernel rules are enforced **inside `MemoryStore`** on every `insert()`, `update()`, `remove()`, and `get()` call. There is no code path that can skip them.

```
┌─────────────────────────────────────────────┐
│  Hook Layer (GovernanceHooks)               │  ← Flexible, agent-aware, runtime-configurable
│  before_remember / on_recall / before_patch │  ← Applied at Memoria API boundary
├─────────────────────────────────────────────┤
│  Kernel Layer (KernelRules)                 │  ← Sealed at build time, unbypassable
│  insert() / update() / remove() / get()    │  ← Every code path goes through here
└─────────────────────────────────────────────┘
```

Kernel rules are **sealed after construction** — once built via `KernelRulesBuilder`, they cannot be modified at runtime.

### 11.1 `KernelRules`

```rust
pub struct KernelRules {
    existence_rules: Vec<KernelExistenceRule>,
    transition_rules: Vec<KernelTransitionRule>,
    deletion_rules: Vec<KernelDeletionRule>,
    visibility_rules: Vec<KernelVisibilityRule>,
    audit: Option<Arc<dyn KernelAuditSink>>,
}

impl KernelRules {
    pub fn empty() -> Arc<Self>;   // no-op rule set (all operations allowed)
    pub fn is_empty(&self) -> bool;

    // Enforcement (called inside MemoryStore):
    pub fn enforce_insert(&self, memory: &Memory) -> MemoriaResult<()>;
    pub fn enforce_update(&self, old: &Memory, new: &Memory) -> MemoriaResult<()>;
    pub fn enforce_delete(&self, memory: &Memory) -> MemoriaResult<()>;
    pub fn redact(&self, memory: &mut Memory);
}
```

### 11.2 Kernel Rule Types

**`KernelExistenceRule`** — Checked in `store.insert()`

```rust
pub struct KernelExistenceRule {
    pub name: String,              // audit/diagnostics
    pub memory_kind: String,       // exact match on memory.kind
    pub deny: bool,                // if true, this kind cannot be created
    pub required_fields: Vec<String>,
    pub when_field: Option<FieldFilter>,  // conditional enforcement
}
```

**`KernelTransitionRule`** — Checked in `store.update()`

```rust
pub struct KernelTransitionRule {
    pub name: String,
    pub memory_kind: String,
    pub field: String,                          // governed field
    pub allowed_transitions: Vec<(String, String)>,  // (from, to) pairs
}
```

No agent scoping — the store doesn't know about agents. Agent-scoped transitions remain in `GovernanceHooks`.

**`KernelDeletionRule`** — Checked in `store.remove()`

```rust
pub struct KernelDeletionRule {
    pub name: String,
    pub memory_kind: String,
    pub protect: bool,             // if true, cannot be deleted
    pub when_field: Option<FieldFilter>,  // conditional protection
}
```

**`KernelVisibilityRule`** — Checked in `store.get()`

```rust
pub struct KernelVisibilityRule {
    pub name: String,
    pub memory_kind: String,
    pub redacted_fields: Vec<String>,  // replaced with null on every read
}
```

Unlike hook-layer `VisibilityRule`, this has no agent scoping — fields are **always** redacted regardless of who reads.

### 11.3 `KernelOp` & `KernelAuditSink`

```rust
pub enum KernelOp { Insert, Update, Delete, Read }

pub trait KernelAuditSink: Send + Sync {
    fn on_deny(&self, rule_name: &str, memory_id: MemoryId, op: KernelOp, reason: &str);
}
```

When the `persistence` feature is enabled, `WalAuditSink` logs denials to the WAL as `AuditDeny (0x0B)` entries for a tamper-evident audit trail.

### 11.4 `KernelRulesBuilder`

```rust
let rules = KernelRulesBuilder::new()
    .deny_kind("secret.raw", "raw secrets cannot be stored")
    .require_fields("audit.log", &["author", "timestamp"])
    .protect_kind("compliance.record")
    .redact_fields("user.profile", &["ssn", "credit_card"])
    .add_transition_rule(KernelTransitionRule {
        name: "status-flow".into(),
        memory_kind: "task".into(),
        field: "status".into(),
        allowed_transitions: vec![
            ("draft".into(), "review".into()),
            ("review".into(), "approved".into()),
        ],
    })
    .audit_sink(my_audit_sink)
    .build();  // returns Arc<KernelRules>
```

| Method | Description |
|--------|-------------|
| `deny_kind(kind, reason)` | Block creation of memories with this kind |
| `require_fields(kind, &[fields])` | Require fields when creating memories of this kind |
| `protect_kind(kind)` | Prevent deletion of memories with this kind |
| `redact_fields(kind, &[fields])` | Always redact these fields on read |
| `add_existence_rule(rule)` | Add a fully specified existence rule |
| `add_transition_rule(rule)` | Add a transition rule governing field changes |
| `add_deletion_rule(rule)` | Add a fully specified deletion rule |
| `add_visibility_rule(rule)` | Add a fully specified visibility rule |
| `audit_sink(sink)` | Set an audit sink for denial events |
| `build()` | Produce an immutable `Arc<KernelRules>` |

### 11.5 Builder Integration

```rust
use memoria::{MemoriaBuilder, KernelRulesBuilder};

let rules = KernelRulesBuilder::new()
    .deny_kind("secret.raw", "no raw secrets")
    .protect_kind("compliance.record")
    .redact_fields("user.profile", &["ssn", "credit_card"])
    .build();

let memoria = MemoriaBuilder::new(128)
    .kernel_rules(rules)
    .build()?;
```

**Errors:** Kernel rule violations return `MemoriaError::KernelDenied { rule, reason }`.

---

## 12. Error Handling

All fallible operations return `MemoriaResult<T>` which is `Result<T, MemoriaError>`.

```rust
#[non_exhaustive]
pub enum MemoriaError {
    // Config & validation
    InvalidEmbeddingDim { dim: usize },
    InvalidParameter { param: &'static str, value: String, reason: &'static str },
    OutOfRange { param: &'static str, value: String, lo: String, hi: String },
    FeatureDisabled { feature: &'static str },

    // Not-found & state
    MemoryNotFound { id: Uuid },
    VersionConflict { id: Uuid, expected: u64, actual: u64 },
    IllegalTransition { from: String, to: String },
    Blocked { hook: String, reason: String },

    // Kernel enforcement
    KernelDenied { rule: String, reason: String },

    // Scope enforcement
    ScopeDenied { agent: String, namespace: String, permission: String },

    // Persistence (feature = "persistence")
    #[cfg(feature = "persistence")] Io(std::io::Error),
    #[cfg(feature = "persistence")] WalChecksum { offset: u64, expected: u32, computed: u32 },
    #[cfg(feature = "persistence")] WalSequence { offset: u64, expected: u64, actual: u64 },
    #[cfg(feature = "persistence")] SnapshotFormat { reason: String },

    // Serialization
    Encode { message: String, source: Option<Box<dyn Error + Send + Sync>> },
    Decode { message: String, source: Option<Box<dyn Error + Send + Sync>> },

    // Internal
    LockOrderingViolation { first: &'static str, second: &'static str, required_order: &'static str },
    InvariantViolation { message: String },
}
```

**Convenience constructors:**

```rust
MemoriaError::encode("context", source_error);
MemoriaError::encode_msg("message only");
MemoriaError::decode("context", source_error);
MemoriaError::decode_msg("message only");
```

The enum is `#[non_exhaustive]` — always include a wildcard arm when matching:

```rust
match err {
    MemoriaError::MemoryNotFound { id } => eprintln!("not found: {id}"),
    MemoriaError::Blocked { hook, reason } => eprintln!("blocked by {hook}: {reason}"),
    _ => eprintln!("other error: {err}"),
}
```

---

## 13. Scope Enforcer — Multi-Tenant Access Control

The scope enforcer provides unbypassable, identity-aware namespace isolation. It runs at the top of every public API method, before hooks.

### 13.1 Configuration

When no grants are configured, the enforcer is a no-op (allow-all) — preserving backward compatibility.

```rust
use memoria::{MemoriaBuilder, ScopeGrant, AgentPattern, Permissions};

let memoria = MemoriaBuilder::new(768)
    .grant(ScopeGrant {
        pattern: AgentPattern {
            agent: Some("agent-*".into()),
            user: Some("alice".into()),
            team: None,
            org: None,
        },
        namespaces: vec!["tenant-a".into(), "shared".into()],
        permissions: Permissions { read: true, write: true, delete: false },
    })
    .build()?;
```

### 13.2 How it works

- **`remember()`** — enforcer checks write permission for the memory's namespace
- **`recall()`** — enforcer rewrites the query's namespace filter to only include allowed namespaces
- **`forget()`** — enforcer checks delete permission
- **`patch()`** — enforcer checks write permission

WAL replay, GC, and snapshot restore bypass the enforcer (they operate on the store directly).

### 13.3 Types

```rust
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    pub delete: bool,
}

pub struct AgentPattern {
    pub agent: Option<String>,   // glob pattern matched against AgentId
    pub user: Option<String>,
    pub team: Option<String>,
    pub org: Option<String>,
}

pub struct ScopeGrant {
    pub pattern: AgentPattern,
    pub namespaces: Vec<String>,
    pub permissions: Permissions,
}

pub enum NamespaceAccess {
    Unrestricted,                    // no grants configured — allow all
    Restricted(Vec<String>),         // only these namespaces
    Denied(String),                  // access denied with reason
}
```

### 13.4 Error

Scope violations return `MemoriaError::ScopeDenied`:

```rust
ScopeDenied { agent: String, namespace: String, permission: String }
```

---

## Complete Usage Example

```rust
use memoria::*;
use memoria::hooks::observability::ObservabilityHooks;
use serde_json::json;

fn main() -> MemoriaResult<()> {
    // 1. Build (restores from WAL/snapshot if files exist)
    let mut memoria = MemoriaBuilder::new(128)
        .persistence_paths("data/memoria.wal", "data/memoria.snap")
        .add_index(FieldIndex::new("author"))
        .register_kind("chat.message", KindOpts {
            importance: 0.8,
            tau: 600.0,
            cascade: false,
            pinned: false,
            max_ttl_secs: None,
        })
        .build()?;

    // 2. Add hooks post-build
    memoria.add_hook(ObservabilityHooks::new());

    let ctx = AgentContext::new(AgentId::new("assistant"));

    // 3. Store memories
    let emb = vec![0.1_f32; 128];
    let m1 = Memory::new("chat.message", emb.clone())
        .with_field("author", json!("alice"))
        .with_field("text", json!("Hello world"));
    let id1 = memoria.remember(m1, &ctx)?;

    let m2 = Memory::new("chat.message", emb.clone())
        .with_field("author", json!("bob"))
        .with_field("text", json!("Hi alice"));
    let id2 = memoria.remember(m2, &ctx)?;

    // 4. Link memories
    memoria.link(id2, id1, "replied_to")?;

    // 5. Query
    let results = memoria.recall(
        &Query::new()
            .with_embedding(emb.clone())
            .with_kind("chat.message")
            .with_limit(5),
        &ctx,
    )?;

    for r in &results {
        println!("{}: {:.3}", r.memory.id, r.score);
        for (scorer, weight) in &r.breakdown {
            println!("  {scorer}: {weight:.4}");
        }
    }

    // 6. Feedback loop
    if let Some(top) = results.first() {
        memoria.feedback(top.memory.id, &ctx, true);
    }

    // 7. Patch a memory
    memoria.patch(id1, MemoryPatch::default()
        .set_field("status", json!("archived"))
        .remove_field("temp_flag"))?;

    // 8. Maintenance tick (also triggers auto-snapshot when needed)
    memoria.tick(1.0);
    memoria.gc();

    // 9. Stats
    let stats = memoria.stats();
    println!("memories: {}, hebbian edges: {}", stats.memory_count, stats.hebbian_edges);

    // 10. Explicit snapshot (optional — auto-snapshot handles this in tick())
    memoria.snapshot()?;
    // On next build() with the same paths, all state is restored automatically.

    Ok(())
}
```
