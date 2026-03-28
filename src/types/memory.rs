use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

pub type MemoryId = Uuid;

/// A memory stored in CozoDB.
///
/// Memories are the fundamental unit of knowledge in Memoria.
/// They can represent raw observations, chat messages, code snippets,
/// or any other piece of information an agent encounters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: MemoryId,
    pub kind: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub fields: Map<String, Value>,
    pub namespace: String,
    pub pinned: bool,
    pub expires_at: Option<i64>,
    pub version: i32,
    pub created_at: i64,
    pub confidence: f64,
    pub provenance: String,
    pub source_ids: Vec<Uuid>,
}

/// A memory with its computed retrieval score.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    /// Log-posterior score from factor message fusion.
    pub score: f64,
    /// Effective confidence (provenance-weighted, time-decayed).
    pub confidence: f64,
    /// Chain of source IDs that this memory derives from.
    pub provenance_chain: Vec<Uuid>,
}

/// A candidate memory from broad HNSW retrieval, before scoring.
#[derive(Debug, Clone)]
pub struct CandidateMemory {
    pub memory: Memory,
    /// Cosine distance from HNSW search.
    pub distance: f64,
    /// Computed activation (recency-weighted access count), if available.
    pub activation: Option<f64>,
    /// Hebbian association weight to current context, if available.
    pub hebbian_weight: Option<f64>,
    /// PageRank importance score, if available.
    pub pagerank: Option<f64>,
    /// Belief precision (confidence × ln(reinforcement_count + 1)), if available.
    /// Used by precision-weighted factor message fusion in Phase 6 AIF.
    pub precision: Option<f64>,
    /// Telos alignment boost — cosine similarity to active goals, weighted by priority.
    pub telos_boost: Option<f64>,
}

impl Memory {
    /// Create a new memory with minimal required fields.
    pub fn new(kind: impl Into<String>, content: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: Uuid::now_v7(),
            kind: kind.into(),
            content: content.into(),
            embedding,
            fields: Map::new(),
            namespace: String::new(),
            pinned: false,
            expires_at: None,
            version: 0,
            created_at: now_ms(),
            confidence: 1.0,
            provenance: "direct".to_string(),
            source_ids: Vec::new(),
        }
    }
}

/// Current time in milliseconds since Unix epoch.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_millis() as i64
}
