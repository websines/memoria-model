use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// How to retrieve memories — the query planner produces these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecallStrategy {
    /// Direct entity/fact lookup.
    Structured {
        entity: String,
        predicate: Option<String>,
    },

    /// Vector similarity search.
    Semantic {
        embedding: Vec<f32>,
        filters: Vec<FieldFilter>,
    },

    /// Temporal diff — what changed since a given time.
    TemporalDiff {
        since: i64,
        entity_filter: Option<String>,
    },

    /// Episode retrieval by time range and/or topic.
    Episodic {
        time_range: Option<TimeRange>,
        topic: Option<String>,
    },

    /// Abstraction/pattern search.
    Abstract { topic: String },

    /// Run multiple strategies and merge results.
    Composite(Vec<RecallStrategy>),
}

/// A time range for temporal queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: i64,
    pub end: i64,
}

/// A filter on memory fields for structured queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldFilter {
    pub field: String,
    pub op: FilterOp,
    pub value: serde_json::Value,
}

/// Filter operations for field queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
}

/// Context provided by the calling agent for memory operations.
#[derive(Debug, Clone, Default)]
pub struct AgentContext {
    pub agent_id: String,
    pub namespace: String,
    pub current_episode: Option<Uuid>,
    /// The current task ID for attribution and audit trail.
    /// Used by ask() to key recall contexts and by feedback() for causal attribution.
    pub task_id: Option<Uuid>,
    pub limit: Option<usize>,
    /// IDs of memories currently in the agent's working context.
    pub context_memory_ids: Vec<Uuid>,
    /// Team ID for multi-agent coordination.
    pub team_id: Option<String>,
    /// Organization ID for multi-tenant isolation.
    pub org_id: Option<String>,
    /// Agent's role (e.g., "planner", "executor").
    pub role: Option<String>,
    /// Session ID for grouping operations within a single interaction.
    pub session_id: Option<String>,
    /// Parent agent ID for hierarchical agent topologies.
    pub parent_agent_id: Option<String>,
    /// Niche hint for skill selection in prime().
    /// When set, uses niche-aware skill selection instead of generic EFE ranking.
    pub niche_hint: Option<String>,
}

/// Result of a tell() operation.
#[derive(Debug, Clone)]
pub struct TellResult {
    pub memory_ids: Vec<Uuid>,
    pub entity_ids: Vec<Uuid>,
    pub fact_ids: Vec<Uuid>,
    pub surprise: f64,
}

/// Result of an ask() operation.
#[derive(Debug, Clone)]
pub struct AskResult {
    pub results: Vec<super::memory::ScoredMemory>,
    pub contradictions: Vec<Contradiction>,
    pub strategy_used: RecallStrategy,
}

/// A detected contradiction between two facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub fact_a: Uuid,
    pub fact_b: Uuid,
    pub entity: Uuid,
    pub predicate: String,
    pub value_a: String,
    pub value_b: String,
}
