//! # Memoria v2 — A Self-Evolving Memory Runtime for AI Agents
//!
//! Memoria provides structured, evolving memory for AI agents. Built on CozoDB
//! (an embedded relational-graph-vector database with Datalog queries and
//! time-travel support), it replaces hand-rolled storage with a unified
//! knowledge substrate.
//!
//! ## Architecture
//!
//! - **CozoDB** handles storage, HNSW vector indexes, graph algorithms,
//!   and bi-temporal history via `Validity` columns.
//! - **Service traits** define pluggable backends for embedding, NER, LLM,
//!   and reranking.
//! - **Hot cache** (DashMap) provides fast access to recently used memories.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use memoria::{CozoStore, MemoriaConfig};
//! use memoria::services::MockEmbedder;
//!
//! let config = MemoriaConfig::default();
//! let store = CozoStore::open_mem(768).unwrap();
//! let embedder = MockEmbedder::new(768);
//! ```

pub mod aif;
pub mod api;
pub mod cache;
pub mod causal;
pub mod config;
pub mod dynamics;
pub mod error;
pub mod pipeline;
pub mod queue;
pub mod rules;
pub mod runtime;
pub mod services;
pub mod skills;
pub mod store;
pub mod types;

// Re-export primary types for convenience
pub use config::{DynamicsConfig, MemoriaConfig};
pub use error::{MemoriaError, Result};
pub use runtime::{FeedbackResult, Memoria, PrimeResult};
pub use store::CozoStore;
pub use types::{
    AgentContext, AskResult, CandidateMemory, Contradiction, Entity, EntityId, EntityMention,
    Episode, EpisodeId, EpisodeMemory, Fact, FactId, FieldFilter, FilterOp, Memory, MemoryId,
    RecallStrategy, ScoredMemory, TellResult, TimeRange,
};

// Phase 3 re-exports
pub use pipeline::verifier::{RelationVerifier, TemporalStatus, VerifiedRelation};
pub use queue::{QueueTask, QueueWorker, TaskQueue, TaskStatus};
pub use rules::{CompiledType, EvalResult, Rule, RuleEnforcer, RuleEvaluation, Severity};

// Phase 4 re-exports
pub use dynamics::{
    CompressionLevel, CompressionResult, CommunityResult, EmbeddingProjection, ImportanceResult,
    Observation, ProjectionStats, ReconsolidationResult, ReflectionResult, SurpriseResult,
};
pub use pipeline::chunker_hierarchical::{ChunkLevel, HierarchicalChunk};

// Phase 5 re-exports
pub use causal::{
    Attribution, AttributionResult, CausalEdge, CausalGraph, CausalMechanism, DoResult,
    NotearsConfig, NotearsResult, PropagatedEffect,
};
pub use pipeline::pes::PesResult;
pub use skills::{
    BootstrapResult, CrystallizeResult, LineageEntry, ScoredSkill, Skill, SkillNiche,
    SkillOutcome, SkillPerformance, SkillProvenance, SkillStep, SkillUsage,
};
pub use skills::storage::{RecallContext, TaskOutcome, VersionComparison, VersionStats};

// Phase 6 re-exports
pub use aif::{
    BeliefUpdateResult, FreeEnergyState, HierarchicalScore, ModelHealth,
    PropagationResult, Trend, FactorMessage, FactorType,
};

// Telos (goal system) re-exports
pub use api::telos_detect::DetectedGoal;
pub use api::telos_multi::{TelosConflict, ConflictType};
pub use types::{
    ScoredTelos, SuccessCriterion, Telos, TelosAttention, TelosEvent, TelosId, TelosProvenance,
    TelosStatus,
};

// Multi-agent re-exports
pub use config::KernelRulesConfig;
pub use types::{
    AgentFilter, AgentRecord, AgentRegistration, AgentStatus,
    AgentPattern, GrantFilter, Permission, ScopeGrant,
    ScratchEntry, ScratchValue, Visibility,
    MemoryEvent,
    AuditEntry, AuditFilter, AuditRecord, AuditVerification, Integrity,
    KernelRule,
};
