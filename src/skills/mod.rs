//! Phase 5: Procedural Memory — skills that crystallize from experience,
//! selected by Expected Free Energy (EFE), improved by causal attribution.
//!
//! Skills are the third memory type: **procedural memory** — "how to do things."
//! Distinct from episodic (what happened) and semantic (what is known).

pub mod lifecycle;
pub mod lineage;
pub mod niche;
pub mod selection;
pub mod storage;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-exports
pub use lifecycle::{BootstrapResult, CrystallizeResult, SpecializeResult};
pub use lineage::LineageEntry;
pub use niche::SkillNiche;
pub use selection::ScoredSkill;
pub use storage::{RecallContext, TaskOutcome};

/// A single step in a skill procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    pub step: usize,
    pub action: String,
}

/// Provenance of a skill — how it came into existence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkillProvenance {
    Bootstrapped,
    Crystallized,
    Discovered,
    Generalized,
    Specialized,
}

impl std::fmt::Display for SkillProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bootstrapped => write!(f, "bootstrapped"),
            Self::Crystallized => write!(f, "crystallized"),
            Self::Discovered => write!(f, "discovered"),
            Self::Generalized => write!(f, "generalized"),
            Self::Specialized => write!(f, "specialized"),
        }
    }
}

impl SkillProvenance {
    pub fn from_str(s: &str) -> Self {
        match s {
            "crystallized" => Self::Crystallized,
            "discovered" => Self::Discovered,
            "generalized" => Self::Generalized,
            "specialized" => Self::Specialized,
            _ => Self::Bootstrapped,
        }
    }
}

/// Outcome of a skill usage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkillOutcome {
    Success,
    Failure,
    Partial,
    Abandoned,
}

impl std::fmt::Display for SkillOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::Failure => write!(f, "failure"),
            Self::Partial => write!(f, "partial"),
            Self::Abandoned => write!(f, "abandoned"),
        }
    }
}

impl SkillOutcome {
    pub fn from_str(s: &str) -> Self {
        match s {
            "success" => Self::Success,
            "failure" => Self::Failure,
            "partial" => Self::Partial,
            _ => Self::Abandoned,
        }
    }
}

/// Skill performance metrics (stored as JSON in CozoDB).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillPerformance {
    #[serde(default)]
    pub success_rate: f64,
    #[serde(default)]
    pub avg_duration_ms: f64,
    #[serde(default)]
    pub usage_count: u64,
    #[serde(default)]
    pub last_used: i64,
}

/// A procedural skill — "how to accomplish a goal."
#[derive(Debug, Clone)]
pub struct Skill {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub steps: Vec<SkillStep>,
    pub preconditions: Vec<serde_json::Value>,
    pub postconditions: Vec<serde_json::Value>,
    pub confidence: f64,
    pub provenance: SkillProvenance,
    pub source_episodes: Vec<Uuid>,
    pub domain: String,
    pub version: i64,
    pub performance: SkillPerformance,
    pub parent_skill: Option<Uuid>,
    pub tags: Vec<String>,
}

impl Skill {
    pub fn new(name: &str, description: &str, steps: Vec<SkillStep>) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.to_string(),
            description: description.to_string(),
            steps,
            preconditions: vec![],
            postconditions: vec![],
            confidence: 0.5,
            provenance: SkillProvenance::Bootstrapped,
            source_episodes: vec![],
            domain: "general".to_string(),
            version: 1,
            performance: SkillPerformance::default(),
            parent_skill: None,
            tags: vec![],
        }
    }
}

/// A record of a skill being used by an agent.
#[derive(Debug, Clone)]
pub struct SkillUsage {
    pub id: Uuid,
    pub ts: i64,
    pub skill_id: Uuid,
    pub skill_version: i64,
    pub episode_id: Option<Uuid>,
    pub agent_id: String,
    pub outcome: SkillOutcome,
    pub duration_ms: Option<i64>,
    pub context_summary: String,
    pub adaptations: Vec<serde_json::Value>,
}

impl SkillUsage {
    pub fn new(skill_id: Uuid, skill_version: i64, agent_id: &str, outcome: SkillOutcome) -> Self {
        Self {
            id: Uuid::now_v7(),
            ts: crate::types::memory::now_ms(),
            skill_id,
            skill_version,
            episode_id: None,
            agent_id: agent_id.to_string(),
            outcome,
            duration_ms: None,
            context_summary: String::new(),
            adaptations: vec![],
        }
    }
}
