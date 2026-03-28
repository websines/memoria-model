use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::memory::now_ms;

pub type TelosId = Uuid;

/// Lifecycle status of a telos (goal).
///
/// Transitions are stored via CozoDB `Validity` — you can query
/// "what was the status of this telos last week?" via time-travel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelosStatus {
    Proposed,
    Active,
    Blocked,
    Stalled,
    Paused,
    Completed,
    Failed,
    Abandoned,
}

impl TelosStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Active => "active",
            Self::Blocked => "blocked",
            Self::Stalled => "stalled",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Abandoned => "abandoned",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "proposed" => Self::Proposed,
            "active" => Self::Active,
            "blocked" => Self::Blocked,
            "stalled" => Self::Stalled,
            "paused" => Self::Paused,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "abandoned" => Self::Abandoned,
            _ => Self::Active,
        }
    }

    /// Whether this status represents a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Abandoned)
    }
}

/// How the telos was created — determines initial confidence and decay rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelosProvenance {
    UserStated,
    AgentProposed,
    Decomposition,
    Inferred,
    Enterprise,
    /// Generated autonomously from surprise patterns via free energy minimization.
    Intrinsic,
}

impl TelosProvenance {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserStated => "user_stated",
            Self::AgentProposed => "agent_proposed",
            Self::Decomposition => "decomposition",
            Self::Inferred => "inferred",
            Self::Enterprise => "enterprise",
            Self::Intrinsic => "intrinsic",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "user_stated" => Self::UserStated,
            "agent_proposed" => Self::AgentProposed,
            "decomposition" => Self::Decomposition,
            "inferred" => Self::Inferred,
            "enterprise" => Self::Enterprise,
            "intrinsic" => Self::Intrinsic,
            _ => Self::UserStated,
        }
    }

    /// Initial confidence for this provenance type.
    pub fn initial_confidence(&self) -> f64 {
        match self {
            Self::UserStated => 1.0,
            Self::AgentProposed => 0.6,
            Self::Decomposition => 0.8,
            Self::Inferred => 0.4,
            Self::Enterprise => 1.0,
            Self::Intrinsic => 0.3,
        }
    }
}

/// A single success criterion for a telos.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    pub id: String,
    pub description: String,
    pub met: bool,
}

/// A persistent goal stored in CozoDB.
///
/// Telos is not a task — it's a direction the agent is pulled toward.
/// It persists, decomposes, competes for attention, reshapes perception,
/// and decays like all other Memoria knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Telos {
    pub id: TelosId,
    pub title: String,
    pub description: String,
    pub embedding: Vec<f32>,

    // Hierarchy
    pub parent: Option<TelosId>,
    /// 0=north star, 1=strategic, 2=tactical, 3=operational, 4=task
    pub depth: i32,

    // Ownership
    pub owner: String,
    pub set_by: String,
    pub namespace: String,

    // Lifecycle
    pub status: TelosStatus,
    pub priority: f64,
    pub urgency: f64,
    pub confidence: f64,
    pub provenance: TelosProvenance,

    // Temporal
    pub deadline: Option<i64>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,

    // Progress
    pub progress: f64,
    pub stalled_since: Option<i64>,

    // Success criteria
    pub success_criteria: Vec<SuccessCriterion>,

    // Context links
    pub related_entities: Vec<Uuid>,
    pub required_skills: Vec<Uuid>,
    pub depends_on: Vec<Uuid>,

    // Attention
    pub last_attended: i64,
    pub attention_count: i64,
}

impl Telos {
    /// Create a new telos with required fields. Everything else gets defaults.
    pub fn new(
        title: impl Into<String>,
        description: impl Into<String>,
        embedding: Vec<f32>,
        owner: impl Into<String>,
        set_by: impl Into<String>,
    ) -> Self {
        let provenance = TelosProvenance::UserStated;
        Self {
            id: Uuid::now_v7(),
            title: title.into(),
            description: description.into(),
            embedding,
            parent: None,
            depth: 0,
            owner: owner.into(),
            set_by: set_by.into(),
            namespace: String::new(),
            status: TelosStatus::Active,
            priority: 0.5,
            urgency: 0.0,
            confidence: provenance.initial_confidence(),
            provenance,
            deadline: None,
            created_at: now_ms(),
            started_at: None,
            completed_at: None,
            progress: 0.0,
            stalled_since: None,
            success_criteria: Vec::new(),
            related_entities: Vec::new(),
            required_skills: Vec::new(),
            depends_on: Vec::new(),
            last_attended: 0,
            attention_count: 0,
        }
    }
}

/// An event recording a meaningful change to a telos or progress toward it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelosEvent {
    pub id: Uuid,
    pub ts: i64,
    pub telos_id: Uuid,
    pub event_type: String,
    pub agent_id: String,
    pub description: String,
    pub impact: f64,
    pub source_memory: Option<Uuid>,
    pub source_episode: Option<Uuid>,
    pub metadata: serde_json::Value,
}

impl TelosEvent {
    pub fn new(telos_id: Uuid, event_type: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            ts: now_ms(),
            telos_id,
            event_type: event_type.into(),
            agent_id: String::new(),
            description: String::new(),
            impact: 0.0,
            source_memory: None,
            source_episode: None,
            metadata: serde_json::Value::Object(Default::default()),
        }
    }
}

/// Tracks when the agent actively focused on a telos.
/// Used for stall detection and attention balancing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelosAttention {
    pub telos_id: Uuid,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub agent_id: String,
    pub episode_id: Option<Uuid>,
    pub outcome: String,
}

/// A telos with its computed attention score.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredTelos {
    pub telos: Telos,
    /// Attention score from the attention allocation formula.
    pub attention_score: f64,
}

impl ScoredTelos {
    /// Urgency level label based on the telos urgency field.
    /// Returns a descriptive string — thresholds are configurable upstream.
    pub fn urgency_level(&self) -> &'static str {
        let u = self.telos.urgency;
        if u >= 0.8 {
            "critical"
        } else if u >= 0.5 {
            "high"
        } else if u >= 0.2 {
            "medium"
        } else {
            "low"
        }
    }

    /// Human-readable deadline display, if a deadline is set.
    pub fn deadline_display(&self) -> Option<String> {
        let deadline_ms = self.telos.deadline?;
        let now = now_ms();
        let delta_ms = deadline_ms - now;

        if delta_ms <= 0 {
            return Some("overdue".to_string());
        }

        let hours = delta_ms / 3_600_000;
        if hours < 24 {
            Some(format!("{hours}h"))
        } else {
            let days = hours / 24;
            Some(format!("{days}d"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telos_status_roundtrip() {
        for status in [
            TelosStatus::Proposed,
            TelosStatus::Active,
            TelosStatus::Blocked,
            TelosStatus::Stalled,
            TelosStatus::Paused,
            TelosStatus::Completed,
            TelosStatus::Failed,
            TelosStatus::Abandoned,
        ] {
            assert_eq!(TelosStatus::from_str(status.as_str()), status);
        }
    }

    #[test]
    fn test_telos_provenance_roundtrip() {
        for prov in [
            TelosProvenance::UserStated,
            TelosProvenance::AgentProposed,
            TelosProvenance::Decomposition,
            TelosProvenance::Inferred,
            TelosProvenance::Enterprise,
            TelosProvenance::Intrinsic,
        ] {
            assert_eq!(TelosProvenance::from_str(prov.as_str()), prov);
        }
    }

    #[test]
    fn test_telos_new_defaults() {
        let t = Telos::new("Ship Q3 deck", "Prepare investor deck", vec![0.1; 4], "agent-1", "user");
        assert_eq!(t.status, TelosStatus::Active);
        assert_eq!(t.depth, 0);
        assert_eq!(t.progress, 0.0);
        assert_eq!(t.confidence, 1.0); // user_stated initial confidence
        assert!(t.parent.is_none());
    }

    #[test]
    fn test_terminal_status() {
        assert!(!TelosStatus::Active.is_terminal());
        assert!(!TelosStatus::Blocked.is_terminal());
        assert!(TelosStatus::Completed.is_terminal());
        assert!(TelosStatus::Failed.is_terminal());
        assert!(TelosStatus::Abandoned.is_terminal());
    }

    #[test]
    fn test_scored_telos_urgency_level() {
        let make = |urgency: f64| ScoredTelos {
            telos: {
                let mut t = Telos::new("test", "", vec![0.1; 4], "a", "u");
                t.urgency = urgency;
                t
            },
            attention_score: 0.0,
        };
        assert_eq!(make(0.9).urgency_level(), "critical");
        assert_eq!(make(0.6).urgency_level(), "high");
        assert_eq!(make(0.3).urgency_level(), "medium");
        assert_eq!(make(0.1).urgency_level(), "low");
    }

    #[test]
    fn test_scored_telos_deadline_display() {
        let now = now_ms();
        let make = |deadline: Option<i64>| ScoredTelos {
            telos: {
                let mut t = Telos::new("test", "", vec![0.1; 4], "a", "u");
                t.deadline = deadline;
                t
            },
            attention_score: 0.0,
        };

        assert!(make(None).deadline_display().is_none());
        assert_eq!(make(Some(now - 1000)).deadline_display().unwrap(), "overdue");
        assert_eq!(make(Some(now + 7_200_000)).deadline_display().unwrap(), "2h");
        assert_eq!(make(Some(now + 172_800_000)).deadline_display().unwrap(), "2d");
    }
}
