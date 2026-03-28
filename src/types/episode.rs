use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::types::memory::now_ms;

pub type EpisodeId = Uuid;

/// An episode — a bounded temporal segment of agent experience.
///
/// Episodes group memories into coherent sequences: a conversation,
/// a debugging session, a task execution. They are the fundamental
/// unit of episodic memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: EpisodeId,
    pub agent_id: String,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub summary: String,
    pub summary_embedding: Option<Vec<f32>>,
    pub episode_type: String,
    pub outcome: Option<String>,
    pub properties: Map<String, Value>,
}

/// A link between an episode and a memory within it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMemory {
    pub episode_id: EpisodeId,
    pub memory_id: Uuid,
    pub position: i64,
    pub role: String,
}

impl Episode {
    pub fn new(agent_id: impl Into<String>, episode_type: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            agent_id: agent_id.into(),
            started_at: now_ms(),
            ended_at: None,
            summary: String::new(),
            summary_embedding: None,
            episode_type: episode_type.into(),
            outcome: None,
            properties: Map::new(),
        }
    }

    pub fn close(&mut self, outcome: impl Into<String>) {
        self.ended_at = Some(now_ms());
        self.outcome = Some(outcome.into());
    }
}
