use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

pub type EntityId = Uuid;

/// An extracted entity in the knowledge graph.
///
/// Entities represent people, organizations, concepts, projects, etc.
/// They are extracted from memories via NER (GLiNER2) and linked to
/// the memories that mention them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub entity_type: String,
    pub namespace: String,
    pub embedding: Vec<f32>,
    pub properties: Map<String, Value>,
    pub mention_count: i64,
    pub confidence: f64,
    pub provenance: String,
    pub source_ids: Vec<Uuid>,
}

/// A link between an entity and a memory that mentions it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMention {
    pub entity_id: EntityId,
    pub memory_id: Uuid,
    pub role: String,
    pub confidence: f64,
}

impl Entity {
    pub fn new(
        name: impl Into<String>,
        entity_type: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.into(),
            entity_type: entity_type.into(),
            namespace: String::new(),
            embedding,
            properties: Map::new(),
            mention_count: 1,
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: Vec::new(),
        }
    }
}
