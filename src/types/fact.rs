use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type FactId = Uuid;

/// An extracted fact in the knowledge graph.
///
/// Facts are structured subject-predicate-object triples extracted from
/// memories. They can have either an entity object or a literal value.
///
/// Examples:
/// - (Alice, works_at, Acme Corp) — entity object
/// - (Alice, prefers, "dark mode") — literal value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    pub subject_entity: Uuid,
    pub predicate: String,
    pub object_entity: Option<Uuid>,
    pub object_value: Option<String>,
    pub namespace: String,
    pub temporal_status: String,
    pub confidence: f64,
    pub provenance: String,
    pub source_ids: Vec<Uuid>,
    pub reinforcement_count: i64,
}

impl Fact {
    /// Create a fact with an entity as the object.
    pub fn with_entity(
        subject: Uuid,
        predicate: impl Into<String>,
        object: Uuid,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            subject_entity: subject,
            predicate: predicate.into(),
            object_entity: Some(object),
            object_value: None,
            namespace: String::new(),
            temporal_status: "current".to_string(),
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: Vec::new(),
            reinforcement_count: 1,
        }
    }

    /// Create a fact with a literal string value.
    pub fn with_value(
        subject: Uuid,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            subject_entity: subject,
            predicate: predicate.into(),
            object_entity: None,
            object_value: Some(value.into()),
            namespace: String::new(),
            temporal_status: "current".to_string(),
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: Vec::new(),
            reinforcement_count: 1,
        }
    }
}
