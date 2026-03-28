use serde::{Deserialize, Serialize};

/// A value stored in the scratchpad.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScratchValue {
    Text(String),
    Number(f64),
    Bool(bool),
    Json(serde_json::Value),
}

impl ScratchValue {
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn from_json_str(s: &str) -> Option<Self> {
        serde_json::from_str(s).ok()
    }
}

/// Visibility level for scratchpad entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Only the owning agent can see this entry.
    Private,
    /// All agents in the same team can see this entry.
    Team,
    /// All agents can see this entry.
    Public,
}

impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Private => write!(f, "private"),
            Self::Team => write!(f, "team"),
            Self::Public => write!(f, "public"),
        }
    }
}

impl Visibility {
    pub fn from_str(s: &str) -> Self {
        match s {
            "private" => Self::Private,
            "team" => Self::Team,
            "public" => Self::Public,
            _ => Self::Private,
        }
    }
}

/// A scratchpad entry as stored in the database.
#[derive(Debug, Clone)]
pub struct ScratchEntry {
    pub namespace: String,
    pub key: String,
    pub value: ScratchValue,
    pub visibility: Visibility,
    pub owner_agent: String,
    pub updated_at: i64,
    pub expires_at: Option<i64>,
}
