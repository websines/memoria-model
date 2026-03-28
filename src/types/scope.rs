use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Permissions that can be granted to agents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    Read,
    Write,
    Delete,
    Admin,
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Delete => write!(f, "delete"),
            Self::Admin => write!(f, "admin"),
        }
    }
}

impl Permission {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "read" => Some(Self::Read),
            "write" => Some(Self::Write),
            "delete" => Some(Self::Delete),
            "admin" => Some(Self::Admin),
            _ => None,
        }
    }
}

/// Pattern for matching agents in scope grants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentPattern {
    /// Matches a specific agent by ID.
    Exact(String),
    /// Matches all agents in a team.
    Team(String),
    /// Matches all agents in an organization.
    Org(String),
    /// Matches all agents with a specific role.
    Role(String),
    /// Matches all agents.
    Any,
}

impl std::fmt::Display for AgentPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact(id) => write!(f, "agent:{id}"),
            Self::Team(id) => write!(f, "team:{id}"),
            Self::Org(id) => write!(f, "org:{id}"),
            Self::Role(r) => write!(f, "role:{r}"),
            Self::Any => write!(f, "*"),
        }
    }
}

/// A scope grant — gives permissions to matching agents on a namespace pattern.
#[derive(Debug, Clone)]
pub struct ScopeGrant {
    pub id: Uuid,
    pub agent_pattern: AgentPattern,
    pub namespace_pattern: String,
    pub permissions: Vec<Permission>,
    pub granted_by: String,
    pub granted_at: i64,
    pub expires_at: Option<i64>,
}

impl ScopeGrant {
    pub fn new(
        agent_pattern: AgentPattern,
        namespace_pattern: impl Into<String>,
        permissions: Vec<Permission>,
        granted_by: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            agent_pattern,
            namespace_pattern: namespace_pattern.into(),
            permissions,
            granted_by: granted_by.into(),
            granted_at: crate::types::memory::now_ms(),
            expires_at: None,
        }
    }

    pub fn with_expiry(mut self, expires_at: i64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
}

/// Filter for querying scope grants.
#[derive(Debug, Clone, Default)]
pub struct GrantFilter {
    pub agent_pattern: Option<String>,
    pub namespace_pattern: Option<String>,
}
