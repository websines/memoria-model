use serde::{Deserialize, Serialize};

/// Status of a registered agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Active,
    Idle,
    Suspended,
    Deregistered,
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Idle => write!(f, "idle"),
            Self::Suspended => write!(f, "suspended"),
            Self::Deregistered => write!(f, "deregistered"),
        }
    }
}

impl AgentStatus {
    pub fn from_str(s: &str) -> Self {
        match s {
            "active" => Self::Active,
            "idle" => Self::Idle,
            "suspended" => Self::Suspended,
            "deregistered" => Self::Deregistered,
            _ => Self::Idle,
        }
    }
}

/// Data required to register a new agent.
#[derive(Debug, Clone)]
pub struct AgentRegistration {
    pub agent_id: String,
    pub display_name: String,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub role: Option<String>,
    pub capabilities: Vec<String>,
    pub metadata: serde_json::Value,
}

impl AgentRegistration {
    pub fn new(agent_id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            display_name: display_name.into(),
            team_id: None,
            org_id: None,
            role: None,
            capabilities: Vec::new(),
            metadata: serde_json::Value::Object(Default::default()),
        }
    }

    pub fn with_team(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = Some(team_id.into());
        self
    }

    pub fn with_org(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn with_capabilities(mut self, caps: Vec<String>) -> Self {
        self.capabilities = caps;
        self
    }
}

/// A registered agent record as stored in the database.
#[derive(Debug, Clone)]
pub struct AgentRecord {
    pub agent_id: String,
    pub display_name: String,
    pub status: AgentStatus,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub role: Option<String>,
    pub capabilities: Vec<String>,
    pub metadata: serde_json::Value,
    pub registered_at: i64,
    pub last_seen_at: i64,
}

/// Filter for querying agents.
#[derive(Debug, Clone, Default)]
pub struct AgentFilter {
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub status: Option<AgentStatus>,
    pub role: Option<String>,
}
