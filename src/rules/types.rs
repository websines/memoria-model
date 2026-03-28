use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Rule severity — determines what happens on violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Block the operation entirely.
    Block,
    /// Warn but allow.
    Warn,
    /// Log only — silent monitoring.
    Log,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Warn => "warn",
            Self::Log => "log",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "block" | "hard" => Self::Block,
            "warn" | "soft" => Self::Warn,
            "log" => Self::Log,
            _ => Self::Warn,
        }
    }
}

/// How a rule is evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompiledType {
    /// Rule is a CozoDB Datalog query. Fast, deterministic.
    Datalog,
    /// Rule is evaluated by an LLM. Flexible, slower.
    Llm,
}

impl CompiledType {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "datalog" => Self::Datalog,
            "llm" => Self::Llm,
            _ => Self::Llm,
        }
    }
}

/// Result of evaluating a rule against an operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalResult {
    Allow,
    Block,
    Warn,
}

impl EvalResult {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block => "block",
            Self::Warn => "warn",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "allow" | "pass" => Self::Allow,
            "block" | "deny" | "reject" => Self::Block,
            "warn" | "warning" => Self::Warn,
            _ => Self::Allow,
        }
    }
}

/// A governance rule stored in CozoDB.
#[derive(Debug, Clone)]
pub struct Rule {
    pub id: Uuid,
    pub text: String,
    pub embedding: Vec<f32>,
    pub category: String,
    pub severity: Severity,
    pub scope: String,
    pub compiled_type: CompiledType,
    pub compiled_predicate: String,
    pub active: bool,
    pub created_by: String,
    pub violation_count: i64,
}

/// A record of evaluating a rule against an operation.
#[derive(Debug, Clone)]
pub struct RuleEvaluation {
    pub id: Uuid,
    pub ts: i64,
    pub rule_id: Uuid,
    pub operation: String,
    pub agent_id: String,
    pub target_id: Option<Uuid>,
    pub result: EvalResult,
    pub reason: String,
    pub evaluation_ms: i64,
}
