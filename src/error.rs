use thiserror::Error;
use uuid::Uuid;

/// All errors that Memoria can produce.
#[derive(Debug, Error)]
pub enum MemoriaError {
    #[error("CozoDB error: {0}")]
    Store(String),

    #[error("embedding service error: {0}")]
    Embedding(String),

    #[error("NER service error: {0}")]
    Ner(String),

    #[error("LLM service error: {0}")]
    Llm(String),

    #[error("reranker error: {0}")]
    Reranker(String),

    #[error("memory not found: {0}")]
    NotFound(Uuid),

    #[error("schema bootstrap failed: {0}")]
    SchemaBootstrap(String),

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("namespace access denied: agent {agent} cannot access namespace {namespace}")]
    AccessDenied { agent: String, namespace: String },

    #[error("rule violation: {0}")]
    RuleViolation(String),

    #[error("task queue error: {0}")]
    TaskQueue(String),

    #[error("skill error: {0}")]
    Skill(String),

    #[error("telos error: {0}")]
    Telos(String),

    #[error("causal attribution error: {0}")]
    Causal(String),

    #[error("active inference error: {0}")]
    ActiveInference(String),

    #[error("permission denied: agent {agent} lacks {permission} on namespace {namespace}")]
    PermissionDenied {
        agent: String,
        permission: String,
        namespace: String,
    },

    #[error("kernel rule denied: {rule} — {message}")]
    KernelDenied { rule: String, message: String },

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<cozo::Error> for MemoriaError {
    fn from(e: cozo::Error) -> Self {
        MemoriaError::Store(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, MemoriaError>;
