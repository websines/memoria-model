use serde::{Deserialize, Serialize};

/// An entry to be inserted into the hash-chained audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub operation: String,
    pub agent_id: String,
    pub namespace: String,
    pub details: serde_json::Value,
}

/// A stored audit record with sequence number and hash chain.
#[derive(Debug, Clone)]
pub struct AuditRecord {
    pub seq: i64,
    pub ts: i64,
    pub operation: String,
    pub agent_id: String,
    pub namespace: String,
    pub details: serde_json::Value,
    pub prev_hash: String,
    pub hash: String,
}

/// Filter for querying the audit chain.
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    pub agent_id: Option<String>,
    pub operation: Option<String>,
    pub namespace: Option<String>,
    pub since_seq: Option<i64>,
    pub limit: Option<usize>,
}

/// Result of verifying the audit chain integrity.
#[derive(Debug, Clone)]
pub struct AuditVerification {
    pub integrity: Integrity,
    pub entries_checked: usize,
    pub first_seq: i64,
    pub last_seq: i64,
    /// If broken, the sequence number where the chain breaks.
    pub broken_at: Option<i64>,
}

/// Audit chain integrity status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Integrity {
    Valid,
    Broken,
    Empty,
}
