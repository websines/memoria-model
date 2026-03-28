use serde::{Deserialize, Serialize};

/// Kernel-level rules that guard memory operations.
///
/// These are evaluated synchronously before writes/deletes and cannot be
/// overridden by scope grants or governance rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelRule {
    /// Requires that a specific field exists and is non-empty before storing.
    /// E.g., `Existence { field: "provenance" }` — all memories must have provenance.
    Existence { field: String },

    /// Guards state transitions for a field (e.g., status can go active→suspended but not deregistered→active).
    Transition {
        field: String,
        from: String,
        to: String,
    },

    /// Prevents deletion of memories matching a predicate.
    /// E.g., `Deletion { namespace_pattern: "audit:*" }` — audit memories are immutable.
    Deletion { namespace_pattern: String },

    /// Controls visibility: memories in matching namespaces can only be read by agents
    /// matching the agent pattern.
    Visibility {
        namespace_pattern: String,
        agent_pattern: String,
    },
}

impl std::fmt::Display for KernelRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Existence { field } => write!(f, "existence({field})"),
            Self::Transition { field, from, to } => {
                write!(f, "transition({field}: {from} -> {to})")
            }
            Self::Deletion { namespace_pattern } => {
                write!(f, "no-delete({namespace_pattern})")
            }
            Self::Visibility {
                namespace_pattern,
                agent_pattern,
            } => write!(f, "visibility({namespace_pattern} -> {agent_pattern})"),
        }
    }
}
