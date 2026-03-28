//! Multi-agent telos operations — team visibility, claiming, delegation, conflict detection.
//!
//! Enables multiple agents to coordinate around shared goals:
//! - **Team telos**: View goals visible to a team/namespace
//! - **Available telos**: Find unclaimed goals an agent can pick up
//! - **Claim**: An agent claims ownership of an available goal
//! - **Delegate**: Create a subtelos assigned to a different agent
//! - **Conflict detection**: Find semantically similar goals that may be duplicates
//! - **Directive conversion**: Convert enterprise directives into telos (depth 0-1)

use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::runtime::Memoria;
use crate::store::CozoStore;
use crate::types::query::AgentContext;
use crate::types::telos::{Telos, TelosEvent, TelosProvenance, TelosStatus};

/// A detected conflict between two telos goals.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TelosConflict {
    pub telos_a: Uuid,
    pub telos_b: Uuid,
    pub title_a: String,
    pub title_b: String,
    pub similarity: f64,
    pub conflict_type: ConflictType,
}

/// Type of conflict between telos goals.
#[derive(Debug, Clone, serde::Serialize)]
pub enum ConflictType {
    /// Two goals are semantically near-identical (likely duplicates).
    Duplicate,
    /// Two goals are highly similar but owned by different agents.
    Overlap,
}

/// List all active telos visible to a team/namespace.
///
/// Returns goals across all owners in the namespace, sorted by attention score.
pub fn team_telos(store: &CozoStore, namespace: &str, limit: usize) -> Result<Vec<Telos>> {
    store.list_active_telos(namespace, limit)
}

/// Find available (unclaimed) telos in a namespace.
///
/// A telos is "available" if:
/// - Status is Active or Proposed
/// - Owner is empty, "unassigned", or "system"
/// - Not in a terminal state
pub fn available_telos(store: &CozoStore, namespace: &str) -> Result<Vec<Telos>> {
    let all = store.list_active_telos(namespace, 100)?;
    Ok(all
        .into_iter()
        .filter(|t| {
            !t.status.is_terminal()
                && (t.owner.is_empty()
                    || t.owner == "unassigned"
                    || t.owner == "system")
        })
        .collect())
}

/// Claim a telos — assign it to the claiming agent.
///
/// Records a `claimed` event. Fails if the telos is already owned by another agent
/// (unless it's unassigned).
pub fn claim_telos(
    store: &CozoStore,
    telos_id: Uuid,
    agent_id: &str,
) -> Result<()> {
    let telos = store
        .get_telos(telos_id)?
        .ok_or_else(|| MemoriaError::NotFound(telos_id))?;

    // Check if already claimed by another agent
    if !telos.owner.is_empty()
        && telos.owner != "unassigned"
        && telos.owner != "system"
        && telos.owner != agent_id
    {
        return Err(MemoriaError::Telos(format!(
            "Telos '{}' is already owned by '{}'",
            telos.title, telos.owner
        )));
    }

    // Update owner
    store.update_telos_owner(telos_id, agent_id)?;

    // If it was proposed, activate it upon claiming
    if telos.status == TelosStatus::Proposed {
        store.update_telos_status(telos_id, "active")?;
    }

    // Record event
    let mut event = TelosEvent::new(telos_id, "claimed");
    event.agent_id = agent_id.to_string();
    event.description = format!("Claimed by agent '{}'", agent_id);
    store.insert_telos_event(&event)?;

    Ok(())
}

/// Delegate a telos — create a subtelos assigned to a target agent.
///
/// The delegated telos becomes a child of the parent with `Decomposition` provenance
/// and the target agent as owner. Records delegation events on both parent and child.
pub async fn delegate_telos(
    memoria: &Memoria,
    parent_telos_id: Uuid,
    target_agent: &str,
    title: &str,
    description: &str,
    ctx: &AgentContext,
) -> Result<Telos> {
    // Verify parent exists
    let parent = memoria
        .store()
        .get_telos(parent_telos_id)?
        .ok_or_else(|| MemoriaError::NotFound(parent_telos_id))?;

    // Create child telos via the standard creation path
    let child_depth = (parent.depth + 1).min(4);
    let mut child = memoria
        .create_telos(
            title,
            description,
            ctx,
            child_depth,
            Some(parent_telos_id),
            parent.deadline, // Inherit parent deadline
            TelosProvenance::Decomposition,
        )
        .await?;

    // Override owner to target agent
    memoria.store().update_telos_owner(child.id, target_agent)?;
    child.owner = target_agent.to_string();

    // Record delegation event on parent
    let mut parent_event = TelosEvent::new(parent_telos_id, "delegated");
    parent_event.agent_id = ctx.agent_id.clone();
    parent_event.description = format!(
        "Delegated subtelos '{}' to agent '{}'",
        title, target_agent
    );
    memoria.store().insert_telos_event(&parent_event)?;

    // Record delegation event on child
    let mut child_event = TelosEvent::new(child.id, "delegation_received");
    child_event.agent_id = target_agent.to_string();
    child_event.description = format!(
        "Delegated from '{}' by agent '{}'",
        parent.title, ctx.agent_id
    );
    memoria.store().insert_telos_event(&child_event)?;

    Ok(child)
}

/// Detect conflicting/duplicate telos goals in a namespace.
///
/// Uses embedding cosine similarity to find goals that are semantically
/// too close together, which may indicate duplicates or overlapping efforts.
///
/// Returns pairs of conflicting telos with their similarity score.
pub fn detect_conflicts(
    store: &CozoStore,
    namespace: &str,
    similarity_threshold: f64,
) -> Result<Vec<TelosConflict>> {
    let active = store.list_active_telos(namespace, 100)?;
    let mut conflicts = Vec::new();

    // Pairwise comparison — O(n²) but n is small (typically <50 active goals)
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let a = &active[i];
            let b = &active[j];

            if a.embedding.is_empty() || b.embedding.is_empty() {
                continue;
            }

            let sim = crate::store::cozo::cosine_similarity(&a.embedding, &b.embedding);

            if sim >= similarity_threshold {
                let conflict_type = if a.owner == b.owner || a.owner.is_empty() || b.owner.is_empty() {
                    ConflictType::Duplicate
                } else {
                    ConflictType::Overlap
                };

                conflicts.push(TelosConflict {
                    telos_a: a.id,
                    telos_b: b.id,
                    title_a: a.title.clone(),
                    title_b: b.title.clone(),
                    similarity: sim,
                    conflict_type,
                });
            }
        }
    }

    // Sort by similarity descending (most similar first)
    conflicts.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(conflicts)
}

/// Convert an enterprise directive into a telos.
///
/// Enterprise directives become depth 0 (north-star) or depth 1 (strategic) goals
/// with enterprise provenance and pinned confidence (never decays).
pub async fn directive_to_telos(
    memoria: &Memoria,
    title: &str,
    description: &str,
    namespace: &str,
    depth: i32,
    deadline: Option<i64>,
    priority: f64,
) -> Result<Telos> {
    let depth = depth.clamp(0, 1); // Enterprise directives are north-star or strategic
    let ctx = AgentContext::new("enterprise", namespace);

    let mut telos = memoria
        .create_telos(
            title,
            description,
            &ctx,
            depth,
            None,
            deadline,
            TelosProvenance::Enterprise,
        )
        .await?;

    // Override priority and confidence (enterprise goals are pinned)
    telos.priority = priority.clamp(0.0, 1.0);
    telos.confidence = 1.0; // Enterprise confidence never decays (τ=∞)
    memoria.store().upsert_telos(&telos)?;

    Ok(telos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_telos_filters() {
        let store = CozoStore::open_mem(4).unwrap();

        // Unassigned goal — should be available
        let mut t1 = Telos::new("Available", "", vec![0.1; 4], "unassigned", "u");
        t1.namespace = "team".to_string();
        t1.status = TelosStatus::Active;
        store.insert_telos(&t1).unwrap();

        // Assigned goal — should NOT be available
        let mut t2 = Telos::new("Owned", "", vec![0.1; 4], "agent-1", "u");
        t2.namespace = "team".to_string();
        t2.status = TelosStatus::Active;
        store.insert_telos(&t2).unwrap();

        // Completed goal — should NOT be available
        let mut t3 = Telos::new("Done", "", vec![0.1; 4], "unassigned", "u");
        t3.namespace = "team".to_string();
        t3.status = TelosStatus::Completed;
        store.insert_telos(&t3).unwrap();

        let available = available_telos(&store, "team").unwrap();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].title, "Available");
    }

    #[test]
    fn test_claim_telos() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut t = Telos::new("Claimable", "", vec![0.1; 4], "unassigned", "u");
        t.namespace = "team".to_string();
        t.status = TelosStatus::Proposed;
        store.insert_telos(&t).unwrap();

        // Claim it
        claim_telos(&store, t.id, "agent-1").unwrap();

        let updated = store.get_telos(t.id).unwrap().unwrap();
        assert_eq!(updated.owner, "agent-1");
        assert_eq!(updated.status, TelosStatus::Active); // Proposed → Active on claim

        // Check event was recorded
        let events = store.get_telos_events(t.id, 10).unwrap();
        assert!(events.iter().any(|e| e.event_type == "claimed"));
    }

    #[test]
    fn test_claim_already_owned_fails() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut t = Telos::new("Owned", "", vec![0.1; 4], "agent-1", "u");
        t.namespace = "team".to_string();
        store.insert_telos(&t).unwrap();

        let result = claim_telos(&store, t.id, "agent-2");
        assert!(result.is_err());
    }

    #[test]
    fn test_claim_own_goal_succeeds() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut t = Telos::new("My goal", "", vec![0.1; 4], "agent-1", "u");
        t.namespace = "team".to_string();
        store.insert_telos(&t).unwrap();

        // Re-claiming own goal should succeed
        claim_telos(&store, t.id, "agent-1").unwrap();
    }

    #[test]
    fn test_detect_conflicts() {
        let store = CozoStore::open_mem(4).unwrap();

        // Two nearly identical goals (same embedding)
        let mut t1 = Telos::new("Ship Q3 deck", "", vec![0.9, 0.1, 0.0, 0.0], "agent-1", "u");
        t1.namespace = "team".to_string();
        store.insert_telos(&t1).unwrap();

        let mut t2 = Telos::new("Prepare Q3 presentation", "", vec![0.88, 0.12, 0.01, 0.0], "agent-2", "u");
        t2.namespace = "team".to_string();
        store.insert_telos(&t2).unwrap();

        // A very different goal
        let mut t3 = Telos::new("Fix CI pipeline", "", vec![0.0, 0.0, 0.9, 0.1], "agent-1", "u");
        t3.namespace = "team".to_string();
        store.insert_telos(&t3).unwrap();

        let conflicts = detect_conflicts(&store, "team", 0.9).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].similarity >= 0.9);
        // Different owners → Overlap
        assert!(matches!(conflicts[0].conflict_type, ConflictType::Overlap));
    }

    #[test]
    fn test_detect_conflicts_duplicate() {
        let store = CozoStore::open_mem(4).unwrap();

        // Two identical goals from same owner
        let mut t1 = Telos::new("Ship it", "", vec![1.0, 0.0, 0.0, 0.0], "agent-1", "u");
        t1.namespace = "team".to_string();
        store.insert_telos(&t1).unwrap();

        let mut t2 = Telos::new("Ship it now", "", vec![1.0, 0.0, 0.0, 0.0], "agent-1", "u");
        t2.namespace = "team".to_string();
        store.insert_telos(&t2).unwrap();

        let conflicts = detect_conflicts(&store, "team", 0.99).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(conflicts[0].conflict_type, ConflictType::Duplicate));
    }

    #[tokio::test]
    async fn test_delegate_telos() {
        let memoria = Memoria::with_mocks(4).unwrap();
        let ctx = AgentContext::new("agent-1", "team");

        // Create a parent goal
        let parent = memoria
            .create_telos("Ship Q3 deck", "Full deck", &ctx, 1, None, None, TelosProvenance::UserStated)
            .await
            .unwrap();

        // Delegate a subtask to agent-2
        let child = delegate_telos(
            &memoria,
            parent.id,
            "agent-2",
            "Draft financial slides",
            "Create slides 5-10 with Q3 financials",
            &ctx,
        )
        .await
        .unwrap();

        assert_eq!(child.owner, "agent-2");
        assert_eq!(child.parent, Some(parent.id));
        assert_eq!(child.depth, 2); // parent depth 1 + 1

        // Check delegation events
        let parent_events = memoria.store().get_telos_events(parent.id, 10).unwrap();
        assert!(parent_events.iter().any(|e| e.event_type == "delegated"));

        let child_events = memoria.store().get_telos_events(child.id, 10).unwrap();
        assert!(child_events.iter().any(|e| e.event_type == "delegation_received"));
    }

    #[tokio::test]
    async fn test_directive_to_telos() {
        let memoria = Memoria::with_mocks(4).unwrap();

        let telos = directive_to_telos(
            &memoria,
            "Achieve SOC2 compliance",
            "Full SOC2 Type II certification by Q4",
            "org",
            0, // north-star
            None,
            0.95,
        )
        .await
        .unwrap();

        assert_eq!(telos.provenance, TelosProvenance::Enterprise);
        assert_eq!(telos.depth, 0);
        assert!((telos.priority - 0.95).abs() < 0.01);
        assert!((telos.confidence - 1.0).abs() < 0.01); // pinned
        assert_eq!(telos.namespace, "org");
    }

    #[tokio::test]
    async fn test_directive_depth_clamped() {
        let memoria = Memoria::with_mocks(4).unwrap();

        let telos = directive_to_telos(
            &memoria,
            "Tactical directive",
            "",
            "org",
            5, // too deep — should clamp to 1
            None,
            0.5,
        )
        .await
        .unwrap();

        assert_eq!(telos.depth, 1); // Clamped to max 1 for enterprise
    }

    #[test]
    fn test_team_telos() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut t1 = Telos::new("Goal A", "", vec![0.1; 4], "agent-1", "u");
        t1.namespace = "team".to_string();
        store.insert_telos(&t1).unwrap();

        let mut t2 = Telos::new("Goal B", "", vec![0.1; 4], "agent-2", "u");
        t2.namespace = "team".to_string();
        store.insert_telos(&t2).unwrap();

        // Different namespace — should not appear
        let mut t3 = Telos::new("Other", "", vec![0.1; 4], "agent-3", "u");
        t3.namespace = "other".to_string();
        store.insert_telos(&t3).unwrap();

        let team = team_telos(&store, "team", 10).unwrap();
        assert_eq!(team.len(), 2);
    }
}
