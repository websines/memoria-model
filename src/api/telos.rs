use uuid::Uuid;

use crate::error::Result;
use crate::runtime::Memoria;
use crate::types::memory::now_ms;
use crate::types::query::AgentContext;
use crate::types::telos::{ScoredTelos, Telos, TelosEvent, TelosProvenance, TelosStatus};

impl Memoria {
    /// Create a new telos (goal) — embeds, stores, records event, enqueues decomposition.
    ///
    /// This is the primary entry point for goal creation. It:
    /// 1. Embeds the title + description for semantic retrieval
    /// 2. Stores the telos in CozoDB
    /// 3. Records a `created` event
    /// 4. Enqueues a `decompose_telos` task if depth < 3
    pub async fn create_telos(
        &self,
        title: &str,
        description: &str,
        ctx: &AgentContext,
        depth: i32,
        parent: Option<Uuid>,
        deadline: Option<i64>,
        provenance: TelosProvenance,
    ) -> Result<Telos> {
        // 1. Embed title + description for semantic search
        let embed_text = if description.is_empty() {
            title.to_string()
        } else {
            format!("{title}: {description}")
        };
        let embeddings = self
            .embedder
            .embed(&[embed_text.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;
        let embedding = embeddings.into_iter().next().unwrap_or_default();

        // 2. Build the telos struct
        let mut telos = Telos::new(title, description, embedding, &ctx.agent_id, &ctx.agent_id);
        telos.namespace = ctx.namespace.clone();
        telos.depth = depth;
        telos.parent = parent;
        telos.deadline = deadline;
        telos.provenance = provenance;
        telos.confidence = provenance.initial_confidence();

        // User-stated goals start active; agent-proposed start as proposed
        telos.status = match provenance {
            TelosProvenance::AgentProposed => TelosStatus::Proposed,
            TelosProvenance::Inferred => TelosStatus::Proposed,
            _ => TelosStatus::Active,
        };

        telos.set_by = match provenance {
            TelosProvenance::UserStated => "user".to_string(),
            TelosProvenance::AgentProposed => format!("agent:{}", ctx.agent_id),
            TelosProvenance::Decomposition => "decomposition".to_string(),
            TelosProvenance::Inferred => format!("agent:{}", ctx.agent_id),
            TelosProvenance::Enterprise => "enterprise".to_string(),
            TelosProvenance::Intrinsic => "intrinsic_dynamics".to_string(),
        };

        // 3. Store in CozoDB
        self.store.insert_telos(&telos)?;

        // 4. Record a `created` event
        let mut event = TelosEvent::new(telos.id, "created");
        event.agent_id = ctx.agent_id.clone();
        event.description = format!("Telos created: {title}");
        self.store.insert_telos_event(&event)?;

        // 5. Enqueue decomposition for non-leaf goals
        if depth < 3 {
            if let Some(ref queue) = self.task_queue {
                let payload = serde_json::json!({
                    "telos_id": telos.id.to_string(),
                    "namespace": ctx.namespace,
                });
                let _ = queue.enqueue("decompose_telos", 8, &payload.to_string(), 3);
            }
        }

        // 5b. Enqueue conflict detection
        if let Some(ref queue) = self.task_queue {
            let payload = serde_json::json!({
                "namespace": ctx.namespace,
            });
            let _ = queue.enqueue("detect_telos_conflicts", 10, &payload.to_string(), 1);
        }

        // 6. Emit event
        self.emit(crate::types::event::MemoryEvent::TelosCreated {
            telos_id: telos.id,
            title: title.to_string(),
            namespace: ctx.namespace.clone(),
            agent_id: ctx.agent_id.clone(),
        });

        Ok(telos)
    }

    /// Get a telos by ID.
    pub fn get_telos(&self, id: Uuid) -> Result<Option<Telos>> {
        self.store.get_telos(id)
    }

    /// Update telos status with event recording.
    pub async fn update_telos_status(
        &self,
        id: Uuid,
        new_status: &str,
        agent_id: &str,
        reason: &str,
    ) -> Result<()> {
        self.store.update_telos_status(id, new_status)?;

        // Record event
        let event_type = match new_status {
            "completed" => "completed",
            "failed" => "failed",
            "abandoned" => "abandoned",
            "paused" => "paused",
            "blocked" => "blocked",
            "active" => "resumed",
            _ => "status_changed",
        };
        let mut event = TelosEvent::new(id, event_type);
        event.agent_id = agent_id.to_string();
        event.description = reason.to_string();
        self.store.insert_telos_event(&event)?;

        // Emit event
        self.emit(crate::types::event::MemoryEvent::TelosStatusChanged {
            telos_id: id,
            new_status: new_status.to_string(),
            agent_id: agent_id.to_string(),
        });

        // Enqueue reflection on completion or failure
        if new_status == "completed" || new_status == "failed" {
            if let Some(ref queue) = self.task_queue {
                let payload = serde_json::json!({
                    "telos_id": id.to_string(),
                    "outcome": new_status,
                });
                let _ = queue.enqueue("telos_reflection", 5, &payload.to_string(), 3);
            }
        }

        Ok(())
    }

    /// Record progress on a telos — updates progress and logs a progress event.
    pub async fn record_telos_progress(
        &self,
        telos_id: Uuid,
        delta: f64,
        description: &str,
        agent_id: &str,
        source_memory: Option<Uuid>,
        source_episode: Option<Uuid>,
    ) -> Result<()> {
        // Get current progress to compute new value
        let telos = self
            .store
            .get_telos(telos_id)?
            .ok_or_else(|| crate::error::MemoriaError::NotFound(telos_id))?;

        let new_progress = (telos.progress + delta).clamp(0.0, 1.0);
        self.store.update_telos_progress(telos_id, new_progress)?;

        // Record progress event
        let mut event = TelosEvent::new(telos_id, "progress");
        event.agent_id = agent_id.to_string();
        event.description = description.to_string();
        event.impact = delta;
        event.source_memory = source_memory;
        event.source_episode = source_episode;
        self.store.insert_telos_event(&event)?;

        // Emit event
        self.emit(crate::types::event::MemoryEvent::TelosProgress {
            telos_id,
            progress: new_progress,
            agent_id: agent_id.to_string(),
        });

        // Auto-complete if progress reaches 1.0
        if new_progress >= 1.0 {
            self.update_telos_status(telos_id, "completed", agent_id, "Progress reached 100%")
                .await?;
        }

        Ok(())
    }

    /// Get active telos for a namespace, scored by attention allocation.
    ///
    /// Returns up to `k` telos sorted by computed attention score:
    /// `priority × (1 + effective_urgency) × recency_penalty + staleness_bonus`
    pub fn active_telos(&self, namespace: &str, k: usize) -> Result<Vec<ScoredTelos>> {
        let telos_list = self.store.list_active_telos(namespace, k * 2)?;
        let now = now_ms();

        let mut scored: Vec<ScoredTelos> = telos_list
            .into_iter()
            .map(|t| {
                let score = compute_attention_score(&t, now);
                ScoredTelos {
                    telos: t,
                    attention_score: score,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.attention_score
                .partial_cmp(&a.attention_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);

        Ok(scored)
    }

    /// Mark a specific success criterion as met.
    pub fn mark_criterion_met(
        &self,
        telos_id: Uuid,
        criterion_id: &str,
        agent_id: &str,
    ) -> Result<()> {
        let mut telos = self
            .store
            .get_telos(telos_id)?
            .ok_or_else(|| crate::error::MemoriaError::NotFound(telos_id))?;

        // Find and mark the criterion
        let mut found = false;
        for criterion in &mut telos.success_criteria {
            if criterion.id == criterion_id {
                criterion.met = true;
                found = true;
                break;
            }
        }

        if !found {
            return Ok(()); // Criterion not found, no-op
        }

        // Recompute criteria-based progress
        let total = telos.success_criteria.len() as f64;
        let met = telos.success_criteria.iter().filter(|c| c.met).count() as f64;
        if total > 0.0 {
            telos.progress = met / total;
        }

        self.store.upsert_telos(&telos)?;

        // Record event
        let mut event = TelosEvent::new(telos_id, "criterion_met");
        event.agent_id = agent_id.to_string();
        event.description = format!("Criterion met: {criterion_id}");
        self.store.insert_telos_event(&event)?;

        Ok(())
    }
}

/// Compute attention score for a telos.
///
/// Formula: `priority × (1 + effective_urgency) × recency_penalty + staleness_bonus`
///
/// No hardcoded thresholds — the formula naturally handles:
/// - Urgency: sigmoid of deadline proximity (peaks as deadline approaches)
/// - Recency: exponential decay penalizes recently-attended goals
/// - Staleness: bonus for stalled goals grows over time
fn compute_attention_score(telos: &Telos, now_ms: i64) -> f64 {
    // Deadline-based urgency (sigmoid, peaks as deadline approaches)
    let deadline_urgency = if let Some(deadline) = telos.deadline {
        // Sigmoid centered 3 days before deadline (259_200_000 ms)
        let x = (now_ms - deadline + 259_200_000) as f64 / 86_400_000.0;
        1.0 / (1.0 + (-x).exp())
    } else {
        0.0
    };
    let effective_urgency = telos.urgency.max(deadline_urgency);

    // Recency penalty: recently attended goals deprioritized (prevents fixation)
    let hours_since_attended = if telos.last_attended > 0 {
        (now_ms - telos.last_attended) as f64 / 3_600_000.0
    } else {
        f64::MAX // Never attended = no penalty
    };
    let recency_penalty = 1.0 - (-hours_since_attended / 24.0).exp();

    // Staleness bonus: stalled goals get attention boost
    let staleness_bonus = if let Some(stalled_since) = telos.stalled_since {
        let days_stalled = (now_ms - stalled_since) as f64 / 86_400_000.0;
        0.3 * days_stalled.min(1.0)
    } else {
        0.0
    };

    telos.priority * (1.0 + effective_urgency) * recency_penalty + staleness_bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::telos::SuccessCriterion;

    #[test]
    fn test_attention_score_basic() {
        let now = now_ms();
        let t = Telos::new("test", "", vec![0.1; 4], "a", "u");
        let score = compute_attention_score(&t, now);
        // With default priority 0.5, no urgency, never attended (max recency penalty = ~1.0)
        assert!(score > 0.0);
    }

    #[test]
    fn test_attention_score_deadline_urgency() {
        let now = now_ms();
        let mut t_urgent = Telos::new("urgent", "", vec![0.1; 4], "a", "u");
        t_urgent.deadline = Some(now + 3_600_000); // 1 hour from now
        t_urgent.priority = 0.8;

        let mut t_relaxed = Telos::new("relaxed", "", vec![0.1; 4], "a", "u");
        t_relaxed.deadline = Some(now + 30 * 86_400_000); // 30 days from now
        t_relaxed.priority = 0.8;

        let score_urgent = compute_attention_score(&t_urgent, now);
        let score_relaxed = compute_attention_score(&t_relaxed, now);
        assert!(
            score_urgent > score_relaxed,
            "urgent={score_urgent} should be > relaxed={score_relaxed}"
        );
    }

    #[test]
    fn test_attention_score_recency_penalty() {
        let now = now_ms();
        let mut t_recent = Telos::new("recent", "", vec![0.1; 4], "a", "u");
        t_recent.last_attended = now - 60_000; // 1 minute ago
        t_recent.priority = 0.8;

        let mut t_stale = Telos::new("stale", "", vec![0.1; 4], "a", "u");
        t_stale.last_attended = now - 48 * 3_600_000; // 48 hours ago
        t_stale.priority = 0.8;

        let score_recent = compute_attention_score(&t_recent, now);
        let score_stale = compute_attention_score(&t_stale, now);
        assert!(
            score_stale > score_recent,
            "stale={score_stale} should be > recent={score_recent}"
        );
    }

    #[test]
    fn test_attention_score_staleness_bonus() {
        let now = now_ms();
        let mut t_normal = Telos::new("normal", "", vec![0.1; 4], "a", "u");
        t_normal.priority = 0.5;

        let mut t_stalled = Telos::new("stalled", "", vec![0.1; 4], "a", "u");
        t_stalled.priority = 0.5;
        t_stalled.stalled_since = Some(now - 86_400_000); // Stalled 1 day ago

        let score_normal = compute_attention_score(&t_normal, now);
        let score_stalled = compute_attention_score(&t_stalled, now);
        assert!(
            score_stalled > score_normal,
            "stalled={score_stalled} should be > normal={score_normal}"
        );
    }

    #[tokio::test]
    async fn test_create_telos() {
        let memoria = Memoria::with_mocks(4).unwrap();
        let ctx = AgentContext {
            agent_id: "test-agent".to_string(),
            namespace: "test".to_string(),
            ..Default::default()
        };

        let telos = memoria
            .create_telos(
                "Ship Q3 deck",
                "Prepare investor presentation",
                &ctx,
                1,
                None,
                None,
                TelosProvenance::UserStated,
            )
            .await
            .unwrap();

        assert_eq!(telos.title, "Ship Q3 deck");
        assert_eq!(telos.status, TelosStatus::Active);
        assert_eq!(telos.depth, 1);
        assert!(!telos.embedding.is_empty());

        // Should be retrievable
        let retrieved = memoria.get_telos(telos.id).unwrap().unwrap();
        assert_eq!(retrieved.title, "Ship Q3 deck");
    }

    #[tokio::test]
    async fn test_record_telos_progress() {
        let memoria = Memoria::with_mocks(4).unwrap();
        let ctx = AgentContext {
            agent_id: "test-agent".to_string(),
            namespace: "test".to_string(),
            ..Default::default()
        };

        let telos = memoria
            .create_telos("Test goal", "", &ctx, 2, None, None, TelosProvenance::UserStated)
            .await
            .unwrap();

        memoria
            .record_telos_progress(telos.id, 0.3, "Made progress", "test-agent", None, None)
            .await
            .unwrap();

        let updated = memoria.get_telos(telos.id).unwrap().unwrap();
        assert!((updated.progress - 0.3).abs() < 0.01);

        // Check event was recorded
        let events = memoria.store.get_telos_events(telos.id, 10).unwrap();
        assert!(events.iter().any(|e| e.event_type == "progress"));
    }

    #[tokio::test]
    async fn test_active_telos_scoring() {
        let memoria = Memoria::with_mocks(4).unwrap();
        let ctx = AgentContext {
            agent_id: "test-agent".to_string(),
            namespace: "test".to_string(),
            ..Default::default()
        };

        let _t1 = memoria
            .create_telos("Low priority", "", &ctx, 1, None, None, TelosProvenance::UserStated)
            .await
            .unwrap();

        // Create high priority telos
        let mut high = memoria
            .create_telos("High priority", "", &ctx, 1, None, None, TelosProvenance::UserStated)
            .await
            .unwrap();
        high.priority = 0.95;
        memoria.store.upsert_telos(&high).unwrap();

        let active = memoria.active_telos("test", 10).unwrap();
        assert_eq!(active.len(), 2);
        // Highest attention score should be first
        assert!(active[0].attention_score >= active[1].attention_score);
    }

    #[tokio::test]
    async fn test_mark_criterion_met() {
        let memoria = Memoria::with_mocks(4).unwrap();
        let ctx = AgentContext {
            agent_id: "test-agent".to_string(),
            namespace: "test".to_string(),
            ..Default::default()
        };

        let mut telos = memoria
            .create_telos("Test goal", "", &ctx, 2, None, None, TelosProvenance::UserStated)
            .await
            .unwrap();

        // Add success criteria
        telos.success_criteria = vec![
            SuccessCriterion { id: "a".to_string(), description: "First".to_string(), met: false },
            SuccessCriterion { id: "b".to_string(), description: "Second".to_string(), met: false },
        ];
        memoria.store.upsert_telos(&telos).unwrap();

        // Verify criteria were stored
        let before = memoria.get_telos(telos.id).unwrap().unwrap();
        assert_eq!(before.success_criteria.len(), 2, "criteria should be stored");

        // Mark one criterion
        memoria.mark_criterion_met(telos.id, "a", "test-agent").unwrap();

        let updated = memoria.get_telos(telos.id).unwrap().unwrap();
        assert!((updated.progress - 0.5).abs() < 0.01, "progress={}", updated.progress); // 1 of 2 met
        assert!(updated.success_criteria[0].met);
        assert!(!updated.success_criteria[1].met);
    }
}
