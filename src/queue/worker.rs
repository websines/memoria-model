use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::pipeline::verifier::RelationVerifier;
use crate::queue::task_queue::TaskQueue;
use crate::services::traits::{Embedder, LlmService};
use crate::store::CozoStore;

/// Background task processor for the Memoria task queue.
///
/// Polls the queue at regular intervals, dispatches tasks by type,
/// and handles completion/failure. Shuts down cleanly via CancellationToken.
pub struct QueueWorker {
    queue: Arc<TaskQueue>,
    verifier: Arc<RelationVerifier>,
    store: CozoStore,
    #[allow(dead_code)]
    embedder: Arc<dyn Embedder>,
    #[allow(dead_code)]
    llm: Arc<dyn LlmService>,
    poll_interval: Duration,
}

impl QueueWorker {
    pub fn new(
        queue: Arc<TaskQueue>,
        verifier: Arc<RelationVerifier>,
        store: CozoStore,
        poll_interval: Duration,
    ) -> Self {
        // Backward-compatible constructor using mock services
        use crate::services::mock::{MockEmbedder, MockLlm};
        let dim = store.dim();
        Self {
            queue,
            verifier,
            store,
            embedder: Arc::new(MockEmbedder::new(dim)),
            llm: Arc::new(MockLlm),
            poll_interval,
        }
    }

    /// Create a QueueWorker with full service access (for Phase 5 tasks).
    pub fn with_services(
        queue: Arc<TaskQueue>,
        verifier: Arc<RelationVerifier>,
        store: CozoStore,
        embedder: Arc<dyn Embedder>,
        llm: Arc<dyn LlmService>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            queue,
            verifier,
            store,
            embedder,
            llm,
            poll_interval,
        }
    }

    /// Run the worker loop until the cancellation token is triggered.
    pub async fn run(&self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    break;
                }
                _ = tokio::time::sleep(self.poll_interval) => {
                    if let Err(e) = self.poll_once().await {
                        eprintln!("QueueWorker error: {e}");
                    }
                }
            }
        }
    }

    /// Poll and process one task from the queue.
    async fn poll_once(&self) -> crate::error::Result<()> {
        let lock_duration_ms = 60_000; // 1 minute lock

        let task = self.queue.dequeue(lock_duration_ms)?;
        let task = match task {
            Some(t) => t,
            None => return Ok(()),
        };

        let result = match task.task_type.as_str() {
            "verify_relations" => self.process_verify_relations(&task.payload).await,
            "confidence_propagation" => self.process_confidence_propagation(&task.payload),
            "recompute_pagerank" => self.process_recompute_pagerank(),
            // ── Phase 4: Memory Dynamics tasks ──
            "reflect" => self.process_reflect(&task.payload).await,
            "reconsolidate" => self.process_reconsolidate(&task.payload).await,
            "compress_memories" => self.process_compress_memories(&task.payload).await,
            // ── Phase 5: Procedural Memory tasks ──
            "causal_attribution" => self.process_causal_attribution(&task.payload).await,
            "crystallize_skill" => self.process_crystallize_skill().await,
            "pes_summary" => self.process_pes_summary(&task.payload).await,
            // ── Skill Lifecycle tasks ──
            "improve_skill" => self.process_improve_skill(&task.payload).await,
            "generalize_skills" => self.process_generalize_skills().await,
            // ── Telos Intelligence tasks ──
            "enforce_telos_deadlines" => self.process_enforce_telos_deadlines(),
            "decompose_telos" => self.process_decompose_telos(&task.payload).await,
            "detect_stalls" => self.process_detect_stalls(&task.payload),
            "estimate_progress" => self.process_estimate_progress(&task.payload).await,
            "telos_reflection" => self.process_telos_reflection(&task.payload).await,
            "detect_telos_conflicts" => self.process_detect_telos_conflicts(&task.payload),
            // ── Maintenance tasks ──
            "scratchpad_gc" => self.process_scratchpad_gc(),
            "periodic_reflect" => self.process_reflect(&task.payload).await,
            // ── Phase 6: Active Inference tasks ──
            "snapshot_model_state" => self.process_snapshot_model_state(&task.payload),
            // ── Meta-Learning ──
            "meta_learning_step" => self.process_meta_learning_step(),
            // ── Predictive Generation ──
            "generate_predictions" => self.process_generate_predictions(&task.payload).await,
            "evaluate_expired_predictions" => self.process_evaluate_expired_predictions(&task.payload),
            // ── Embedding Projection ──
            "train_projection" => self.process_train_projection().await,
            // ── Intrinsic Goal Generation ──
            "generate_intrinsic_goals" => self.process_generate_intrinsic_goals(&task.payload).await,
            // ── Structural Causal Models ──
            "accumulate_causal_edges" => self.process_accumulate_causal_edges(&task.payload).await,
            "run_notears_validation" => self.process_run_notears_validation(&task.payload),
            other => Err(crate::error::MemoriaError::TaskQueue(format!(
                "unknown task type: {other}"
            ))),
        };

        match result {
            Ok(r) => {
                self.queue.complete(task.id, task.enqueued_at, &r)?;
            }
            Err(e) => {
                self.queue.fail(
                    task.id,
                    task.enqueued_at,
                    &e.to_string(),
                    task.max_attempts,
                    task.attempts,
                )?;
            }
        }

        Ok(())
    }

    // ── Phase 4: Memory Dynamics task processors ──

    /// Run the reflection pipeline: extract patterns from recent episodes.
    async fn process_reflect(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct ReflectPayload {
            #[serde(default = "default_threshold")]
            surprise_threshold: f64,
        }
        fn default_threshold() -> f64 { 5.0 }

        let payload: ReflectPayload = serde_json::from_str(payload)
            .unwrap_or(ReflectPayload { surprise_threshold: default_threshold() });

        let result = crate::dynamics::reflection::run_reflection(
            &self.store,
            &self.llm,
            &self.embedder,
            payload.surprise_threshold,
        )
        .await?;

        match result {
            Some(r) => Ok(format!(
                "reflection: {} episodes reviewed, {} abstractions created in {}ms",
                r.episodes_reviewed.len(),
                r.abstractions_created,
                r.duration_ms,
            )),
            None => Ok("reflection: below surprise threshold, skipped".to_string()),
        }
    }

    /// Reconsolidate a specific memory: rewrite if newer knowledge contradicts it.
    async fn process_reconsolidate(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct ReconsolidatePayload {
            memory_id: String,
        }

        let payload: ReconsolidatePayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let memory_id = uuid::Uuid::parse_str(&payload.memory_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad memory_id: {e}")))?;

        let memory = self
            .store
            .get_memory(memory_id)?
            .ok_or(crate::error::MemoriaError::NotFound(memory_id))?;

        let result = crate::dynamics::reconsolidation::reconsolidate(
            &self.store,
            &memory,
            &self.llm,
            &self.embedder,
        )
        .await?;

        match result {
            Some(r) => Ok(format!(
                "reconsolidated memory {}: resolved {} contradictions",
                memory_id, r.contradictions_resolved,
            )),
            None => Ok(format!("reconsolidate {}: no contradictions found", memory_id)),
        }
    }

    /// Compress co-activated memory clusters into summaries.
    async fn process_compress_memories(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct CompressPayload {
            namespace: String,
            #[serde(default = "default_min_cluster")]
            min_cluster_size: usize,
        }
        fn default_min_cluster() -> usize { 3 }

        let payload: CompressPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let results = crate::dynamics::compression::compress_clusters(
            &self.store,
            &self.llm,
            &self.embedder,
            payload.min_cluster_size,
            &payload.namespace,
        )
        .await?;

        let total_sources: usize = results.iter().map(|r| r.source_ids.len()).sum();
        Ok(format!(
            "compressed: {} clusters → {} summaries from {} source memories",
            results.len(),
            results.len(),
            total_sources,
        ))
    }

    /// Recompute PageRank and community detection for all memories.
    fn process_recompute_pagerank(&self) -> crate::error::Result<String> {
        let results = crate::dynamics::graph_metrics::compute_pagerank(&self.store)?;
        Ok(format!("pagerank: computed for {} nodes", results.len()))
    }

    // ── Phase 5: Procedural Memory task processors ──

    async fn process_causal_attribution(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct CausalPayload {
            task_id: String,
            task_description: String,
        }

        let payload: CausalPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let task_id = uuid::Uuid::parse_str(&payload.task_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad task_id: {e}")))?;

        let result = crate::causal::attribution::attribute_failure(
            &self.store,
            &self.embedder,
            task_id,
            &payload.task_description,
        )
        .await?;

        Ok(format!(
            "attributed {} memories, analyzed {} tasks",
            result.attributions.len(),
            result.total_tasks_analyzed
        ))
    }

    async fn process_crystallize_skill(&self) -> crate::error::Result<String> {
        let result = crate::skills::lifecycle::crystallize_skills(
            &self.store,
            &self.llm,
            &self.embedder,
        )
        .await?;

        Ok(format!(
            "crystallized {} new skills, reinforced {}",
            result.new_skill_ids.len(),
            result.reinforced_skill_ids.len()
        ))
    }

    async fn process_pes_summary(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct PesSummaryPayload {
            task_id: String,
            plan: String,
            outcome: String,
            failure_reason: Option<String>,
            agent_id: String,
            duration_ms: Option<i64>,
            skills_used: Vec<String>,
        }

        let payload: PesSummaryPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let task_id = uuid::Uuid::parse_str(&payload.task_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad task_id: {e}")))?;

        let skills_used: Vec<uuid::Uuid> = payload
            .skills_used
            .iter()
            .filter_map(|s| uuid::Uuid::parse_str(s).ok())
            .collect();

        let ctx = crate::types::query::AgentContext::new(&payload.agent_id, "default");

        let result = crate::pipeline::pes::summarize(
            &self.store,
            &self.llm,
            &self.embedder,
            task_id,
            &payload.plan,
            &payload.outcome,
            payload.failure_reason.as_deref(),
            &ctx,
            payload.duration_ms,
            &skills_used,
        )
        .await?;

        Ok(format!("PES summary stored for task {}: {}", task_id, result.summary))
    }

    fn process_confidence_propagation(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct PropagationPayload {
            source_id: String,
            new_confidence: f64,
        }

        let payload: PropagationPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let source_id = uuid::Uuid::parse_str(&payload.source_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad source_id: {e}")))?;

        let result = crate::aif::propagation::propagate_confidence(
            &self.store,
            source_id,
            payload.new_confidence,
        )?;

        Ok(format!(
            "propagated: {} memories, {} facts updated, {} nodes visited",
            result.memories_updated, result.facts_updated, result.nodes_visited
        ))
    }

    fn process_snapshot_model_state(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct SnapshotPayload {
            agent_id: String,
        }

        let payload: SnapshotPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let state =
            crate::aif::model_state::snapshot_model_state(&self.store, &payload.agent_id)?;

        Ok(format!(
            "snapshot: FE={:.4}, beta={:.4}",
            state.free_energy, state.beta
        ))
    }

    // ── Meta-Learning task processor ──

    /// Run one meta-learning step: load/create learner, feed current free energy,
    /// let the learner decide whether to adjust parameters.
    fn process_meta_learning_step(&self) -> crate::error::Result<String> {
        use crate::dynamics::meta_learning;

        // Load or create the meta-learner (persisted across restarts)
        let mut learner = meta_learning::load_optimizer_state(&self.store)?
            .unwrap_or_else(|| {
                // Initialize from config defaults (observation_window, bo_budget
                // are read from DynamicsConfig but we don't have access to it here,
                // so we use sensible defaults that match the config defaults)
                meta_learning::MetaLearner::new(5, 150)
            });

        // Get current free energy state
        let fe_state = crate::aif::compute_bethe_free_energy(&self.store)?;
        let surprise = crate::dynamics::surprise::accumulated_unresolved_surprise(&self.store)
            .unwrap_or(0.0);

        // Advance one step
        match learner.step(&self.store, fe_state.free_energy, fe_state.beta, surprise)? {
            Some(result) => {
                let adj_count = result.adjustments.len();
                let gen = result.generation;
                Ok(format!(
                    "meta-learning gen {gen}: {adj_count} adjustments, FE={:.4}, phase={:?}",
                    result.free_energy, result.phase,
                ))
            }
            None => Ok("meta-learning: waiting for observation window".into()),
        }
    }

    // ── Telos Intelligence task processors ──

    /// Decompose a telos into subtelos using LLM.
    async fn process_decompose_telos(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct DecomposePayload {
            telos_id: String,
            #[serde(default = "default_namespace")]
            namespace: String,
        }
        fn default_namespace() -> String {
            "default".to_string()
        }

        let payload: DecomposePayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let telos_id = uuid::Uuid::parse_str(&payload.telos_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad telos_id: {e}")))?;

        let children = crate::api::telos_decompose::decompose_telos(
            &self.store,
            &self.llm,
            &self.embedder,
            telos_id,
            &payload.namespace,
        )
        .await?;

        Ok(format!(
            "decomposed telos {} into {} subtelos",
            telos_id,
            children.len()
        ))
    }

    /// Detect stalled telos goals based on inactivity thresholds.
    fn process_detect_stalls(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct StallPayload {
            #[serde(default = "default_namespace")]
            namespace: String,
        }
        fn default_namespace() -> String {
            "default".to_string()
        }

        let payload: StallPayload = serde_json::from_str(payload)
            .unwrap_or(StallPayload { namespace: default_namespace() });

        let active = self.store.list_active_telos(&payload.namespace, 100)?;
        let now = crate::types::memory::now_ms();
        let hour_ms: i64 = 3_600_000;
        let mut stalled_count = 0;

        for telos in &active {
            // Skip already-stalled goals
            if telos.stalled_since.is_some() {
                continue;
            }

            // Stall threshold scales inversely with urgency
            let threshold_ms = if telos.urgency >= 0.8 {
                2 * hour_ms
            } else if telos.urgency >= 0.5 {
                4 * hour_ms
            } else if telos.urgency >= 0.2 {
                12 * hour_ms
            } else {
                24 * hour_ms
            };

            let idle_ms = now - telos.last_attended;
            if idle_ms >= threshold_ms {
                self.store.update_telos_status(telos.id, "stalled")?;

                let mut event = crate::types::telos::TelosEvent::new(telos.id, "stalled");
                event.agent_id = "detect_stalls".to_string();
                event.description = format!(
                    "No activity for {} hours (threshold: {}h)",
                    idle_ms / hour_ms,
                    threshold_ms / hour_ms,
                );
                self.store.insert_telos_event(&event)?;

                stalled_count += 1;
            }
        }

        Ok(format!(
            "detect_stalls: checked {} active telos, {} newly stalled",
            active.len(),
            stalled_count
        ))
    }

    /// Enforce deadlines on active telos — stalls any goal whose deadline is in the past.
    fn process_enforce_telos_deadlines(&self) -> crate::error::Result<String> {
        let stalled = crate::dynamics::deadline::enforce_deadlines(&self.store)?;
        Ok(format!(
            "enforce_telos_deadlines: {} telos marked stalled (deadline exceeded)",
            stalled
        ))
    }

    /// Estimate progress for a telos using deterministic signals + optional LLM.
    async fn process_estimate_progress(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct ProgressPayload {
            telos_id: String,
            #[serde(default)]
            use_llm: bool,
        }

        let payload: ProgressPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let telos_id = uuid::Uuid::parse_str(&payload.telos_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad telos_id: {e}")))?;

        let estimate = crate::api::telos_progress::estimate_progress(&self.store, telos_id)?;

        // If deterministic signals are insufficient and LLM is requested, use cold path
        let final_estimate = if estimate.method == crate::api::telos_progress::ProgressMethod::None
            && payload.use_llm
        {
            crate::api::telos_progress::estimate_progress_llm(
                &self.store,
                &self.llm,
                telos_id,
            )
            .await?
        } else {
            estimate
        };

        // Update the telos progress if we got a meaningful estimate
        if final_estimate.confidence > 0.0 {
            self.store
                .update_telos_progress(telos_id, final_estimate.progress)?;
        }

        Ok(format!(
            "progress estimate for {}: {:.1}% via {:?} (confidence: {:.2})",
            telos_id,
            final_estimate.progress * 100.0,
            final_estimate.method,
            final_estimate.confidence,
        ))
    }

    /// Reflect on a completed/failed telos — extract lessons learned.
    async fn process_telos_reflection(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct ReflectionPayload {
            telos_id: String,
            #[serde(default)]
            outcome: String,
        }

        let payload: ReflectionPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let telos_id = uuid::Uuid::parse_str(&payload.telos_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad telos_id: {e}")))?;

        let telos = self
            .store
            .get_telos(telos_id)?
            .ok_or_else(|| crate::error::MemoriaError::NotFound(telos_id))?;

        let events = self.store.get_telos_events(telos_id, 50)?;
        let event_summary: String = events
            .iter()
            .map(|e| format!("- [{}] {}: {}", e.event_type, e.agent_id, e.description))
            .collect::<Vec<_>>()
            .join("\n");

        let outcome = if payload.outcome.is_empty() {
            telos.status.as_str().to_string()
        } else {
            payload.outcome.clone()
        };

        let prompt = format!(
            concat!(
                "A goal has reached outcome '{outcome}'. Extract lessons learned.\n\n",
                "Goal: {title}\n",
                "Description: {description}\n",
                "Final progress: {progress:.0}%\n\n",
                "Event history:\n{events}\n\n",
                "Return a JSON object with:\n",
                "- \"lessons\": array of lesson strings\n",
                "- \"should_retry\": boolean (for failed goals)\n",
                "- \"pattern\": one-sentence pattern for future similar goals\n",
                "No markdown fences."
            ),
            outcome = outcome,
            title = telos.title,
            description = if telos.description.is_empty() {
                &telos.title
            } else {
                &telos.description
            },
            progress = telos.progress * 100.0,
            events = if event_summary.is_empty() {
                "No events.".to_string()
            } else {
                event_summary
            },
        );

        use crate::services::traits::Message;
        let llm_response = self
            .llm
            .complete(
                &[
                    Message {
                        role: "system".into(),
                        content: "You extract lessons from completed goals. Return valid JSON."
                            .into(),
                    },
                    Message {
                        role: "user".into(),
                        content: prompt,
                    },
                ],
                512,
            )
            .await
            .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

        // Store the reflection as a telos event
        let mut event = crate::types::telos::TelosEvent::new(telos_id, "reflection");
        event.agent_id = "telos_reflection".to_string();
        event.description = llm_response.content.clone();
        self.store.insert_telos_event(&event)?;

        // Store the reflection as a memory (abstraction) via tell
        let reflection_text = format!(
            "TELOS REFLECTION ({}): Goal '{}' — {}. Lessons: {}",
            outcome,
            telos.title,
            if telos.progress >= 1.0 {
                "completed successfully"
            } else {
                "did not complete"
            },
            &llm_response.content[..llm_response.content.len().min(500)],
        );

        let ctx = crate::types::query::AgentContext::new("telos_reflection", &telos.namespace);
        // Embed and store as an abstraction memory
        let embeddings = self
            .embedder
            .embed(&[reflection_text.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

        if let Some(embedding) = embeddings.into_iter().next() {
            let mut memory = crate::types::memory::Memory::new(
                "abstraction.telos_reflection",
                &reflection_text,
                embedding,
            );
            memory.namespace = ctx.namespace.clone();
            memory.fields.insert(
                "telos_id".to_string(),
                serde_json::Value::String(telos_id.to_string()),
            );
            memory.fields.insert(
                "outcome".to_string(),
                serde_json::Value::String(outcome),
            );
            self.store.insert_memory(&memory)?;
        }

        Ok(format!(
            "telos reflection for '{}': stored lessons",
            telos.title
        ))
    }

    /// Detect conflicting/duplicate telos goals in a namespace.
    fn process_detect_telos_conflicts(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct ConflictPayload {
            #[serde(default = "default_namespace")]
            namespace: String,
            #[serde(default = "default_threshold")]
            similarity_threshold: f64,
        }
        fn default_namespace() -> String {
            "default".to_string()
        }
        fn default_threshold() -> f64 {
            0.85
        }

        let payload: ConflictPayload = serde_json::from_str(payload)
            .unwrap_or(ConflictPayload {
                namespace: default_namespace(),
                similarity_threshold: default_threshold(),
            });

        let conflicts = crate::api::telos_multi::detect_conflicts(
            &self.store,
            &payload.namespace,
            payload.similarity_threshold,
        )?;

        Ok(format!(
            "detect_telos_conflicts: found {} conflicts in namespace '{}'",
            conflicts.len(),
            payload.namespace,
        ))
    }

    // ── Skill Lifecycle task processors ──

    async fn process_improve_skill(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct ImprovePayload {
            skill_id: String,
        }

        let payload: ImprovePayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let skill_id = uuid::Uuid::parse_str(&payload.skill_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad skill_id: {e}")))?;

        let result = crate::skills::lifecycle::improve_skill(
            &self.store,
            &self.llm,
            &self.embedder,
            skill_id,
        )
        .await?;

        match result {
            Some(new_version_id) => Ok(format!(
                "improved skill {skill_id} → new version {new_version_id}"
            )),
            None => Ok(format!("improve skill {skill_id}: no improvement needed")),
        }
    }

    async fn process_generalize_skills(&self) -> crate::error::Result<String> {
        let new_skill_ids = crate::skills::lifecycle::generalize_skills(
            &self.store,
            &self.llm,
            &self.embedder,
        )
        .await?;

        // Classify new generalized skills into niches
        for &skill_id in &new_skill_ids {
            let _ = crate::skills::niche::classify_niche(
                &self.store,
                &self.llm,
                skill_id,
            )
            .await;
        }

        Ok(format!("generalized: {} new skills created", new_skill_ids.len()))
    }

    // ── Predictive Generation task processors ──

    /// Generate predictions from all generators (PPM-C, ETS, BOCPD, episodic).
    async fn process_generate_predictions(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct PredictionPayload {
            #[serde(default = "default_namespace")]
            namespace: String,
            #[serde(default = "default_agent_id")]
            agent_id: String,
            #[serde(default = "default_max_depth")]
            ppm_max_depth: usize,
            #[serde(default)]
            cycle: u64,
        }
        fn default_namespace() -> String { "default".to_string() }
        fn default_agent_id() -> String { "system".to_string() }
        fn default_max_depth() -> usize { 5 }

        let payload: PredictionPayload = serde_json::from_str(payload)
            .unwrap_or(PredictionPayload {
                namespace: default_namespace(),
                agent_id: default_agent_id(),
                ppm_max_depth: default_max_depth(),
                cycle: 0,
            });

        use crate::dynamics::prediction;
        let mut all_predictions = Vec::new();

        // Generator 1: PPM-C sequence predictions
        match prediction::generate_sequence_predictions(
            &self.store, &payload.agent_id, &payload.namespace, payload.ppm_max_depth,
        ) {
            Ok(preds) => all_predictions.extend(preds),
            Err(e) => eprintln!("PPM-C prediction error: {e}"),
        }

        // Generator 2: ETS telos progress predictions
        match prediction::generate_telos_predictions(&self.store, &payload.namespace) {
            Ok(preds) => all_predictions.extend(preds),
            Err(e) => eprintln!("ETS prediction error: {e}"),
        }

        // Generator 3: BOCPD regime change detection
        let regime_stable = match prediction::detect_regime_changes(
            &self.store, &payload.namespace,
        ) {
            Ok((preds, stable)) => {
                all_predictions.extend(preds);
                stable
            }
            Err(e) => {
                eprintln!("BOCPD detection error: {e}");
                true
            }
        };

        // Generator 4: LLM episodic pattern prediction (every 5th cycle)
        if payload.cycle % 5 == 0 {
            match prediction::generate_episodic_predictions(
                &self.store, &*self.llm, &*self.embedder, &payload.namespace,
            ).await {
                Ok(preds) => all_predictions.extend(preds),
                Err(e) => eprintln!("Episodic prediction error: {e}"),
            }
        }

        // Store all generated predictions
        let count = all_predictions.len();
        if !all_predictions.is_empty() {
            prediction::store_predictions(&self.store, &all_predictions)?;
        }

        Ok(format!(
            "predictions: generated {count}, regime_stable={regime_stable}"
        ))
    }

    /// Evaluate expired predictions as misses and compute prediction accuracy.
    fn process_evaluate_expired_predictions(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct EvalPayload {
            #[serde(default = "default_namespace")]
            namespace: String,
        }
        fn default_namespace() -> String { "default".to_string() }

        let payload: EvalPayload = serde_json::from_str(payload)
            .unwrap_or(EvalPayload { namespace: default_namespace() });

        use crate::dynamics::prediction;
        let now = crate::types::memory::now_ms();
        let expired = prediction::get_expired_unresolved(&self.store, &payload.namespace, now)?;
        let expired_count = expired.len();

        for pred in &expired {
            prediction::resolve_prediction(&self.store, pred.id, false, None, 1.0)?;
        }

        let accuracy = prediction::prediction_accuracy(&self.store, &payload.namespace)?;

        Ok(format!(
            "evaluate_expired: {expired_count} resolved as misses, accuracy={accuracy:.3}"
        ))
    }

    /// Generate intrinsic telos from surprise hotspots.
    async fn process_generate_intrinsic_goals(
        &self,
        payload: &str,
    ) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct IntrinsicPayload {
            #[serde(default = "default_cooldown")]
            cooldown_ms: i64,
        }
        fn default_cooldown() -> i64 {
            300_000 // 5 minutes
        }

        let payload: IntrinsicPayload = serde_json::from_str(payload)
            .unwrap_or(IntrinsicPayload {
                cooldown_ms: default_cooldown(),
            });

        let result = crate::dynamics::intrinsic::generate_intrinsic_goals(
            &self.store,
            &*self.embedder,
            &*self.llm,
            payload.cooldown_ms,
        )
        .await?;

        Ok(format!(
            "intrinsic_goals: {} hotspots, {} created, {} activated, {} dupes skipped, {} cooldown",
            result.hotspots_found,
            result.goals_created,
            result.goals_activated,
            result.duplicates_skipped,
            result.cooldown_skipped,
        ))
    }

    /// Garbage-collect expired scratchpad entries.
    fn process_scratchpad_gc(&self) -> crate::error::Result<String> {
        let removed = self.store.scratch_gc()?;
        Ok(format!("scratchpad_gc: removed {} expired entries", removed))
    }

    /// Train the embedding projection on triplets from recall outcomes.
    async fn process_train_projection(&self) -> crate::error::Result<String> {
        use crate::dynamics::projection;

        let dim = self.store.dim();
        let min_triplets = 50; // from config default
        let epochs = 10;
        let lr = 0.01f32;

        // 1. Collect triplets from recall_contexts + task_outcomes
        let triplets = projection::collect_triplets(&self.store, &*self.embedder, min_triplets).await?;
        if triplets.len() < min_triplets {
            return Ok(format!(
                "train_projection: only {} triplets (need {}), skipped",
                triplets.len(),
                min_triplets,
            ));
        }

        let start = std::time::Instant::now();

        // 2. Load existing projection or create identity
        let mut proj = projection::load_projection(&self.store, dim)?
            .unwrap_or_else(|| projection::EmbeddingProjection::identity(dim));

        // 3. Compute loss before training
        let loss_before = proj.triplet_loss(&triplets);

        // 4. Train
        let loss_after = proj.train(&triplets, epochs, lr);

        let duration_ms = start.elapsed().as_millis() as u64;

        // 5. Save projection
        projection::save_projection(&self.store, &proj)?;

        // 6. Save stats
        let stats = projection::ProjectionStats {
            loss_before,
            loss_after,
            triplet_count: triplets.len(),
            epochs,
            duration_ms,
        };
        projection::save_training_stats(&self.store, &stats)?;

        // 7. Compute improvement
        let improvement = if loss_before > 0.0 {
            (loss_before - loss_after) / loss_before
        } else {
            0.0
        };

        Ok(format!(
            "train_projection: {} triplets, loss {:.4} → {:.4} ({:.1}% improvement, {}ms)",
            triplets.len(),
            loss_before,
            loss_after,
            improvement * 100.0,
            duration_ms,
        ))
    }

    // ── Structural Causal Models task processors ──

    /// Process causal attribution and accumulate edges.
    ///
    /// Re-runs attribution for a failed task and stores resulting causal edges.
    /// This is typically enqueued after each task failure.
    async fn process_accumulate_causal_edges(
        &self,
        payload: &str,
    ) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct AccumulatePayload {
            task_id: String,
            task_description: String,
        }

        let payload: AccumulatePayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let task_id = uuid::Uuid::parse_str(&payload.task_id)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad task_id: {e}")))?;

        // Run attribution — this now automatically accumulates causal edges
        let result = crate::causal::attribution::attribute_failure(
            &self.store,
            &self.embedder,
            task_id,
            &payload.task_description,
        )
        .await?;

        let edge_count = crate::causal::graph::count_edges(&self.store, "")?;

        Ok(format!(
            "accumulate_causal_edges: {} attributions → {} total causal edges",
            result.attributions.len(),
            edge_count,
        ))
    }

    /// Run NOTEARS structure learning to validate and discover causal edges.
    ///
    /// Periodically (every ~100 ticks), run NOTEARS on accumulated observations
    /// to discover new causal relationships and validate existing ones.
    fn process_run_notears_validation(
        &self,
        payload: &str,
    ) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct NotearsPayload {
            #[serde(default = "default_namespace")]
            namespace: String,
            #[serde(default = "default_max_vars")]
            max_variables: usize,
        }
        fn default_namespace() -> String {
            "".to_string()
        }
        fn default_max_vars() -> usize {
            50
        }

        let payload: NotearsPayload = serde_json::from_str(payload)
            .unwrap_or(NotearsPayload {
                namespace: default_namespace(),
                max_variables: default_max_vars(),
            });

        // Build observation matrix from existing data
        let obs = crate::causal::notears::build_observation_matrix(
            &self.store,
            &payload.namespace,
            payload.max_variables,
        )?;

        let (data, variable_ids) = match obs {
            Some((d, v)) => (d, v),
            None => {
                return Ok(
                    "run_notears_validation: insufficient data for structure learning".to_string(),
                )
            }
        };

        // Run NOTEARS
        let config = crate::causal::notears::NotearsConfig {
            lambda_l1: 0.1,
            max_outer_iter: 10,
            max_inner_iter: 100,
            ..Default::default()
        };
        let result = crate::causal::notears::notears(&data, &config);

        // Cross-reference discovered edges with existing causal graph
        let existing_edges = crate::causal::graph::load_all_edges(&self.store, &payload.namespace)?;
        let now = crate::types::memory::now_ms();
        let mut edges_discovered = 0;
        let mut edges_confirmed = 0;
        let d = variable_ids.len();

        for i in 0..d {
            for j in 0..d {
                let weight = result.adjacency[[i, j]];
                if weight.abs() < 0.01 {
                    continue;
                }

                let cause = variable_ids[i];
                let effect = variable_ids[j];

                // Check if this edge already exists
                let existing = existing_edges
                    .iter()
                    .find(|e| e.cause_id == cause && e.effect_id == effect);

                if let Some(_existing_edge) = existing {
                    // NOTEARS confirms existing edge — boost confidence
                    crate::causal::graph::adjust_edge_confidence(
                        &self.store,
                        cause,
                        effect,
                        true,
                    )?;
                    edges_confirmed += 1;
                } else {
                    // NOTEARS discovered novel edge
                    let edge = crate::causal::graph::CausalEdge {
                        cause_id: cause,
                        effect_id: effect,
                        causal_strength: weight.abs().min(1.0),
                        observations: 1,
                        last_observed: now,
                        mechanism: crate::causal::graph::CausalMechanism::NotearsDiscovered,
                        confidence: 0.4, // moderate confidence for data-driven discovery
                        namespace: payload.namespace.clone(),
                    };
                    crate::causal::graph::accumulate_causal_edge(&self.store, &edge)?;
                    edges_discovered += 1;
                }
            }
        }

        // Check for edges that NOTEARS contradicts (exist but NOTEARS found no weight)
        for existing_edge in &existing_edges {
            let cause_idx = variable_ids.iter().position(|&id| id == existing_edge.cause_id);
            let effect_idx = variable_ids
                .iter()
                .position(|&id| id == existing_edge.effect_id);

            if let (Some(i), Some(j)) = (cause_idx, effect_idx) {
                if result.adjacency[[i, j]].abs() < 0.01 {
                    // NOTEARS contradicts this edge — reduce confidence
                    crate::causal::graph::adjust_edge_confidence(
                        &self.store,
                        existing_edge.cause_id,
                        existing_edge.effect_id,
                        false,
                    )?;
                }
            }
        }

        Ok(format!(
            "run_notears_validation: {} discovered, {} confirmed, converged={}",
            edges_discovered, edges_confirmed, result.converged,
        ))
    }

    /// Process a verify_relations task.
    async fn process_verify_relations(&self, payload: &str) -> crate::error::Result<String> {
        #[derive(serde::Deserialize)]
        struct VerifyPayload {
            text: String,
            relations: Vec<crate::services::traits::ExtractedRelation>,
        }

        let payload: VerifyPayload = serde_json::from_str(payload)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("bad payload: {e}")))?;

        let verified = self.verifier.verify(&payload.text, &payload.relations).await?;

        serde_json::to_string(&verified)
            .map_err(|e| crate::error::MemoriaError::TaskQueue(format!("serialize result: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::mock::MockLlm;

    #[tokio::test]
    async fn test_worker_processes_task() {
        let store = CozoStore::open_mem(4).unwrap();
        let queue = Arc::new(TaskQueue::new(CozoStore::open_mem(4).unwrap()));
        let verifier = Arc::new(RelationVerifier::new(Arc::new(MockLlm)));
        let worker = QueueWorker::new(
            Arc::clone(&queue),
            verifier,
            store,
            Duration::from_millis(100),
        );

        // Enqueue a task
        queue
            .enqueue(
                "verify_relations",
                0,
                r#"{"text": "Alice works at Acme", "relations": [{"head": "Alice", "tail": "Acme", "label": "works_at", "confidence": 0.9}]}"#,
                3,
            )
            .unwrap();

        // Process it
        worker.poll_once().await.unwrap();

        // Queue should be empty now
        assert_eq!(queue.count_pending().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_worker_shutdown() {
        let store = CozoStore::open_mem(4).unwrap();
        let queue = Arc::new(TaskQueue::new(CozoStore::open_mem(4).unwrap()));
        let verifier = Arc::new(RelationVerifier::new(Arc::new(MockLlm)));
        let worker = QueueWorker::new(
            Arc::clone(&queue),
            verifier,
            store,
            Duration::from_millis(50),
        );

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let handle = tokio::spawn(async move {
            worker.run(cancel_clone).await;
        });

        // Let it run briefly
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        // Should shut down cleanly
        handle.await.unwrap();
    }
}
