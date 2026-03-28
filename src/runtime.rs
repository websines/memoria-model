use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use arc_swap::ArcSwap;
use tokio::sync::broadcast;

use crate::cache::hot::HotCache;
use crate::config::MemoriaConfig;
use crate::pipeline::chunker::SemanticChunker;
use crate::pipeline::scoring::FactorScorer;
use crate::pipeline::verifier::RelationVerifier;
use crate::services::traits::{Embedder, LlmService, NerService, Reranker};
use crate::store::CozoStore;
use crate::types::event::MemoryEvent;
use crate::types::kernel::KernelRule;

/// Result of the `prime()` API — pre-task context for an agent.
pub struct PrimeResult {
    /// EFE-ranked skills for the task.
    pub skills: Vec<crate::skills::selection::ScoredSkill>,
    /// Pre-activated memories relevant to the task.
    pub memories: crate::types::query::AskResult,
    /// Current auto-tuned β (exploration/exploitation balance).
    pub beta: f64,
    /// Current system free energy.
    pub free_energy: f64,
    /// Accumulated unresolved surprise (high = needs attention).
    pub unresolved_surprise: f64,
    /// Predicted next task types from sequential pattern mining (type, confidence).
    pub predicted_next_tasks: Vec<(String, f64)>,
    /// Active telos (goals) ranked by attention score.
    pub active_telos: Vec<crate::types::telos::ScoredTelos>,
    /// Active predictions — what the system currently expects to happen.
    pub active_predictions: Vec<crate::dynamics::Prediction>,
    /// Rolling prediction accuracy (0.0 = all wrong, 1.0 = all correct).
    pub prediction_accuracy: f64,
    /// Whether BOCPD indicates the current regime is stable (no recent changepoint).
    pub regime_stable: bool,
}

/// Result of the `feedback()` API — post-task belief updates.
pub struct FeedbackResult {
    /// Updated free energy after incorporating the outcome.
    pub free_energy: f64,
    /// Updated β after the snapshot.
    pub beta: f64,
    /// Current unresolved surprise level.
    pub unresolved_surprise: f64,
    /// Whether the outcome surprise exceeded the adaptive threshold.
    pub surprise_triggered: bool,
    /// Whether consolidation (reflection) was actually run.
    pub consolidation_triggered: bool,
}

/// Result of a single tick iteration.
#[derive(Debug, Clone, Default)]
pub struct TickResult {
    /// Whether a compression task was enqueued.
    pub compression_enqueued: bool,
    /// Number of expired scratchpad entries removed.
    pub scratchpad_gc_removed: usize,
    /// Whether a reflection task was enqueued.
    pub reflection_enqueued: bool,
    /// Whether a stall detection task was enqueued.
    pub stall_detection_enqueued: bool,
    /// Whether an intrinsic goal generation task was enqueued.
    pub intrinsic_goal_enqueued: bool,
}

/// The main Memoria runtime — holds all services and drives the tell/ask pipelines.
///
/// Create one per process. Thread-safe (all fields are Send + Sync).
pub struct Memoria {
    pub(crate) store: CozoStore,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) ner: Arc<dyn NerService>,
    pub(crate) reranker: Arc<dyn Reranker>,
    #[allow(dead_code)]
    pub(crate) llm: Arc<dyn LlmService>,
    /// Base config — the original, never mutated by meta-learning.
    /// Use `effective_config` for runtime reads.
    pub(crate) base_config: MemoriaConfig,
    /// Live-tuned config, swapped lock-free by the meta-learning subsystem.
    /// All runtime code should read from this, not `base_config`.
    pub(crate) config: ArcSwap<MemoriaConfig>,
    pub(crate) chunker: SemanticChunker,
    pub(crate) scorer: FactorScorer,
    pub(crate) verifier: RelationVerifier,
    #[allow(dead_code)]
    pub(crate) cache: HotCache,
    /// Optional task queue for async offloading (e.g., verification).
    pub(crate) task_queue: Option<Arc<crate::queue::TaskQueue>>,
    /// Broadcast channel for memory events (Phase 5: Event Channel).
    pub(crate) event_tx: broadcast::Sender<MemoryEvent>,
    /// Kernel rules for guarded operations (Phase 7).
    /// Wrapped in RwLock for interior mutability (allows add_kernel_rule via &self / Arc<Memoria>).
    pub(crate) kernel_rules: RwLock<Vec<KernelRule>>,
    /// Monotonic tick counter for meta-learning scheduling.
    pub(crate) tick_count: AtomicU64,
    /// Learned embedding projection — None means identity (no-op).
    /// Swapped lock-free after training completes.
    pub(crate) projection: ArcSwap<Option<crate::dynamics::projection::EmbeddingProjection>>,
}

impl Memoria {
    /// Create a new Memoria runtime with the given services and config.
    pub fn new(
        store: CozoStore,
        embedder: Arc<dyn Embedder>,
        ner: Arc<dyn NerService>,
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn LlmService>,
        config: MemoriaConfig,
    ) -> Self {
        let verifier = RelationVerifier::new(Arc::clone(&llm));
        let (event_tx, _) = broadcast::channel(config.event_channel_capacity);
        let kernel_rules = RwLock::new(config.kernel_rules.rules.clone());
        let effective = ArcSwap::from_pointee(config.clone());
        Self {
            store,
            embedder,
            ner,
            reranker,
            llm,
            base_config: config,
            config: effective,
            chunker: SemanticChunker::default(),
            scorer: FactorScorer,
            verifier,
            cache: HotCache::new(10_000),
            task_queue: None,
            event_tx,
            kernel_rules,
            tick_count: AtomicU64::new(0),
            projection: ArcSwap::from_pointee(None),
        }
    }

    /// Create a Memoria runtime with mock services (for testing).
    pub fn with_mocks(dim: usize) -> crate::error::Result<Self> {
        use crate::services::mock::{MockEmbedder, MockLlm, MockNer, MockReranker};

        let store = CozoStore::open_mem(dim)?;
        Ok(Self::new(
            store,
            Arc::new(MockEmbedder::new(dim)),
            Arc::new(MockNer),
            Arc::new(MockReranker),
            Arc::new(MockLlm),
            MemoriaConfig::default(),
        ))
    }

    /// Set the task queue for async offloading.
    pub fn set_task_queue(&mut self, queue: Arc<crate::queue::TaskQueue>) {
        self.task_queue = Some(queue);
    }

    /// Load the learned embedding projection from CozoDB at boot time.
    ///
    /// Call this after construction to restore a previously trained projection.
    pub fn load_projection(&self) {
        let dim = self.store.dim();
        if let Ok(Some(proj)) = crate::dynamics::projection::load_projection(&self.store, dim) {
            self.projection.store(Arc::new(Some(proj)));
        }
    }

    /// Embed a text string using the configured embedder.
    ///
    /// Thin wrapper exposing the embedder for external consumers (e.g. NoteTool).
    pub async fn embed_text(&self, text: &str) -> crate::error::Result<Vec<f32>> {
        let embeddings = self
            .embedder
            .embed(&[text])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::error::MemoriaError::Embedding("empty embedding result".into()))
    }

    /// Read a snapshot of the current effective config.
    pub fn effective_config(&self) -> arc_swap::Guard<Arc<MemoriaConfig>> {
        self.config.load()
    }

    /// Read the base (untuned) config.
    pub fn base_config(&self) -> &MemoriaConfig {
        &self.base_config
    }

    /// Schedule periodic maintenance tasks (scratchpad GC, compression, reflection).
    pub fn schedule_periodic_tasks(&self) -> crate::error::Result<()> {
        if let Some(ref queue) = self.task_queue {
            // Scratchpad GC
            let _ = queue.enqueue("scratchpad_gc", 3, "{}", 1);

            // Periodic reflection
            let payload = serde_json::json!({
                "surprise_threshold": self.config.load().consolidation_threshold,
            });
            let _ = queue.enqueue("reflect", 2, &payload.to_string(), 2);

            // PageRank recomputation
            let _ = queue.enqueue("recompute_pagerank", 4, "{}", 1);

            // Periodic free energy snapshot for historical trends
            let snap_payload = serde_json::json!({ "agent_id": "system" });
            let _ = queue.enqueue("snapshot_model_state", 3, &snap_payload.to_string(), 1);

            // Deadline enforcement — stall active telos whose deadline has passed
            let _ = queue.enqueue("enforce_telos_deadlines", 5, "{}", 1);
        }
        Ok(())
    }

    /// Run a single tick: check thresholds and enqueue maintenance tasks.
    ///
    /// - Checks if any namespace exceeds the compression memory threshold
    /// - Runs scratchpad GC inline (lightweight)
    /// - Checks if accumulated surprise exceeds consolidation threshold
    pub fn tick(&self) -> crate::error::Result<TickResult> {
        let mut result = TickResult::default();

        // 1. Check compression threshold for default namespace
        if let Some(ref queue) = self.task_queue {
            if let Ok(count) = self.store.count_memories_in_namespace("default") {
                let cfg = self.config.load();
                let threshold = cfg.dynamics.compression_threshold();
                if count > threshold {
                    let payload = serde_json::json!({
                        "namespace": "default",
                        "min_cluster_size": cfg.dynamics.min_cluster_size,
                    });
                    let _ = queue.enqueue("compress_memories", 2, &payload.to_string(), 2);
                    result.compression_enqueued = true;
                }
            }
        }

        // 2. Scratchpad GC (inline — lightweight)
        if let Ok(removed) = self.store.scratch_gc() {
            result.scratchpad_gc_removed = removed;
        }

        // 3. Check reflection threshold
        if let Some(ref queue) = self.task_queue {
            if let Ok(unresolved) = crate::dynamics::surprise::accumulated_unresolved_surprise(&self.store) {
                if let Ok(threshold) = self.effective_consolidation_threshold() {
                    if unresolved > threshold {
                        let payload = serde_json::json!({
                            "surprise_threshold": threshold,
                        });
                        let _ = queue.enqueue("reflect", 2, &payload.to_string(), 2);
                        result.reflection_enqueued = true;
                    }
                }
            }
        }

        // 4. Telos stall detection
        if let Some(ref queue) = self.task_queue {
            let payload = serde_json::json!({
                "namespace": "default",
            });
            let _ = queue.enqueue("detect_stalls", 3, &payload.to_string(), 1);
            result.stall_detection_enqueued = true;

            // Deadline enforcement — stall active telos whose deadline has passed
            let _ = queue.enqueue("enforce_telos_deadlines", 5, "{}", 1);
        }

        // 5. Auto-abandon telos with decayed confidence below threshold
        {
            let now = crate::types::memory::now_ms();
            if let Ok(confidences) = crate::dynamics::confidence::compute_all_telos_confidences(
                &self.store, "default", Some(now),
            ) {
                for (telos_id, eff_confidence) in &confidences {
                    if *eff_confidence < 0.2 {
                        // Auto-abandon: confidence has decayed too low
                        let _ = self.store.update_telos_status(*telos_id, "abandoned");
                        let mut event = crate::types::telos::TelosEvent::new(*telos_id, "auto_abandoned");
                        event.agent_id = "tick".to_string();
                        event.description = format!(
                            "Auto-abandoned: effective confidence {:.3} below threshold",
                            eff_confidence
                        );
                        let _ = self.store.insert_telos_event(&event);
                    }
                }
            }
        }

        // 6. Skill lifecycle: check for skills eligible for improvement
        //    Skills with 3+ successful adaptations are candidates for improvement
        if let Some(ref queue) = self.task_queue {
            if let Ok(query_result) = self.store.run_query(
                r#"?[skill_id, count(id)] :=
                    *skill_usages{id, skill_id, outcome},
                    outcome = "success"
                    :order -count(id)
                    :limit 5"#,
                std::collections::BTreeMap::new(),
            ) {
                for row in &query_result.rows {
                    let count = row[1].get_int().unwrap_or(0);
                    if count >= 3 {
                        if let Ok(skill_id) = crate::store::cozo::parse_uuid_pub(&row[0]) {
                            let payload = serde_json::json!({ "skill_id": skill_id.to_string() });
                            let _ = queue.enqueue("improve_skill", 3, &payload.to_string(), 1);
                        }
                    }
                }
            }

            // 7. Periodically enqueue skill generalization
            let _ = queue.enqueue("generalize_skills", 3, "{}", 1);

            // 8. Crystallize skills — worker deduplicates internally
            let _ = queue.enqueue("crystallize_skill", 3, "{}", 1);

            // 9. Recompute pagerank for graph importance ranking
            let _ = queue.enqueue("recompute_pagerank", 4, "{}", 1);

            // 10. Estimate progress for active telos goals
            if let Ok(active) = self.store.list_active_telos("default", 50) {
                for telos in &active {
                    let payload = serde_json::json!({
                        "telos_id": telos.id.to_string(),
                    });
                    let _ = queue.enqueue("estimate_progress", 3, &payload.to_string(), 1);
                }
            }

            // 11. Periodic free energy snapshot for trend history
            let snap_payload = serde_json::json!({ "agent_id": "system" });
            let _ = queue.enqueue("snapshot_model_state", 3, &snap_payload.to_string(), 1);

            // 12. Predictive generation: run prediction generators.
            //     Runs every `prediction_interval` ticks.
            let tick_num = self.tick_count.fetch_add(1, Ordering::Relaxed);
            let cfg = self.config.load();
            if cfg.dynamics.prediction_enabled {
                let pred_interval = cfg.dynamics.prediction_interval;
                if pred_interval > 0 && tick_num % pred_interval == 0 {
                    let pred_payload = serde_json::json!({
                        "namespace": "default",
                        "agent_id": "system",
                        "ppm_max_depth": cfg.dynamics.ppm_max_depth,
                        "cycle": tick_num / pred_interval,
                    });
                    let _ = queue.enqueue(
                        "generate_predictions",
                        3,
                        &pred_payload.to_string(),
                        1,
                    );
                    // Also evaluate expired predictions
                    let eval_payload = serde_json::json!({ "namespace": "default" });
                    let _ = queue.enqueue(
                        "evaluate_expired_predictions",
                        3,
                        &eval_payload.to_string(),
                        1,
                    );
                }
            }

            // 13. Meta-learning: tune dynamics parameters based on free energy trend.
            //     Runs every `meta_learning_interval` ticks (not every tick).
            if cfg.dynamics.meta_learning_enabled {
                let interval = cfg.dynamics.meta_learning_interval;
                if interval > 0 && tick_num % interval == 0 {
                    let _ = queue.enqueue(
                        "meta_learning_step",
                        4,
                        &serde_json::json!({ "namespace": "default" }).to_string(),
                        1,
                    );
                }
            }

            // 14. Embedding projection training.
            //     Runs every `projection_train_interval` ticks.
            if cfg.dynamics.projection_enabled {
                let interval = cfg.dynamics.projection_train_interval;
                if interval > 0 && tick_num % interval == 0 {
                    let _ = queue.enqueue(
                        "train_projection",
                        4,
                        &serde_json::json!({ "namespace": "default" }).to_string(),
                        1,
                    );
                }
            }

            // 15. Intrinsic goal generation from surprise hotspots.
            //     Creates exploratory telos when persistent surprise patterns are detected.
            //     Gated by β and free energy — no hardcoded thresholds.
            if cfg.dynamics.intrinsic_goal_enabled {
                let interval = cfg.dynamics.intrinsic_goal_interval;
                if interval > 0 && tick_num % interval == 0 {
                    let cooldown_ms = cfg.dynamics.tick_interval_secs * 1000 * 5;
                    let payload = serde_json::json!({
                        "cooldown_ms": cooldown_ms,
                    });
                    let _ = queue.enqueue(
                        "generate_intrinsic_goals",
                        1, // lowest priority — exploration is subordinate
                        &payload.to_string(),
                        1,
                    );
                    result.intrinsic_goal_enqueued = true;
                }
            }

        }

        // 13. If meta-learning has written new params, apply them to effective config.
        //     This runs even without a task queue — the reload is pure CozoDB read + config swap.
        if let Ok(meta_params) = crate::dynamics::meta_learning::load_meta_params(&self.store) {
            if !meta_params.is_empty() {
                let mut new_cfg = self.base_config.clone();
                for (name, value) in &meta_params {
                    apply_meta_param(&mut new_cfg, name, *value);
                }
                self.config.store(Arc::new(new_cfg));
            }
        }

        // 15. If projection training has produced a newer generation, hot-reload it.
        if self.config.load().dynamics.projection_enabled {
            if let Ok(Some(proj)) = crate::dynamics::projection::load_projection(
                &self.store,
                self.store.dim(),
            ) {
                let current = self.projection.load();
                let current_count = current.as_ref().as_ref().map(|p| p.train_count).unwrap_or(0);
                if proj.train_count > current_count {
                    self.projection.store(Arc::new(Some(proj)));
                }
            }
        }

        Ok(result)
    }

    /// Start a background tick loop that runs `tick()` at the configured interval.
    ///
    /// Returns immediately. The loop runs until the cancellation token is triggered.
    pub fn start_tick_loop(
        self: &Arc<Self>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let this = Arc::clone(self);
        let interval_secs = this.config.load().dynamics.tick_interval_secs;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(interval_secs),
            );
            // Skip the first immediate tick
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = interval.tick() => {
                        if let Err(e) = this.tick() {
                            eprintln!("tick error: {e}");
                        }
                    }
                }
            }
        })
    }

    /// Get a reference to the underlying store.
    pub fn store(&self) -> &CozoStore {
        &self.store
    }

    // ── Phase 5: Event Channel ──

    /// Emit a memory event. Best-effort — never blocks.
    pub(crate) fn emit(&self, event: MemoryEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Subscribe to all memory events.
    pub fn subscribe_all(&self) -> broadcast::Receiver<MemoryEvent> {
        self.event_tx.subscribe()
    }

    // ── Phase 7: Kernel Rules ──

    /// Add a kernel rule at runtime.
    ///
    /// Uses interior mutability (RwLock) so this works through `Arc<Memoria>`.
    pub fn add_kernel_rule(&self, rule: KernelRule) {
        self.kernel_rules.write().unwrap().push(rule);
    }

    /// Evaluate kernel rules for a memory write.
    ///
    /// Returns Ok(()) if all rules pass, or Err with the first failing rule.
    pub(crate) fn evaluate_kernel_rules_for_write(
        &self,
        content: &str,
        namespace: &str,
        fields: &serde_json::Map<String, serde_json::Value>,
    ) -> crate::error::Result<()> {
        let rules = self.kernel_rules.read().unwrap();
        for rule in rules.iter() {
            match rule {
                KernelRule::Existence { field } => {
                    // Check that the required field is non-empty
                    let has_field = match field.as_str() {
                        "content" => !content.is_empty(),
                        "namespace" => !namespace.is_empty(),
                        other => fields
                            .get(other)
                            .map(|v| !v.is_null() && v.as_str().map_or(true, |s| !s.is_empty()))
                            .unwrap_or(false),
                    };
                    if !has_field {
                        return Err(crate::error::MemoriaError::KernelDenied {
                            rule: rule.to_string(),
                            message: format!("required field '{field}' is missing or empty"),
                        });
                    }
                }
                KernelRule::Deletion { .. } | KernelRule::Transition { .. } | KernelRule::Visibility { .. } => {
                    // These rules are evaluated in other contexts (delete, update, read)
                }
            }
        }
        Ok(())
    }

    /// Evaluate kernel rules for a memory deletion.
    #[allow(dead_code)]
    pub(crate) fn evaluate_kernel_rules_for_delete(
        &self,
        namespace: &str,
    ) -> crate::error::Result<()> {
        let rules = self.kernel_rules.read().unwrap();
        for rule in rules.iter() {
            if let KernelRule::Deletion { namespace_pattern } = rule {
                if crate::store::cozo::namespace_matches_pub(namespace_pattern, namespace) {
                    return Err(crate::error::MemoriaError::KernelDenied {
                        rule: rule.to_string(),
                        message: format!("deletion not allowed in namespace '{namespace}'"),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check governance rules for an operation. Returns an error if any rule blocks.
    pub(crate) async fn enforce_rules(
        &self,
        operation: &str,
        agent_id: &str,
        scope: &str,
        content: &str,
        target_id: Option<uuid::Uuid>,
    ) -> crate::error::Result<Vec<crate::rules::RuleEvaluation>> {
        let enforcer = crate::rules::RuleEnforcer::new(
            self.store.clone(),
            Arc::clone(&self.llm),
        );

        let evals = enforcer
            .check_rules(operation, agent_id, scope, content, target_id)
            .await?;

        // Check for blocks
        for eval in &evals {
            if eval.result == crate::rules::EvalResult::Block {
                return Err(crate::error::MemoriaError::RuleViolation(format!(
                    "Rule '{}' blocks this {}: {}",
                    eval.rule_id, operation, eval.reason
                )));
            }
        }

        Ok(evals)
    }

    // ── Episode Lifecycle ──

    /// Create a new episode in the store.
    pub fn create_episode(
        &self,
        agent_id: &str,
        episode_type: &str,
        properties: serde_json::Map<String, serde_json::Value>,
    ) -> crate::error::Result<crate::types::episode::Episode> {
        let mut episode = crate::types::episode::Episode::new(agent_id, episode_type);
        episode.properties = properties;
        self.store.create_episode(&episode)?;
        Ok(episode)
    }

    /// Close an episode: set ended_at, outcome, and optionally embed the summary.
    pub async fn close_episode(
        &self,
        episode_id: uuid::Uuid,
        outcome: &str,
        summary: &str,
    ) -> crate::error::Result<()> {
        let summary_embedding = if !summary.is_empty() {
            let embeddings = self.embedder.embed(&[summary]).await
                .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;
            embeddings.into_iter().next()
        } else {
            None
        };

        self.store.close_episode(
            episode_id,
            outcome,
            summary,
            summary_embedding.as_deref(),
        )?;

        Ok(())
    }

    /// Get an episode by ID.
    pub fn get_episode(
        &self,
        episode_id: uuid::Uuid,
    ) -> crate::error::Result<Option<crate::types::episode::Episode>> {
        self.store.get_episode(episode_id)
    }

    // ── Phase 4: Memory Dynamics ──

    /// Compute surprise for a new observation against existing beliefs.
    ///
    /// Finds matching facts by predicate and computes precision-weighted
    /// prediction error. Logs surprise events and returns the total.
    pub fn compute_surprise_for_facts(
        &self,
        entity_id: uuid::Uuid,
        observation: &crate::dynamics::surprise::Observation,
    ) -> crate::error::Result<crate::dynamics::surprise::SurpriseResult> {
        let facts = self.store.find_facts_by_entity(entity_id)?;

        let mut total_surprise = crate::dynamics::surprise::SurpriseResult::default();

        for fact in &facts {
            let result = crate::dynamics::surprise::compute_surprise(fact, observation);
            if result.surprise > total_surprise.surprise {
                total_surprise = result.clone();
            }
            if result.surprise > 0.0 {
                crate::dynamics::surprise::log_surprise(
                    &self.store,
                    result.surprise,
                    &observation.source,
                    Some(fact.id),
                    &observation.content,
                )?;
            }
        }

        Ok(total_surprise)
    }

    /// Run the reflection pipeline if accumulated surprise exceeds threshold.
    ///
    /// Extracts patterns from recent episodes and stores as abstractions.
    pub async fn run_reflection(
        &self,
    ) -> crate::error::Result<Option<crate::dynamics::reflection::ReflectionResult>> {
        crate::dynamics::reflection::run_reflection(
            &self.store,
            &self.llm,
            &self.embedder,
            self.config.load().consolidation_threshold,
        )
        .await
    }

    /// Reconsolidate a memory: rewrite it if newer knowledge contradicts it.
    pub async fn reconsolidate(
        &self,
        memory_id: uuid::Uuid,
    ) -> crate::error::Result<Option<crate::dynamics::reconsolidation::ReconsolidationResult>> {
        let memory = self
            .store
            .get_memory(memory_id)?
            .ok_or(crate::error::MemoriaError::NotFound(memory_id))?;

        crate::dynamics::reconsolidation::reconsolidate(
            &self.store,
            &memory,
            &self.llm,
            &self.embedder,
        )
        .await
    }

    /// Compress co-activated memory clusters into summaries.
    pub async fn compress(
        &self,
        namespace: &str,
        min_cluster_size: usize,
    ) -> crate::error::Result<Vec<crate::dynamics::compression::CompressionResult>> {
        crate::dynamics::compression::compress_clusters(
            &self.store,
            &self.llm,
            &self.embedder,
            min_cluster_size,
            namespace,
        )
        .await
    }

    /// Compute and cache PageRank + community detection for all memories.
    pub fn compute_graph_metrics(
        &self,
    ) -> crate::error::Result<Vec<crate::dynamics::graph_metrics::ImportanceResult>> {
        crate::dynamics::graph_metrics::compute_pagerank(&self.store)
    }

    /// Get effective confidence for a memory, accounting for provenance decay.
    pub fn effective_confidence(&self, memory_id: uuid::Uuid) -> crate::error::Result<f64> {
        let mem = self
            .store
            .get_memory(memory_id)?
            .ok_or(crate::error::MemoriaError::NotFound(memory_id))?;

        let now = crate::types::memory::now_ms();
        Ok(crate::dynamics::confidence::effective_confidence(
            mem.confidence,
            &mem.provenance,
            mem.created_at,
            now,
        ))
    }

    /// Get accumulated unresolved surprise.
    pub fn accumulated_surprise(&self) -> crate::error::Result<f64> {
        crate::dynamics::surprise::accumulated_unresolved_surprise(&self.store)
    }

    // ── Phase 5: Procedural Memory ──

    /// Bootstrap skills from a SKILL.md document.
    pub async fn bootstrap_skills(
        &self,
        markdown: &str,
        ctx: &crate::types::query::AgentContext,
    ) -> crate::error::Result<crate::skills::lifecycle::BootstrapResult> {
        crate::skills::lifecycle::bootstrap_from_markdown(
            &self.store,
            &self.llm,
            &self.embedder,
            markdown,
            &ctx.agent_id,
        )
        .await
    }

    /// Select skills for a task using Expected Free Energy.
    pub async fn select_skills(
        &self,
        task_description: &str,
        beta: f64,
        k: usize,
    ) -> crate::error::Result<Vec<crate::skills::selection::ScoredSkill>> {
        let gamma = self.config.load().dynamics.telos_gamma;
        crate::skills::selection::select_skills_with_gamma(
            &self.store,
            &self.embedder,
            task_description,
            beta,
            gamma,
            k,
        )
        .await
    }

    /// Store a skill from a text declaration (embeds and stores in the skills table).
    ///
    /// This is the preferred way to store skills from external sources (management API,
    /// enterprise push, user declarations) — it uses the typed Skill struct and real
    /// embeddings instead of the fragile pipe-delimited string format via tell().
    pub async fn store_skill_from_text(
        &self,
        name: &str,
        description: &str,
        triggers: &[String],
        tools: &[String],
        source: &str,
    ) -> crate::error::Result<uuid::Uuid> {
        use crate::skills::{Skill, SkillStep, SkillProvenance};

        let steps: Vec<SkillStep> = vec![SkillStep {
            step: 1,
            action: description.to_string(),
        }];

        let mut skill = Skill::new(name, description, steps);
        skill.provenance = SkillProvenance::Discovered;
        skill.confidence = 0.5;
        skill.domain = source.to_string();
        skill.tags = triggers.to_vec();
        if !tools.is_empty() {
            skill.tags.extend(tools.iter().cloned());
        }

        // Embed the skill description
        let embed_text = format!("{name}: {description}");
        let embeddings = self
            .embedder
            .embed(&[embed_text.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

        let emb = embeddings
            .first()
            .ok_or_else(|| crate::error::MemoriaError::Embedding("no embedding returned".into()))?;

        let skill_id = skill.id;
        crate::skills::storage::store_skill(&self.store, &skill, emb)?;
        Ok(skill_id)
    }

    /// List all skills in the skills table.
    pub fn list_all_skills(&self) -> crate::error::Result<Vec<crate::skills::Skill>> {
        crate::skills::storage::list_all_skills(&self.store)
    }

    /// Record a skill usage.
    pub fn record_skill_usage(
        &self,
        usage: &crate::skills::SkillUsage,
    ) -> crate::error::Result<()> {
        crate::skills::storage::store_skill_usage(&self.store, usage)
    }

    /// Run causal attribution on a failed task.
    pub async fn attribute_failure(
        &self,
        task_id: uuid::Uuid,
        task_description: &str,
    ) -> crate::error::Result<crate::causal::AttributionResult> {
        crate::causal::attribution::attribute_failure(
            &self.store,
            &self.embedder,
            task_id,
            task_description,
        )
        .await
    }

    /// Plan phase of Plan-Execute-Summarize loop.
    pub async fn pes_plan(
        &self,
        task_id: uuid::Uuid,
        task_description: &str,
        ctx: &crate::types::query::AgentContext,
        beta: f64,
    ) -> crate::error::Result<crate::pipeline::pes::PesResult> {
        crate::pipeline::pes::plan(
            &self.store,
            &self.llm,
            &self.embedder,
            task_id,
            task_description,
            ctx,
            beta,
        )
        .await
    }

    /// Summarize phase of PES loop (after task completion).
    pub async fn pes_summarize(
        &self,
        task_id: uuid::Uuid,
        plan: &str,
        outcome: &str,
        failure_reason: Option<&str>,
        ctx: &crate::types::query::AgentContext,
        duration_ms: Option<i64>,
        skills_used: &[uuid::Uuid],
    ) -> crate::error::Result<crate::skills::storage::TaskOutcome> {
        crate::pipeline::pes::summarize(
            &self.store,
            &self.llm,
            &self.embedder,
            task_id,
            plan,
            outcome,
            failure_reason,
            ctx,
            duration_ms,
            skills_used,
        )
        .await
    }

    /// Crystallize skills from recurring successful patterns.
    pub async fn crystallize_skills(
        &self,
    ) -> crate::error::Result<crate::skills::lifecycle::CrystallizeResult> {
        crate::skills::lifecycle::crystallize_skills(&self.store, &self.llm, &self.embedder).await
    }

    /// Specialize a general skill to a target domain.
    pub async fn specialize_skill(
        &self,
        general_skill_id: uuid::Uuid,
        target_domain: &str,
        context_examples: &[&str],
    ) -> crate::error::Result<crate::skills::lifecycle::SpecializeResult> {
        crate::skills::lifecycle::specialize_skill(
            &self.store,
            &self.llm,
            &self.embedder,
            general_skill_id,
            target_domain,
            context_examples,
        )
        .await
    }

    /// Get a skill by ID.
    pub fn get_skill(
        &self,
        skill_id: uuid::Uuid,
    ) -> crate::error::Result<Option<crate::skills::Skill>> {
        crate::skills::storage::get_skill(&self.store, skill_id)
    }

    /// Count all skills in the store.
    pub fn count_skills(&self) -> crate::error::Result<usize> {
        crate::skills::storage::count_skills(&self.store)
    }

    /// Compare two skill versions for A/B testing.
    ///
    /// Returns performance stats for each version and whether the difference
    /// is statistically significant (chi-squared test, p < 0.05).
    pub fn compare_skill_versions(
        &self,
        skill_id: uuid::Uuid,
        version_a: i64,
        version_b: i64,
    ) -> crate::error::Result<crate::skills::storage::VersionComparison> {
        crate::skills::storage::compare_skill_versions(
            &self.store,
            skill_id,
            version_a,
            version_b,
        )
    }

    // ── Phase 6: Active Inference ──

    /// Get the overall health of the Memoria model (convenience wrapper with no queue info).
    pub fn health(&self) -> crate::error::Result<crate::aif::health::ModelHealth> {
        self.model_health(0, 0)
    }

    /// Get the overall health of the Memoria model.
    pub fn model_health(
        &self,
        queue_pending: usize,
        queue_dead: usize,
    ) -> crate::error::Result<crate::aif::health::ModelHealth> {
        crate::aif::health::compute_health(&self.store, queue_pending, queue_dead)
    }

    /// Take a snapshot of the current model state (free energy, β, etc.).
    pub fn snapshot_model_state(
        &self,
        agent_id: &str,
    ) -> crate::error::Result<crate::aif::free_energy::FreeEnergyState> {
        crate::aif::model_state::snapshot_model_state(&self.store, agent_id)
    }

    /// Get the auto-tuned β parameter from the latest model state snapshot.
    /// Returns 1.0 (maximum exploration) if no snapshots exist.
    pub fn auto_beta(&self) -> crate::error::Result<f64> {
        crate::aif::model_state::get_latest_beta(&self.store)
    }

    /// Select skills for a task using the auto-tuned β parameter.
    pub async fn select_skills_auto(
        &self,
        task_description: &str,
        k: usize,
    ) -> crate::error::Result<Vec<crate::skills::selection::ScoredSkill>> {
        let beta = self.auto_beta()?;
        self.select_skills(task_description, beta, k).await
    }

    /// Get the effective consolidation threshold, adapted by β when `adaptive_beta` is enabled.
    ///
    /// When β is high (model is uncertain), the threshold drops — triggering
    /// more aggressive consolidation. When β is low (confident), the base
    /// threshold is used as-is.
    ///
    /// Formula: `effective = base / (1 + β)`
    pub fn effective_consolidation_threshold(&self) -> crate::error::Result<f64> {
        let cfg = self.config.load();
        let base = cfg.consolidation_threshold;
        if !cfg.dynamics.adaptive_beta {
            return Ok(base);
        }
        let beta = self.auto_beta()?;
        Ok(base / (1.0 + beta))
    }

    /// Get behavioral diversity metrics across all skill niches.
    ///
    /// Returns coverage (number of niches), fitness spread, and usage uniformity.
    pub fn diversity_metrics(
        &self,
    ) -> crate::error::Result<crate::skills::niche::DiversityMetrics> {
        crate::skills::niche::behavioral_diversity_metrics(&self.store)
    }

    /// Pre-task priming: returns EFE-ranked skills, model state, and warnings.
    ///
    /// This is the `prime()` API from §15.13. Before executing a task, call this
    /// to get the agent's best skills, current model health, and any high-surprise
    /// items needing attention.
    pub async fn prime(
        &self,
        task_description: &str,
        ctx: &crate::types::query::AgentContext,
        k: usize,
    ) -> crate::error::Result<PrimeResult> {
        let beta = self.auto_beta()?;

        // 1. EFE-ranked skills — use niche-aware selection if hint is provided
        let skills = if let Some(ref niche) = ctx.niche_hint {
            crate::skills::selection::select_skills_for_niche(
                &self.store,
                &self.embedder,
                task_description,
                niche,
                beta,
                k,
            ).await?
        } else {
            self.select_skills(task_description, beta, k).await?
        };

        // 2. Predictive graph traversal — forward inference on knowledge graph
        //    to pre-activate connected entities before standard retrieval
        let prediction = crate::api::predict::predict_relevant_memories(
            &self.store,
            task_description,
            &ctx.namespace,
            2, // 2-hop traversal
            k * 2,
        )?;

        // 3. Standard retrieval (ask) — combine with predictions
        let mut enriched_ctx = ctx.clone();
        // Add predicted memory IDs as context for Hebbian boosting
        for candidate in &prediction.predicted_memories {
            if !enriched_ctx.context_memory_ids.contains(&candidate.memory.id) {
                enriched_ctx.context_memory_ids.push(candidate.memory.id);
            }
        }
        let memories = self.ask(task_description, &enriched_ctx).await?;

        // 4. Model state snapshot
        let state = crate::aif::free_energy::compute_bethe_free_energy(&self.store)?;

        // 5. Unresolved surprise as warnings
        let unresolved_surprise =
            crate::dynamics::surprise::accumulated_unresolved_surprise(&self.store)?;

        // 6. Sequence mining — predict next tasks from historical patterns
        let predicted_next_tasks = crate::causal::sequence_mining::predict_next_task(
            &self.store,
            task_description,
        )
        .unwrap_or_default()
        .into_iter()
        .filter(|(_, conf)| *conf > 0.3)
        .collect();

        // 7. Active telos — goals ranked by attention score for context injection
        let active_telos = self.active_telos(&ctx.namespace, k)
            .unwrap_or_default();

        // 8. Active predictions — what the system expects to happen
        let now = crate::types::memory::now_ms();
        let active_predictions = crate::dynamics::prediction::get_pending_predictions(
            &self.store, &ctx.namespace, now,
        ).unwrap_or_default();

        // 9. Prediction accuracy — rolling average
        let prediction_accuracy = crate::dynamics::prediction::prediction_accuracy(
            &self.store, &ctx.namespace,
        ).unwrap_or(0.5);

        // 10. Regime stability — check if BOCPD detected recent changepoint
        let regime_stable = crate::dynamics::prediction::detect_regime_changes(
            &self.store, &ctx.namespace,
        ).map(|(_, stable)| stable).unwrap_or(true);

        Ok(PrimeResult {
            skills,
            memories,
            beta,
            free_energy: state.free_energy,
            unresolved_surprise,
            predicted_next_tasks,
            active_telos,
            active_predictions,
            prediction_accuracy,
            regime_stable,
        })
    }

    /// Post-task feedback: feeds outcome back into the factor graph.
    ///
    /// This is the `feedback()` API from §15.13. After task completion:
    /// 1. Records skill usage outcome
    /// 2. Runs causal attribution if task failed (uses task_id to find recall context)
    /// 3. Takes a model state snapshot
    /// 4. Triggers consolidation if accumulated surprise exceeds adaptive threshold
    /// 5. Audit log entry for full traceability
    pub async fn feedback(
        &self,
        task_id: uuid::Uuid,
        outcome: &crate::skills::SkillOutcome,
        agent_id: &str,
        skills_used: &[uuid::Uuid],
    ) -> crate::error::Result<FeedbackResult> {
        let mut surprise_triggered = false;
        let mut consolidation_triggered = false;

        // 1. Record skill usages — keyed to the real task_id
        for &skill_id in skills_used {
            let usage = crate::skills::SkillUsage::new(
                skill_id,
                1,
                agent_id,
                outcome.clone(),
            );
            self.record_skill_usage(&usage)?;
        }

        // 2. On failure, run causal attribution
        //    Prefer async queue if available, fall back to direct call
        let mut attribution_penalty = 0.0;
        if matches!(outcome, crate::skills::SkillOutcome::Failure) {
            if let Some(ref queue) = self.task_queue {
                // Async: enqueue causal attribution for background processing
                let payload = serde_json::json!({
                    "task_id": task_id.to_string(),
                    "agent_id": agent_id,
                });
                let _ = queue.enqueue("causal_attribution", 1, &payload.to_string(), 2);
            } else {
                // Fallback: direct call if no queue
                if let Ok(result) = self.attribute_failure(task_id, &format!("Task {} failed", task_id)).await {
                    attribution_penalty = result.attributions.iter()
                        .map(|a| a.causal_impact.abs())
                        .sum::<f64>();
                    let _ = crate::causal::attribution::apply_attribution(&self.store, &result, 0.1);
                }
            }
        }

        // 2b. On failure, trigger reconsolidation for memories with reduced confidence
        if matches!(outcome, crate::skills::SkillOutcome::Failure) {
            if let Some(ref queue) = self.task_queue {
                // Find recall context for this task to get memories that were used
                if let Ok(recall_contexts) = crate::skills::storage::find_recall_contexts_for_task(
                    &self.store,
                    task_id,
                ) {
                    for ctx_mem_id in recall_contexts {
                        let payload = serde_json::json!({ "memory_id": ctx_mem_id.to_string() });
                        let _ = queue.enqueue("reconsolidate", 1, &payload.to_string(), 2);
                    }
                }
            }
        }

        // 2c. Update agent trust profile
        {
            let is_success = matches!(outcome, crate::skills::SkillOutcome::Success);
            // Best-effort trust update — don't fail the feedback call
            let _ = self.store.update_agent_trust(agent_id, is_success, attribution_penalty);
        }

        // 2d. On success with no skills used, discover a new skill from the episode
        if matches!(outcome, crate::skills::SkillOutcome::Success) && skills_used.is_empty() {
            if let Ok(Some(episode)) = self.store.find_latest_episode_for_agent(agent_id) {
                if !episode.summary.is_empty() {
                    // Discover skill from episode summary
                    if let Ok(Some(skill_id)) = crate::skills::lifecycle::discover_skill(
                        &self.store,
                        &self.llm,
                        &self.embedder,
                        episode.id,
                        &episode.summary,
                    ).await {
                        // Classify into a behavioral niche
                        let _ = crate::skills::niche::classify_niche(
                            &self.store,
                            &self.llm,
                            skill_id,
                        ).await;

                        self.emit(crate::types::event::MemoryEvent::SkillDiscovered { skill_id });
                    }
                }
            }
        }

        // 3. Snapshot model state — enqueue async if queue exists, but always
        //    compute inline too (needed for the immediate return value).
        //    The enqueue ensures a historical snapshot is persisted even if
        //    the inline computation here has slightly stale data.
        if let Some(ref queue) = self.task_queue {
            let payload = serde_json::json!({ "agent_id": agent_id });
            let _ = queue.enqueue("snapshot_model_state", 3, &payload.to_string(), 1);
        }
        let state = self.snapshot_model_state(agent_id)?;

        // 4. Check if surprise exceeds adaptive threshold
        let unresolved =
            crate::dynamics::surprise::accumulated_unresolved_surprise(&self.store)?;
        let threshold = self.effective_consolidation_threshold()?;

        if unresolved > threshold {
            surprise_triggered = true;

            // Emit surprise event
            self.emit(crate::types::event::MemoryEvent::SurpriseThresholdExceeded {
                surprise: unresolved,
                threshold,
            });

            // Trigger consolidation via reflection
            if let Some(ref reflection) = self.run_reflection().await? {
                consolidation_triggered = true;
                self.emit(crate::types::event::MemoryEvent::ConsolidationTriggered {
                    episodes_reviewed: reflection.episodes_reviewed.len(),
                    abstractions_created: reflection.abstractions_created,
                });
            }

            // After reflection, also schedule compression and GC
            if let Some(ref queue) = self.task_queue {
                let payload = serde_json::json!({
                    "namespace": "default",
                    "min_cluster_size": self.config.load().dynamics.min_cluster_size,
                });
                let _ = queue.enqueue("compress_memories", 2, &payload.to_string(), 2);
                let _ = queue.enqueue("scratchpad_gc", 3, "{}", 1);
            }
        }

        // 5. Hash-chained audit entry — record the task_id for full traceability
        let _ = self.store.insert_audit_entry(
            &crate::types::audit::AuditEntry {
                operation: "feedback".to_string(),
                agent_id: agent_id.to_string(),
                namespace: String::new(),
                details: serde_json::json!({
                    "task_id": task_id.to_string(),
                    "outcome": outcome.to_string(),
                    "skills_used": skills_used.len(),
                    "surprise_triggered": surprise_triggered,
                    "consolidation_triggered": consolidation_triggered,
                }),
            },
        );

        // 6. Trigger projection training if enabled — the worker will skip
        //    if there aren't enough triplets yet.
        if self.config.load().dynamics.projection_enabled {
            if let Some(ref queue) = self.task_queue {
                let _ = queue.enqueue("train_projection", 4, "{}", 1);
            }
        }

        Ok(FeedbackResult {
            free_energy: state.free_energy,
            beta: state.beta,
            unresolved_surprise: unresolved,
            surprise_triggered,
            consolidation_triggered,
        })
    }

    /// Propagate a confidence change through the source chain.
    ///
    /// When a memory's confidence changes (e.g., after reconsolidation),
    /// call this to cascade the change to all derived memories and facts.
    pub fn propagate_confidence(
        &self,
        source_id: uuid::Uuid,
        new_confidence: f64,
    ) -> crate::error::Result<crate::aif::propagation::PropagationResult> {
        crate::aif::propagation::propagate_confidence(&self.store, source_id, new_confidence)
    }
}

/// Apply a single meta-learned parameter value to a config struct.
fn apply_meta_param(cfg: &mut MemoriaConfig, name: &str, value: f64) {
    match name {
        "consolidation_threshold" => cfg.consolidation_threshold = value,
        "compression_memory_threshold" => {
            cfg.dynamics.compression_memory_threshold = value as usize;
        }
        "telos_gamma" => cfg.dynamics.telos_gamma = value,
        "activation_tau" => cfg.activation_tau = value,
        "min_cluster_size" => cfg.dynamics.min_cluster_size = value as usize,
        "max_distance" => cfg.max_distance = value,
        "promotion_threshold" => cfg.dynamics.promotion_threshold = value as i64,
        _ => {} // unknown param — ignore
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::query::AgentContext;

    #[tokio::test]
    async fn test_tell_stores_memory() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        let result = m.tell("Alice works at Acme Corp", &ctx).await.unwrap();
        assert_eq!(result.memory_ids.len(), 1);

        // Verify memory was stored
        let mem = m.store.get_memory(result.memory_ids[0]).unwrap().unwrap();
        assert_eq!(mem.content, "Alice works at Acme Corp");
        assert_eq!(mem.namespace, "default");
    }

    #[tokio::test]
    async fn test_tell_empty_text_returns_empty() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        let result = m.tell("", &ctx).await.unwrap();
        assert!(result.memory_ids.is_empty());
    }

    #[tokio::test]
    async fn test_tell_long_text_creates_multiple_chunks() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        // Build text longer than the default 1500 char chunk size
        let mut long_text = String::new();
        for i in 0..100 {
            long_text.push_str(&format!(
                "This is sentence number {} which contains enough words to be meaningful. ",
                i
            ));
        }

        let result = m.tell(&long_text, &ctx).await.unwrap();
        assert!(
            result.memory_ids.len() > 1,
            "expected multiple memories for long text, got {}",
            result.memory_ids.len()
        );

        // Verify all memories were stored
        for id in &result.memory_ids {
            let mem = m.store.get_memory(*id).unwrap();
            assert!(mem.is_some(), "memory {} should exist", id);
        }
    }

    #[tokio::test]
    async fn test_ask_retrieves_stored_memories() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        // Store some facts
        m.tell("Alice works at Acme Corp", &ctx).await.unwrap();
        m.tell("Bob manages the engineering team", &ctx).await.unwrap();
        m.tell("Acme Corp builds rockets", &ctx).await.unwrap();

        // Ask a question
        let result = m.ask("Where does Alice work?", &ctx).await.unwrap();

        // Should return results (with mock embedder, similarity is pseudo-random
        // but vector search should find stored memories)
        assert!(
            !result.results.is_empty(),
            "should retrieve at least one result"
        );
    }

    #[tokio::test]
    async fn test_ask_empty_store_returns_empty() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        let result = m.ask("Where does Alice work?", &ctx).await.unwrap();
        assert!(result.results.is_empty());
    }

    #[tokio::test]
    async fn test_tell_records_access_events() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        let result = m.tell("Test content", &ctx).await.unwrap();
        let memory_id = result.memory_ids[0];

        // Compute activations — should find the access we just recorded
        let activations = m.store.compute_activations(
            &[memory_id],
            86_400_000.0,
            crate::types::memory::now_ms(),
        ).unwrap();

        assert!(!activations.is_empty(), "should have activation from store access");
    }

    #[tokio::test]
    async fn test_ask_strengthens_hebbian_associations() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        // Store memories
        let r1 = m.tell("Alice works at Acme Corp", &ctx).await.unwrap();
        let r2 = m.tell("Bob works at Acme Corp too", &ctx).await.unwrap();

        // Ask — both memories might be co-retrieved, which should strengthen associations
        let ask_result = m.ask("Who works at Acme?", &ctx).await.unwrap();

        if ask_result.results.len() >= 2 {
            // If both were retrieved, check that associations were strengthened
            let id_a = r1.memory_ids[0];
            let id_b = r2.memory_ids[0];
            let (a, b) = if id_a < id_b { (id_a, id_b) } else { (id_b, id_a) };

            let weights = m.store.get_association_weights(&[a], &[b]).unwrap();
            // Associations may or may not exist depending on whether both memories
            // were in the result set — this is a best-effort check
            if !weights.is_empty() {
                assert!(weights[0].1 > 0.0, "hebbian weight should be positive");
            }
        }
    }

    #[tokio::test]
    async fn test_tell_and_ask_round_trip() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        // Tell 10 facts
        let facts = [
            "Alice is a software engineer",
            "Bob manages the data team",
            "Charlie works on machine learning",
            "Diana is the CTO of Acme Corp",
            "Eve handles security operations",
            "Frank builds frontend components",
            "Grace reviews pull requests",
            "Heidi writes documentation",
            "Ivan maintains the CI pipeline",
            "Judy manages product roadmap",
        ];

        for fact in &facts {
            m.tell(fact, &ctx).await.unwrap();
        }

        // Verify all stored
        let count = m.store.count_memories().unwrap();
        assert_eq!(count, 10, "all 10 memories should be stored");

        // Ask and verify we get ranked results
        let result = m.ask("Who works on engineering?", &ctx).await.unwrap();
        assert!(!result.results.is_empty(), "should find relevant memories");

        // Results should have scores
        for r in &result.results {
            assert!(r.score.is_finite(), "scores should be finite");
        }
    }

    #[tokio::test]
    async fn test_namespace_isolation() {
        let m = Memoria::with_mocks(128).unwrap();

        let ctx_a = AgentContext::new("agent-a", "project-alpha");
        let ctx_b = AgentContext::new("agent-b", "project-beta");

        m.tell("Secret alpha data", &ctx_a).await.unwrap();
        m.tell("Secret beta data", &ctx_b).await.unwrap();

        // Ask in namespace alpha — should only find alpha data
        let result = m.ask("secret data", &ctx_a).await.unwrap();
        for r in &result.results {
            assert_eq!(
                r.memory.namespace, "project-alpha",
                "should only return memories from the correct namespace"
            );
        }
    }

    // ── Phase 4: Dynamics Integration Tests ──

    #[test]
    fn test_accumulated_surprise_starts_at_zero() {
        let m = Memoria::with_mocks(128).unwrap();
        let surprise = m.accumulated_surprise().unwrap();
        assert_eq!(surprise, 0.0);
    }

    #[test]
    fn test_effective_confidence_for_memory() {
        let m = Memoria::with_mocks(128).unwrap();

        // Store a memory via direct insert (sync)
        let mem = crate::types::memory::Memory::new(
            "test",
            "hello",
            vec![0.0; 128],
        );
        let id = mem.id;
        m.store.insert_memory(&mem).unwrap();

        // Just-created memory should have ~1.0 confidence
        let conf = m.effective_confidence(id).unwrap();
        assert!(conf > 0.99, "just-created memory should have full confidence, got {conf}");
    }

    #[test]
    fn test_compute_graph_metrics_empty() {
        let m = Memoria::with_mocks(128).unwrap();
        let results = m.compute_graph_metrics().unwrap();
        assert!(results.is_empty(), "no edges → no PageRank results");
    }

    #[tokio::test]
    async fn test_reflection_below_threshold() {
        let m = Memoria::with_mocks(128).unwrap();
        let result = m.run_reflection().await.unwrap();
        assert!(result.is_none(), "no surprise → no reflection");
    }

    #[test]
    fn test_surprise_for_unknown_entity_returns_default() {
        let m = Memoria::with_mocks(128).unwrap();
        let obs = crate::dynamics::surprise::Observation {
            content: "Alice is now a manager".into(),
            predicate: Some("role".into()),
            object_value: Some("manager".into()),
            confidence: 1.0,
            provenance: "direct".into(),
            source: "test".into(),
        };

        // No facts exist for this entity → no surprise
        let result = m.compute_surprise_for_facts(uuid::Uuid::now_v7(), &obs).unwrap();
        assert_eq!(result.surprise, 0.0);
    }

    // ── Wire Integration Tests ──

    #[tokio::test]
    async fn test_hash_chained_audit_round_trip() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("test-agent", "default");

        // tell → ask → feedback should each create an audit entry
        m.tell("Alice works at Acme", &ctx).await.unwrap();
        m.ask("Where does Alice work?", &ctx).await.unwrap();
        m.feedback(
            uuid::Uuid::now_v7(),
            &crate::skills::SkillOutcome::Success,
            "test-agent",
            &[],
        )
        .await
        .unwrap();

        // Verify the audit chain integrity
        let verification = m.store.verify_audit_chain(0).unwrap();
        assert_eq!(
            verification.integrity,
            crate::types::audit::Integrity::Valid,
            "audit chain should be valid after tell+ask+feedback"
        );
        // At minimum tell + feedback produce audit entries (ask may return early if no candidates)
        assert!(
            verification.entries_checked >= 2,
            "should have at least 2 audit entries, got {}",
            verification.entries_checked
        );
    }

    #[test]
    fn test_kernel_rules_block_empty_content() {
        let m = Memoria::with_mocks(128).unwrap();
        m.add_kernel_rule(crate::types::kernel::KernelRule::Existence {
            field: "content".to_string(),
        });

        // evaluate_kernel_rules_for_write should reject empty content
        let result = m.evaluate_kernel_rules_for_write("", "default", &serde_json::Map::new());
        assert!(result.is_err(), "empty content should be rejected by Existence rule");
    }

    #[test]
    fn test_tick_runs_without_error() {
        let m = Memoria::with_mocks(128).unwrap();
        let result = m.tick().unwrap();
        // With no data, nothing should be enqueued
        assert!(!result.compression_enqueued);
        assert!(!result.reflection_enqueued);
    }

    #[test]
    fn test_agent_trust_default() {
        let m = Memoria::with_mocks(128).unwrap();
        let trust = m.store.get_agent_trust("unknown-agent").unwrap();
        assert_eq!(trust.trust_score, 1.0, "default trust should be 1.0");
        assert_eq!(trust.success_count, 0);
        assert_eq!(trust.failure_count, 0);
    }

    #[test]
    fn test_agent_trust_update() {
        let m = Memoria::with_mocks(128).unwrap();

        // Record a success
        let trust = m.store.update_agent_trust("agent-a", true, 0.0).unwrap();
        assert!(trust.trust_score > 0.0);
        assert_eq!(trust.success_count, 1);
        assert_eq!(trust.failure_count, 0);

        // Record a failure
        let trust = m.store.update_agent_trust("agent-a", false, 0.5).unwrap();
        assert_eq!(trust.success_count, 1);
        assert_eq!(trust.failure_count, 1);
        assert!(trust.trust_score < 1.0, "trust should decrease after failure");
    }

    #[tokio::test]
    async fn test_feedback_updates_trust() {
        let m = Memoria::with_mocks(128).unwrap();

        m.feedback(
            uuid::Uuid::now_v7(),
            &crate::skills::SkillOutcome::Success,
            "trust-agent",
            &[],
        )
        .await
        .unwrap();

        let trust = m.store.get_agent_trust("trust-agent").unwrap();
        assert_eq!(trust.success_count, 1, "should record success");
    }

    #[test]
    fn test_niche_hint_in_context() {
        // Verify that niche_hint can be set on AgentContext
        let ctx = AgentContext::new("agent", "ns")
            .with_niche_hint("coding");
        assert_eq!(ctx.niche_hint.as_deref(), Some("coding"));
    }
}
