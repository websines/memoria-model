use std::collections::HashMap;

use crate::api::query_planner;
use crate::error::Result;
use crate::pipeline::hebbian;
use crate::runtime::Memoria;
use crate::skills::storage::{self, RecallContext};
use crate::types::memory::{now_ms, ScoredMemory};
use crate::types::query::{AgentContext, AskResult};

impl Memoria {
    /// Retrieve knowledge — agent asks a question, gets ranked results.
    ///
    /// Pipeline:
    /// 1. Query planner selects recall strategy(ies)
    /// 2. Execute strategy → broad candidates
    /// 3. Enrich candidates with activation + Hebbian weights + precision
    /// 4. Factor message fusion → scored & ranked results
    /// 5. Reranker pass (cross-encoder refinement)
    /// 6. Record access events for future activation
    /// 7. Strengthen Hebbian associations for co-retrieved memories
    /// 8. Record recall context + audit log for full auditability
    ///
    /// Returns AskResult with ranked results, contradictions, and strategy used.
    pub async fn ask(&self, text: &str, ctx: &AgentContext) -> Result<AskResult> {
        // 0. Update agent last_seen (best-effort)
        let _ = self.store.update_agent_last_seen(&ctx.agent_id);

        // 0a. Scope enforcement — check Read permission
        self.check_scope(ctx, &crate::types::scope::Permission::Read)?;

        // 0b. Governance check — enforce rules before retrieval
        self.enforce_rules("ask", &ctx.agent_id, &ctx.namespace, text, None)
            .await?;

        let limit = ctx.limit.unwrap_or(10);

        // 1. Query planner selects strategy
        let strategy = query_planner::plan_query(self, text, ctx).await?;

        // 2. Execute strategy → broad candidates
        let mut candidates = query_planner::execute_strategy(
            self,
            &strategy,
            ctx,
            self.config.load().max_candidates,
        )?;

        if candidates.is_empty() {
            return Ok(AskResult {
                results: Vec::new(),
                contradictions: Vec::new(),
                strategy_used: strategy,
            });
        }

        // 3a. Compute activations from access log
        let candidate_ids: Vec<_> = candidates.iter().map(|c| c.memory.id).collect();
        let activations = self.store.compute_activations(
            &candidate_ids,
            self.config.load().activation_tau,
            now_ms(),
        )?;

        let activation_map: HashMap<_, _> = activations.into_iter().collect();
        for candidate in &mut candidates {
            candidate.activation = activation_map.get(&candidate.memory.id).copied();
        }

        // 3b. Get Hebbian association weights (if context provided)
        if !ctx.context_memory_ids.is_empty() {
            let weights = self.store.get_association_weights(
                &ctx.context_memory_ids,
                &candidate_ids,
            )?;

            let weight_map: HashMap<_, _> = weights.into_iter().collect();
            for candidate in &mut candidates {
                candidate.hebbian_weight = weight_map.get(&candidate.memory.id).copied();
            }
        }

        // 3c. Apply provenance decay with weakest-source semantics + compute belief precision
        let now = now_ms();
        for candidate in &mut candidates {
            // For memories with provenance chains, apply weakest-source semantics
            let eff_conf = if !candidate.memory.source_ids.is_empty() {
                // Resolve chain and use weakest-source
                let chain = crate::dynamics::confidence::resolve_provenance_chain(
                    &self.store,
                    candidate.memory.id,
                )
                .unwrap_or_default();
                let chain_confs: Vec<f64> = chain.iter().map(|(_, c)| *c).collect();
                crate::dynamics::confidence::effective_confidence_with_chain(
                    candidate.memory.confidence,
                    &candidate.memory.provenance,
                    candidate.memory.created_at,
                    now,
                    &chain_confs,
                )
            } else {
                crate::dynamics::confidence::effective_confidence(
                    candidate.memory.confidence,
                    &candidate.memory.provenance,
                    candidate.memory.created_at,
                    now,
                )
            };
            // Update the candidate's confidence to the decayed value
            // so downstream scoring uses query-time confidence
            candidate.memory.confidence = eff_conf;

            let version = candidate.memory.version as i64;
            candidate.precision =
                Some(crate::aif::belief_update::precision(eff_conf, version));
        }

        // 3d. Hierarchical scoring — boost candidates with chunk_parent edges
        //     Uses bidirectional message passing: bottom-up evidence × top-down prediction
        {
            // Check if any candidates have hierarchical chunk kinds
            let sentence_scores: Vec<(uuid::Uuid, f64)> = candidates
                .iter()
                .filter(|c| c.memory.kind.starts_with("chunk."))
                .map(|c| (c.memory.id, 1.0 - c.distance.min(1.0)))
                .collect();

            if !sentence_scores.is_empty() {
                // Bottom-up: sentence matches propagate to parents
                if let Ok(bu) = crate::aif::hierarchical::compute_bottom_up(
                    &self.store,
                    &sentence_scores,
                ) {
                    // Top-down: find document-level nodes and propagate down
                    let doc_scores: Vec<(uuid::Uuid, f64)> = candidates
                        .iter()
                        .filter(|c| c.memory.kind == "chunk.document")
                        .map(|c| (c.memory.id, 1.0 - c.distance.min(1.0)))
                        .collect();

                    let td = if !doc_scores.is_empty() {
                        crate::aif::hierarchical::compute_top_down(&self.store, &doc_scores)
                            .unwrap_or_default()
                    } else {
                        std::collections::BTreeMap::new()
                    };

                    let combined = crate::aif::hierarchical::combine_hierarchical_scores(&bu, &td);
                    let score_map: std::collections::HashMap<uuid::Uuid, f64> = combined
                        .into_iter()
                        .map(|s| (s.memory_id, s.combined))
                        .collect();

                    // Apply hierarchical boost to candidate distances
                    for candidate in &mut candidates {
                        if let Some(&hier_score) = score_map.get(&candidate.memory.id) {
                            // Reduce distance by hierarchical score (better = lower distance)
                            candidate.distance *= 1.0 - (hier_score * 0.3);
                        }
                    }
                }
            }
        }

        // 3e. Cross-namespace trust: discount memories from other namespaces
        //     by the source agent's trust score
        for candidate in &mut candidates {
            if candidate.memory.namespace != ctx.namespace {
                // Cross-namespace memory — apply agent trust multiplier
                if let Ok(trust) = self.store.get_agent_trust(&ctx.agent_id) {
                    candidate.memory.confidence *= trust.trust_score;
                }
            }
        }

        // 3f. Telos alignment boost — up-weight memories relevant to active goals
        {
            let telos_boosts =
                self.store.compute_telos_boost(&candidate_ids, &ctx.namespace)?;
            if !telos_boosts.is_empty() {
                let boost_map: HashMap<_, _> = telos_boosts.into_iter().collect();
                for candidate in &mut candidates {
                    candidate.telos_boost = boost_map.get(&candidate.memory.id).copied();
                }
            }
        }

        // 3g. PageRank enrichment — populate cached graph importance scores
        {
            let pageranks = self.store.get_cached_pagerank(&candidate_ids)?;
            if !pageranks.is_empty() {
                for candidate in &mut candidates {
                    candidate.pagerank = pageranks.get(&candidate.memory.id).copied();
                }
            }
        }

        // 4. Factor message fusion → scored results
        let scored = self.scorer.fuse(&candidates);

        // 5. Reranker pass (cross-encoder)
        let to_rerank: Vec<&str> = scored
            .iter()
            .take(limit * 2) // rerank 2x the desired limit
            .map(|s| s.memory.content.as_str())
            .collect();

        let reranked = self
            .reranker
            .rerank(text, &to_rerank, limit)
            .await
            .map_err(|e| crate::error::MemoriaError::Reranker(e.to_string()))?;

        // Reorder scored results by reranker output
        let results: Vec<ScoredMemory> = reranked
            .iter()
            .filter_map(|r| scored.get(r.index).cloned())
            .collect();

        // 6. Record access events
        let result_ids: Vec<_> = results.iter().map(|r| r.memory.id).collect();
        self.store
            .record_accesses(&result_ids, &ctx.agent_id, "recall")?;

        // 7. Strengthen Hebbian associations for co-retrieved memories
        if result_ids.len() >= 2 {
            hebbian::strengthen_associations(
                &self.store,
                &result_ids,
                hebbian::DEFAULT_LEARNING_RATE,
            )?;
        }

        // 7b. Precision-weighted reconsolidation gate
        //     For each result, run belief_update. If Kalman gain > 0.3,
        //     the observation is strong enough to warrant reconsolidation.
        for r in &results {
            let obs_prov_weight = crate::dynamics::surprise::provenance_weight(&r.memory.provenance);
            let bu = crate::aif::belief_update::belief_update(
                r.memory.confidence,
                r.memory.version as i64,
                r.memory.confidence, // self-observation: query-time confidence
                obs_prov_weight,
                0.3, // reconsolidation threshold
            );
            if bu.should_reconsolidate {
                if let Some(ref queue) = self.task_queue {
                    let payload = serde_json::json!({ "memory_id": r.memory.id.to_string() });
                    let _ = queue.enqueue("reconsolidate", 1, &payload.to_string(), 2);
                }
            }
        }

        // 8. Detect contradictions among retrieved memories' entities
        let entity_ids = self.store.collect_entity_ids_for_memories(&result_ids)?;
        let contradictions = if entity_ids.is_empty() {
            Vec::new()
        } else {
            self.store.find_contradictions(&entity_ids)?
        };

        // 8b. Trigger reconsolidation for memories involved in contradictions
        if !contradictions.is_empty() {
            // Collect unique memory IDs linked to contradicted entities
            let mut reconsolidate_ids = std::collections::HashSet::new();
            for c in &contradictions {
                // Find memories linked to the contradicted entity
                for r in &results {
                    if let Ok(entity_ids_for_mem) =
                        self.store.collect_entity_ids_for_memories(&[r.memory.id])
                    {
                        if entity_ids_for_mem.contains(&c.entity) {
                            reconsolidate_ids.insert(r.memory.id);
                        }
                    }
                }
            }

            for mem_id in reconsolidate_ids {
                // Emit event
                self.emit(crate::types::event::MemoryEvent::ReconsolidationNeeded {
                    memory_id: mem_id,
                    contradiction_count: contradictions.len(),
                });

                if let Some(ref queue) = self.task_queue {
                    let payload = serde_json::json!({ "memory_id": mem_id.to_string() });
                    let _ = queue.enqueue("reconsolidate", 1, &payload.to_string(), 2);
                }
            }
        }

        // 9. Record recall context for audit trail
        //    Use real task_id from context (for attribution loop); fall back to episode, then fresh ID
        let task_id = ctx.task_id
            .or(ctx.current_episode)
            .unwrap_or_else(uuid::Uuid::now_v7);

        // Collect all relevant fact IDs: contradictions + facts linked to retrieved memories' entities
        let mut all_fact_ids: Vec<uuid::Uuid> = contradictions.iter().map(|c| c.fact_a).collect();
        all_fact_ids.extend(contradictions.iter().map(|c| c.fact_b));
        // Also include facts from entity_ids for richer attribution context
        for &eid in &entity_ids {
            if let Ok(facts) = self.store.find_facts_by_entity(eid) {
                all_fact_ids.extend(facts.iter().map(|f| f.id));
            }
        }
        all_fact_ids.sort();
        all_fact_ids.dedup();

        let recall_ctx = RecallContext {
            task_id,
            recall_id: uuid::Uuid::now_v7(),
            memory_ids: result_ids.clone(),
            fact_ids: all_fact_ids,
            query_text: text.to_string(),
            ts: now_ms(),
        };
        // Best-effort audit — don't fail the query if audit write fails
        let _ = storage::store_recall_context(&self.store, &recall_ctx);

        // 10. Hash-chained audit entry
        let _ = self.store.insert_audit_entry(
            &crate::types::audit::AuditEntry {
                operation: "ask".to_string(),
                agent_id: ctx.agent_id.clone(),
                namespace: ctx.namespace.clone(),
                details: serde_json::json!({
                    "query": text,
                    "results_count": results.len(),
                    "strategy": format!("{:?}", strategy),
                }),
            },
        );

        Ok(AskResult {
            results,
            contradictions,
            strategy_used: strategy,
        })
    }
}
