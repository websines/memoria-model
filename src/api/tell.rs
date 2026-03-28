use uuid::Uuid;

use crate::error::Result;
use crate::runtime::Memoria;
use crate::services::traits::{ExtractedEntity, TextInput};
use crate::types::entity::Entity;
use crate::types::fact::Fact;
use crate::types::query::{AgentContext, TellResult};

impl Memoria {
    /// Resolve a relation head/tail name to an entity, with fuzzy fallback.
    ///
    /// NER often extracts entity text like `"company"` but relation heads like
    /// `"The company"`. Exact `find_entity_by_name` fails in that case.
    /// We fall back to case-insensitive containment against extracted entities.
    fn resolve_entity_for_relation(
        &self,
        rel_name: &str,
        namespace: &str,
        extracted_entities: &[ExtractedEntity],
    ) -> Result<Option<Entity>> {
        // Exact match first
        if let Some(entity) = self.store.find_entity_by_name(rel_name, namespace)? {
            return Ok(Some(entity));
        }
        // Fuzzy: check if any extracted entity text is contained in rel_name or vice versa
        let name_lower = rel_name.to_lowercase();
        let mut best: Option<&str> = None;
        let mut best_len = 0;
        for e in extracted_entities {
            let e_lower = e.text.to_lowercase();
            if (name_lower.contains(&e_lower) || e_lower.contains(&name_lower))
                && e.text.len() > best_len
            {
                best = Some(&e.text);
                best_len = e.text.len();
            }
        }
        match best {
            Some(name) => self.store.find_entity_by_name(name, namespace),
            None => Ok(None),
        }
    }

    /// Store knowledge — agent says what happened, Memoria figures out the rest.
    ///
    /// Pipeline:
    /// 1. Chunk text (if long) using semantic boundaries
    /// 2. Embed all chunks in one batch call
    /// 3. Store each chunk as a memory in CozoDB
    /// 4. Link sibling chunks via edges (if split occurred)
    /// 5. Extract entities via NER service
    /// 6. Store entities + link to source memories
    /// 7. Link memories to current episode (if any)
    /// 8. Record access events for activation tracking
    ///
    /// Returns TellResult with memory IDs, entity IDs, and surprise value.
    pub async fn tell(&self, text: &str, ctx: &AgentContext) -> Result<TellResult> {
        // 0. Update agent last_seen (best-effort, don't fail if agent not registered)
        let _ = self.store.update_agent_last_seen(&ctx.agent_id);

        // 0a. Scope enforcement — check Write permission
        self.check_scope(ctx, &crate::types::scope::Permission::Write)?;

        // 0b. Governance check — enforce rules before storing
        self.enforce_rules("tell", &ctx.agent_id, &ctx.namespace, text, None)
            .await?;

        // 0c. Kernel rules — validate required fields before storing
        self.evaluate_kernel_rules_for_write(text, &ctx.namespace, &serde_json::Map::new())?;

        // 1. Chunk text — hierarchical or flat depending on config
        let memory_ids = if self.config.load().dynamics.use_hierarchical_chunking {
            // Hierarchical chunking: sentence → paragraph → section → document
            let chunks = crate::pipeline::chunker_hierarchical::build_hierarchy(text);
            if chunks.is_empty() {
                return Ok(TellResult {
                    memory_ids: Vec::new(),
                    entity_ids: Vec::new(),
                    fact_ids: Vec::new(),
                    surprise: 0.0,
                });
            }

            // 2. Embed all hierarchical chunks in one batch
            let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
            let embeddings = self.embedder.embed(&texts).await.map_err(|e| {
                crate::error::MemoriaError::Embedding(e.to_string())
            })?;

            // 3. Store with parent-child edges
            let stored = crate::pipeline::chunker_hierarchical::store_hierarchical_chunks(
                &self.store,
                &chunks,
                &embeddings,
                &ctx.namespace,
            )?;

            stored.into_iter().map(|(_, id)| id).collect::<Vec<_>>()
        } else {
            // Flat semantic chunking (original path)
            let chunk_result = self.chunker.chunk(text);
            if chunk_result.chunks.is_empty() {
                return Ok(TellResult {
                    memory_ids: Vec::new(),
                    entity_ids: Vec::new(),
                    fact_ids: Vec::new(),
                    surprise: 0.0,
                });
            }

            // 2. Embed all chunks in one batch
            let texts: Vec<&str> = chunk_result.chunks.iter().map(|c| c.as_str()).collect();
            let embeddings = self.embedder.embed(&texts).await.map_err(|e| {
                crate::error::MemoriaError::Embedding(e.to_string())
            })?;

            // 3. Store each chunk as a memory
            let ids = self.store.store_memories(
                &chunk_result.chunks,
                &embeddings,
                &ctx.namespace,
                "observation",
            )?;

            // 4. Link sibling chunks if text was split
            if chunk_result.was_split && ids.len() > 1 {
                for i in 0..ids.len() - 1 {
                    self.store.insert_edge(
                        ids[i],
                        ids[i + 1],
                        "chunk_sibling",
                        1.0,
                    )?;
                }
            }

            ids
        };

        // 5. Extract entities via NER
        let cfg = self.config.load();
        let entity_labels: Vec<&str> = cfg.entity_labels.iter().map(|s| s.as_str()).collect();
        let relation_labels: Vec<&str> = cfg.relation_labels.iter().map(|s| s.as_str()).collect();
        let extraction = self
            .ner
            .extract(
                &[TextInput {
                    text: text.to_string(),
                    id: None,
                }],
                &entity_labels,
                &relation_labels,
            )
            .await
            .map_err(|e| crate::error::MemoriaError::Ner(e.to_string()))?;

        // 6. Store entities + link to memories (namespace-scoped)
        let mut entity_ids = Vec::new();
        if let Some(result) = extraction.first() {
            for extracted in &result.entities {
                let entity_id = self
                    .store_or_update_entity(extracted, &memory_ids, &ctx.namespace)
                    .await?;
                entity_ids.push(entity_id);
            }
        }

        // 7. Verify relations and store as facts
        let mut fact_ids = Vec::new();
        let mut verified_relations = Vec::new();
        let mut total_surprise = 0.0f64;
        if let Some(result) = extraction.first() {
            if !result.relations.is_empty() {
                if self.task_queue.is_some() {
                    // Async path: store as pending facts, enqueue verification
                    for rel in &result.relations {
                        // Resolve head entity (fuzzy match against NER entities)
                        let head_entity =
                            self.resolve_entity_for_relation(&rel.head, &ctx.namespace, &result.entities)?;
                        let Some(head) = head_entity else { continue };
                        let tail_entity =
                            self.resolve_entity_for_relation(&rel.tail, &ctx.namespace, &result.entities)?;

                        // Check for existing matching fact → reinforce instead of duplicate
                        let existing_fact = self.store.find_matching_fact(
                            head.id,
                            &rel.label,
                            tail_entity.as_ref().map(|t| t.id),
                            if tail_entity.is_none() { Some(&rel.tail) } else { None },
                            &ctx.namespace,
                        )?;

                        if let Some(existing) = existing_fact {
                            // Reinforce existing fact
                            self.store.reinforce_fact(
                                &existing,
                                rel.confidence,
                                &memory_ids,
                            )?;
                            fact_ids.push(existing.id);
                        } else {
                            // Store as pending fact with halved confidence
                            let mut fact = if let Some(ref tail) = tail_entity {
                                let mut f = Fact::with_entity(head.id, &rel.label, tail.id);
                                f.object_value = Some(rel.tail.clone());
                                f
                            } else {
                                Fact::with_value(head.id, &rel.label, &rel.tail)
                            };
                            fact.namespace = ctx.namespace.clone();
                            fact.confidence = rel.confidence * 0.5; // Halved until verified
                            fact.temporal_status = "pending_verification".to_string();
                            fact.source_ids = memory_ids.clone();

                            self.store.insert_fact(&fact)?;
                            fact_ids.push(fact.id);
                        }
                    }

                    // Enqueue verification task
                    let payload = serde_json::json!({
                        "text": text,
                        "relations": result.relations,
                        "fact_ids": fact_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                        "namespace": ctx.namespace,
                    });
                    let _ = self.task_queue.as_ref().unwrap().enqueue(
                        "verify_relations",
                        0,
                        &payload.to_string(),
                        3,
                    );

                    // Compute surprise from pending facts (async path).
                    // verified_relations is empty in the async path, so compute
                    // surprise directly from extracted relations here.
                    for rel in &result.relations {
                        let observation = crate::dynamics::surprise::Observation {
                            content: format!("{} {} {}", rel.head, rel.label, rel.tail),
                            predicate: Some(rel.label.clone()),
                            object_value: Some(rel.tail.clone()),
                            confidence: rel.confidence * 0.5, // halved like pending facts
                            provenance: "extracted".to_string(),
                            source: ctx.agent_id.clone(),
                        };
                        if let Some(head) = self.resolve_entity_for_relation(
                            &rel.head,
                            &ctx.namespace,
                            &result.entities,
                        )? {
                            if let Ok(sr) =
                                self.compute_surprise_for_facts(head.id, &observation)
                            {
                                total_surprise = total_surprise.max(sr.surprise);
                            }
                        }
                    }
                } else {
                    // Synchronous fallback: refine + verify inline
                    verified_relations = self
                        .verifier
                        .refine(text, &result.relations, &result.entities)
                        .await?;

                    for vr in &verified_relations {
                        if !vr.verified {
                            continue;
                        }

                        // Resolve head entity to UUID (fuzzy match against NER entities)
                        let head_entity =
                            self.resolve_entity_for_relation(&vr.head, &ctx.namespace, &result.entities)?;
                        let Some(head) = head_entity else { continue };

                        // Try to resolve tail as entity; otherwise store as value
                        let tail_entity =
                            self.resolve_entity_for_relation(&vr.tail, &ctx.namespace, &result.entities)?;

                        // Use the original NER label as the canonical predicate
                        // for fact storage and surprise comparison. NER labels are
                        // deterministic; LLM-refined labels may vary between calls.
                        let predicate = if vr.original_label.is_empty() {
                            &vr.label
                        } else {
                            &vr.original_label
                        };

                        // Check for existing matching fact → reinforce instead of duplicate
                        let existing_fact = self.store.find_matching_fact(
                            head.id,
                            predicate,
                            tail_entity.as_ref().map(|t| t.id),
                            if tail_entity.is_none() { Some(&vr.tail) } else { None },
                            &ctx.namespace,
                        )?;

                        if let Some(existing) = existing_fact {
                            // Reinforce existing fact
                            self.store.reinforce_fact(
                                &existing,
                                vr.confidence,
                                &memory_ids,
                            )?;
                            fact_ids.push(existing.id);
                        } else {
                            // Create new fact
                            let mut fact = if let Some(tail) = tail_entity {
                                let mut f = Fact::with_entity(head.id, predicate, tail.id);
                                // Also store tail name as object_value for surprise comparison.
                                // Entity-only facts with object_value=None are invisible to
                                // contradiction detection.
                                f.object_value = Some(vr.tail.clone());
                                f
                            } else {
                                Fact::with_value(head.id, predicate, &vr.tail)
                            };

                            fact.namespace = ctx.namespace.clone();
                            fact.confidence = vr.confidence;
                            fact.temporal_status = vr.temporal_status.as_str().to_string();
                            fact.source_ids = memory_ids.clone();

                            self.store.insert_fact(&fact)?;
                            fact_ids.push(fact.id);
                        }
                    }
                }
            }
        }

        // 7b. Detect goals from natural language and create telos
        if let Some(detected) = crate::api::telos_detect::detect_goal(text) {
            let provenance = if detected.confidence >= 0.8 {
                crate::types::telos::TelosProvenance::UserStated
            } else {
                crate::types::telos::TelosProvenance::Inferred
            };
            let _ = self
                .create_telos(
                    &detected.title,
                    text,
                    ctx,
                    detected.depth,
                    None,
                    detected.deadline,
                    provenance,
                )
                .await;
        }

        // 7c. Match observation against pending predictions (prediction error → surprise)
        if self.config.load().dynamics.prediction_enabled {
            if let Ok(pred_match) = crate::dynamics::prediction::match_observation_against_predictions(
                &self.store,
                text,
                &[], // no embedding for full text — sequence predictions use content matching
                *memory_ids.first().unwrap_or(&uuid::Uuid::nil()),
                &ctx.namespace,
            ) {
                total_surprise += pred_match.surprise_delta;
            }
        }

        // 8. Compute surprise for new/updated facts against existing beliefs
        //    (sync path only — async path already computed surprise above)
        for vr in verified_relations.iter().filter(|vr| vr.verified) {
                // Use original NER label for consistent predicate matching
                let pred = if vr.original_label.is_empty() {
                    &vr.label
                } else {
                    &vr.original_label
                };
                // Build an observation from the verified relation
                let observation = crate::dynamics::surprise::Observation {
                    content: format!("{} {} {}", vr.head, pred, vr.tail),
                    predicate: Some(pred.clone()),
                    object_value: Some(vr.tail.clone()),
                    confidence: vr.confidence,
                    provenance: "extracted".to_string(),
                    source: ctx.agent_id.clone(),
                };

                // Find the head entity and compute surprise against its existing facts
                let entities = extraction.first().map(|r| r.entities.as_slice()).unwrap_or(&[]);
                if let Ok(Some(head_entity)) =
                    self.resolve_entity_for_relation(&vr.head, &ctx.namespace, entities)
                {
                    if let Ok(surprise_result) =
                        self.compute_surprise_for_facts(head_entity.id, &observation)
                    {
                        total_surprise = total_surprise.max(surprise_result.surprise);
                    }
                }
        }

        // 9. Link memories to current episode
        if let Some(episode_id) = ctx.current_episode {
            self.store.link_to_episode(episode_id, &memory_ids)?;
        }

        // 10. Record access events
        self.store
            .record_accesses(&memory_ids, &ctx.agent_id, "store")?;

        // 11. Hash-chained audit entry
        let _ = self.store.insert_audit_entry(
            &crate::types::audit::AuditEntry {
                operation: "tell".to_string(),
                agent_id: ctx.agent_id.clone(),
                namespace: ctx.namespace.clone(),
                details: serde_json::json!({
                    "memories_stored": memory_ids.len(),
                    "entities_extracted": entity_ids.len(),
                    "facts_created": fact_ids.len(),
                    "surprise": total_surprise,
                }),
            },
        );

        // 12. Auto-trigger compression if namespace has grown large
        if let Some(ref queue) = self.task_queue {
            if let Ok(count) = self.store.count_memories_in_namespace(&ctx.namespace) {
                let cfg = self.config.load();
                let threshold = cfg.dynamics.compression_threshold();
                if count > threshold {
                    let payload = serde_json::json!({
                        "namespace": ctx.namespace,
                        "min_cluster_size": cfg.dynamics.min_cluster_size,
                    });
                    let _ = queue.enqueue("compress_memories", 2, &payload.to_string(), 2);
                }
            }
        }

        // 13. Emit MemoryCreated event
        self.emit(crate::types::event::MemoryEvent::Created {
            memory_ids: memory_ids.clone(),
            namespace: ctx.namespace.clone(),
            agent_id: ctx.agent_id.clone(),
        });

        Ok(TellResult {
            memory_ids,
            entity_ids,
            fact_ids,
            surprise: total_surprise,
        })
    }

    /// Store a new entity or update an existing one.
    ///
    /// For existing entities: increment mention_count, update confidence (running average),
    /// merge source memory IDs, and link to new memories.
    /// For new entities: create with embedding, link to source memories.
    async fn store_or_update_entity(
        &self,
        extracted: &ExtractedEntity,
        memory_ids: &[Uuid],
        namespace: &str,
    ) -> Result<Uuid> {
        // Check if entity already exists by name within namespace
        if let Some(existing) = self.store.find_entity_by_name(&extracted.text, namespace)? {
            // Update entity: increment mention_count, update confidence, merge sources
            self.store.update_entity_on_mention(
                &existing,
                extracted.confidence,
                memory_ids,
            )?;

            // Link existing entity to new memories
            for &mem_id in memory_ids {
                self.store.link_entity_to_memory(
                    existing.id,
                    mem_id,
                    "mentioned",
                    extracted.confidence,
                )?;
            }
            return Ok(existing.id);
        }

        // Create new entity with embedding
        let embedding = self
            .embedder
            .embed(&[extracted.text.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

        let entity = Entity {
            id: Uuid::now_v7(),
            name: extracted.text.clone(),
            entity_type: extracted.label.clone(),
            namespace: namespace.to_string(),
            embedding: embedding.into_iter().next().unwrap_or_default(),
            properties: serde_json::Map::new(),
            mention_count: 1,
            confidence: extracted.confidence,
            provenance: "extracted".to_string(),
            source_ids: memory_ids.to_vec(),
        };

        let entity_id = entity.id;
        self.store.insert_entity(&entity)?;

        // Link entity to source memories
        for &mem_id in memory_ids {
            self.store.link_entity_to_memory(
                entity_id,
                mem_id,
                "mentioned",
                extracted.confidence,
            )?;
        }

        Ok(entity_id)
    }
}
