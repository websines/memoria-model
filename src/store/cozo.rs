use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};
use ndarray::Array1;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::store::schema;
use crate::types::entity::{Entity, EntityId};
use crate::types::fact::Fact;
use crate::types::memory::{CandidateMemory, Memory, MemoryId};
use crate::types::query::Contradiction;
use crate::types::telos::{SuccessCriterion, Telos, TelosEvent, TelosProvenance, TelosStatus};

/// The CozoDB-backed store for all Memoria data.
///
/// Wraps a `DbInstance` and provides typed methods for memory operations.
/// All methods are synchronous because CozoDB is embedded and fast.
#[derive(Clone)]
pub struct CozoStore {
    db: Arc<DbInstance>,
    dim: usize,
}

impl CozoStore {
    /// Open or create a CozoDB with RocksDB backend.
    ///
    /// If the schema doesn't exist yet, it will be bootstrapped.
    pub fn open(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let db = DbInstance::new("rocksdb", &path_str, Default::default())
            .map_err(|e| MemoriaError::SchemaBootstrap(format!("opening db: {e}")))?;

        if !schema::schema_exists(&db) {
            schema::bootstrap_schema(&db, dim)?;
        }

        Ok(Self {
            db: Arc::new(db),
            dim,
        })
    }

    /// Create an in-memory CozoDB (for testing).
    pub fn open_mem(dim: usize) -> Result<Self> {
        let db = DbInstance::new("mem", "", Default::default())
            .map_err(|e| MemoriaError::SchemaBootstrap(format!("opening mem db: {e}")))?;

        schema::bootstrap_schema(&db, dim)?;

        Ok(Self {
            db: Arc::new(db),
            dim,
        })
    }

    /// Get a reference to the underlying CozoDB instance.
    pub fn db(&self) -> &DbInstance {
        &self.db
    }

    /// Get the embedding dimension this store was created with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Run a CozoScript query with parameters (mutable).
    pub fn run_script(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows> {
        self.db
            .run_script(script, params, ScriptMutability::Mutable)
            .map_err(MemoriaError::from)
    }

    /// Run an immutable (read-only) CozoScript query.
    pub fn run_query(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows> {
        self.db
            .run_script(script, params, ScriptMutability::Immutable)
            .map_err(MemoriaError::from)
    }

    /// Insert a memory into the store.
    pub fn insert_memory(&self, memory: &Memory) -> Result<()> {
        let embedding_vals: Vec<DataValue> = memory
            .embedding
            .iter()
            .map(|&v| DataValue::from(v as f64))
            .collect();

        let source_id_vals: Vec<DataValue> = memory
            .source_ids
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(memory.id.to_string()));
        params.insert("kind".into(), DataValue::from(memory.kind.as_str()));
        params.insert("content".into(), DataValue::from(memory.content.as_str()));
        params.insert("embedding".into(), DataValue::List(embedding_vals));
        params.insert(
            "fields".into(),
            DataValue::from(serde_json::to_string(&memory.fields)?),
        );
        params.insert(
            "namespace".into(),
            DataValue::from(memory.namespace.as_str()),
        );
        params.insert("pinned".into(), DataValue::from(memory.pinned));
        params.insert("version".into(), DataValue::from(memory.version as i64));
        params.insert("created_at".into(), DataValue::from(memory.created_at));
        params.insert("confidence".into(), DataValue::from(memory.confidence));
        params.insert(
            "provenance".into(),
            DataValue::from(memory.provenance.as_str()),
        );
        params.insert("source_ids".into(), DataValue::List(source_id_vals));

        if let Some(exp) = memory.expires_at {
            params.insert("expires_at".into(), DataValue::from(exp));
        } else {
            params.insert("expires_at".into(), DataValue::Null);
        }

        let script = concat!(
            "?[id, valid_at, kind, content, embedding, fields, namespace, pinned, ",
            "expires_at, version, created_at, confidence, provenance, source_ids] <- ",
            "[[$id, 'ASSERT', $kind, $content, $embedding, $fields, $namespace, $pinned, ",
            "$expires_at, $version, $created_at, $confidence, $provenance, $source_ids]] ",
            ":put memories {id, valid_at => kind, content, embedding, fields, namespace, pinned, ",
            "expires_at, version, created_at, confidence, provenance, source_ids}",
        );
        self.run_script(script, params)?;

        Ok(())
    }

    /// Get a memory by ID (latest version).
    pub fn get_memory(&self, id: MemoryId) -> Result<Option<Memory>> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(id.to_string()));

        let result = self.run_query(
            r#"?[id, kind, content, embedding, fields, namespace, pinned,
                expires_at, version, created_at, confidence, provenance, source_ids] :=
                *memories{id, kind, content, embedding, fields, namespace, pinned,
                          expires_at, version, created_at, confidence, provenance,
                          source_ids}, id = to_uuid($id)"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        Ok(Some(parse_memory_row(row)?))
    }

    /// Count total memories in the store.
    pub fn count_memories(&self) -> Result<usize> {
        let result = self.run_query(
            "?[count(id)] := *memories{id}",
            BTreeMap::new(),
        )?;

        if result.rows.is_empty() {
            return Ok(0);
        }

        Ok(result.rows[0][0].get_int().unwrap_or(0) as usize)
    }

    // ── Phase 2: tell() / ask() support ──

    /// Batch insert memories from chunks + embeddings. Returns the assigned IDs.
    pub fn store_memories(
        &self,
        chunks: &[String],
        embeddings: &[Vec<f32>],
        namespace: &str,
        kind: &str,
    ) -> Result<Vec<MemoryId>> {
        assert_eq!(chunks.len(), embeddings.len());

        let mut ids = Vec::with_capacity(chunks.len());
        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let mut memory = Memory::new(kind, chunk.as_str(), embedding.clone());
            memory.namespace = namespace.to_string();
            let id = memory.id;
            self.insert_memory(&memory)?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Insert an entity into the entities relation.
    pub fn insert_entity(&self, entity: &Entity) -> Result<()> {
        let embedding_vals: Vec<DataValue> = entity
            .embedding
            .iter()
            .map(|&v| DataValue::from(v as f64))
            .collect();

        let source_id_vals: Vec<DataValue> = entity
            .source_ids
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(entity.id.to_string()));
        params.insert("name".into(), DataValue::from(entity.name.as_str()));
        params.insert(
            "entity_type".into(),
            DataValue::from(entity.entity_type.as_str()),
        );
        params.insert(
            "namespace".into(),
            DataValue::from(entity.namespace.as_str()),
        );
        params.insert("embedding".into(), DataValue::List(embedding_vals));
        params.insert(
            "properties".into(),
            DataValue::from(serde_json::to_string(&entity.properties)?),
        );
        params.insert(
            "mention_count".into(),
            DataValue::from(entity.mention_count),
        );
        params.insert("confidence".into(), DataValue::from(entity.confidence));
        params.insert(
            "provenance".into(),
            DataValue::from(entity.provenance.as_str()),
        );
        params.insert("source_ids".into(), DataValue::List(source_id_vals));

        self.run_script(
            concat!(
                "?[id, valid_at, name, entity_type, namespace, embedding, properties, ",
                "mention_count, confidence, provenance, source_ids] <- ",
                "[[$id, 'ASSERT', $name, $entity_type, $namespace, $embedding, $properties, ",
                "$mention_count, $confidence, $provenance, $source_ids]] ",
                ":put entities {id, valid_at => name, entity_type, namespace, embedding, ",
                "properties, mention_count, confidence, provenance, source_ids}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Find an entity by name within a namespace (exact match, latest version).
    pub fn find_entity_by_name(&self, name: &str, namespace: &str) -> Result<Option<Entity>> {
        let mut params = BTreeMap::new();
        params.insert("name".into(), DataValue::from(name));
        params.insert("namespace".into(), DataValue::from(namespace));

        let result = self.run_query(
            r#"?[id, name, entity_type, namespace, embedding, properties,
                mention_count, confidence, provenance, source_ids] :=
                *entities{id, name, entity_type, namespace, embedding, properties,
                          mention_count, confidence, provenance, source_ids},
                name = $name, namespace = $namespace"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        Ok(Some(parse_entity_row(row)?))
    }

    /// Update an existing entity on re-mention: increment mention_count,
    /// update confidence (running average), and append new source memory IDs.
    pub fn update_entity_on_mention(
        &self,
        existing: &Entity,
        new_confidence: f64,
        new_source_ids: &[Uuid],
    ) -> Result<()> {
        let new_mention_count = existing.mention_count + 1;
        // Running average: (old * count + new) / (count + 1)
        let updated_confidence = (existing.confidence * existing.mention_count as f64
            + new_confidence)
            / new_mention_count as f64;

        // Merge source_ids
        let mut merged_sources = existing.source_ids.clone();
        for &id in new_source_ids {
            if !merged_sources.contains(&id) {
                merged_sources.push(id);
            }
        }

        let embedding_vals: Vec<DataValue> = existing
            .embedding
            .iter()
            .map(|&v| DataValue::from(v as f64))
            .collect();
        let source_id_vals: Vec<DataValue> = merged_sources
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(existing.id.to_string()));
        params.insert("name".into(), DataValue::from(existing.name.as_str()));
        params.insert(
            "entity_type".into(),
            DataValue::from(existing.entity_type.as_str()),
        );
        params.insert(
            "namespace".into(),
            DataValue::from(existing.namespace.as_str()),
        );
        params.insert("embedding".into(), DataValue::List(embedding_vals));
        params.insert(
            "properties".into(),
            DataValue::from(serde_json::to_string(&existing.properties)?),
        );
        params.insert(
            "mention_count".into(),
            DataValue::from(new_mention_count),
        );
        params.insert("confidence".into(), DataValue::from(updated_confidence));
        params.insert(
            "provenance".into(),
            DataValue::from(existing.provenance.as_str()),
        );
        params.insert("source_ids".into(), DataValue::List(source_id_vals));

        self.run_script(
            concat!(
                "?[id, valid_at, name, entity_type, namespace, embedding, properties, ",
                "mention_count, confidence, provenance, source_ids] <- ",
                "[[$id, 'ASSERT', $name, $entity_type, $namespace, $embedding, $properties, ",
                "$mention_count, $confidence, $provenance, $source_ids]] ",
                ":put entities {id, valid_at => name, entity_type, namespace, embedding, ",
                "properties, mention_count, confidence, provenance, source_ids}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Link an entity to a memory via entity_mentions.
    pub fn link_entity_to_memory(
        &self,
        entity_id: EntityId,
        memory_id: MemoryId,
        role: &str,
        confidence: f64,
    ) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert(
            "entity_id".into(),
            DataValue::from(entity_id.to_string()),
        );
        params.insert(
            "memory_id".into(),
            DataValue::from(memory_id.to_string()),
        );
        params.insert("role".into(), DataValue::from(role));
        params.insert("confidence".into(), DataValue::from(confidence));

        self.run_script(
            concat!(
                "?[entity_id, memory_id, valid_at, role, confidence] <- ",
                "[[$entity_id, $memory_id, 'ASSERT', $role, $confidence]] ",
                ":put entity_mentions {entity_id, memory_id, valid_at => role, confidence}",
            ),
            params,
        )?;

        Ok(())
    }

    /// HNSW vector search on memories. Returns candidates with distance.
    pub fn vector_search(
        &self,
        query_embedding: &[f32],
        namespace: &str,
        max_k: usize,
        max_dist: f64,
    ) -> Result<Vec<CandidateMemory>> {
        // HNSW requires DataValue::Vec (ndarray), not DataValue::List
        let arr = Array1::from_vec(query_embedding.to_vec());
        let query_vec = DataValue::Vec(Vector::F32(arr));

        let mut params = BTreeMap::new();
        params.insert("q_vec".into(), query_vec);
        params.insert("k".into(), DataValue::from(max_k as i64));
        params.insert("max_dist".into(), DataValue::from(max_dist));
        params.insert("ns".into(), DataValue::from(namespace));

        // HNSW search with namespace filter, then join to get full memory data
        let result = self.run_query(
            r#"?[id, dist, kind, content, embedding, fields, namespace, pinned,
                expires_at, version, created_at, confidence, provenance, source_ids] :=
                ~memories:vec_idx{id | query: $q_vec, k: $k, ef: 50, bind_distance: dist},
                *memories{id, kind, content, embedding, fields, namespace, pinned,
                          expires_at, version, created_at, confidence, provenance, source_ids},
                namespace = $ns,
                dist < $max_dist
            :sort dist
            :limit $k"#,
            params,
        )?;

        let mut candidates = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let distance = row[1].get_float().unwrap_or(1.0);
            // Row layout: [id, dist, kind, content, embedding, fields, namespace,
            //              pinned, expires_at, version, created_at, confidence,
            //              provenance, source_ids]
            // We need to reconstruct: [id, kind, content, ...] for parse_memory_row
            let mut mem_row = Vec::with_capacity(13);
            mem_row.push(row[0].clone()); // id
            mem_row.extend_from_slice(&row[2..]); // kind..source_ids
            let memory = parse_memory_row(&mem_row)?;

            candidates.push(CandidateMemory {
                memory,
                distance,
                activation: None,
                hebbian_weight: None,
                pagerank: None,
                precision: None,
                telos_boost: None,
            });
        }

        Ok(candidates)
    }

    /// Compute activation values for a set of memories from the access log.
    ///
    /// activation = Σ weight × exp(-(now - ts) / τ)
    pub fn compute_activations(
        &self,
        memory_ids: &[Uuid],
        tau: f64,
        now: i64,
    ) -> Result<Vec<(Uuid, f64)>> {
        if memory_ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_vals: Vec<DataValue> = memory_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ids".into(), DataValue::List(id_vals));
        params.insert("tau".into(), DataValue::from(tau));
        params.insert("now".into(), DataValue::from(now));

        let result = self.run_query(
            r#"input_raw[raw_id] <- $ids
               input[uid] := input_raw[raw_id], uid = to_uuid(raw_id)
               ?[memory_id, sum(contribution)] :=
                   input[memory_id],
                   *accesses{memory_id, ts, event_type},
                   w = if(event_type == "spike", 2.0,
                       if(event_type == "feedback", 1.5, 1.0)),
                   contribution = w * exp(-($now - ts) / $tau)"#,
            params,
        )?;

        let mut activations = Vec::new();
        for row in &result.rows {
            let id = parse_uuid(&row[0])?;
            let act = row[1].get_float().unwrap_or(0.0);
            activations.push((id, act));
        }

        Ok(activations)
    }

    /// Get Hebbian association weights between context memories and candidate memories.
    ///
    /// Returns (candidate_id, weight) pairs.
    pub fn get_association_weights(
        &self,
        context_ids: &[Uuid],
        candidate_ids: &[Uuid],
    ) -> Result<Vec<(Uuid, f64)>> {
        if context_ids.is_empty() || candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let ctx_vals: Vec<DataValue> = context_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();
        let cand_vals: Vec<DataValue> = candidate_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ctx_ids".into(), DataValue::List(ctx_vals));
        params.insert("cand_ids".into(), DataValue::List(cand_vals));

        let result = self.run_query(
            r#"ctx_raw[raw_id] <- $ctx_ids
               ctx[uid] := ctx_raw[raw_id], uid = to_uuid(raw_id)
               cand_raw[raw_id] <- $cand_ids
               cand[uid] := cand_raw[raw_id], uid = to_uuid(raw_id)
               ?[b, max(weight)] :=
                   ctx[a], cand[b],
                   *associations{a, b, weight}"#,
            params,
        )?;

        let mut weights = Vec::new();
        for row in &result.rows {
            let id = parse_uuid(&row[0])?;
            let weight = row[1].get_float().unwrap_or(0.0);
            weights.push((id, weight));
        }

        Ok(weights)
    }

    /// Record access events for memories (for activation computation).
    pub fn record_accesses(
        &self,
        memory_ids: &[Uuid],
        agent_id: &str,
        event_type: &str,
    ) -> Result<()> {
        if memory_ids.is_empty() {
            return Ok(());
        }

        let now = crate::types::memory::now_ms();

        for id in memory_ids {
            let mut params = BTreeMap::new();
            params.insert("memory_id".into(), DataValue::from(id.to_string()));
            params.insert("ts".into(), DataValue::from(now));
            params.insert("event_type".into(), DataValue::from(event_type));
            params.insert("agent_id".into(), DataValue::from(agent_id));

            self.run_script(
                concat!(
                    "?[memory_id, ts, event_type, agent_id] <- ",
                    "[[$memory_id, $ts, $event_type, $agent_id]] ",
                    ":put accesses {memory_id, ts => event_type, agent_id}",
                ),
                params,
            )?;
        }

        Ok(())
    }

    /// Upsert a Hebbian association between two memories.
    ///
    /// Uses the update rule: new_weight = old_weight + lr × (1 - old_weight)
    pub fn upsert_association(
        &self,
        a: Uuid,
        b: Uuid,
        learning_rate: f64,
        now: i64,
    ) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("a".into(), DataValue::from(a.to_string()));
        params.insert("b".into(), DataValue::from(b.to_string()));
        params.insert("lr".into(), DataValue::from(learning_rate));
        params.insert("now".into(), DataValue::from(now));

        // Try read existing, compute new weight in Rust, then write.
        // CozoDB doesn't support read-modify-write in one script easily,
        // so we do it in two steps.
        let existing = self.run_query(
            r#"?[weight, co_access_count, last_access] :=
                *associations{a, b, weight, co_access_count, last_access},
                a = to_uuid($a), b = to_uuid($b)"#,
            params.clone(),
        )?;

        let (new_weight, new_count) = if existing.rows.is_empty() {
            (learning_rate, 1i64)
        } else {
            let old_weight = existing.rows[0][0].get_float().unwrap_or(0.0);
            let old_count = existing.rows[0][1].get_int().unwrap_or(0);
            let updated = old_weight + learning_rate * (1.0 - old_weight);
            (updated, old_count + 1)
        };

        params.insert("weight".into(), DataValue::from(new_weight));
        params.insert("count".into(), DataValue::from(new_count));

        self.run_script(
            concat!(
                "?[a, b, valid_at, weight, co_access_count, last_access] <- ",
                "[[$a, $b, 'ASSERT', $weight, $count, $now]] ",
                ":put associations {a, b, valid_at => weight, co_access_count, last_access}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Upsert a co-activation record between two memories.
    ///
    /// This feeds the `co_activations` table that `find_coactivation_clusters()`
    /// uses for memory compression. Called alongside `upsert_association` in
    /// Hebbian learning so that co-retrieved memories form compression clusters.
    pub fn upsert_co_activation(&self, a: Uuid, b: Uuid, now: i64) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("a".into(), DataValue::from(a.to_string()));
        params.insert("b".into(), DataValue::from(b.to_string()));
        params.insert("now".into(), DataValue::from(now));

        let existing = self.run_query(
            r#"?[count] := *co_activations{a, b, count}, a = to_uuid($a), b = to_uuid($b)"#,
            params.clone(),
        )?;

        let new_count = if existing.rows.is_empty() {
            1i64
        } else {
            existing.rows[0][0].get_int().unwrap_or(0) + 1
        };
        params.insert("count".into(), DataValue::from(new_count));

        self.run_script(
            concat!(
                "?[a, b, valid_at, count, last_seen] <- [[$a, $b, 'ASSERT', $count, $now]] ",
                ":put co_activations {a, b, valid_at => count, last_seen}",
            ),
            params,
        )?;
        Ok(())
    }

    /// Link memories to an episode.
    pub fn link_to_episode(
        &self,
        episode_id: Uuid,
        memory_ids: &[Uuid],
    ) -> Result<()> {
        for (i, memory_id) in memory_ids.iter().enumerate() {
            let mut params = BTreeMap::new();
            params.insert(
                "episode_id".into(),
                DataValue::from(episode_id.to_string()),
            );
            params.insert(
                "memory_id".into(),
                DataValue::from(memory_id.to_string()),
            );
            params.insert("position".into(), DataValue::from(i as i64));

            self.run_script(
                concat!(
                    "?[episode_id, memory_id, position, role] <- ",
                    "[[$episode_id, $memory_id, $position, 'member']] ",
                    ":put episode_memories {episode_id, memory_id => position, role}",
                ),
                params,
            )?;
        }

        Ok(())
    }

    /// Insert a knowledge graph edge between two nodes.
    pub fn insert_edge(
        &self,
        source: Uuid,
        target: Uuid,
        kind: &str,
        weight: f64,
    ) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("source".into(), DataValue::from(source.to_string()));
        params.insert("target".into(), DataValue::from(target.to_string()));
        params.insert("kind".into(), DataValue::from(kind));
        params.insert("weight".into(), DataValue::from(weight));

        self.run_script(
            concat!(
                "?[source, target, kind, valid_at, weight, fields] <- ",
                "[[$source, $target, $kind, 'ASSERT', $weight, '{}']] ",
                ":put edges {source, target, kind, valid_at => weight, fields}",
            ),
            params,
        )?;

        Ok(())
    }
    // ── Phase 3: Knowledge Graph & Verification ──

    /// Insert a fact into the facts relation.
    pub fn insert_fact(&self, fact: &Fact) -> Result<()> {
        let source_id_vals: Vec<DataValue> = fact
            .source_ids
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(fact.id.to_string()));
        params.insert(
            "subject_entity".into(),
            DataValue::from(fact.subject_entity.to_string()),
        );
        params.insert("predicate".into(), DataValue::from(fact.predicate.as_str()));

        if let Some(obj) = fact.object_entity {
            params.insert("object_entity".into(), DataValue::from(obj.to_string()));
        } else {
            params.insert("object_entity".into(), DataValue::Null);
        }
        if let Some(ref val) = fact.object_value {
            params.insert("object_value".into(), DataValue::from(val.as_str()));
        } else {
            params.insert("object_value".into(), DataValue::Null);
        }

        params.insert(
            "namespace".into(),
            DataValue::from(fact.namespace.as_str()),
        );
        params.insert(
            "temporal_status".into(),
            DataValue::from(fact.temporal_status.as_str()),
        );
        params.insert("confidence".into(), DataValue::from(fact.confidence));
        params.insert(
            "provenance".into(),
            DataValue::from(fact.provenance.as_str()),
        );
        params.insert("source_ids".into(), DataValue::List(source_id_vals));
        params.insert(
            "reinforcement_count".into(),
            DataValue::from(fact.reinforcement_count),
        );

        self.run_script(
            concat!(
                "?[id, valid_at, subject_entity, predicate, object_entity, object_value, ",
                "namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] <- ",
                "[[$id, 'ASSERT', $subject_entity, $predicate, $object_entity, $object_value, ",
                "$namespace, $temporal_status, $confidence, $provenance, $source_ids, $reinforcement_count]] ",
                ":put facts {id, valid_at => subject_entity, predicate, object_entity, object_value, ",
                "namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Find an existing fact matching subject+predicate+object within a namespace.
    ///
    /// Used to detect when a newly extracted fact matches an existing one,
    /// enabling reinforcement instead of duplicate insertion.
    pub fn find_matching_fact(
        &self,
        subject: Uuid,
        predicate: &str,
        object_entity: Option<Uuid>,
        object_value: Option<&str>,
        namespace: &str,
    ) -> Result<Option<Fact>> {
        let mut params = BTreeMap::new();
        params.insert("subject".into(), DataValue::from(subject.to_string()));
        params.insert("predicate".into(), DataValue::from(predicate));
        params.insert("namespace".into(), DataValue::from(namespace));

        // Build the query based on whether object is entity or value
        let query = if let Some(obj_entity) = object_entity {
            params.insert("object_entity".into(), DataValue::from(obj_entity.to_string()));
            r#"?[id, subject_entity, predicate, object_entity, object_value,
                namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] :=
                *facts{id, subject_entity, predicate, object_entity, object_value,
                       namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count},
                subject_entity = to_uuid($subject),
                predicate = $predicate,
                object_entity = to_uuid($object_entity),
                namespace = $namespace"#
        } else if let Some(obj_val) = object_value {
            params.insert("object_value".into(), DataValue::from(obj_val));
            r#"?[id, subject_entity, predicate, object_entity, object_value,
                namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] :=
                *facts{id, subject_entity, predicate, object_entity, object_value,
                       namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count},
                subject_entity = to_uuid($subject),
                predicate = $predicate,
                object_value = $object_value,
                namespace = $namespace"#
        } else {
            return Ok(None);
        };

        let result = self.run_query(query, params)?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        Ok(Some(parse_fact_row(&result.rows[0])?))
    }

    /// Reinforce an existing fact: increment reinforcement_count, update confidence,
    /// and append new source memory IDs.
    pub fn reinforce_fact(
        &self,
        existing: &Fact,
        new_confidence: f64,
        new_source_ids: &[Uuid],
    ) -> Result<()> {
        let new_rc = existing.reinforcement_count + 1;
        // Running average confidence
        let updated_confidence = (existing.confidence * existing.reinforcement_count as f64
            + new_confidence)
            / new_rc as f64;

        // Merge source_ids
        let mut merged_sources = existing.source_ids.clone();
        for &id in new_source_ids {
            if !merged_sources.contains(&id) {
                merged_sources.push(id);
            }
        }

        let source_id_vals: Vec<DataValue> = merged_sources
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(existing.id.to_string()));
        params.insert(
            "subject_entity".into(),
            DataValue::from(existing.subject_entity.to_string()),
        );
        params.insert("predicate".into(), DataValue::from(existing.predicate.as_str()));

        if let Some(obj) = existing.object_entity {
            params.insert("object_entity".into(), DataValue::from(obj.to_string()));
        } else {
            params.insert("object_entity".into(), DataValue::Null);
        }
        if let Some(ref val) = existing.object_value {
            params.insert("object_value".into(), DataValue::from(val.as_str()));
        } else {
            params.insert("object_value".into(), DataValue::Null);
        }

        params.insert(
            "namespace".into(),
            DataValue::from(existing.namespace.as_str()),
        );
        params.insert(
            "temporal_status".into(),
            DataValue::from(existing.temporal_status.as_str()),
        );
        params.insert("confidence".into(), DataValue::from(updated_confidence));
        params.insert(
            "provenance".into(),
            DataValue::from(existing.provenance.as_str()),
        );
        params.insert("source_ids".into(), DataValue::List(source_id_vals));
        params.insert("reinforcement_count".into(), DataValue::from(new_rc));

        self.run_script(
            concat!(
                "?[id, valid_at, subject_entity, predicate, object_entity, object_value, ",
                "namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] <- ",
                "[[$id, 'ASSERT', $subject_entity, $predicate, $object_entity, $object_value, ",
                "$namespace, $temporal_status, $confidence, $provenance, $source_ids, $reinforcement_count]] ",
                ":put facts {id, valid_at => subject_entity, predicate, object_entity, object_value, ",
                "namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Find all facts where the given entity is the subject.
    pub fn find_facts_by_entity(&self, entity_id: Uuid) -> Result<Vec<Fact>> {
        let mut params = BTreeMap::new();
        params.insert("entity_id".into(), DataValue::from(entity_id.to_string()));

        let result = self.run_query(
            r#"?[id, subject_entity, predicate, object_entity, object_value,
                namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] :=
                *facts{id, subject_entity, predicate, object_entity, object_value,
                       namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count},
                subject_entity = to_uuid($entity_id)"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_fact_row(row)).collect()
    }

    /// Find contradictions: facts about the same entity with the same predicate
    /// but different values, where both have temporal_status = "current".
    pub fn find_contradictions(&self, entity_ids: &[Uuid]) -> Result<Vec<Contradiction>> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_vals: Vec<DataValue> = entity_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("entity_ids".into(), DataValue::List(id_vals));

        let result = self.run_query(
            r#"input_raw[raw_id] <- $entity_ids
               input[uid] := input_raw[raw_id], uid = to_uuid(raw_id)
               current_facts[fid, entity, pred, val] :=
                   input[entity],
                   *facts{id: fid, subject_entity: entity, predicate: pred,
                          object_value: val, temporal_status: ts},
                   ts == "current",
                   val != null
               ?[fact_a, fact_b, entity, predicate, value_a, value_b] :=
                   current_facts[fact_a, entity, predicate, value_a],
                   current_facts[fact_b, entity, predicate, value_b],
                   fact_a != fact_b,
                   value_a != value_b,
                   value_a < value_b"#,
            params,
        )?;

        let mut contradictions = Vec::new();
        for row in &result.rows {
            contradictions.push(Contradiction {
                fact_a: parse_uuid(&row[0])?,
                fact_b: parse_uuid(&row[1])?,
                entity: parse_uuid(&row[2])?,
                predicate: parse_string(&row[3]),
                value_a: parse_string(&row[4]),
                value_b: parse_string(&row[5]),
            });
        }

        Ok(contradictions)
    }

    /// Collect entity IDs linked to the given memories via entity_mentions.
    pub fn collect_entity_ids_for_memories(&self, memory_ids: &[Uuid]) -> Result<Vec<Uuid>> {
        if memory_ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_vals: Vec<DataValue> = memory_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("memory_ids".into(), DataValue::List(id_vals));

        let result = self.run_query(
            r#"input_raw[raw_id] <- $memory_ids
               input[uid] := input_raw[raw_id], uid = to_uuid(raw_id)
               ?[entity_id] := input[memory_id], *entity_mentions{entity_id, memory_id}"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_uuid(&row[0])).collect()
    }

    /// Retrieve memories by a list of IDs.
    pub fn get_memories_by_ids(&self, ids: &[Uuid]) -> Result<Vec<Memory>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_vals: Vec<DataValue> = ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ids".into(), DataValue::List(id_vals));

        let result = self.run_query(
            r#"input_raw[raw_id] <- $ids
               input[uid] := input_raw[raw_id], uid = to_uuid(raw_id)
               ?[id, kind, content, embedding, fields, namespace, pinned,
                 expires_at, version, created_at, confidence, provenance, source_ids] :=
                 input[id],
                 *memories{id, kind, content, embedding, fields, namespace, pinned,
                           expires_at, version, created_at, confidence, provenance, source_ids}"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_memory_row(row)).collect()
    }

    /// Insert an entry into the audit log.
    pub fn insert_audit_log(
        &self,
        operation: &str,
        agent_id: &str,
        target_id: Option<Uuid>,
        details: &serde_json::Value,
    ) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("ts".into(), DataValue::from(crate::types::memory::now_ms()));
        params.insert("operation".into(), DataValue::from(operation));
        params.insert("agent_id".into(), DataValue::from(agent_id));
        if let Some(tid) = target_id {
            params.insert("target_id".into(), DataValue::from(tid.to_string()));
        } else {
            params.insert("target_id".into(), DataValue::Null);
        }
        params.insert(
            "details".into(),
            DataValue::from(serde_json::to_string(details).unwrap_or_default()),
        );

        self.run_script(
            concat!(
                "?[id, ts, operation, agent_id, target_id, details] <- ",
                "[[rand_uuid_v7(), $ts, $operation, $agent_id, $target_id, $details]] ",
                ":put audit_log {id, ts => operation, agent_id, target_id, details}",
            ),
            params,
        )?;

        Ok(())
    }

    // ── Episode Lifecycle ──

    /// Create a new episode in the store.
    pub fn create_episode(&self, episode: &crate::types::episode::Episode) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(episode.id.to_string()));
        params.insert("agent_id".into(), DataValue::from(episode.agent_id.as_str()));
        params.insert("started_at".into(), DataValue::from(episode.started_at));
        params.insert("ended_at".into(), DataValue::Null);
        params.insert("summary".into(), DataValue::from(episode.summary.as_str()));
        params.insert("summary_embedding".into(), DataValue::Null);
        params.insert("episode_type".into(), DataValue::from(episode.episode_type.as_str()));
        params.insert("outcome".into(), DataValue::Null);
        params.insert(
            "properties".into(),
            DataValue::from(serde_json::to_string(&episode.properties).unwrap_or_default()),
        );

        self.run_script(
            concat!(
                "?[id, valid_at, agent_id, started_at, ended_at, summary, summary_embedding, ",
                "episode_type, outcome, properties] <- ",
                "[[$id, 'ASSERT', $agent_id, $started_at, $ended_at, $summary, $summary_embedding, ",
                "$episode_type, $outcome, $properties]] ",
                ":put episodes {id, valid_at => agent_id, started_at, ended_at, summary, ",
                "summary_embedding, episode_type, outcome, properties}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Close an episode: set ended_at, outcome, and optionally a summary with embedding.
    pub fn close_episode(
        &self,
        episode_id: Uuid,
        outcome: &str,
        summary: &str,
        summary_embedding: Option<&[f32]>,
    ) -> Result<()> {
        // Read existing episode to preserve fields
        let existing = self.get_episode(episode_id)?
            .ok_or(crate::error::MemoriaError::NotFound(episode_id))?;

        let now = crate::types::memory::now_ms();
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(episode_id.to_string()));
        params.insert("agent_id".into(), DataValue::from(existing.agent_id.as_str()));
        params.insert("started_at".into(), DataValue::from(existing.started_at));
        params.insert("ended_at".into(), DataValue::from(now));
        params.insert("summary".into(), DataValue::from(summary));
        params.insert("episode_type".into(), DataValue::from(existing.episode_type.as_str()));
        params.insert("outcome".into(), DataValue::from(outcome));
        params.insert(
            "properties".into(),
            DataValue::from(serde_json::to_string(&existing.properties).unwrap_or_default()),
        );

        if let Some(emb) = summary_embedding {
            let arr = ndarray::Array1::from_vec(emb.to_vec());
            params.insert(
                "summary_embedding".into(),
                DataValue::Vec(Vector::F32(arr)),
            );
        } else {
            params.insert("summary_embedding".into(), DataValue::Null);
        }

        self.run_script(
            concat!(
                "?[id, valid_at, agent_id, started_at, ended_at, summary, summary_embedding, ",
                "episode_type, outcome, properties] <- ",
                "[[$id, 'ASSERT', $agent_id, $started_at, $ended_at, $summary, $summary_embedding, ",
                "$episode_type, $outcome, $properties]] ",
                ":put episodes {id, valid_at => agent_id, started_at, ended_at, summary, ",
                "summary_embedding, episode_type, outcome, properties}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Get an episode by ID (latest version).
    pub fn get_episode(&self, episode_id: Uuid) -> Result<Option<crate::types::episode::Episode>> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(episode_id.to_string()));

        let result = self.run_query(
            r#"?[id, agent_id, started_at, ended_at, summary, summary_embedding,
                episode_type, outcome, properties] :=
                *episodes{id, agent_id, started_at, ended_at, summary, summary_embedding,
                          episode_type, outcome, properties, @ 'NOW'},
                id = to_uuid($id)"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        let id = parse_uuid(&row[0])?;
        let agent_id = parse_string(&row[1]);
        let started_at = row[2].get_int().unwrap_or(0);
        let ended_at = row[3].get_int();
        let summary = parse_string(&row[4]);
        let summary_embedding = parse_f32_vec_opt(&row[5]);
        let episode_type = parse_string(&row[6]);
        let outcome = if row[7] == DataValue::Null { None } else { Some(parse_string(&row[7])) };
        let properties: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&parse_string(&row[8])).unwrap_or_default();

        Ok(Some(crate::types::episode::Episode {
            id,
            agent_id,
            started_at,
            ended_at,
            summary,
            summary_embedding,
            episode_type,
            outcome,
            properties,
        }))
    }

    /// Find the latest closed episode for an agent (most recent by ended_at).
    pub fn find_latest_episode_for_agent(
        &self,
        agent_id: &str,
    ) -> Result<Option<crate::types::episode::Episode>> {
        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));

        let result = self.run_query(
            r#"?[id, agent_id, started_at, ended_at, summary, summary_embedding,
                episode_type, outcome, properties] :=
                *episodes{id, agent_id, started_at, ended_at, summary, summary_embedding,
                          episode_type, outcome, properties, @ 'NOW'},
                agent_id = $agent_id,
                ended_at != null
                :order -ended_at
                :limit 1"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        let id = parse_uuid(&row[0])?;
        let agent_id_val = parse_string(&row[1]);
        let started_at = row[2].get_int().unwrap_or(0);
        let ended_at = row[3].get_int();
        let summary = parse_string(&row[4]);
        let summary_embedding = parse_f32_vec_opt(&row[5]);
        let episode_type = parse_string(&row[6]);
        let outcome = if row[7] == DataValue::Null { None } else { Some(parse_string(&row[7])) };
        let properties: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&parse_string(&row[8])).unwrap_or_default();

        Ok(Some(crate::types::episode::Episode {
            id,
            agent_id: agent_id_val,
            started_at,
            ended_at,
            summary,
            summary_embedding,
            episode_type,
            outcome,
            properties,
        }))
    }

    /// Count memories in a namespace.
    pub fn count_memories_in_namespace(&self, namespace: &str) -> Result<usize> {
        let mut params = BTreeMap::new();
        params.insert("ns".into(), DataValue::from(namespace));

        let result = self.run_query(
            r#"?[count(id)] := *memories{id, namespace}, namespace = $ns"#,
            params,
        )?;

        Ok(result.rows.first()
            .and_then(|r| r[0].get_int())
            .unwrap_or(0) as usize)
    }

    /// Update agent last_seen_at timestamp.
    pub fn update_agent_last_seen(&self, agent_id: &str) -> Result<()> {
        let now = crate::types::memory::now_ms();
        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));
        params.insert("now".into(), DataValue::from(now));

        // Best-effort update — only touches last_seen_at if agent exists
        let _ = self.run_script(
            r#"existing[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at] :=
                *agent_registry{agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at},
                agent_id = $agent_id
            ?[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at] :=
                existing[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at],
                last_seen_at = $now
            :put agent_registry {
                agent_id => display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at
            }"#,
            params,
        );

        Ok(())
    }

    // ── Multi-Agent: Agent Registry ──

    /// Register a new agent.
    pub fn register_agent(
        &self,
        reg: &crate::types::agent::AgentRegistration,
    ) -> Result<()> {
        let mut params = BTreeMap::new();
        let now = crate::types::memory::now_ms();
        params.insert("agent_id".into(), DataValue::from(&*reg.agent_id));
        params.insert("display_name".into(), DataValue::from(&*reg.display_name));
        params.insert("status".into(), DataValue::from("active"));
        params.insert(
            "team_id".into(),
            DataValue::from(reg.team_id.as_deref().unwrap_or("")),
        );
        params.insert(
            "org_id".into(),
            DataValue::from(reg.org_id.as_deref().unwrap_or("")),
        );
        params.insert(
            "role".into(),
            DataValue::from(reg.role.as_deref().unwrap_or("")),
        );
        params.insert(
            "capabilities".into(),
            DataValue::from(serde_json::to_string(&reg.capabilities).unwrap_or_default()),
        );
        params.insert(
            "metadata".into(),
            DataValue::from(serde_json::to_string(&reg.metadata).unwrap_or_default()),
        );
        params.insert("registered_at".into(), DataValue::from(now));
        params.insert("last_seen_at".into(), DataValue::from(now));

        self.run_script(
            r#"?[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at] <- [[
                $agent_id, $display_name, $status, $team_id, $org_id, $role, $capabilities, $metadata, $registered_at, $last_seen_at
            ]]
            :put agent_registry {
                agent_id => display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at
            }"#,
            params,
        )?;

        // Add team membership if team_id is set
        if let Some(team_id) = &reg.team_id {
            let mut tp = BTreeMap::new();
            tp.insert("agent_id".into(), DataValue::from(&*reg.agent_id));
            tp.insert("team_id".into(), DataValue::from(team_id.as_str()));
            tp.insert("joined_at".into(), DataValue::from(now));
            self.run_script(
                r#"?[agent_id, team_id, joined_at] <- [[$agent_id, $team_id, $joined_at]]
                :put agent_team_membership { agent_id, team_id => joined_at }"#,
                tp,
            )?;
        }

        Ok(())
    }

    /// Get an agent record by ID.
    pub fn get_agent(&self, agent_id: &str) -> Result<Option<crate::types::agent::AgentRecord>> {
        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));

        let result = self.run_query(
            r#"?[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at] :=
                *agent_registry{agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at},
                agent_id == $agent_id"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        Ok(Some(crate::types::agent::AgentRecord {
            agent_id: parse_string(&row[0]),
            display_name: parse_string(&row[1]),
            status: crate::types::agent::AgentStatus::from_str(&parse_string(&row[2])),
            team_id: non_empty_opt(parse_string(&row[3])),
            org_id: non_empty_opt(parse_string(&row[4])),
            role: non_empty_opt(parse_string(&row[5])),
            capabilities: serde_json::from_str(&parse_string(&row[6])).unwrap_or_default(),
            metadata: serde_json::from_str(&parse_string(&row[7]))
                .unwrap_or(serde_json::Value::Object(Default::default())),
            registered_at: row[8].get_int().unwrap_or(0),
            last_seen_at: row[9].get_int().unwrap_or(0),
        }))
    }

    /// Update an agent's status.
    pub fn update_agent_status(
        &self,
        agent_id: &str,
        status: &crate::types::agent::AgentStatus,
    ) -> Result<()> {
        // Read-modify-write: get current record, update status
        let agent = self
            .get_agent(agent_id)?
            .ok_or_else(|| MemoriaError::Store(format!("agent not found: {agent_id}")))?;

        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));
        params.insert("display_name".into(), DataValue::from(&*agent.display_name));
        params.insert("status".into(), DataValue::from(status.to_string().as_str()));
        params.insert(
            "team_id".into(),
            DataValue::from(agent.team_id.as_deref().unwrap_or("")),
        );
        params.insert(
            "org_id".into(),
            DataValue::from(agent.org_id.as_deref().unwrap_or("")),
        );
        params.insert(
            "role".into(),
            DataValue::from(agent.role.as_deref().unwrap_or("")),
        );
        params.insert(
            "capabilities".into(),
            DataValue::from(serde_json::to_string(&agent.capabilities).unwrap_or_default()),
        );
        params.insert(
            "metadata".into(),
            DataValue::from(serde_json::to_string(&agent.metadata).unwrap_or_default()),
        );
        params.insert("registered_at".into(), DataValue::from(agent.registered_at));
        params.insert(
            "last_seen_at".into(),
            DataValue::from(crate::types::memory::now_ms()),
        );

        self.run_script(
            r#"?[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at] <- [[
                $agent_id, $display_name, $status, $team_id, $org_id, $role, $capabilities, $metadata, $registered_at, $last_seen_at
            ]]
            :put agent_registry {
                agent_id => display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at
            }"#,
            params,
        )?;

        Ok(())
    }

    /// Query agents with optional filters.
    pub fn query_agents(
        &self,
        filter: &crate::types::agent::AgentFilter,
    ) -> Result<Vec<crate::types::agent::AgentRecord>> {
        // Build dynamic filter conditions
        let mut conditions = Vec::new();
        let mut params = BTreeMap::new();

        if let Some(team) = &filter.team_id {
            conditions.push("team_id == $filter_team".to_string());
            params.insert("filter_team".into(), DataValue::from(team.as_str()));
        }
        if let Some(org) = &filter.org_id {
            conditions.push("org_id == $filter_org".to_string());
            params.insert("filter_org".into(), DataValue::from(org.as_str()));
        }
        if let Some(status) = &filter.status {
            conditions.push("status == $filter_status".to_string());
            params.insert(
                "filter_status".into(),
                DataValue::from(status.to_string().as_str()),
            );
        }
        if let Some(role) = &filter.role {
            conditions.push("role == $filter_role".to_string());
            params.insert("filter_role".into(), DataValue::from(role.as_str()));
        }

        let filter_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(", {}", conditions.join(", "))
        };

        let query = format!(
            r#"?[agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at] :=
                *agent_registry{{agent_id, display_name, status, team_id, org_id, role, capabilities, metadata, registered_at, last_seen_at}}{filter_clause}"#,
        );

        let result = self.run_query(&query, params)?;

        result
            .rows
            .iter()
            .map(|row| {
                Ok(crate::types::agent::AgentRecord {
                    agent_id: parse_string(&row[0]),
                    display_name: parse_string(&row[1]),
                    status: crate::types::agent::AgentStatus::from_str(&parse_string(&row[2])),
                    team_id: non_empty_opt(parse_string(&row[3])),
                    org_id: non_empty_opt(parse_string(&row[4])),
                    role: non_empty_opt(parse_string(&row[5])),
                    capabilities: serde_json::from_str(&parse_string(&row[6])).unwrap_or_default(),
                    metadata: serde_json::from_str(&parse_string(&row[7]))
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                    registered_at: row[8].get_int().unwrap_or(0),
                    last_seen_at: row[9].get_int().unwrap_or(0),
                })
            })
            .collect()
    }

    /// Deregister an agent (sets status to deregistered).
    pub fn deregister_agent(&self, agent_id: &str) -> Result<()> {
        self.update_agent_status(
            agent_id,
            &crate::types::agent::AgentStatus::Deregistered,
        )
    }

    /// Add a team membership.
    pub fn add_team_membership(&self, agent_id: &str, team_id: &str) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));
        params.insert("team_id".into(), DataValue::from(team_id));
        params.insert(
            "joined_at".into(),
            DataValue::from(crate::types::memory::now_ms()),
        );

        self.run_script(
            r#"?[agent_id, team_id, joined_at] <- [[$agent_id, $team_id, $joined_at]]
            :put agent_team_membership { agent_id, team_id => joined_at }"#,
            params,
        )?;

        Ok(())
    }

    /// Remove a team membership.
    pub fn remove_team_membership(&self, agent_id: &str, team_id: &str) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));
        params.insert("team_id".into(), DataValue::from(team_id));

        self.run_script(
            r#"?[agent_id, team_id] <- [[$agent_id, $team_id]]
            :rm agent_team_membership { agent_id, team_id }"#,
            params,
        )?;

        Ok(())
    }

    // ── Multi-Agent: Scope Grants ──

    /// Insert a scope grant.
    pub fn insert_grant(&self, grant: &crate::types::scope::ScopeGrant) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(grant.id.to_string()));
        params.insert(
            "agent_pattern".into(),
            DataValue::from(grant.agent_pattern.to_string().as_str()),
        );
        params.insert(
            "namespace_pattern".into(),
            DataValue::from(&*grant.namespace_pattern),
        );
        params.insert(
            "permissions".into(),
            DataValue::from(
                serde_json::to_string(&grant.permissions).unwrap_or_default(),
            ),
        );
        params.insert("granted_by".into(), DataValue::from(&*grant.granted_by));
        params.insert("granted_at".into(), DataValue::from(grant.granted_at));
        if let Some(exp) = grant.expires_at {
            params.insert("expires_at".into(), DataValue::from(exp));
        } else {
            params.insert("expires_at".into(), DataValue::Null);
        }
        params.insert("revoked".into(), DataValue::from(false));

        self.run_script(
            r#"?[id, agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at, revoked] <- [[
                to_uuid($id), $agent_pattern, $namespace_pattern, $permissions, $granted_by, $granted_at, $expires_at, $revoked
            ]]
            :put scope_grants {
                id => agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at, revoked
            }"#,
            params,
        )?;

        Ok(())
    }

    /// Revoke a scope grant by marking it as revoked.
    pub fn revoke_grant(&self, grant_id: uuid::Uuid) -> Result<()> {
        // Read the existing grant
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(grant_id.to_string()));

        let result = self.run_query(
            r#"?[id, agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at] :=
                *scope_grants{id, agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at},
                id == to_uuid($id)"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Err(MemoriaError::Store(format!(
                "grant not found: {grant_id}"
            )));
        }

        // Re-put with revoked=true
        let row = &result.rows[0];
        let mut params2 = BTreeMap::new();
        params2.insert("id".into(), DataValue::from(grant_id.to_string()));
        params2.insert("agent_pattern".into(), row[1].clone());
        params2.insert("namespace_pattern".into(), row[2].clone());
        params2.insert("permissions".into(), row[3].clone());
        params2.insert("granted_by".into(), row[4].clone());
        params2.insert("granted_at".into(), row[5].clone());
        params2.insert("expires_at".into(), row[6].clone());
        params2.insert("revoked".into(), DataValue::from(true));

        self.run_script(
            r#"?[id, agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at, revoked] <- [[
                to_uuid($id), $agent_pattern, $namespace_pattern, $permissions, $granted_by, $granted_at, $expires_at, $revoked
            ]]
            :put scope_grants {
                id => agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at, revoked
            }"#,
            params2,
        )?;

        Ok(())
    }

    /// Query active (non-revoked, non-expired) grants.
    pub fn query_grants(
        &self,
        filter: &crate::types::scope::GrantFilter,
    ) -> Result<Vec<crate::types::scope::ScopeGrant>> {
        let now = crate::types::memory::now_ms();

        // Query all grants and filter in Rust to avoid CozoDB expression quirks
        let result = self.run_query(
            r#"?[id, agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at, revoked] :=
                *scope_grants{id, agent_pattern, namespace_pattern, permissions, granted_by, granted_at, expires_at, revoked}"#,
            BTreeMap::new(),
        )?;

        let mut grants = Vec::new();
        for row in &result.rows {
            let revoked = row[7].get_bool().unwrap_or(false);
            if revoked {
                continue;
            }

            let expires_at = row[6].get_int();
            if let Some(exp) = expires_at {
                if exp <= now {
                    continue;
                }
            }

            let agent_pattern_str = parse_string(&row[1]);
            let namespace_pattern = parse_string(&row[2]);

            // Apply optional filters
            if let Some(ref ap) = filter.agent_pattern {
                if &agent_pattern_str != ap {
                    continue;
                }
            }
            if let Some(ref ns) = filter.namespace_pattern {
                if &namespace_pattern != ns {
                    continue;
                }
            }

            let id = parse_uuid(&row[0])?;
            let agent_pattern = parse_agent_pattern(&agent_pattern_str);
            let permissions: Vec<crate::types::scope::Permission> =
                serde_json::from_str(&parse_string(&row[3])).unwrap_or_default();
            let granted_by = parse_string(&row[4]);
            let granted_at = row[5].get_int().unwrap_or(0);

            grants.push(crate::types::scope::ScopeGrant {
                id,
                agent_pattern,
                namespace_pattern,
                permissions,
                granted_by,
                granted_at,
                expires_at,
            });
        }

        Ok(grants)
    }

    /// Check if an agent has a specific permission on a namespace.
    /// Returns true if any matching grant provides the permission.
    pub fn check_permission(
        &self,
        agent_id: &str,
        agent_team: Option<&str>,
        agent_org: Option<&str>,
        agent_role: Option<&str>,
        namespace: &str,
        permission: &crate::types::scope::Permission,
    ) -> Result<bool> {
        let grants = self.query_grants(&crate::types::scope::GrantFilter::default())?;

        for grant in &grants {
            // Check agent pattern match
            let agent_matches = match &grant.agent_pattern {
                crate::types::scope::AgentPattern::Any => true,
                crate::types::scope::AgentPattern::Exact(id) => id == agent_id,
                crate::types::scope::AgentPattern::Team(t) => {
                    agent_team.map_or(false, |at| at == t)
                }
                crate::types::scope::AgentPattern::Org(o) => {
                    agent_org.map_or(false, |ao| ao == o)
                }
                crate::types::scope::AgentPattern::Role(r) => {
                    agent_role.map_or(false, |ar| ar == r)
                }
            };

            if !agent_matches {
                continue;
            }

            // Check namespace pattern match (simple glob: * matches everything, prefix:* matches prefix)
            let ns_matches = namespace_matches(&grant.namespace_pattern, namespace);

            if !ns_matches {
                continue;
            }

            // Check permission
            if grant.permissions.contains(permission) {
                return Ok(true);
            }

            // Admin implies all permissions
            if grant
                .permissions
                .contains(&crate::types::scope::Permission::Admin)
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    // ── Multi-Agent: Scratchpad ──

    /// Put a value into the scratchpad (upsert).
    pub fn scratch_put(
        &self,
        namespace: &str,
        key: &str,
        value: &crate::types::scratch::ScratchValue,
        owner_agent: &str,
        visibility: &crate::types::scratch::Visibility,
        expires_at: Option<i64>,
    ) -> Result<()> {
        let now = crate::types::memory::now_ms();
        let mut params = BTreeMap::new();
        params.insert("namespace".into(), DataValue::from(namespace));
        params.insert("key".into(), DataValue::from(key));
        params.insert("value".into(), DataValue::from(value.to_json_string().as_str()));
        params.insert("visibility".into(), DataValue::from(visibility.to_string().as_str()));
        params.insert("owner_agent".into(), DataValue::from(owner_agent));
        params.insert("updated_at".into(), DataValue::from(now));
        if let Some(exp) = expires_at {
            params.insert("expires_at".into(), DataValue::from(exp));
        } else {
            params.insert("expires_at".into(), DataValue::Null);
        }

        self.run_script(
            r#"?[namespace, key, value, visibility, owner_agent, updated_at, expires_at] <- [[
                $namespace, $key, $value, $visibility, $owner_agent, $updated_at, $expires_at
            ]]
            :put scratchpad {
                namespace, key => value, visibility, owner_agent, updated_at, expires_at
            }"#,
            params,
        )?;

        Ok(())
    }

    /// Get a scratchpad entry by namespace and key.
    pub fn scratch_get(
        &self,
        namespace: &str,
        key: &str,
    ) -> Result<Option<crate::types::scratch::ScratchEntry>> {
        let mut params = BTreeMap::new();
        params.insert("namespace".into(), DataValue::from(namespace));
        params.insert("key".into(), DataValue::from(key));

        let result = self.run_query(
            r#"?[namespace, key, value, visibility, owner_agent, updated_at, expires_at] :=
                *scratchpad{namespace, key, value, visibility, owner_agent, updated_at, expires_at},
                namespace == $namespace, key == $key"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        Ok(Some(crate::types::scratch::ScratchEntry {
            namespace: parse_string(&row[0]),
            key: parse_string(&row[1]),
            value: crate::types::scratch::ScratchValue::from_json_str(&parse_string(&row[2]))
                .unwrap_or(crate::types::scratch::ScratchValue::Text(parse_string(&row[2]))),
            visibility: crate::types::scratch::Visibility::from_str(&parse_string(&row[3])),
            owner_agent: parse_string(&row[4]),
            updated_at: row[5].get_int().unwrap_or(0),
            expires_at: row[6].get_int(),
        }))
    }

    /// List all scratchpad entries in a namespace.
    pub fn scratch_list(
        &self,
        namespace: &str,
    ) -> Result<Vec<crate::types::scratch::ScratchEntry>> {
        let mut params = BTreeMap::new();
        params.insert("namespace".into(), DataValue::from(namespace));

        let result = self.run_query(
            r#"?[namespace, key, value, visibility, owner_agent, updated_at, expires_at] :=
                *scratchpad{namespace, key, value, visibility, owner_agent, updated_at, expires_at},
                namespace == $namespace"#,
            params,
        )?;

        result
            .rows
            .iter()
            .map(|row| {
                Ok(crate::types::scratch::ScratchEntry {
                    namespace: parse_string(&row[0]),
                    key: parse_string(&row[1]),
                    value: crate::types::scratch::ScratchValue::from_json_str(
                        &parse_string(&row[2]),
                    )
                    .unwrap_or(crate::types::scratch::ScratchValue::Text(parse_string(&row[2]))),
                    visibility: crate::types::scratch::Visibility::from_str(
                        &parse_string(&row[3]),
                    ),
                    owner_agent: parse_string(&row[4]),
                    updated_at: row[5].get_int().unwrap_or(0),
                    expires_at: row[6].get_int(),
                })
            })
            .collect()
    }

    /// Delete a scratchpad entry.
    pub fn scratch_delete(&self, namespace: &str, key: &str) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("namespace".into(), DataValue::from(namespace));
        params.insert("key".into(), DataValue::from(key));

        self.run_script(
            r#"?[namespace, key] <- [[$namespace, $key]]
            :rm scratchpad { namespace, key }"#,
            params,
        )?;

        Ok(())
    }

    /// Clear all scratchpad entries in a namespace.
    pub fn scratch_clear(&self, namespace: &str) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("namespace".into(), DataValue::from(namespace));

        self.run_script(
            r#"?[namespace, key] := *scratchpad{namespace, key}, namespace == $namespace
            :rm scratchpad { namespace, key }"#,
            params,
        )?;

        Ok(())
    }

    /// Garbage-collect expired scratchpad entries.
    pub fn scratch_gc(&self) -> Result<usize> {
        let now = crate::types::memory::now_ms();
        let mut params = BTreeMap::new();
        params.insert("now".into(), DataValue::from(now));

        // Count expired entries first
        let count_result = self.run_query(
            r#"?[count(key)] := *scratchpad{namespace, key, expires_at},
                !is_null(expires_at), expires_at <= $now"#,
            params.clone(),
        )?;

        let count = count_result
            .rows
            .first()
            .and_then(|r| r[0].get_int())
            .unwrap_or(0) as usize;

        if count > 0 {
            self.run_script(
                r#"?[namespace, key] := *scratchpad{namespace, key, expires_at},
                    !is_null(expires_at), expires_at <= $now
                :rm scratchpad { namespace, key }"#,
                params,
            )?;
        }

        Ok(count)
    }

    // ── Multi-Agent: Hash-Chained Audit Log ──

    /// Insert an entry into the hash-chained audit log.
    ///
    /// Reads the current chain head, computes SHA-256 hash, and appends.
    pub fn insert_audit_entry(
        &self,
        entry: &crate::types::audit::AuditEntry,
    ) -> Result<crate::types::audit::AuditRecord> {
        use sha2::{Digest, Sha256};

        let now = crate::types::memory::now_ms();

        // Read the chain head (highest seq)
        let head = self.run_query(
            r#"?[seq, hash] := *audit_chain{seq, hash}
            :order -seq
            :limit 1"#,
            BTreeMap::new(),
        )?;

        let (next_seq, prev_hash) = if head.rows.is_empty() {
            (0i64, "genesis".to_string())
        } else {
            let seq = head.rows[0][0].get_int().unwrap_or(0);
            let hash = parse_string(&head.rows[0][1]);
            (seq + 1, hash)
        };

        // Compute SHA-256 hash: H(seq || prev_hash || operation || agent_id || details)
        let details_str = serde_json::to_string(&entry.details).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(next_seq.to_le_bytes());
        hasher.update(prev_hash.as_bytes());
        hasher.update(entry.operation.as_bytes());
        hasher.update(entry.agent_id.as_bytes());
        hasher.update(details_str.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        let mut params = BTreeMap::new();
        params.insert("seq".into(), DataValue::from(next_seq));
        params.insert("ts".into(), DataValue::from(now));
        params.insert("operation".into(), DataValue::from(&*entry.operation));
        params.insert("agent_id".into(), DataValue::from(&*entry.agent_id));
        params.insert("namespace".into(), DataValue::from(&*entry.namespace));
        params.insert(
            "details".into(),
            DataValue::from(details_str.as_str()),
        );
        params.insert("prev_hash".into(), DataValue::from(prev_hash.as_str()));
        params.insert("hash".into(), DataValue::from(hash.as_str()));

        self.run_script(
            r#"?[seq, ts, operation, agent_id, namespace, details, prev_hash, hash] <- [[
                $seq, $ts, $operation, $agent_id, $namespace, $details, $prev_hash, $hash
            ]]
            :put audit_chain {
                seq => ts, operation, agent_id, namespace, details, prev_hash, hash
            }"#,
            params,
        )?;

        Ok(crate::types::audit::AuditRecord {
            seq: next_seq,
            ts: now,
            operation: entry.operation.clone(),
            agent_id: entry.agent_id.clone(),
            namespace: entry.namespace.clone(),
            details: entry.details.clone(),
            prev_hash,
            hash,
        })
    }

    /// Verify the integrity of the audit chain from a given starting sequence.
    pub fn verify_audit_chain(
        &self,
        from_seq: i64,
    ) -> Result<crate::types::audit::AuditVerification> {
        use sha2::{Digest, Sha256};

        let mut params = BTreeMap::new();
        params.insert("from_seq".into(), DataValue::from(from_seq));

        let result = self.run_query(
            r#"?[seq, ts, operation, agent_id, namespace, details, prev_hash, hash] :=
                *audit_chain{seq, ts, operation, agent_id, namespace, details, prev_hash, hash},
                seq >= $from_seq
            :order seq"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(crate::types::audit::AuditVerification {
                integrity: crate::types::audit::Integrity::Empty,
                entries_checked: 0,
                first_seq: 0,
                last_seq: 0,
                broken_at: None,
            });
        }

        let first_seq = result.rows[0][0].get_int().unwrap_or(0);
        let last_seq = result.rows.last().unwrap()[0].get_int().unwrap_or(0);
        let mut entries_checked = 0;

        for row in &result.rows {
            let seq = row[0].get_int().unwrap_or(0);
            let operation = parse_string(&row[2]);
            let agent_id = parse_string(&row[3]);
            let details = parse_string(&row[5]);
            let prev_hash = parse_string(&row[6]);
            let stored_hash = parse_string(&row[7]);

            // Recompute hash
            let mut hasher = Sha256::new();
            hasher.update(seq.to_le_bytes());
            hasher.update(prev_hash.as_bytes());
            hasher.update(operation.as_bytes());
            hasher.update(agent_id.as_bytes());
            hasher.update(details.as_bytes());
            let computed_hash = format!("{:x}", hasher.finalize());

            if computed_hash != stored_hash {
                return Ok(crate::types::audit::AuditVerification {
                    integrity: crate::types::audit::Integrity::Broken,
                    entries_checked,
                    first_seq,
                    last_seq,
                    broken_at: Some(seq),
                });
            }

            entries_checked += 1;
        }

        Ok(crate::types::audit::AuditVerification {
            integrity: crate::types::audit::Integrity::Valid,
            entries_checked,
            first_seq,
            last_seq,
            broken_at: None,
        })
    }

    /// Query the audit chain with optional filters.
    pub fn query_audit_chain(
        &self,
        filter: &crate::types::audit::AuditFilter,
    ) -> Result<Vec<crate::types::audit::AuditRecord>> {
        let mut conditions = Vec::new();
        let mut params = BTreeMap::new();

        if let Some(agent) = &filter.agent_id {
            conditions.push("agent_id == $filter_agent".to_string());
            params.insert("filter_agent".into(), DataValue::from(agent.as_str()));
        }
        if let Some(op) = &filter.operation {
            conditions.push("operation == $filter_op".to_string());
            params.insert("filter_op".into(), DataValue::from(op.as_str()));
        }
        if let Some(ns) = &filter.namespace {
            conditions.push("namespace == $filter_ns".to_string());
            params.insert("filter_ns".into(), DataValue::from(ns.as_str()));
        }
        if let Some(since) = filter.since_seq {
            conditions.push("seq >= $since_seq".to_string());
            params.insert("since_seq".into(), DataValue::from(since));
        }

        let filter_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(", {}", conditions.join(", "))
        };

        let limit_clause = filter
            .limit
            .map(|l| format!(":limit {l}"))
            .unwrap_or_default();

        let query = format!(
            r#"?[seq, ts, operation, agent_id, namespace, details, prev_hash, hash] :=
                *audit_chain{{seq, ts, operation, agent_id, namespace, details, prev_hash, hash}}{filter_clause}
            :order seq
            {limit_clause}"#,
        );

        let result = self.run_query(&query, params)?;

        result
            .rows
            .iter()
            .map(|row| {
                let details_str = parse_string(&row[5]);
                let details: serde_json::Value =
                    serde_json::from_str(&details_str).unwrap_or(serde_json::Value::Null);
                Ok(crate::types::audit::AuditRecord {
                    seq: row[0].get_int().unwrap_or(0),
                    ts: row[1].get_int().unwrap_or(0),
                    operation: parse_string(&row[2]),
                    agent_id: parse_string(&row[3]),
                    namespace: parse_string(&row[4]),
                    details,
                    prev_hash: parse_string(&row[6]),
                    hash: parse_string(&row[7]),
                })
            })
            .collect()
    }

    // ── Agent Trust Profiles ──

    /// Get the trust profile for an agent. Returns default trust (1.0) if not found.
    pub fn get_agent_trust(&self, agent_id: &str) -> Result<AgentTrustProfile> {
        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));

        let result = self.run_query(
            r#"?[trust_score, success_count, failure_count, attribution_penalty_sum] :=
                *agent_trust_profiles{agent_id, trust_score, success_count, failure_count, attribution_penalty_sum},
                agent_id = $agent_id"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(AgentTrustProfile {
                agent_id: agent_id.to_string(),
                trust_score: 1.0,
                success_count: 0,
                failure_count: 0,
                attribution_penalty_sum: 0.0,
            });
        }

        let row = &result.rows[0];
        Ok(AgentTrustProfile {
            agent_id: agent_id.to_string(),
            trust_score: row[0].get_float().unwrap_or(1.0),
            success_count: row[1].get_int().unwrap_or(0),
            failure_count: row[2].get_int().unwrap_or(0),
            attribution_penalty_sum: row[3].get_float().unwrap_or(0.0),
        })
    }

    /// Update agent trust after a task outcome.
    /// Recomputes trust = (success / (success + failure + 1)) * (1.0 - penalty_decay).clamp(0.1, 1.0)
    pub fn update_agent_trust(
        &self,
        agent_id: &str,
        is_success: bool,
        penalty: f64,
    ) -> Result<AgentTrustProfile> {
        let current = self.get_agent_trust(agent_id)?;

        let new_success = current.success_count + if is_success { 1 } else { 0 };
        let new_failure = current.failure_count + if is_success { 0 } else { 1 };
        let new_penalty_sum = current.attribution_penalty_sum + penalty;

        // Compute trust score with Laplace smoothing and penalty decay
        let success_ratio = new_success as f64 / (new_success as f64 + new_failure as f64 + 1.0);
        let penalty_decay = (new_penalty_sum * 0.1).min(0.9); // cap at 0.9 so trust floor is 0.1
        let trust_score = (success_ratio * (1.0 - penalty_decay)).clamp(0.1, 1.0);

        let mut params = BTreeMap::new();
        params.insert("agent_id".into(), DataValue::from(agent_id));
        params.insert("trust_score".into(), DataValue::from(trust_score));
        params.insert("success_count".into(), DataValue::from(new_success));
        params.insert("failure_count".into(), DataValue::from(new_failure));
        params.insert("attribution_penalty_sum".into(), DataValue::from(new_penalty_sum));

        self.run_script(
            r#"?[agent_id, trust_score, success_count, failure_count, attribution_penalty_sum] <-
                [[$agent_id, $trust_score, $success_count, $failure_count, $attribution_penalty_sum]]
            :put agent_trust_profiles {
                agent_id
                =>
                trust_score, success_count, failure_count, attribution_penalty_sum
            }"#,
            params,
        )?;

        Ok(AgentTrustProfile {
            agent_id: agent_id.to_string(),
            trust_score,
            success_count: new_success,
            failure_count: new_failure,
            attribution_penalty_sum: new_penalty_sum,
        })
    }

    // ── Telos: Goal System CRUD ──

    /// Insert a telos (goal) into the store.
    pub fn insert_telos(&self, telos: &Telos) -> Result<()> {
        let embedding_vals: Vec<DataValue> = telos
            .embedding
            .iter()
            .map(|&v| DataValue::from(v as f64))
            .collect();

        let related_entities_vals: Vec<DataValue> = telos
            .related_entities
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();
        let required_skills_vals: Vec<DataValue> = telos
            .required_skills
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();
        let depends_on_vals: Vec<DataValue> = telos
            .depends_on
            .iter()
            .map(|id| DataValue::from(id.to_string()))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(telos.id.to_string()));
        params.insert("title".into(), DataValue::from(telos.title.as_str()));
        params.insert(
            "description".into(),
            DataValue::from(telos.description.as_str()),
        );
        params.insert("embedding".into(), DataValue::List(embedding_vals));
        params.insert(
            "parent".into(),
            telos
                .parent
                .map(|p| DataValue::from(p.to_string()))
                .unwrap_or(DataValue::Null),
        );
        params.insert("depth".into(), DataValue::from(telos.depth as i64));
        params.insert("owner".into(), DataValue::from(telos.owner.as_str()));
        params.insert("set_by".into(), DataValue::from(telos.set_by.as_str()));
        params.insert(
            "namespace".into(),
            DataValue::from(telos.namespace.as_str()),
        );
        params.insert(
            "status".into(),
            DataValue::from(telos.status.as_str()),
        );
        params.insert("priority".into(), DataValue::from(telos.priority));
        params.insert("urgency".into(), DataValue::from(telos.urgency));
        params.insert("confidence".into(), DataValue::from(telos.confidence));
        params.insert(
            "provenance".into(),
            DataValue::from(telos.provenance.as_str()),
        );
        params.insert(
            "deadline".into(),
            telos
                .deadline
                .map(DataValue::from)
                .unwrap_or(DataValue::Null),
        );
        params.insert("created_at".into(), DataValue::from(telos.created_at));
        params.insert(
            "started_at".into(),
            telos
                .started_at
                .map(DataValue::from)
                .unwrap_or(DataValue::Null),
        );
        params.insert(
            "completed_at".into(),
            telos
                .completed_at
                .map(DataValue::from)
                .unwrap_or(DataValue::Null),
        );
        params.insert("progress".into(), DataValue::from(telos.progress));
        params.insert(
            "stalled_since".into(),
            telos
                .stalled_since
                .map(DataValue::from)
                .unwrap_or(DataValue::Null),
        );
        params.insert(
            "success_criteria".into(),
            DataValue::from(serde_json::to_string(&telos.success_criteria)?),
        );
        params.insert(
            "related_entities".into(),
            DataValue::List(related_entities_vals),
        );
        params.insert(
            "required_skills".into(),
            DataValue::List(required_skills_vals),
        );
        params.insert("depends_on".into(), DataValue::List(depends_on_vals));
        params.insert(
            "last_attended".into(),
            DataValue::from(telos.last_attended),
        );
        params.insert(
            "attention_count".into(),
            DataValue::from(telos.attention_count),
        );

        let script = concat!(
            "?[id, valid_at, title, description, embedding, parent, depth, owner, set_by, ",
            "namespace, status, priority, urgency, confidence, provenance, deadline, ",
            "created_at, started_at, completed_at, progress, stalled_since, ",
            "success_criteria, related_entities, required_skills, depends_on, ",
            "last_attended, attention_count] <- ",
            "[[$id, 'ASSERT', $title, $description, $embedding, $parent, $depth, $owner, $set_by, ",
            "$namespace, $status, $priority, $urgency, $confidence, $provenance, $deadline, ",
            "$created_at, $started_at, $completed_at, $progress, $stalled_since, ",
            "$success_criteria, $related_entities, $required_skills, $depends_on, ",
            "$last_attended, $attention_count]] ",
            ":put telos {id, valid_at => title, description, embedding, parent, depth, owner, ",
            "set_by, namespace, status, priority, urgency, confidence, provenance, deadline, ",
            "created_at, started_at, completed_at, progress, stalled_since, ",
            "success_criteria, related_entities, required_skills, depends_on, ",
            "last_attended, attention_count}",
        );
        self.run_script(script, params)?;
        Ok(())
    }

    /// Get a telos by ID (latest version).
    pub fn get_telos(&self, id: Uuid) -> Result<Option<Telos>> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(id.to_string()));

        let result = self.run_query(
            r#"?[id, title, description, embedding, parent, depth, owner, set_by,
                namespace, status, priority, urgency, confidence, provenance, deadline,
                created_at, started_at, completed_at, progress, stalled_since,
                success_criteria, related_entities, required_skills, depends_on,
                last_attended, attention_count] :=
                *telos{id, title, description, embedding, parent, depth, owner, set_by,
                       namespace, status, priority, urgency, confidence, provenance,
                       deadline, created_at, started_at, completed_at, progress,
                       stalled_since, success_criteria, related_entities, required_skills,
                       depends_on, last_attended, attention_count},
                id = to_uuid($id)"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        Ok(Some(parse_telos_row(&result.rows[0])?))
    }

    /// Remove all temporal versions of a telos so a fresh version can be inserted.
    /// No-op if the telos doesn't exist.
    fn remove_telos_versions(&self, id: Uuid) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(id.to_string()));
        // Remove all temporal rows for this id
        let _ = self.run_script(
            r#"?[id, valid_at] := *telos{id, valid_at}, id = to_uuid($id)
            :rm telos {id, valid_at}"#,
            params,
        );
        Ok(())
    }

    /// Update a telos by removing old versions and inserting a fresh one.
    pub fn upsert_telos(&self, telos: &Telos) -> Result<()> {
        self.remove_telos_versions(telos.id)?;
        self.insert_telos(telos)
    }

    /// Update the status of a telos (creates new temporal version via Validity).
    pub fn update_telos_status(&self, id: Uuid, status: &str) -> Result<()> {
        let mut telos = self
            .get_telos(id)?
            .ok_or_else(|| MemoriaError::NotFound(id))?;

        telos.status = TelosStatus::from_str(status);

        // If transitioning to completed/failed/abandoned, set completed_at
        if telos.status.is_terminal() && telos.completed_at.is_none() {
            telos.completed_at = Some(crate::types::memory::now_ms());
        }

        self.upsert_telos(&telos)
    }

    /// Update the progress of a telos.
    pub fn update_telos_progress(&self, id: Uuid, progress: f64) -> Result<()> {
        let mut telos = self
            .get_telos(id)?
            .ok_or_else(|| MemoriaError::NotFound(id))?;

        let old_progress = telos.progress;
        telos.progress = progress.clamp(0.0, 1.0);

        // Clear stalled_since when progress is made
        if telos.progress > old_progress {
            telos.stalled_since = None;
        }

        self.upsert_telos(&telos)
    }

    /// Update the owner of a telos (for claiming/delegation).
    pub fn update_telos_owner(&self, id: Uuid, owner: &str) -> Result<()> {
        let mut telos = self
            .get_telos(id)?
            .ok_or_else(|| MemoriaError::NotFound(id))?;

        telos.owner = owner.to_string();

        self.upsert_telos(&telos)
    }

    /// List active telos in a namespace, ordered by priority descending.
    pub fn list_active_telos(&self, namespace: &str, limit: usize) -> Result<Vec<Telos>> {
        let mut params = BTreeMap::new();
        params.insert("namespace".into(), DataValue::from(namespace));
        params.insert("limit".into(), DataValue::from(limit as i64));

        let result = self.run_query(
            r#"?[id, title, description, embedding, parent, depth, owner, set_by,
                namespace, status, priority, urgency, confidence, provenance, deadline,
                created_at, started_at, completed_at, progress, stalled_since,
                success_criteria, related_entities, required_skills, depends_on,
                last_attended, attention_count] :=
                *telos{id, title, description, embedding, parent, depth, owner, set_by,
                       namespace, status, priority, urgency, confidence, provenance,
                       deadline, created_at, started_at, completed_at, progress,
                       stalled_since, success_criteria, related_entities, required_skills,
                       depends_on, last_attended, attention_count},
                namespace = $namespace,
                status = "active"
            :sort -priority
            :limit $limit"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_telos_row(row)).collect()
    }

    /// Find telos by embedding similarity via HNSW index.
    pub fn find_telos_by_embedding(
        &self,
        emb: &[f32],
        k: usize,
    ) -> Result<Vec<(Telos, f64)>> {
        // HNSW requires DataValue::Vec (ndarray), not DataValue::List
        let arr = Array1::from_vec(emb.to_vec());
        let query_vec = DataValue::Vec(Vector::F32(arr));

        let mut params = BTreeMap::new();
        params.insert("query_emb".into(), query_vec);
        params.insert("k".into(), DataValue::from(k as i64));

        let result = self.run_query(
            r#"?[id, title, description, embedding, parent, depth, owner, set_by,
                namespace, status, priority, urgency, confidence, provenance, deadline,
                created_at, started_at, completed_at, progress, stalled_since,
                success_criteria, related_entities, required_skills, depends_on,
                last_attended, attention_count, dist] :=
                ~telos:telos_vec_idx{id | query: $query_emb, k: $k, ef: 50, bind_distance: dist},
                *telos{id, title, description, embedding, parent, depth, owner, set_by,
                       namespace, status, priority, urgency, confidence, provenance,
                       deadline, created_at, started_at, completed_at, progress,
                       stalled_since, success_criteria, related_entities, required_skills,
                       depends_on, last_attended, attention_count}"#,
            params,
        )?;

        let mut results = Vec::new();
        for row in &result.rows {
            let telos = parse_telos_row(row)?;
            let dist = row[26].get_float().unwrap_or(1.0);
            results.push((telos, dist));
        }
        Ok(results)
    }

    /// List active telos whose deadline has passed (overdue goals).
    ///
    /// Returns all telos with status="active", deadline IS NOT NULL, and deadline < now_ms.
    /// Used by the deadline enforcement background task.
    pub fn list_overdue_telos(&self, now_ms: i64) -> Result<Vec<Telos>> {
        let mut params = BTreeMap::new();
        params.insert("now_ms".into(), DataValue::from(now_ms));

        let result = self.run_query(
            r#"?[id, title, description, embedding, parent, depth, owner, set_by,
                namespace, status, priority, urgency, confidence, provenance, deadline,
                created_at, started_at, completed_at, progress, stalled_since,
                success_criteria, related_entities, required_skills, depends_on,
                last_attended, attention_count] :=
                *telos{id, title, description, embedding, parent, depth, owner, set_by,
                       namespace, status, priority, urgency, confidence, provenance,
                       deadline, created_at, started_at, completed_at, progress,
                       stalled_since, success_criteria, related_entities, required_skills,
                       depends_on, last_attended, attention_count},
                status = "active",
                !is_null(deadline),
                deadline < $now_ms
            :sort deadline"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_telos_row(row)).collect()
    }

    /// Update the stalled_since timestamp of a telos (used after deadline enforcement).
    ///
    /// If `stalled_since` is already set this is a no-op to avoid overwriting the
    /// original stall time.
    pub fn update_telos_stalled_since(&self, id: Uuid, now_ms: i64) -> Result<()> {
        let mut telos = self
            .get_telos(id)?
            .ok_or_else(|| MemoriaError::NotFound(id))?;

        // Only set if not already stalled
        if telos.stalled_since.is_none() {
            telos.stalled_since = Some(now_ms);
            self.upsert_telos(&telos)?;
        }

        Ok(())
    }

    /// Get children telos of a parent.
    pub fn get_children_telos(&self, parent_id: Uuid) -> Result<Vec<Telos>> {
        let mut params = BTreeMap::new();
        params.insert("parent_id".into(), DataValue::from(parent_id.to_string()));

        let result = self.run_query(
            r#"?[id, title, description, embedding, parent, depth, owner, set_by,
                namespace, status, priority, urgency, confidence, provenance, deadline,
                created_at, started_at, completed_at, progress, stalled_since,
                success_criteria, related_entities, required_skills, depends_on,
                last_attended, attention_count] :=
                *telos{id, title, description, embedding, parent, depth, owner, set_by,
                       namespace, status, priority, urgency, confidence, provenance,
                       deadline, created_at, started_at, completed_at, progress,
                       stalled_since, success_criteria, related_entities, required_skills,
                       depends_on, last_attended, attention_count},
                parent = to_uuid($parent_id)
            :sort -priority"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_telos_row(row)).collect()
    }

    /// Insert a telos event.
    pub fn insert_telos_event(&self, event: &TelosEvent) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(event.id.to_string()));
        params.insert("ts".into(), DataValue::from(event.ts));
        params.insert(
            "telos_id".into(),
            DataValue::from(event.telos_id.to_string()),
        );
        params.insert(
            "event_type".into(),
            DataValue::from(event.event_type.as_str()),
        );
        params.insert(
            "agent_id".into(),
            DataValue::from(event.agent_id.as_str()),
        );
        params.insert(
            "description".into(),
            DataValue::from(event.description.as_str()),
        );
        params.insert("impact".into(), DataValue::from(event.impact));
        params.insert(
            "source_memory".into(),
            event
                .source_memory
                .map(|id| DataValue::from(id.to_string()))
                .unwrap_or(DataValue::Null),
        );
        params.insert(
            "source_episode".into(),
            event
                .source_episode
                .map(|id| DataValue::from(id.to_string()))
                .unwrap_or(DataValue::Null),
        );
        params.insert(
            "metadata".into(),
            DataValue::from(serde_json::to_string(&event.metadata)?),
        );

        self.run_script(
            concat!(
                "?[id, ts, telos_id, event_type, agent_id, description, impact, ",
                "source_memory, source_episode, metadata] <- ",
                "[[$id, $ts, $telos_id, $event_type, $agent_id, $description, $impact, ",
                "$source_memory, $source_episode, $metadata]] ",
                ":put telos_events {id, ts => telos_id, event_type, agent_id, description, ",
                "impact, source_memory, source_episode, metadata}",
            ),
            params,
        )?;
        Ok(())
    }

    /// Get events for a telos, ordered by timestamp descending.
    pub fn get_telos_events(
        &self,
        telos_id: Uuid,
        limit: usize,
    ) -> Result<Vec<TelosEvent>> {
        let mut params = BTreeMap::new();
        params.insert(
            "telos_id".into(),
            DataValue::from(telos_id.to_string()),
        );
        params.insert("limit".into(), DataValue::from(limit as i64));

        let result = self.run_query(
            r#"?[id, ts, telos_id, event_type, agent_id, description, impact,
                source_memory, source_episode, metadata] :=
                *telos_events{id, ts, telos_id, event_type, agent_id, description,
                              impact, source_memory, source_episode, metadata},
                telos_id = to_uuid($telos_id)
            :sort -ts
            :limit $limit"#,
            params,
        )?;

        let mut events = Vec::new();
        for row in &result.rows {
            events.push(parse_telos_event_row(row)?);
        }
        Ok(events)
    }

    /// Start an attention span on a telos.
    pub fn start_telos_attention(
        &self,
        telos_id: Uuid,
        agent_id: &str,
        episode_id: Option<Uuid>,
    ) -> Result<()> {
        let now = crate::types::memory::now_ms();
        let mut params = BTreeMap::new();
        params.insert(
            "telos_id".into(),
            DataValue::from(telos_id.to_string()),
        );
        params.insert("started_at".into(), DataValue::from(now));
        params.insert("agent_id".into(), DataValue::from(agent_id));
        params.insert(
            "episode_id".into(),
            episode_id
                .map(|id| DataValue::from(id.to_string()))
                .unwrap_or(DataValue::Null),
        );

        self.run_script(
            concat!(
                "?[telos_id, started_at, agent_id, episode_id, outcome] <- ",
                "[[$telos_id, $started_at, $agent_id, $episode_id, 'ongoing']] ",
                ":put telos_attention {telos_id, started_at => ended_at, agent_id, ",
                "episode_id, outcome}",
            ),
            params,
        )?;

        // Also update last_attended and attention_count on the telos
        if let Some(mut telos) = self.get_telos(telos_id)? {
            telos.last_attended = now;
            telos.attention_count += 1;
            self.upsert_telos(&telos)?;
        }

        Ok(())
    }

    /// End an attention span on a telos.
    pub fn end_telos_attention(
        &self,
        telos_id: Uuid,
        started_at: i64,
        outcome: &str,
    ) -> Result<()> {
        let now = crate::types::memory::now_ms();

        // Read existing row to preserve agent_id and episode_id
        let mut params = BTreeMap::new();
        params.insert(
            "telos_id".into(),
            DataValue::from(telos_id.to_string()),
        );
        params.insert("started_at".into(), DataValue::from(started_at));

        let result = self.run_query(
            r#"?[telos_id, started_at, ended_at, agent_id, episode_id, outcome] :=
                *telos_attention{telos_id, started_at, ended_at, agent_id, episode_id, outcome},
                telos_id = to_uuid($telos_id), started_at = $started_at"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Err(MemoriaError::NotFound(telos_id));
        }

        let row = &result.rows[0];
        let agent_id = parse_string(&row[3]);
        let episode_id_val = &row[4];

        let mut params = BTreeMap::new();
        params.insert(
            "telos_id".into(),
            DataValue::from(telos_id.to_string()),
        );
        params.insert("started_at".into(), DataValue::from(started_at));
        params.insert("ended_at".into(), DataValue::from(now));
        params.insert("agent_id".into(), DataValue::from(agent_id.as_str()));
        params.insert("episode_id".into(), episode_id_val.clone());
        params.insert("outcome".into(), DataValue::from(outcome));

        self.run_script(
            concat!(
                "?[telos_id, started_at, ended_at, agent_id, episode_id, outcome] <- ",
                "[[$telos_id, $started_at, $ended_at, $agent_id, $episode_id, $outcome]] ",
                ":put telos_attention {telos_id, started_at => ended_at, agent_id, ",
                "episode_id, outcome}",
            ),
            params,
        )?;

        Ok(())
    }

    /// Compute telos alignment boost for a set of candidate memory IDs.
    ///
    /// For each memory, computes the maximum cosine similarity to any active
    /// telos embedding, weighted by the telos's `priority × confidence`.
    /// Returns (memory_id, boost) pairs where boost > 0.
    pub fn compute_telos_boost(
        &self,
        candidate_ids: &[Uuid],
        namespace: &str,
    ) -> Result<Vec<(Uuid, f64)>> {
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Get active telos embeddings with their priority × confidence weights
        let active = self.list_active_telos(namespace, 20)?;
        if active.is_empty() {
            return Ok(Vec::new());
        }

        let telos_vecs: Vec<(&[f32], f64)> = active
            .iter()
            .filter(|t| !t.embedding.is_empty())
            .map(|t| (t.embedding.as_slice(), t.priority * t.confidence))
            .collect();

        if telos_vecs.is_empty() {
            return Ok(Vec::new());
        }

        // Batch-fetch memory embeddings
        let id_list: Vec<String> = candidate_ids
            .iter()
            .map(|id| format!("to_uuid('{}')", id))
            .collect();
        let query = format!(
            "?[id, embedding] := *memories{{id, embedding}}, id in [{}]",
            id_list.join(", ")
        );
        let result = self.run_query(&query, BTreeMap::new())?;

        let mut boosts = Vec::new();
        for row in &result.rows {
            let id = parse_uuid(&row[0])?;
            let emb = parse_f32_vec(&row[1]);
            if emb.is_empty() {
                continue;
            }

            // Max weighted cosine similarity across all active telos
            let max_boost = telos_vecs
                .iter()
                .map(|(telos_emb, weight)| {
                    let sim = cosine_similarity(&emb, telos_emb);
                    sim.max(0.0) * weight
                })
                .fold(0.0_f64, f64::max);

            if max_boost > 0.0 {
                boosts.push((id, max_boost));
            }
        }

        Ok(boosts)
    }

    // ── Context Engine: store methods for tiered memory + tool-based recall ──

    /// Get all pinned memories for a namespace (core memory tier).
    pub fn get_pinned_memories(&self, namespace: &str, limit: usize) -> Result<Vec<Memory>> {
        let mut params = BTreeMap::new();
        params.insert("ns".into(), DataValue::from(namespace));
        params.insert("limit".into(), DataValue::from(limit as i64));

        let result = self.run_query(
            r#"?[id, kind, content, embedding, fields, namespace, pinned,
                expires_at, version, created_at, confidence, provenance, source_ids] :=
                *memories{id, kind, content, embedding, fields, namespace, pinned,
                          expires_at, version, created_at, confidence, provenance, source_ids},
                namespace = $ns,
                pinned = true
            :limit $limit"#,
            params,
        )?;

        let mut memories = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            memories.push(parse_memory_row(row)?);
        }
        Ok(memories)
    }

    /// Get edges between a set of memory IDs (subgraph extraction for formatter).
    ///
    /// Returns edges where both source AND target are in the provided set.
    pub fn get_edges_between(&self, memory_ids: &[Uuid]) -> Result<Vec<(Uuid, Uuid, String, f64)>> {
        if memory_ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_vals: Vec<DataValue> = memory_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ids".into(), DataValue::List(id_vals));

        let result = self.run_query(
            r#"input_raw[raw_id] <- $ids
               input[uid] := input_raw[raw_id], uid = to_uuid(raw_id)
               ?[source, target, kind, weight] :=
                   input[source],
                   input[target],
                   *edges{source, target, kind, weight}"#,
            params,
        )?;

        let mut edges = Vec::new();
        for row in &result.rows {
            let source = parse_uuid(&row[0])?;
            let target = parse_uuid(&row[1])?;
            let kind = parse_string(&row[2]);
            let weight = row[3].get_float().unwrap_or(0.0);
            edges.push((source, target, kind, weight));
        }
        Ok(edges)
    }

    /// Get cached PageRank values for a set of memory IDs.
    ///
    /// Returns a map of memory_id -> pagerank score.
    /// Entries not in `memory_importance` are simply absent from the result.
    pub fn get_cached_pagerank(&self, memory_ids: &[Uuid]) -> Result<std::collections::HashMap<Uuid, f64>> {
        if memory_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        let id_vals: Vec<DataValue> = memory_ids
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ids".into(), DataValue::List(id_vals));

        let result = self.run_query(
            r#"input_raw[raw_id] <- $ids
               input[uid] := input_raw[raw_id], uid = to_uuid(raw_id)
               ?[memory_id, pagerank] :=
                   input[memory_id],
                   *memory_importance{memory_id, pagerank}"#,
            params,
        )?;

        let mut map = std::collections::HashMap::new();
        for row in &result.rows {
            let id = parse_uuid(&row[0])?;
            let pr = row[1].get_float().unwrap_or(0.0);
            map.insert(id, pr);
        }
        Ok(map)
    }

    /// Set or unset the pinned flag on a memory.
    pub fn set_memory_pinned(&self, id: MemoryId, pinned: bool) -> Result<()> {
        let mut memory = self
            .get_memory(id)?
            .ok_or_else(|| MemoriaError::NotFound(id))?;
        memory.pinned = pinned;
        self.insert_memory(&memory)?;
        Ok(())
    }

    /// Update a memory's content and embedding in-place, bumping version.
    pub fn update_memory_content(
        &self,
        id: MemoryId,
        new_content: &str,
        new_embedding: &[f32],
    ) -> Result<()> {
        let mut memory = self
            .get_memory(id)?
            .ok_or_else(|| MemoriaError::NotFound(id))?;
        memory.content = new_content.to_string();
        memory.embedding = new_embedding.to_vec();
        memory.version += 1;
        self.insert_memory(&memory)?;
        Ok(())
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Agent trust profile data.
#[derive(Debug, Clone)]
pub struct AgentTrustProfile {
    pub agent_id: String,
    pub trust_score: f64,
    pub success_count: i64,
    pub failure_count: i64,
    pub attribution_penalty_sum: f64,
}

/// Public UUID parser for use by dynamics modules.
pub fn parse_uuid_pub(val: &DataValue) -> Result<uuid::Uuid> {
    parse_uuid(val)
}

/// Public memory row parser for use by query planner.
pub fn parse_memory_row_pub(row: &[DataValue]) -> Result<Memory> {
    parse_memory_row(row)
}

/// Parse a CozoDB row into a Memory struct.
fn parse_memory_row(row: &[DataValue]) -> Result<Memory> {
    let id = parse_uuid(&row[0])?;
    let kind = parse_string(&row[1]);
    let content = parse_string(&row[2]);
    let embedding = parse_f32_vec(&row[3]);
    let fields_str = parse_string(&row[4]);
    let fields: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&fields_str).unwrap_or_default();
    let namespace = parse_string(&row[5]);
    let pinned = row[6].get_bool().unwrap_or(false);
    let expires_at = row[7].get_int();
    let version = row[8].get_int().unwrap_or(0) as i32;
    let created_at = row[9].get_int().unwrap_or(0);
    let confidence = row[10].get_float().unwrap_or(0.0);
    let provenance = parse_string(&row[11]);
    let source_ids = parse_uuid_list(&row[12]);

    Ok(Memory {
        id,
        kind,
        content,
        embedding,
        fields,
        namespace,
        pinned,
        expires_at,
        version,
        created_at,
        confidence,
        provenance,
        source_ids,
    })
}

// ── DataValue Parsing Helpers ──

fn parse_uuid(val: &DataValue) -> Result<uuid::Uuid> {
    match val {
        DataValue::Str(s) => uuid::Uuid::parse_str(s.as_ref())
            .map_err(|e| MemoriaError::SchemaBootstrap(format!("parsing uuid: {e}"))),
        DataValue::Uuid(u) => {
            // UuidWrapper(pub Uuid) — access inner uuid and convert via bytes
            let inner = &u.0;
            Ok(uuid::Uuid::from_bytes(*inner.as_bytes()))
        }
        _ => Err(MemoriaError::SchemaBootstrap(format!(
            "expected uuid, got {val:?}"
        ))),
    }
}

fn parse_string(val: &DataValue) -> String {
    val.get_str().unwrap_or("").to_string()
}

/// Parse a DataValue that may be a Json column.
/// CozoDB Json columns can return as Str (if we stored a serialized string)
/// or as the native JSON structure (List, Map, etc). This helper handles both.
fn parse_json_string(val: &DataValue) -> String {
    match val {
        DataValue::Str(s) => s.to_string(),
        DataValue::Json(jd) => {
            // CozoDB Json columns parse their input. If we stored a JSON string,
            // the inner serde_json::Value is Value::String — extract the raw string.
            // If it's a parsed structure (Array/Object), serialize it back.
            match &jd.0 {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            }
        }
        DataValue::Null => "{}".to_string(),
        other => serde_json::to_string(&datavalue_to_json(other)).unwrap_or_default(),
    }
}

/// Convert a CozoDB DataValue to a serde_json::Value.
fn datavalue_to_json(val: &DataValue) -> serde_json::Value {
    match val {
        DataValue::Null => serde_json::Value::Null,
        DataValue::Bool(b) => serde_json::Value::Bool(*b),
        DataValue::Str(s) => serde_json::Value::String(s.to_string()),
        DataValue::Json(jd) => jd.0.clone(),
        DataValue::List(items) => {
            serde_json::Value::Array(items.iter().map(datavalue_to_json).collect())
        }
        _ => {
            if let Some(i) = val.get_int() {
                serde_json::Value::Number(serde_json::Number::from(i))
            } else if let Some(f) = val.get_float() {
                serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::String(format!("{val:?}"))
            }
        }
    }
}

fn parse_f32_vec_opt(val: &DataValue) -> Option<Vec<f32>> {
    match val {
        DataValue::Null => None,
        other => {
            let v = parse_f32_vec(other);
            if v.is_empty() { None } else { Some(v) }
        }
    }
}

fn parse_f32_vec(val: &DataValue) -> Vec<f32> {
    match val {
        DataValue::List(list) => list
            .iter()
            .map(|v| v.get_float().unwrap_or(0.0) as f32)
            .collect(),
        DataValue::Vec(v) => match v {
            cozo::Vector::F32(arr) => arr.to_vec(),
            cozo::Vector::F64(arr) => arr.iter().map(|&x| x as f32).collect(),
        },
        _ => Vec::new(),
    }
}

fn parse_uuid_list(val: &DataValue) -> Vec<uuid::Uuid> {
    match val {
        DataValue::List(list) => list
            .iter()
            .filter_map(|v| match v {
                DataValue::Str(s) => uuid::Uuid::parse_str(s.as_ref()).ok(),
                DataValue::Uuid(u) => Some(uuid::Uuid::from_bytes(*u.0.as_bytes())),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn non_empty_opt(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

fn parse_agent_pattern(s: &str) -> crate::types::scope::AgentPattern {
    if s == "*" {
        return crate::types::scope::AgentPattern::Any;
    }
    if let Some(id) = s.strip_prefix("agent:") {
        return crate::types::scope::AgentPattern::Exact(id.to_string());
    }
    if let Some(id) = s.strip_prefix("team:") {
        return crate::types::scope::AgentPattern::Team(id.to_string());
    }
    if let Some(id) = s.strip_prefix("org:") {
        return crate::types::scope::AgentPattern::Org(id.to_string());
    }
    if let Some(id) = s.strip_prefix("role:") {
        return crate::types::scope::AgentPattern::Role(id.to_string());
    }
    crate::types::scope::AgentPattern::Exact(s.to_string())
}

/// Check if a namespace matches a pattern (supports trailing `*` for prefix matching).
pub fn namespace_matches_pub(pattern: &str, namespace: &str) -> bool {
    namespace_matches(pattern, namespace)
}

fn namespace_matches(pattern: &str, namespace: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return namespace.starts_with(prefix);
    }
    pattern == namespace
}

/// Parse a CozoDB row into an Entity struct.
///
/// Column layout: id, name, entity_type, embedding, properties,
/// mention_count, confidence, provenance, source_ids
fn parse_entity_row(row: &[DataValue]) -> Result<Entity> {
    let id = parse_uuid(&row[0])?;
    let name = parse_string(&row[1]);
    let entity_type = parse_string(&row[2]);
    let namespace = parse_string(&row[3]);
    let embedding = parse_f32_vec(&row[4]);
    let properties_str = parse_string(&row[5]);
    let properties: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&properties_str).unwrap_or_default();
    let mention_count = row[6].get_int().unwrap_or(1);
    let confidence = row[7].get_float().unwrap_or(1.0);
    let provenance = parse_string(&row[8]);
    let source_ids = parse_uuid_list(&row[9]);

    Ok(Entity {
        id,
        name,
        entity_type,
        namespace,
        embedding,
        properties,
        mention_count,
        confidence,
        provenance,
        source_ids,
    })
}

/// Parse a CozoDB row into a Fact struct.
///
/// Column layout: id, subject_entity, predicate, object_entity, object_value,
/// temporal_status, confidence, provenance, source_ids, reinforcement_count
fn parse_fact_row(row: &[DataValue]) -> Result<Fact> {
    let id = parse_uuid(&row[0])?;
    let subject_entity = parse_uuid(&row[1])?;
    let predicate = parse_string(&row[2]);
    let object_entity = if row[3] == DataValue::Null {
        None
    } else {
        Some(parse_uuid(&row[3])?)
    };
    let object_value = if row[4] == DataValue::Null {
        None
    } else {
        Some(parse_string(&row[4]))
    };
    let namespace = parse_string(&row[5]);
    let temporal_status = parse_string(&row[6]);
    let confidence = row[7].get_float().unwrap_or(1.0);
    let provenance = parse_string(&row[8]);
    let source_ids = parse_uuid_list(&row[9]);
    let reinforcement_count = row[10].get_int().unwrap_or(1);

    Ok(Fact {
        id,
        subject_entity,
        predicate,
        object_entity,
        object_value,
        namespace,
        temporal_status,
        confidence,
        provenance,
        source_ids,
        reinforcement_count,
    })
}

/// Parse a CozoDB row into a Telos struct.
///
/// Column layout: id, title, description, embedding, parent, depth, owner, set_by,
/// namespace, status, priority, urgency, confidence, provenance, deadline,
/// created_at, started_at, completed_at, progress, stalled_since,
/// success_criteria, related_entities, required_skills, depends_on,
/// last_attended, attention_count
fn parse_telos_row(row: &[DataValue]) -> Result<Telos> {
    let id = parse_uuid(&row[0])?;
    let title = parse_string(&row[1]);
    let description = parse_string(&row[2]);
    let embedding = parse_f32_vec(&row[3]);
    let parent = if row[4] == DataValue::Null {
        None
    } else {
        Some(parse_uuid(&row[4])?)
    };
    let depth = row[5].get_int().unwrap_or(0) as i32;
    let owner = parse_string(&row[6]);
    let set_by = parse_string(&row[7]);
    let namespace = parse_string(&row[8]);
    let status = TelosStatus::from_str(&parse_string(&row[9]));
    let priority = row[10].get_float().unwrap_or(0.5);
    let urgency = row[11].get_float().unwrap_or(0.0);
    let confidence = row[12].get_float().unwrap_or(1.0);
    let provenance = TelosProvenance::from_str(&parse_string(&row[13]));
    let deadline = row[14].get_int();
    let created_at = row[15].get_int().unwrap_or(0);
    let started_at = row[16].get_int();
    let completed_at = row[17].get_int();
    let progress = row[18].get_float().unwrap_or(0.0);
    let stalled_since = row[19].get_int();
    let success_criteria_str = parse_json_string(&row[20]);
    let success_criteria: Vec<SuccessCriterion> =
        serde_json::from_str(&success_criteria_str).unwrap_or_default();
    let related_entities = parse_uuid_list(&row[21]);
    let required_skills = parse_uuid_list(&row[22]);
    let depends_on = parse_uuid_list(&row[23]);
    let last_attended = row[24].get_int().unwrap_or(0);
    let attention_count = row[25].get_int().unwrap_or(0);

    Ok(Telos {
        id,
        title,
        description,
        embedding,
        parent,
        depth,
        owner,
        set_by,
        namespace,
        status,
        priority,
        urgency,
        confidence,
        provenance,
        deadline,
        created_at,
        started_at,
        completed_at,
        progress,
        stalled_since,
        success_criteria,
        related_entities,
        required_skills,
        depends_on,
        last_attended,
        attention_count,
    })
}

/// Parse a CozoDB row into a TelosEvent struct.
///
/// Column layout: id, ts, telos_id, event_type, agent_id, description,
/// impact, source_memory, source_episode, metadata
fn parse_telos_event_row(row: &[DataValue]) -> Result<TelosEvent> {
    let id = parse_uuid(&row[0])?;
    let ts = row[1].get_int().unwrap_or(0);
    let telos_id = parse_uuid(&row[2])?;
    let event_type = parse_string(&row[3]);
    let agent_id = parse_string(&row[4]);
    let description = parse_string(&row[5]);
    let impact = row[6].get_float().unwrap_or(0.0);
    let source_memory = if row[7] == DataValue::Null {
        None
    } else {
        Some(parse_uuid(&row[7])?)
    };
    let source_episode = if row[8] == DataValue::Null {
        None
    } else {
        Some(parse_uuid(&row[8])?)
    };
    let metadata_str = parse_json_string(&row[9]);
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_str).unwrap_or(serde_json::Value::Object(Default::default()));

    Ok(TelosEvent {
        id,
        ts,
        telos_id,
        event_type,
        agent_id,
        description,
        impact,
        source_memory,
        source_episode,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_mem_store() {
        let store = CozoStore::open_mem(128).unwrap();
        assert_eq!(store.dim(), 128);
    }

    #[test]
    fn test_insert_and_get_memory() {
        let store = CozoStore::open_mem(4).unwrap();

        let memory = Memory::new("test.note", "Hello, world!", vec![0.1, 0.2, 0.3, 0.4]);
        let id = memory.id;

        store.insert_memory(&memory).unwrap();

        let count = store.count_memories().unwrap();
        assert_eq!(count, 1);

        let retrieved = store.get_memory(id).unwrap();
        assert!(retrieved.is_some(), "should find memory with id {}", id);
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.content, "Hello, world!");
        assert_eq!(retrieved.kind, "test.note");
    }

    #[test]
    fn test_count_memories() {
        let store = CozoStore::open_mem(4).unwrap();
        assert_eq!(store.count_memories().unwrap(), 0);

        store
            .insert_memory(&Memory::new("a", "one", vec![0.1, 0.2, 0.3, 0.4]))
            .unwrap();
        store
            .insert_memory(&Memory::new("b", "two", vec![0.5, 0.6, 0.7, 0.8]))
            .unwrap();

        assert_eq!(store.count_memories().unwrap(), 2);
    }

    #[test]
    fn test_get_nonexistent_memory() {
        let store = CozoStore::open_mem(4).unwrap();
        let result = store.get_memory(uuid::Uuid::now_v7()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_upsert_association() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = uuid::Uuid::now_v7();
        let b = uuid::Uuid::now_v7();
        let now = crate::types::memory::now_ms();

        // First upsert — should create
        store.upsert_association(a, b, 0.1, now).unwrap();

        // Second upsert — should update
        store.upsert_association(a, b, 0.1, now).unwrap();
    }

    #[test]
    fn test_record_and_compute_activations() {
        let store = CozoStore::open_mem(4).unwrap();
        let mem = Memory::new("test", "hello", vec![0.1, 0.2, 0.3, 0.4]);
        let id = mem.id;
        store.insert_memory(&mem).unwrap();

        store.record_accesses(&[id], "agent", "recall").unwrap();

        let now = crate::types::memory::now_ms();
        let acts = store.compute_activations(&[id], 86_400_000.0, now).unwrap();
        assert!(!acts.is_empty(), "should compute activation");
    }

    // ── Phase 3 Tests ──

    #[test]
    fn test_insert_and_find_fact() {
        let store = CozoStore::open_mem(4).unwrap();

        // Create entities first
        let alice = crate::types::entity::Entity {
            id: uuid::Uuid::now_v7(),
            name: "Alice".to_string(),
            entity_type: "person".to_string(),
            namespace: String::new(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            properties: Default::default(),
            mention_count: 1,
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: vec![],
        };
        let acme = crate::types::entity::Entity {
            id: uuid::Uuid::now_v7(),
            name: "Acme".to_string(),
            entity_type: "organization".to_string(),
            namespace: String::new(),
            embedding: vec![0.5, 0.6, 0.7, 0.8],
            properties: Default::default(),
            mention_count: 1,
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: vec![],
        };
        store.insert_entity(&alice).unwrap();
        store.insert_entity(&acme).unwrap();

        let fact = crate::types::fact::Fact::with_entity(alice.id, "works_at", acme.id);
        store.insert_fact(&fact).unwrap();

        let facts = store.find_facts_by_entity(alice.id).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "works_at");
        assert_eq!(facts[0].temporal_status, "current");
    }

    #[test]
    fn test_find_contradictions_detects_conflict() {
        let store = CozoStore::open_mem(4).unwrap();

        let alice_id = uuid::Uuid::now_v7();

        // Two facts with same predicate, different values → contradiction
        let mut f1 = crate::types::fact::Fact::with_value(alice_id, "role", "engineer");
        f1.temporal_status = "current".to_string();
        store.insert_fact(&f1).unwrap();

        let mut f2 = crate::types::fact::Fact::with_value(alice_id, "role", "manager");
        f2.temporal_status = "current".to_string();
        store.insert_fact(&f2).unwrap();

        let contradictions = store.find_contradictions(&[alice_id]).unwrap();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].predicate, "role");
    }

    #[test]
    fn test_find_contradictions_ignores_past() {
        let store = CozoStore::open_mem(4).unwrap();

        let alice_id = uuid::Uuid::now_v7();

        let mut f1 = crate::types::fact::Fact::with_value(alice_id, "role", "engineer");
        f1.temporal_status = "past".to_string();
        store.insert_fact(&f1).unwrap();

        let mut f2 = crate::types::fact::Fact::with_value(alice_id, "role", "manager");
        f2.temporal_status = "current".to_string();
        store.insert_fact(&f2).unwrap();

        let contradictions = store.find_contradictions(&[alice_id]).unwrap();
        assert!(
            contradictions.is_empty(),
            "past vs current should not be a contradiction"
        );
    }

    #[test]
    fn test_find_contradictions_no_false_positives() {
        let store = CozoStore::open_mem(4).unwrap();

        let alice_id = uuid::Uuid::now_v7();

        // Different predicates — not a contradiction
        let f1 = crate::types::fact::Fact::with_value(alice_id, "role", "engineer");
        store.insert_fact(&f1).unwrap();

        let f2 = crate::types::fact::Fact::with_value(alice_id, "location", "NYC");
        store.insert_fact(&f2).unwrap();

        let contradictions = store.find_contradictions(&[alice_id]).unwrap();
        assert!(
            contradictions.is_empty(),
            "different predicates should not contradict"
        );
    }

    #[test]
    fn test_collect_entity_ids_for_memories() {
        let store = CozoStore::open_mem(4).unwrap();

        let mem = Memory::new("test", "Alice works at Acme", vec![0.1, 0.2, 0.3, 0.4]);
        let mem_id = mem.id;
        store.insert_memory(&mem).unwrap();

        let entity = crate::types::entity::Entity {
            id: uuid::Uuid::now_v7(),
            name: "Alice".to_string(),
            entity_type: "person".to_string(),
            namespace: String::new(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            properties: Default::default(),
            mention_count: 1,
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: vec![],
        };
        let eid = entity.id;
        store.insert_entity(&entity).unwrap();
        store
            .link_entity_to_memory(eid, mem_id, "mentioned", 1.0)
            .unwrap();

        let entity_ids = store.collect_entity_ids_for_memories(&[mem_id]).unwrap();
        assert_eq!(entity_ids.len(), 1);
        assert_eq!(entity_ids[0], eid);
    }

    // ── Telos Tests ──

    #[test]
    fn test_telos_insert_and_get() {
        let store = CozoStore::open_mem(4).unwrap();

        let telos = crate::types::telos::Telos::new(
            "Ship Q3 deck",
            "Prepare the investor deck for Q3",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        let id = telos.id;

        store.insert_telos(&telos).unwrap();

        let retrieved = store.get_telos(id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.title, "Ship Q3 deck");
        assert_eq!(retrieved.description, "Prepare the investor deck for Q3");
        assert_eq!(retrieved.status, crate::types::telos::TelosStatus::Active);
        assert_eq!(retrieved.depth, 0);
        assert_eq!(retrieved.progress, 0.0);
    }

    #[test]
    fn test_telos_json_criteria_roundtrip() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut telos = crate::types::telos::Telos::new(
            "Test goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        telos.success_criteria = vec![
            crate::types::telos::SuccessCriterion {
                id: "a".to_string(),
                description: "First".to_string(),
                met: false,
            },
            crate::types::telos::SuccessCriterion {
                id: "b".to_string(),
                description: "Second".to_string(),
                met: true,
            },
        ];
        let id = telos.id;
        store.insert_telos(&telos).unwrap();

        let retrieved = store.get_telos(id).unwrap().unwrap();
        assert_eq!(retrieved.success_criteria.len(), 2, "criteria should round-trip through CozoDB");
        assert_eq!(retrieved.success_criteria[0].id, "a");
        assert!(!retrieved.success_criteria[0].met);
        assert_eq!(retrieved.success_criteria[1].id, "b");
        assert!(retrieved.success_criteria[1].met);
    }

    #[test]
    fn test_telos_get_nonexistent() {
        let store = CozoStore::open_mem(4).unwrap();
        let result = store.get_telos(uuid::Uuid::now_v7()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_telos_update_status() {
        let store = CozoStore::open_mem(4).unwrap();

        let telos = crate::types::telos::Telos::new(
            "Test goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        let id = telos.id;
        store.insert_telos(&telos).unwrap();

        store.update_telos_status(id, "completed").unwrap();
        let updated = store.get_telos(id).unwrap().unwrap();
        assert_eq!(updated.status, crate::types::telos::TelosStatus::Completed);
        assert!(updated.completed_at.is_some());
    }

    #[test]
    fn test_telos_update_progress() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut telos = crate::types::telos::Telos::new(
            "Test goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        telos.stalled_since = Some(crate::types::memory::now_ms());
        let id = telos.id;
        store.insert_telos(&telos).unwrap();

        store.update_telos_progress(id, 0.5).unwrap();
        let updated = store.get_telos(id).unwrap().unwrap();
        assert!((updated.progress - 0.5).abs() < f64::EPSILON);
        // stalled_since should be cleared on progress
        assert!(updated.stalled_since.is_none());
    }

    #[test]
    fn test_telos_update_owner() {
        let store = CozoStore::open_mem(4).unwrap();

        let telos = crate::types::telos::Telos::new(
            "Test goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        let id = telos.id;
        store.insert_telos(&telos).unwrap();

        store.update_telos_owner(id, "agent-2").unwrap();
        let updated = store.get_telos(id).unwrap().unwrap();
        assert_eq!(updated.owner, "agent-2");
    }

    #[test]
    fn test_telos_list_active() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut t1 = crate::types::telos::Telos::new(
            "Goal A",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        t1.namespace = "test".to_string();
        t1.priority = 0.9;

        let mut t2 = crate::types::telos::Telos::new(
            "Goal B",
            "",
            vec![0.5, 0.6, 0.7, 0.8],
            "agent-1",
            "user",
        );
        t2.namespace = "test".to_string();
        t2.priority = 0.3;

        // A completed telos should not appear
        let mut t3 = crate::types::telos::Telos::new(
            "Goal C",
            "",
            vec![0.2, 0.3, 0.4, 0.5],
            "agent-1",
            "user",
        );
        t3.namespace = "test".to_string();
        t3.status = crate::types::telos::TelosStatus::Completed;

        store.insert_telos(&t1).unwrap();
        store.insert_telos(&t2).unwrap();
        store.insert_telos(&t3).unwrap();

        let active = store.list_active_telos("test", 10).unwrap();
        assert_eq!(active.len(), 2);
        // Should be sorted by priority descending
        assert_eq!(active[0].title, "Goal A");
        assert_eq!(active[1].title, "Goal B");
    }

    #[test]
    fn test_telos_children() {
        let store = CozoStore::open_mem(4).unwrap();

        let parent = crate::types::telos::Telos::new(
            "Parent goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        let parent_id = parent.id;
        store.insert_telos(&parent).unwrap();

        let mut child = crate::types::telos::Telos::new(
            "Child goal",
            "",
            vec![0.5, 0.6, 0.7, 0.8],
            "agent-1",
            "decomposition",
        );
        child.parent = Some(parent_id);
        child.depth = 1;
        store.insert_telos(&child).unwrap();

        let children = store.get_children_telos(parent_id).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].title, "Child goal");
    }

    #[test]
    fn test_telos_embedding_search() {
        let store = CozoStore::open_mem(4).unwrap();

        let t1 = crate::types::telos::Telos::new(
            "Ship Q3 deck",
            "",
            vec![1.0, 0.0, 0.0, 0.0],
            "agent-1",
            "user",
        );
        let t2 = crate::types::telos::Telos::new(
            "Reduce churn",
            "",
            vec![0.0, 1.0, 0.0, 0.0],
            "agent-1",
            "user",
        );
        store.insert_telos(&t1).unwrap();
        store.insert_telos(&t2).unwrap();

        let results = store
            .find_telos_by_embedding(&[0.9, 0.1, 0.0, 0.0], 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        // Closest should be "Ship Q3 deck"
        assert_eq!(results[0].0.title, "Ship Q3 deck");
    }

    #[test]
    fn test_telos_events() {
        let store = CozoStore::open_mem(4).unwrap();

        let telos = crate::types::telos::Telos::new(
            "Test goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        let telos_id = telos.id;
        store.insert_telos(&telos).unwrap();

        let mut event = crate::types::telos::TelosEvent::new(telos_id, "created");
        event.agent_id = "agent-1".to_string();
        event.description = "Goal created by user".to_string();
        event.ts = 1000; // explicit earlier timestamp
        store.insert_telos_event(&event).unwrap();

        let mut event2 = crate::types::telos::TelosEvent::new(telos_id, "progress");
        event2.agent_id = "agent-1".to_string();
        event2.impact = 0.3;
        event2.ts = 2000; // explicit later timestamp
        store.insert_telos_event(&event2).unwrap();

        let events = store.get_telos_events(telos_id, 10).unwrap();
        assert_eq!(events.len(), 2);
        // Most recent first (sorted by -ts)
        assert_eq!(events[0].event_type, "progress");
        assert_eq!(events[1].event_type, "created");
    }

    #[test]
    fn test_telos_attention() {
        let store = CozoStore::open_mem(4).unwrap();

        let telos = crate::types::telos::Telos::new(
            "Test goal",
            "",
            vec![0.1, 0.2, 0.3, 0.4],
            "agent-1",
            "user",
        );
        let telos_id = telos.id;
        store.insert_telos(&telos).unwrap();

        store
            .start_telos_attention(telos_id, "agent-1", None)
            .unwrap();

        // Verify last_attended and attention_count updated
        let updated = store.get_telos(telos_id).unwrap().unwrap();
        assert!(updated.last_attended > 0);
        assert_eq!(updated.attention_count, 1);
    }
}
