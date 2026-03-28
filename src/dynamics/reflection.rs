//! Reflection pipeline — extracts insights from accumulated experience.
//!
//! Triggered when accumulated unresolved surprise exceeds a threshold.
//!
//! Pipeline:
//! 1. Gather recent closed episodes (since last reflection)
//! 2. For each cluster, LLM extracts: facts, entity updates, abstractions
//! 3. Link extracted knowledge to source episodes
//! 4. Mark surprise entries as resolved

use std::collections::BTreeMap;
use std::sync::Arc;

use cozo::DataValue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::services::traits::{Embedder, LlmService, Message};
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// Result of a reflection cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResult {
    pub reflection_id: Uuid,
    pub episodes_reviewed: Vec<Uuid>,
    pub facts_created: usize,
    pub facts_updated: usize,
    pub entities_created: usize,
    pub abstractions_created: usize,
    pub causal_edges_proposed: usize,
    pub duration_ms: i64,
}

/// An abstraction extracted during reflection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Abstraction {
    pub id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub confidence: f64,
    pub evidence_count: i64,
    pub source_episodes: Vec<Uuid>,
    pub category: String,
}

/// Run the reflection pipeline.
///
/// 1. Check if accumulated surprise exceeds the threshold
/// 2. Gather recent closed episodes since last reflection
/// 3. Summarize episodes via LLM and extract patterns
/// 4. Store abstractions and resolve surprise
pub async fn run_reflection(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    surprise_threshold: f64,
) -> Result<Option<ReflectionResult>> {
    let start = now_ms();

    // 1. Check accumulated surprise
    let accumulated = super::surprise::accumulated_unresolved_surprise(store)?;
    if accumulated < surprise_threshold {
        return Ok(None);
    }

    // 2. Gather recent closed episodes since last reflection
    let episodes = get_unreflected_episodes(store)?;
    if episodes.is_empty() {
        // Resolve surprise even if no episodes (prevents infinite buildup)
        super::surprise::resolve_all_surprise(store)?;
        return Ok(None);
    }

    // 2b. Prioritize episodes by telos alignment
    // Episodes more relevant to active goals get processed first
    let episodes = prioritize_episodes_by_telos(store, episodes);

    // 3. Get episode summaries and memories for LLM extraction
    let episode_summaries = gather_episode_content(store, &episodes)?;

    // 4. LLM extracts patterns/abstractions
    let abstractions = extract_abstractions(llm, embedder, &episode_summaries, &episodes).await?;

    // 4b. Extract facts and entities from episode content
    let (facts_created, facts_updated, entities_created) =
        extract_and_store_facts_entities(store, llm, embedder, &episode_summaries).await?;

    // 4c. LLM proposes causal relationships from episode content (Source 2)
    let causal_edges_proposed =
        propose_causal_edges(store, llm, &episode_summaries).await.unwrap_or(0);

    // 5. Store abstractions
    let abstractions_created = abstractions.len();
    for abs in &abstractions {
        store_abstraction(store, abs)?;
    }

    // 6. Log the reflection
    let reflection_id = Uuid::now_v7();
    log_reflection(
        store,
        reflection_id,
        &episodes,
        facts_created,
        facts_updated,
        entities_created,
        abstractions_created,
        now_ms() - start,
    )?;

    // 7. Resolve all accumulated surprise
    super::surprise::resolve_all_surprise(store)?;

    let duration_ms = now_ms() - start;

    Ok(Some(ReflectionResult {
        reflection_id,
        episodes_reviewed: episodes,
        facts_created,
        facts_updated,
        entities_created,
        abstractions_created,
        causal_edges_proposed,
        duration_ms,
    }))
}

/// Extract facts and entities from episode summaries via LLM.
///
/// Uses a structured prompt to extract subject-predicate-object triples and
/// entity mentions. Returns (facts_created, facts_updated, entities_created).
async fn extract_and_store_facts_entities(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    summaries: &[String],
) -> Result<(usize, usize, usize)> {
    if summaries.is_empty() {
        return Ok((0, 0, 0));
    }

    let combined = summaries.join("\n---\n");
    let prompt = format!(
        "Analyze these episodes and extract structured knowledge.\n\
         Output ONLY valid JSON with this schema:\n\
         {{\"facts\": [{{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\", \"confidence\": 0.8}}], \
         \"entities\": [{{\"name\": \"...\", \"type\": \"...\"}}]}}\n\n\
         Rules:\n\
         - subject and object should be entity names\n\
         - predicate should be a verb phrase (e.g. \"works_at\", \"manages\")\n\
         - type should be one of: person, organization, location, concept, event\n\
         - confidence between 0.0 and 1.0\n\n\
         Episodes:\n{combined}"
    );

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "You extract structured facts and entities from text. \
                              Respond ONLY with valid JSON."
                        .into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            2048,
        )
        .await
        .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

    // Parse the LLM response
    #[derive(serde::Deserialize, Default)]
    struct ExtractedKnowledge {
        #[serde(default)]
        facts: Vec<ExtractedFact>,
        #[serde(default)]
        entities: Vec<ExtractedEntityInfo>,
    }

    #[derive(serde::Deserialize)]
    struct ExtractedFact {
        subject: String,
        predicate: String,
        object: String,
        #[serde(default = "default_confidence")]
        confidence: f64,
    }

    #[derive(serde::Deserialize)]
    struct ExtractedEntityInfo {
        name: String,
        #[serde(rename = "type")]
        entity_type: String,
    }

    fn default_confidence() -> f64 {
        0.7
    }

    let content = response.content.trim();
    let parsed: ExtractedKnowledge = serde_json::from_str(content)
        .or_else(|_| {
            // Try extracting JSON from markdown
            if let Some(start) = content.find('{') {
                if let Some(end) = content.rfind('}') {
                    return serde_json::from_str(&content[start..=end]);
                }
            }
            Ok(ExtractedKnowledge::default())
        })
        .unwrap_or_default();

    let mut facts_created = 0usize;
    let mut facts_updated = 0usize;
    let mut entities_created = 0usize;

    // Store entities first (so facts can reference them)
    for entity_info in &parsed.entities {
        if entity_info.name.is_empty() {
            continue;
        }
        // Check if entity already exists
        if store.find_entity_by_name(&entity_info.name, "")?.is_some() {
            continue;
        }

        // Create embedding for entity
        let emb = embedder
            .embed(&[entity_info.name.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

        let entity = crate::types::entity::Entity {
            id: Uuid::now_v7(),
            name: entity_info.name.clone(),
            entity_type: entity_info.entity_type.clone(),
            namespace: String::new(),
            embedding: emb.into_iter().next().unwrap_or_default(),
            properties: serde_json::Map::new(),
            mention_count: 1,
            confidence: 0.7,
            provenance: "reflected".to_string(),
            source_ids: vec![],
        };
        store.insert_entity(&entity)?;
        entities_created += 1;
    }

    // Store facts
    for extracted_fact in &parsed.facts {
        if extracted_fact.subject.is_empty() || extracted_fact.predicate.is_empty() {
            continue;
        }

        // Find or skip subject entity
        let subject = match store.find_entity_by_name(&extracted_fact.subject, "")? {
            Some(e) => e,
            None => continue,
        };

        // Try to resolve object as entity
        let object_entity = store.find_entity_by_name(&extracted_fact.object, "")?;

        // Check for existing fact to reinforce
        let existing = store.find_matching_fact(
            subject.id,
            &extracted_fact.predicate,
            object_entity.as_ref().map(|e| e.id),
            if object_entity.is_none() {
                Some(&extracted_fact.object)
            } else {
                None
            },
            "",
        )?;

        if let Some(existing_fact) = existing {
            store.reinforce_fact(&existing_fact, extracted_fact.confidence, &[])?;
            facts_updated += 1;
        } else {
            let fact = if let Some(obj_entity) = object_entity {
                let mut f = crate::types::fact::Fact::with_entity(
                    subject.id,
                    &extracted_fact.predicate,
                    obj_entity.id,
                );
                f.confidence = extracted_fact.confidence;
                f.provenance = "reflected".to_string();
                f
            } else {
                let mut f = crate::types::fact::Fact::with_value(
                    subject.id,
                    &extracted_fact.predicate,
                    &extracted_fact.object,
                );
                f.confidence = extracted_fact.confidence;
                f.provenance = "reflected".to_string();
                f
            };
            store.insert_fact(&fact)?;
            facts_created += 1;
        }
    }

    Ok((facts_created, facts_updated, entities_created))
}

/// Get episode IDs that haven't been reflected on yet.
fn get_unreflected_episodes(store: &CozoStore) -> Result<Vec<Uuid>> {
    // Get the timestamp of the last reflection
    let last_reflection_ts = store
        .run_query(
            r#"?[max(ts)] := *reflections{ts}"#,
            BTreeMap::new(),
        )?;

    let since_ts = if last_reflection_ts.rows.is_empty() {
        0i64
    } else {
        last_reflection_ts.rows[0][0].get_int().unwrap_or(0)
    };

    let mut params = BTreeMap::new();
    params.insert("since".into(), DataValue::from(since_ts));

    let result = store.run_query(
        r#"?[id] := *episodes{id, ended_at, started_at},
            not is_null(ended_at),
            started_at > $since
           :limit 50"#,
        params,
    )?;

    result
        .rows
        .iter()
        .map(|row| crate::store::cozo::parse_uuid_pub(&row[0]))
        .collect()
}

/// Prioritize episodes by relevance to active telos goals.
///
/// Computes cosine similarity between episode memories and active telos embeddings,
/// weighted by telos priority. Episodes more relevant to active goals sort first.
/// Falls back to original order if no telos are active or on error.
fn prioritize_episodes_by_telos(store: &CozoStore, episodes: Vec<Uuid>) -> Vec<Uuid> {
    // Get active telos embeddings
    let active_telos = match store.list_active_telos("default", 10) {
        Ok(t) => t,
        Err(_) => return episodes,
    };

    if active_telos.is_empty() {
        return episodes;
    }

    // Score each episode by max telos alignment
    let mut scored: Vec<(Uuid, f64)> = episodes
        .iter()
        .map(|ep_id| {
            let score = episode_telos_score(store, *ep_id, &active_telos);
            (*ep_id, score)
        })
        .collect();

    // Sort by telos alignment score (highest first)
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

/// Compute the telos alignment score for a single episode.
///
/// Returns the max across all active telos of:
/// cos_similarity(episode_memory_emb, telos_emb) × telos.priority
fn episode_telos_score(
    store: &CozoStore,
    episode_id: Uuid,
    active_telos: &[crate::types::telos::Telos],
) -> f64 {
    // Get the episode's memory embeddings
    let mut params = BTreeMap::new();
    params.insert("ep_id".into(), DataValue::from(episode_id.to_string()));

    let memories = match store.run_query(
        r#"?[embedding] :=
            *episode_memories{episode_id, memory_id},
            episode_id = to_uuid($ep_id),
            *memories{id, embedding},
            id = memory_id
           :limit 5"#,
        params,
    ) {
        Ok(r) => r,
        Err(_) => return 0.0,
    };

    if memories.rows.is_empty() {
        return 0.0;
    }

    // Compute max alignment across all telos
    let mut max_score = 0.0;
    for row in &memories.rows {
        let mem_emb: Vec<f32> = match &row[0] {
            DataValue::List(vals) => vals
                .iter()
                .filter_map(|v| v.get_float().map(|f| f as f32))
                .collect(),
            _ => continue,
        };

        if mem_emb.is_empty() {
            continue;
        }

        for telos in active_telos {
            if telos.embedding.is_empty() {
                continue;
            }
            let sim = crate::store::cozo::cosine_similarity(&mem_emb, &telos.embedding);
            let score = sim * telos.priority;
            if score > max_score {
                max_score = score;
            }
        }
    }

    max_score
}

/// Gather content from episode memories for LLM processing.
fn gather_episode_content(
    store: &CozoStore,
    episode_ids: &[Uuid],
) -> Result<Vec<String>> {
    let mut summaries = Vec::new();

    for ep_id in episode_ids {
        let mut params = BTreeMap::new();
        params.insert("ep_id".into(), DataValue::from(ep_id.to_string()));

        // Get episode summary
        let ep_result = store.run_query(
            r#"?[summary, episode_type, outcome] :=
                *episodes{id, summary, episode_type, outcome},
                id = to_uuid($ep_id)"#,
            params.clone(),
        )?;

        let summary = if !ep_result.rows.is_empty() {
            let s = ep_result.rows[0][0].get_str().unwrap_or("").to_string();
            let ep_type = ep_result.rows[0][1].get_str().unwrap_or("session").to_string();
            let outcome = ep_result.rows[0][2]
                .get_str()
                .unwrap_or("unknown")
                .to_string();
            format!("[{ep_type}/{outcome}] {s}")
        } else {
            continue;
        };

        // Get memories in this episode
        let mem_result = store.run_query(
            r#"?[content] :=
                *episode_memories{episode_id, memory_id},
                *memories{id: memory_id, content},
                episode_id = to_uuid($ep_id)
               :limit 20"#,
            params,
        )?;

        let memory_texts: Vec<String> = mem_result
            .rows
            .iter()
            .map(|r| r[0].get_str().unwrap_or("").to_string())
            .collect();

        let combined = if memory_texts.is_empty() {
            summary
        } else {
            format!("{summary}\nMemories:\n- {}", memory_texts.join("\n- "))
        };

        summaries.push(combined);
    }

    Ok(summaries)
}

/// Use LLM to extract abstractions from episode summaries.
async fn extract_abstractions(
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    summaries: &[String],
    episode_ids: &[Uuid],
) -> Result<Vec<Abstraction>> {
    if summaries.is_empty() {
        return Ok(Vec::new());
    }

    let combined = summaries.join("\n---\n");
    let prompt = format!(
        "Analyze these agent episodes and extract general patterns or abstractions. \
         For each pattern, provide a single clear sentence.\n\
         Output one pattern per line, nothing else.\n\n{combined}"
    );

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "You extract general patterns from agent experiences. \
                              Output one pattern per line, no numbering or bullets."
                        .into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            2048,
        )
        .await
        .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

    let patterns: Vec<&str> = response
        .content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if patterns.is_empty() {
        return Ok(Vec::new());
    }

    // Embed all patterns
    let pattern_strs: Vec<&str> = patterns.iter().copied().collect();
    let embeddings = embedder
        .embed(&pattern_strs)
        .await
        .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

    let mut abstractions = Vec::new();
    for (i, pattern) in patterns.iter().enumerate() {
        if i >= embeddings.len() {
            break;
        }
        abstractions.push(Abstraction {
            id: Uuid::now_v7(),
            content: pattern.to_string(),
            embedding: embeddings[i].clone(),
            confidence: 0.5,
            evidence_count: episode_ids.len() as i64,
            source_episodes: episode_ids.to_vec(),
            category: "pattern".to_string(),
        });
    }

    Ok(abstractions)
}

/// Use LLM to propose causal relationships from episode content (Source 2).
///
/// The LLM identifies 0-3 causal relationships per reflection cycle.
/// These are stored with mechanism: LlmProposed and low initial confidence (0.3).
/// NOTEARS periodic validation will later confirm or reduce their confidence.
async fn propose_causal_edges(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    summaries: &[String],
) -> Result<usize> {
    if summaries.is_empty() {
        return Ok(0);
    }

    let combined = summaries.join("\n---\n");
    let prompt = format!(
        "Given these events:\n{combined}\n\n\
         Identify 0-3 causal relationships (A caused B).\n\
         Only include causation, not correlation.\n\
         Return ONLY valid JSON array: \
         [{{\"cause\": \"entity or concept name\", \"effect\": \"entity or concept name\", \
         \"mechanism\": \"brief explanation\", \"confidence\": 0.X}}]\n\
         If no causal relationships are evident, return []"
    );

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "You identify causal relationships from agent experiences. \
                              Return ONLY a JSON array. Be conservative — only include \
                              causation, not correlation."
                        .into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            1024,
        )
        .await
        .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

    #[derive(serde::Deserialize)]
    struct ProposedCausalEdge {
        cause: String,
        effect: String,
        #[serde(default)]
        _mechanism: String,
        #[serde(default = "default_causal_confidence")]
        confidence: f64,
    }

    fn default_causal_confidence() -> f64 {
        0.3
    }

    let content = response.content.trim();
    let proposed: Vec<ProposedCausalEdge> = serde_json::from_str(content)
        .or_else(|_| {
            // Try extracting JSON array from markdown
            if let Some(start) = content.find('[') {
                if let Some(end) = content.rfind(']') {
                    return serde_json::from_str(&content[start..=end]);
                }
            }
            Ok(Vec::new())
        })
        .unwrap_or_default();

    let mut stored = 0;
    let now = crate::types::memory::now_ms();

    for proposed_edge in &proposed {
        if proposed_edge.cause.is_empty() || proposed_edge.effect.is_empty() {
            continue;
        }

        // Resolve cause and effect to entity IDs
        let cause_entity = store.find_entity_by_name(&proposed_edge.cause, "")?;
        let effect_entity = store.find_entity_by_name(&proposed_edge.effect, "")?;

        if let (Some(cause), Some(effect)) = (cause_entity, effect_entity) {
            let edge = crate::causal::graph::CausalEdge {
                cause_id: cause.id,
                effect_id: effect.id,
                causal_strength: proposed_edge.confidence.min(0.5), // cap initial strength
                observations: 1,
                last_observed: now,
                mechanism: crate::causal::graph::CausalMechanism::LlmProposed,
                confidence: 0.3, // low initial confidence — NOTEARS will validate
                namespace: String::new(),
            };

            if crate::causal::graph::accumulate_causal_edge(store, &edge).is_ok() {
                stored += 1;
            }
        }
    }

    Ok(stored)
}

/// Store an abstraction in CozoDB.
fn store_abstraction(store: &CozoStore, abs: &Abstraction) -> Result<()> {
    let embedding_vals: Vec<DataValue> = abs
        .embedding
        .iter()
        .map(|&v| DataValue::from(v as f64))
        .collect();

    let source_ep_vals: Vec<DataValue> = abs
        .source_episodes
        .iter()
        .map(|id| DataValue::from(id.to_string()))
        .collect();

    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(abs.id.to_string()));
    params.insert("content".into(), DataValue::from(abs.content.as_str()));
    params.insert("embedding".into(), DataValue::List(embedding_vals));
    params.insert("confidence".into(), DataValue::from(abs.confidence));
    params.insert(
        "evidence_count".into(),
        DataValue::from(abs.evidence_count),
    );
    params.insert(
        "source_episodes".into(),
        DataValue::List(source_ep_vals),
    );
    params.insert("category".into(), DataValue::from(abs.category.as_str()));

    store.run_script(
        concat!(
            "?[id, valid_at, content, embedding, confidence, evidence_count, ",
            "source_episodes, category] <- ",
            "[[$id, 'ASSERT', $content, $embedding, $confidence, $evidence_count, ",
            "$source_episodes, $category]] ",
            ":put abstractions {id, valid_at => content, embedding, confidence, ",
            "evidence_count, source_episodes, category}",
        ),
        params,
    )?;

    Ok(())
}

/// Log a reflection event.
fn log_reflection(
    store: &CozoStore,
    id: Uuid,
    episodes: &[Uuid],
    facts_created: usize,
    facts_updated: usize,
    entities_created: usize,
    abstractions_created: usize,
    duration_ms: i64,
) -> Result<()> {
    let ep_vals: Vec<DataValue> = episodes
        .iter()
        .map(|id| DataValue::from(id.to_string()))
        .collect();

    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(id.to_string()));
    params.insert("episodes_reviewed".into(), DataValue::List(ep_vals));
    params.insert(
        "facts_created".into(),
        DataValue::from(facts_created as i64),
    );
    params.insert(
        "facts_updated".into(),
        DataValue::from(facts_updated as i64),
    );
    params.insert(
        "entities_created".into(),
        DataValue::from(entities_created as i64),
    );
    params.insert(
        "abstractions_created".into(),
        DataValue::from(abstractions_created as i64),
    );
    params.insert("duration_ms".into(), DataValue::from(duration_ms));

    store.run_script(
        concat!(
            "?[id, ts, episodes_reviewed, facts_created, facts_updated, ",
            "entities_created, abstractions_created, duration_ms] <- ",
            "[[$id, now(), $episodes_reviewed, $facts_created, $facts_updated, ",
            "$entities_created, $abstractions_created, $duration_ms]] ",
            ":put reflections {id, ts => episodes_reviewed, facts_created, ",
            "facts_updated, entities_created, abstractions_created, duration_ms}",
        ),
        params,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::mock::{MockEmbedder, MockLlm};

    #[test]
    fn test_get_unreflected_episodes_empty() {
        let store = CozoStore::open_mem(4).unwrap();
        let eps = get_unreflected_episodes(&store).unwrap();
        assert!(eps.is_empty());
    }

    #[tokio::test]
    async fn test_reflection_skips_when_below_threshold() {
        let store = CozoStore::open_mem(128).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(128));

        let result = run_reflection(&store, &llm, &embedder, 5.0).await.unwrap();
        assert!(result.is_none(), "should skip when no surprise accumulated");
    }

    #[tokio::test]
    async fn test_reflection_resolves_surprise_even_without_episodes() {
        let store = CozoStore::open_mem(128).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(128));

        // Log some surprise
        super::super::surprise::log_surprise(&store, 10.0, "test", None, "test").unwrap();

        let result = run_reflection(&store, &llm, &embedder, 5.0).await.unwrap();
        // Should return None (no episodes) but surprise should be resolved
        assert!(result.is_none());

        let remaining = super::super::surprise::accumulated_unresolved_surprise(&store).unwrap();
        assert_eq!(remaining, 0.0);
    }
}
