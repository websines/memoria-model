//! Query planner — classifies queries and dispatches to the appropriate recall strategy.
//!
//! Instead of always using semantic (HNSW) search, the planner examines the query
//! text and selects structured entity lookup, temporal diff, episodic retrieval,
//! abstraction search, or semantic search — or a composite of multiple strategies.
//!
//! For complex queries (multiple clauses, conjunctions), the planner uses the LLM
//! to decompose them into sub-queries, each planned independently and combined
//! as a Composite strategy.

use crate::error::Result;
use crate::runtime::Memoria;
use crate::types::memory::CandidateMemory;
use crate::types::query::{AgentContext, RecallStrategy};
use uuid::Uuid;

/// Plan a recall strategy for the given query text.
///
/// For simple queries, uses lightweight heuristic classification.
/// For complex queries (conjunctions, multiple clauses), decomposes via LLM
/// into sub-queries, plans each independently, and combines as Composite.
pub async fn plan_query(
    memoria: &Memoria,
    text: &str,
    _ctx: &AgentContext,
) -> Result<RecallStrategy> {
    // Check if query is complex enough to warrant LLM decomposition
    if query_complexity(text) >= 2 {
        if let Ok(sub_queries) = decompose_query(&*memoria.llm, text).await {
            if sub_queries.len() > 1 {
                // Plan each sub-query independently, then combine
                let mut all_strategies = Vec::new();
                for sub_q in &sub_queries {
                    let sub_strategy = plan_single(memoria, sub_q).await?;
                    match sub_strategy {
                        RecallStrategy::Composite(subs) => all_strategies.extend(subs),
                        other => all_strategies.push(other),
                    }
                }
                // Deduplicate semantic strategies (keep only one embedding search)
                dedup_semantic(&mut all_strategies);
                if all_strategies.len() == 1 {
                    return Ok(all_strategies.into_iter().next().unwrap());
                }
                return Ok(RecallStrategy::Composite(all_strategies));
            }
        }
        // LLM decomposition failed or returned single query — fall through to heuristic
    }

    plan_single(memoria, text).await
}

/// Classify a query using LLM for more accurate strategy selection.
///
/// Returns the strategy type and any extracted entities/predicates.
/// Falls back to None on LLM failure, letting the heuristic path handle it.
async fn llm_classify_query(
    llm: &dyn crate::services::traits::LlmService,
    text: &str,
) -> Option<LlmQueryClassification> {
    let prompt = format!(
        concat!(
            "Classify this memory retrieval query. Return ONLY a JSON object:\n",
            r#"{{"strategy": "<type>", "entity": "<name or null>", "predicate": "<predicate or null>", "topic": "<topic or null>"}}"#,
            "\n\nStrategy types:\n",
            "- \"structured\": asks about a specific entity (person, org, project)\n",
            "- \"temporal\": asks about changes over time, recent events\n",
            "- \"episodic\": asks about past sessions, conversations, episodes\n",
            "- \"abstract\": asks about patterns, trends, best practices\n",
            "- \"semantic\": general knowledge retrieval\n\n",
            "Query: \"{}\""
        ),
        text
    );

    let response = llm
        .complete(
            &[crate::services::traits::Message {
                role: "user".to_string(),
                content: prompt,
            }],
            256,
        )
        .await
        .ok()?;

    let content = response.content.trim();
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    let json_str = &content[start..=end];

    serde_json::from_str::<LlmQueryClassification>(json_str).ok()
}

#[derive(serde::Deserialize)]
struct LlmQueryClassification {
    strategy: String,
    entity: Option<String>,
    predicate: Option<String>,
    topic: Option<String>,
}

/// Plan a single query using LLM classification with heuristic fallback.
///
/// Tries LLM classification first for accuracy; if LLM call fails or returns
/// an unrecognized strategy, falls back to keyword heuristics.
async fn plan_single(
    memoria: &Memoria,
    text: &str,
) -> Result<RecallStrategy> {
    // Try LLM classification first
    if let Some(classification) = llm_classify_query(&*memoria.llm, text).await {
        let query_embedding = memoria
            .embedder
            .embed(&[text])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?
            .into_iter()
            .next()
            .unwrap_or_default();

        let mut strategies = Vec::new();

        match classification.strategy.as_str() {
            "structured" => {
                if let Some(entity) = classification.entity.filter(|e| !e.is_empty()) {
                    strategies.push(RecallStrategy::Structured {
                        entity,
                        predicate: classification.predicate,
                    });
                }
            }
            "temporal" => {
                let now = crate::types::memory::now_ms();
                strategies.push(RecallStrategy::TemporalDiff {
                    since: now - 86_400_000, // default 24h
                    entity_filter: classification.entity,
                });
            }
            "episodic" => {
                strategies.push(RecallStrategy::Episodic {
                    time_range: None,
                    topic: classification.topic.or_else(|| Some(text.to_string())),
                });
            }
            "abstract" => {
                strategies.push(RecallStrategy::Abstract {
                    topic: classification.topic.unwrap_or_else(|| text.to_string()),
                });
            }
            _ => {} // "semantic" or unknown — just use embedding fallback
        }

        // Always include semantic as fallback/complement
        strategies.push(RecallStrategy::Semantic {
            embedding: query_embedding,
            filters: vec![],
        });

        return if strategies.len() == 1 {
            Ok(strategies.into_iter().next().unwrap())
        } else {
            Ok(RecallStrategy::Composite(strategies))
        };
    }

    // LLM classification failed — fall back to heuristic
    plan_single_heuristic(memoria, text).await
}

/// Heuristic query classification — examines the text to select a recall strategy.
///
/// Returns a `RecallStrategy` variant (possibly `Composite`) based on keyword patterns.
async fn plan_single_heuristic(
    memoria: &Memoria,
    text: &str,
) -> Result<RecallStrategy> {
    let lower = text.to_lowercase();

    // Detect structured entity queries: "Who is X?", "Where does X work?",
    // "What does X do?", "Tell me about X", "What is X's role?"
    let is_structured = lower.starts_with("who is ")
        || lower.starts_with("what is ")
        || lower.starts_with("tell me about ")
        || lower.starts_with("where does ")
        || lower.starts_with("where is ")
        || lower.starts_with("what does ")
        || lower.contains("'s role")
        || lower.contains("'s job")
        || lower.contains("works at")
        || lower.contains("works on")
        || lower.contains("works for")
        || lower.contains("entity ")
        || lower.contains("person ")
        || lower.contains("organization ");

    // Detect temporal queries: "since", "after", "what changed", "recent", "last week"
    let is_temporal = lower.contains("since ")
        || lower.contains("after ")
        || lower.contains("what changed")
        || lower.contains("has anything changed")
        || lower.contains("what's new")
        || lower.contains("what is new")
        || lower.contains("last week")
        || lower.contains("last month")
        || lower.starts_with("recent ");

    // Detect episodic queries: "last session", "episode", "conversation",
    // "what happened", "yesterday", "debug session", "last time"
    let is_episodic = lower.contains("last session")
        || lower.contains("episode")
        || lower.contains("conversation")
        || lower.contains("last time")
        || lower.contains("what happened")
        || lower.contains("yesterday")
        || lower.contains("debug session")
        || lower.contains("this morning")
        || lower.contains("earlier today");

    // Detect abstraction queries: "pattern", "trend", "generally", "usually",
    // "how do we", "how does the team", "best practice"
    let is_abstract = lower.contains("pattern")
        || lower.contains("trend")
        || lower.contains("generally")
        || lower.contains("usually")
        || lower.contains("abstraction")
        || lower.starts_with("how do we ")
        || lower.starts_with("how does the ")
        || lower.contains("best practice")
        || lower.contains("common approach")
        || lower.contains("typically");

    // Build strategies list
    let mut strategies = Vec::new();

    if is_structured {
        // Extract entity name from common patterns
        let entity_name = extract_entity_name(&lower);
        if let Some(name) = entity_name {
            strategies.push(RecallStrategy::Structured {
                entity: name,
                predicate: None,
            });
        }
    }

    if is_temporal {
        // Parse time range from query keywords
        let now = crate::types::memory::now_ms();
        let since = if lower.contains("last week") {
            now - 7 * 86_400_000
        } else if lower.contains("last month") {
            now - 30 * 86_400_000
        } else {
            // Default to last 24 hours
            now - 86_400_000
        };

        // Try to extract entity filter from temporal query
        // e.g. "Has anything changed about the API since last week?" → "Api"
        let entity_filter = extract_temporal_entity_filter(&lower);

        strategies.push(RecallStrategy::TemporalDiff {
            since,
            entity_filter,
        });
    }

    if is_episodic {
        strategies.push(RecallStrategy::Episodic {
            time_range: None,
            topic: Some(text.to_string()),
        });
    }

    if is_abstract {
        strategies.push(RecallStrategy::Abstract {
            topic: text.to_string(),
        });
    }

    // Always include semantic as fallback/complement
    let query_embeddings = memoria
        .embedder
        .embed(&[text])
        .await
        .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;
    let query_embedding = query_embeddings.into_iter().next().unwrap_or_default();

    strategies.push(RecallStrategy::Semantic {
        embedding: query_embedding,
        filters: vec![],
    });

    if strategies.len() == 1 {
        // Only semantic — return it directly
        Ok(strategies.into_iter().next().unwrap())
    } else {
        Ok(RecallStrategy::Composite(strategies))
    }
}

/// Execute a single recall strategy and return candidate memories.
pub fn execute_strategy(
    memoria: &Memoria,
    strategy: &RecallStrategy,
    ctx: &AgentContext,
    max_k: usize,
) -> Result<Vec<CandidateMemory>> {
    match strategy {
        RecallStrategy::Semantic {
            embedding, filters: _,
        } => {
            let cfg = memoria.config.load();
            let mut candidates = memoria.store.vector_search(
                embedding,
                &ctx.namespace,
                max_k,
                cfg.max_distance,
            )?;

            // Dual search: if projection is available, also search with projected query
            if cfg.dynamics.projection_enabled {
                if let Some(ref proj) = **memoria.projection.load() {
                    let projected = proj.project(embedding);
                    if let Ok(projected_candidates) = memoria.store.vector_search(
                        &projected,
                        &ctx.namespace,
                        max_k,
                        cfg.max_distance,
                    ) {
                        let mut seen: std::collections::HashSet<uuid::Uuid> =
                            candidates.iter().map(|c| c.memory.id).collect();
                        for pc in projected_candidates {
                            if seen.insert(pc.memory.id) {
                                candidates.push(pc);
                            }
                        }
                    }
                }
            }

            Ok(candidates)
        }

        RecallStrategy::Structured { entity, predicate } => {
            execute_structured(&memoria.store, entity, predicate.as_deref(), &ctx.namespace, max_k)
        }

        RecallStrategy::TemporalDiff {
            since,
            entity_filter,
        } => execute_temporal_diff(&memoria.store, *since, entity_filter.as_deref(), &ctx.namespace, max_k),

        RecallStrategy::Episodic { time_range, topic } => {
            execute_episodic(&memoria.store, time_range, topic.as_deref(), &ctx.namespace, max_k)
        }

        RecallStrategy::Abstract { topic } => {
            execute_abstract(&memoria.store, topic, &ctx.namespace, max_k)
        }

        RecallStrategy::Composite(strategies) => {
            let mut all = Vec::new();
            let mut seen = std::collections::HashSet::new();
            let per_strategy_k = max_k; // each sub-strategy gets full budget
            for sub in strategies {
                let mut candidates = execute_strategy(memoria, sub, ctx, per_strategy_k)?;
                // Deduplicate across strategies
                candidates.retain(|c| seen.insert(c.memory.id));
                all.extend(candidates);
            }
            Ok(all)
        }
    }
}

/// Structured entity lookup — find entity by name, get linked memories via entity_mentions.
fn execute_structured(
    store: &crate::store::CozoStore,
    entity_name: &str,
    predicate: Option<&str>,
    namespace: &str,
    max_k: usize,
) -> Result<Vec<CandidateMemory>> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let entity = store.find_entity_by_name(entity_name, namespace)?;
    let Some(entity) = entity else {
        return Ok(Vec::new());
    };

    // Get memories linked to this entity
    let mut params = BTreeMap::new();
    params.insert(
        "entity_id".into(),
        DataValue::from(entity.id.to_string()),
    );
    params.insert("limit".into(), DataValue::from(max_k as i64));

    let result = store.run_query(
        r#"?[id, kind, content, embedding, fields, namespace, pinned,
            expires_at, version, created_at, confidence, provenance, source_ids] :=
            *entity_mentions{entity_id, memory_id: id},
            entity_id = to_uuid($entity_id),
            *memories{id, kind, content, embedding, fields, namespace, pinned,
                      expires_at, version, created_at, confidence, provenance, source_ids}
        :limit $limit"#,
        params,
    )?;

    let mut candidates = Vec::new();
    for row in &result.rows {
        let memory = crate::store::cozo::parse_memory_row_pub(row)?;
        candidates.push(CandidateMemory {
            memory,
            distance: 0.0, // direct entity match — highest relevance
            activation: None,
            hebbian_weight: None,
            pagerank: None,
            precision: None,
            telos_boost: None,
        });
    }

    // If predicate filter requested, also get facts
    if let Some(_pred) = predicate {
        let facts = store.find_facts_by_entity(entity.id)?;
        // Facts don't map to CandidateMemory directly; their source_ids link to memories
        let fact_memory_ids: Vec<Uuid> = facts
            .iter()
            .flat_map(|f| f.source_ids.iter().copied())
            .collect();
        if !fact_memory_ids.is_empty() {
            let mut seen: std::collections::HashSet<Uuid> =
                candidates.iter().map(|c| c.memory.id).collect();
            let fact_mems = store.get_memories_by_ids(&fact_memory_ids)?;
            for mem in fact_mems {
                if seen.insert(mem.id) {
                    candidates.push(CandidateMemory {
                        memory: mem,
                        distance: 0.1,
                        activation: None,
                        hebbian_weight: None,
                        pagerank: None,
                        precision: None,
            telos_boost: None,
                    });
                }
            }
        }
    }

    Ok(candidates)
}

/// Temporal diff — memories created/modified since a given timestamp.
///
/// If `entity_filter` is provided, restricts to memories linked to that entity.
fn execute_temporal_diff(
    store: &crate::store::CozoStore,
    since: i64,
    entity_filter: Option<&str>,
    namespace: &str,
    max_k: usize,
) -> Result<Vec<CandidateMemory>> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("since".into(), DataValue::from(since));
    params.insert("ns".into(), DataValue::from(namespace));
    params.insert("limit".into(), DataValue::from(max_k as i64));

    // If entity filter provided, restrict to memories linked to that entity
    if let Some(entity_name) = entity_filter {
        if let Some(entity) = store.find_entity_by_name(entity_name, namespace)? {
            params.insert("entity_id".into(), DataValue::from(entity.id.to_string()));

            let result = store.run_query(
                r#"?[id, kind, content, embedding, fields, namespace, pinned,
                    expires_at, version, created_at, confidence, provenance, source_ids] :=
                    *entity_mentions{entity_id, memory_id: id},
                    entity_id = to_uuid($entity_id),
                    *memories{id, kind, content, embedding, fields, namespace, pinned,
                              expires_at, version, created_at, confidence, provenance, source_ids},
                    namespace = $ns,
                    created_at >= $since
                :sort -created_at
                :limit $limit"#,
                params,
            )?;

            let mut candidates = Vec::new();
            for row in &result.rows {
                let memory = crate::store::cozo::parse_memory_row_pub(row)?;
                candidates.push(CandidateMemory {
                    memory,
                    distance: 0.0,
                    activation: None,
                    hebbian_weight: None,
                    pagerank: None,
                    precision: None,
            telos_boost: None,
                });
            }
            return Ok(candidates);
        }
    }

    // No entity filter — return all recent memories
    let result = store.run_query(
        r#"?[id, kind, content, embedding, fields, namespace, pinned,
            expires_at, version, created_at, confidence, provenance, source_ids] :=
            *memories{id, kind, content, embedding, fields, namespace, pinned,
                      expires_at, version, created_at, confidence, provenance, source_ids},
            namespace = $ns,
            created_at >= $since
        :sort -created_at
        :limit $limit"#,
        params,
    )?;

    let mut candidates = Vec::new();
    for row in &result.rows {
        let memory = crate::store::cozo::parse_memory_row_pub(row)?;
        candidates.push(CandidateMemory {
            memory,
            distance: 0.0,
            activation: None,
            hebbian_weight: None,
            pagerank: None,
            precision: None,
            telos_boost: None,
        });
    }

    Ok(candidates)
}

/// Episodic retrieval — find memories linked to episodes, filtered by time range and/or topic.
fn execute_episodic(
    store: &crate::store::CozoStore,
    time_range: &Option<crate::types::query::TimeRange>,
    topic: Option<&str>,
    namespace: &str,
    max_k: usize,
) -> Result<Vec<CandidateMemory>> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("ns".into(), DataValue::from(namespace));
    params.insert("limit".into(), DataValue::from(max_k as i64));

    // Build time-range condition
    let (time_start, time_end) = if let Some(tr) = time_range {
        (tr.start, tr.end)
    } else {
        (0i64, i64::MAX)
    };
    params.insert("time_start".into(), DataValue::from(time_start));
    params.insert("time_end".into(), DataValue::from(time_end));

    // If topic provided, filter memories by content containing the topic keywords
    let has_topic = topic.is_some() && !topic.unwrap_or("").is_empty();
    let topic_lower = topic.unwrap_or("").to_lowercase();

    let result = store.run_query(
        r#"?[id, kind, content, embedding, fields, namespace, pinned,
            expires_at, version, created_at, confidence, provenance, source_ids] :=
            *episode_memories{episode_id, memory_id: id},
            *episodes{id: episode_id, started_at, ended_at},
            started_at >= $time_start,
            (is_null(ended_at) || ended_at <= $time_end),
            *memories{id, kind, content, embedding, fields, namespace, pinned,
                      expires_at, version, created_at, confidence, provenance, source_ids},
            namespace = $ns
        :sort -created_at
        :limit $limit"#,
        params,
    )?;

    let mut candidates = Vec::new();
    for row in &result.rows {
        let memory = crate::store::cozo::parse_memory_row_pub(row)?;

        // Client-side topic filter: check if content is relevant to topic
        if has_topic {
            let content_lower = memory.content.to_lowercase();
            // Keep memory if any topic keyword appears in content
            let topic_words: Vec<&str> = topic_lower
                .split_whitespace()
                .filter(|w| w.len() > 2) // skip tiny words like "the", "is"
                .collect();
            let matches = topic_words.iter().any(|w| content_lower.contains(w));
            if !matches {
                continue;
            }
        }

        candidates.push(CandidateMemory {
            memory,
            distance: 0.0,
            activation: None,
            hebbian_weight: None,
            pagerank: None,
            precision: None,
            telos_boost: None,
        });
    }

    Ok(candidates)
}

/// Abstraction search — find abstracted patterns/generalizations.
///
/// Searches BOTH:
/// 1. `memories` relation with kind = "abstraction" (from compression pipeline)
/// 2. `abstractions` relation (from reflection pipeline)
///
/// Topic keywords are used for client-side relevance filtering.
fn execute_abstract(
    store: &crate::store::CozoStore,
    topic: &str,
    namespace: &str,
    max_k: usize,
) -> Result<Vec<CandidateMemory>> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let topic_lower = topic.to_lowercase();
    let topic_words: Vec<&str> = topic_lower
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();

    let mut candidates = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // 1. Search memories with kind = "abstraction"
    let mut params = BTreeMap::new();
    params.insert("ns".into(), DataValue::from(namespace));
    params.insert("limit".into(), DataValue::from(max_k as i64));

    let result = store.run_query(
        r#"?[id, kind, content, embedding, fields, namespace, pinned,
            expires_at, version, created_at, confidence, provenance, source_ids] :=
            *memories{id, kind, content, embedding, fields, namespace, pinned,
                      expires_at, version, created_at, confidence, provenance, source_ids},
            namespace = $ns,
            kind = "abstraction"
        :sort -confidence
        :limit $limit"#,
        params,
    )?;

    for row in &result.rows {
        let memory = crate::store::cozo::parse_memory_row_pub(row)?;
        // Topic relevance filter (if topic has meaningful keywords)
        if !topic_words.is_empty() {
            let content_lower = memory.content.to_lowercase();
            if !topic_words.iter().any(|w| content_lower.contains(w)) {
                continue;
            }
        }
        if seen.insert(memory.id) {
            candidates.push(CandidateMemory {
                memory,
                distance: 0.0,
                activation: None,
                hebbian_weight: None,
                pagerank: None,
                precision: None,
            telos_boost: None,
            });
        }
    }

    // 2. Search the dedicated abstractions relation (from reflection)
    let abs_result = store.run_query(
        r#"?[id, content, confidence] :=
            *abstractions{id, content, confidence}
        :sort -confidence
        :limit $limit"#,
        {
            let mut p = BTreeMap::new();
            p.insert("limit".into(), DataValue::from(max_k as i64));
            p
        },
    );

    if let Ok(abs_rows) = abs_result {
        for row in &abs_rows.rows {
            if let (Ok(id), Some(content)) = (
                crate::store::cozo::parse_uuid_pub(&row[0]),
                row[1].get_str(),
            ) {
                // Topic relevance filter
                if !topic_words.is_empty() {
                    let content_lower = content.to_lowercase();
                    if !topic_words.iter().any(|w| content_lower.contains(w)) {
                        continue;
                    }
                }
                if seen.insert(id) {
                    let confidence = row[2].get_float().unwrap_or(0.5);
                    // Wrap abstraction as a synthetic CandidateMemory
                    let memory = crate::types::memory::Memory {
                        id,
                        kind: "abstraction.reflection".to_string(),
                        content: content.to_string(),
                        embedding: Vec::new(), // no embedding needed for candidates
                        fields: serde_json::Map::new(),
                        namespace: namespace.to_string(),
                        pinned: false,
                        expires_at: None,
                        version: 1,
                        created_at: 0,
                        confidence,
                        provenance: "reflected".to_string(),
                        source_ids: Vec::new(),
                    };
                    candidates.push(CandidateMemory {
                        memory,
                        distance: 0.0,
                        activation: None,
                        hebbian_weight: None,
                        pagerank: None,
                        precision: None,
            telos_boost: None,
                    });
                }
            }
        }
    }

    Ok(candidates)
}

/// Heuristic complexity score for a query.
///
/// Returns 0 for simple queries, higher for more complex ones.
/// Complexity >= 2 triggers LLM decomposition.
fn query_complexity(text: &str) -> usize {
    let lower = text.to_lowercase();
    let mut score = 0;

    // Conjunctions suggest multiple sub-queries (each counts as +2 since it bridges two clauses)
    let conjunctions = [" and ", " also ", " plus ", " as well as "];
    for conj in &conjunctions {
        let count = lower.matches(conj).count();
        if count > 0 {
            score += count + 1; // one conjunction = two clauses
        }
    }

    // Multiple question marks suggest compound questions
    let question_marks = text.matches('?').count();
    if question_marks > 1 {
        score += question_marks; // 2 questions = 2 sub-queries
    }

    // Multiple sentences (periods followed by capital letter) suggest compound queries
    let sentences = text.split(". ").count();
    if sentences > 1 {
        score += sentences;
    }

    // "compared to", "relationship between", "how does X relate to Y" patterns
    if lower.contains("compared to")
        || lower.contains("relationship between")
        || lower.contains("relate to")
        || lower.contains("difference between")
        || lower.contains("connection between")
    {
        score += 1;
    }

    score
}

/// Use the LLM to decompose a complex query into simpler sub-queries.
///
/// Returns a list of sub-query strings. If the LLM fails or returns a single
/// query, the caller falls back to heuristic planning.
async fn decompose_query(
    llm: &dyn crate::services::traits::LlmService,
    text: &str,
) -> anyhow::Result<Vec<String>> {
    let prompt = format!(
        "Break this complex question into simpler sub-questions for a memory retrieval system.\n\
         Each sub-question should target a single piece of knowledge.\n\
         Return ONLY a JSON array of strings, nothing else.\n\n\
         Question: \"{text}\"\n\n\
         Example input: \"What is Alice's role and how has the Acme project changed since Tuesday?\"\n\
         Example output: [\"What is Alice's role?\", \"How has the Acme project changed since Tuesday?\"]"
    );

    let response = llm
        .complete(
            &[crate::services::traits::Message {
                role: "user".to_string(),
                content: prompt,
            }],
            256,
        )
        .await?;

    // Parse JSON array from response
    let content = response.content.trim();
    // Find the JSON array in the response (LLMs sometimes add surrounding text)
    let start = content.find('[').ok_or_else(|| anyhow::anyhow!("No JSON array in response"))?;
    let end = content.rfind(']').ok_or_else(|| anyhow::anyhow!("No JSON array end in response"))?;
    let json_str = &content[start..=end];

    let sub_queries: Vec<String> = serde_json::from_str(json_str)?;

    // Filter out empty strings
    let sub_queries: Vec<String> = sub_queries.into_iter().filter(|s| !s.trim().is_empty()).collect();

    if sub_queries.is_empty() {
        anyhow::bail!("LLM returned empty decomposition");
    }

    Ok(sub_queries)
}

/// Remove duplicate Semantic strategies, keeping only the last one (broadest embedding).
fn dedup_semantic(strategies: &mut Vec<RecallStrategy>) {
    let semantic_count = strategies
        .iter()
        .filter(|s| matches!(s, RecallStrategy::Semantic { .. }))
        .count();
    if semantic_count <= 1 {
        return;
    }
    // Keep only the last semantic strategy
    let mut seen_first = false;
    strategies.retain(|s| {
        if matches!(s, RecallStrategy::Semantic { .. }) {
            if seen_first {
                return true; // keep subsequent ones (we'll reverse logic)
            }
            seen_first = true;
            false // drop the first one
        } else {
            true
        }
    });
}

/// Extract entity name from common query patterns.
fn extract_entity_name(lower: &str) -> Option<String> {
    // Direct prefix patterns: "who is Alice?" → "Alice"
    for prefix in &["who is ", "what is ", "tell me about "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let name = rest.trim_end_matches('?').trim_end_matches('.').trim();
            if !name.is_empty() {
                return Some(title_case(name));
            }
        }
    }

    // "where does Alice work?" → "Alice"
    // "what does Bob do?" → "Bob"
    for prefix in &["where does ", "what does "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            // Take the subject (first word or multi-word name before the verb)
            let name = rest.split(|c: char| c == ' ').next().unwrap_or("").trim();
            if !name.is_empty() {
                return Some(title_case(name));
            }
        }
    }

    // "where is Alice?" → "Alice"
    if let Some(rest) = lower.strip_prefix("where is ") {
        let name = rest.trim_end_matches('?').trim_end_matches('.').trim();
        if !name.is_empty() {
            return Some(title_case(name));
        }
    }

    None
}

/// Extract entity name from temporal query patterns.
/// e.g., "Has anything changed about the API?" → "Api"
/// e.g., "What changed with Alice since last week?" → "Alice"
fn extract_temporal_entity_filter(lower: &str) -> Option<String> {
    for pattern in &["about the ", "about ", "with ", "for ", "regarding "] {
        if let Some(pos) = lower.find(pattern) {
            let rest = &lower[pos + pattern.len()..];
            // Take until common temporal/stop words
            let name = rest
                .split(|c: char| matches!(c, '?' | '.' | ','))
                .next()
                .unwrap_or("")
                .trim();
            // Remove trailing temporal words (iterate until stable)
            let mut name = name;
            loop {
                let trimmed = name
                    .trim_end_matches(" since last week")
                    .trim_end_matches(" since last month")
                    .trim_end_matches(" since")
                    .trim_end_matches(" after")
                    .trim_end_matches(" last week")
                    .trim_end_matches(" last month")
                    .trim_end_matches(" recently")
                    .trim();
                if trimmed == name {
                    break;
                }
                name = trimmed;
            }
            if !name.is_empty() && name.len() < 50 {
                return Some(title_case(name));
            }
        }
    }
    None
}

pub(crate) fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => {
                    let mut s = c.to_uppercase().to_string();
                    s.push_str(&chars.as_str().to_lowercase());
                    s
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_entity_name() {
        assert_eq!(extract_entity_name("who is alice?"), Some("Alice".to_string()));
        assert_eq!(extract_entity_name("tell me about acme corp"), Some("Acme Corp".to_string()));
        assert_eq!(extract_entity_name("what is kubernetes?"), Some("Kubernetes".to_string()));
        assert_eq!(extract_entity_name("where does alice work?"), Some("Alice".to_string()));
        assert_eq!(extract_entity_name("what does bob do?"), Some("Bob".to_string()));
        assert_eq!(extract_entity_name("how does it work?"), None);
    }

    #[test]
    fn test_extract_temporal_entity_filter() {
        assert_eq!(
            extract_temporal_entity_filter("has anything changed about the api since last week?"),
            Some("Api".to_string())
        );
        assert_eq!(
            extract_temporal_entity_filter("what changed with alice recently?"),
            Some("Alice".to_string())
        );
        assert_eq!(
            extract_temporal_entity_filter("what happened recently?"),
            None
        );
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("alice"), "Alice");
        assert_eq!(title_case("acme corp"), "Acme Corp");
    }

    #[test]
    fn test_query_complexity_simple() {
        assert_eq!(query_complexity("Who is Alice?"), 0);
        assert_eq!(query_complexity("What happened recently?"), 0);
        assert_eq!(query_complexity("Tell me about Acme"), 0);
    }

    #[test]
    fn test_query_complexity_compound() {
        // Conjunction
        assert!(query_complexity("What is Alice's role and how has the project changed?") >= 2);
        // Multiple questions
        assert!(query_complexity("Who is Alice? What does she do?") >= 2);
        // Comparison
        assert!(query_complexity("What is the difference between project A and project B?") >= 2);
        // Multiple sentences
        assert!(query_complexity("Tell me about Alice. Also explain Bob's role.") >= 2);
    }

    #[test]
    fn test_dedup_semantic() {
        let mut strategies = vec![
            RecallStrategy::Structured {
                entity: "Alice".to_string(),
                predicate: None,
            },
            RecallStrategy::Semantic {
                embedding: vec![1.0],
                filters: vec![],
            },
            RecallStrategy::Semantic {
                embedding: vec![2.0],
                filters: vec![],
            },
        ];
        dedup_semantic(&mut strategies);
        let semantic_count = strategies
            .iter()
            .filter(|s| matches!(s, RecallStrategy::Semantic { .. }))
            .count();
        assert_eq!(semantic_count, 1);
        assert_eq!(strategies.len(), 2);
    }
}
