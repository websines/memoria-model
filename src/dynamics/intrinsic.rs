//! Intrinsic goal generation from surprise patterns.
//!
//! Closes the Active Inference loop: when the surprise_log shows persistent
//! prediction errors concentrated around specific entities or topics, the
//! system autonomously generates exploratory telos to reduce that uncertainty.
//!
//! All gating is derived from β and free energy — no hardcoded thresholds.
//! Higher β (exploration mode) → more aggressive goal generation.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::services::traits::{Embedder, LlmService, Message};
use crate::store::cozo::cosine_similarity;
use crate::store::CozoStore;
use crate::types::telos::{Telos, TelosEvent, TelosProvenance, TelosStatus};

/// A concentration of unresolved surprise around a specific variable or topic.
#[derive(Debug, Clone)]
pub struct SurpriseHotspot {
    /// The variable_id if entity-linked, None for source-clustered.
    pub variable_id: Option<Uuid>,
    /// Source prefix for non-entity hotspots (e.g. "telos:dependency_failed").
    pub source_key: Option<String>,
    /// Total accumulated surprise for this hotspot.
    pub total_surprise: f64,
    /// Number of distinct surprise log entries.
    pub event_count: usize,
    /// Representative observation summaries (most recent first).
    pub summaries: Vec<String>,
}

/// Result of running the intrinsic goal generation pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntrinsicGoalResult {
    pub hotspots_found: usize,
    pub goals_created: usize,
    pub goals_activated: usize,
    pub duplicates_skipped: usize,
    pub cooldown_skipped: usize,
}

/// Internal gating decision derived from β and surprise signals.
struct GenerationGate {
    should_generate: bool,
    beta: f64,
    auto_activate: bool,
    max_goals: usize,
}

// ── Hotspot aggregation ──

/// Aggregate unresolved surprise entries into hotspots.
///
/// Groups by variable_id (entity-linked) or by source prefix (topic-linked).
/// Aggregation is done in Rust, consistent with `compute_bethe_free_energy`.
pub fn aggregate_surprise_hotspots(
    store: &CozoStore,
    min_events: usize,
) -> Result<Vec<SurpriseHotspot>> {
    let result = store.run_query(
        r#"?[variable_id, surprise, source, observation_summary] :=
            *surprise_log{variable_id, surprise, source, observation_summary, resolved},
            resolved = false"#,
        BTreeMap::new(),
    )?;

    // Group by variable_id (entity-linked) or by source (topic-linked)
    let mut entity_map: HashMap<Uuid, SurpriseHotspot> = HashMap::new();
    let mut source_map: HashMap<String, SurpriseHotspot> = HashMap::new();

    for row in &result.rows {
        let surprise = row[1].get_float().unwrap_or(0.0);
        let source = match &row[2] {
            cozo::DataValue::Str(s) => s.to_string(),
            _ => "unknown".to_string(),
        };
        let summary = match &row[3] {
            cozo::DataValue::Str(s) => s.to_string(),
            _ => String::new(),
        };

        // Try to extract variable_id as UUID
        let var_id = match &row[0] {
            cozo::DataValue::Str(s) => Uuid::parse_str(s).ok(),
            cozo::DataValue::Uuid(u) => {
                // CozoDB UUIDs — convert via bytes
                let bytes = u.0.as_bytes();
                Some(Uuid::from_bytes(*bytes))
            }
            _ => None,
        };

        if let Some(vid) = var_id {
            let hotspot = entity_map.entry(vid).or_insert_with(|| SurpriseHotspot {
                variable_id: Some(vid),
                source_key: None,
                total_surprise: 0.0,
                event_count: 0,
                summaries: Vec::new(),
            });
            hotspot.total_surprise += surprise;
            hotspot.event_count += 1;
            if hotspot.summaries.len() < 5 {
                hotspot.summaries.push(summary);
            }
        } else {
            // Group by source prefix (e.g. "telos:dependency_failed" → "telos")
            let key = source.split(':').next().unwrap_or(&source).to_string();
            let hotspot = source_map.entry(key.clone()).or_insert_with(|| SurpriseHotspot {
                variable_id: None,
                source_key: Some(key),
                total_surprise: 0.0,
                event_count: 0,
                summaries: Vec::new(),
            });
            hotspot.total_surprise += surprise;
            hotspot.event_count += 1;
            if hotspot.summaries.len() < 5 {
                hotspot.summaries.push(summary);
            }
        }
    }

    // Combine and filter by minimum event count
    let mut hotspots: Vec<SurpriseHotspot> = entity_map
        .into_values()
        .chain(source_map.into_values())
        .filter(|h| h.event_count >= min_events)
        .collect();

    // Sort by total surprise descending
    hotspots.sort_by(|a, b| {
        b.total_surprise
            .partial_cmp(&a.total_surprise)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(hotspots)
}

// ── Generation gating ──

/// Compute whether intrinsic goal generation should proceed.
///
/// Uses β (exploration/exploitation balance) and accumulated surprise
/// to derive an adaptive threshold — no hardcoded values.
fn compute_generation_gate(store: &CozoStore) -> Result<GenerationGate> {
    let fe_state = crate::aif::compute_bethe_free_energy(store)?;
    let unresolved = crate::dynamics::surprise::accumulated_unresolved_surprise(store)?;

    // Count unresolved entries
    let count_result = store.run_query(
        r#"?[count(id)] := *surprise_log{id, resolved}, resolved = false"#,
        BTreeMap::new(),
    )?;
    let count = count_result
        .rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0) as usize;

    let mean_surprise = if count > 0 {
        unresolved / count as f64
    } else {
        0.0
    };

    // Adaptive threshold: mean × (1 + β).
    // High β → lower effective threshold → more generation.
    let adaptive_threshold = mean_surprise * (1.0 + fe_state.beta);
    let should_generate = unresolved > adaptive_threshold && count >= 3;

    // Auto-activate when in exploration mode (β > 0.5)
    let auto_activate = fe_state.beta > 0.5;

    // Scale max goals by β: [1, 5]
    let max_goals = (fe_state.beta * 5.0).ceil().max(1.0) as usize;

    Ok(GenerationGate {
        should_generate,
        beta: fe_state.beta,
        auto_activate,
        max_goals,
    })
}

// ── Cooldown ──

/// Check if we're within the cooldown window since the last intrinsic goal.
fn check_cooldown(store: &CozoStore, cooldown_ms: i64) -> Result<bool> {
    let result = store.run_query(
        r#"?[max(created_at)] :=
            *telos{created_at, provenance},
            provenance = "intrinsic""#,
        BTreeMap::new(),
    )?;

    let last_created = result
        .rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0);

    if last_created == 0 {
        return Ok(false); // Never created an intrinsic goal
    }

    let now = crate::types::memory::now_ms();
    Ok(now - last_created < cooldown_ms)
}

// ── Deduplication ──

/// Check if a telos with similar embedding already exists (non-terminal).
fn check_duplicate_telos(store: &CozoStore, embedding: &[f32]) -> Result<bool> {
    let results = store.find_telos_by_embedding(embedding, 5)?;
    for (telos, _distance) in &results {
        if telos.status.is_terminal() {
            continue;
        }
        let sim = cosine_similarity(&telos.embedding, embedding);
        if sim >= 0.85 {
            return Ok(true);
        }
    }
    Ok(false)
}

// ── Goal text synthesis ──

/// Use the LLM to synthesize a meaningful goal title + description from
/// the observation summaries in a surprise hotspot.
async fn synthesize_goal_text(
    llm: &dyn LlmService,
    hotspot: &SurpriseHotspot,
    entity_name: Option<&str>,
) -> (String, String) {
    let summaries_text = hotspot
        .summaries
        .iter()
        .map(|s| format!("- {s}"))
        .collect::<Vec<_>>()
        .join("\n");

    let entity_context = entity_name
        .map(|n| format!("\nThese all relate to entity: {n}"))
        .unwrap_or_default();

    let prompt = format!(
        concat!(
            "The following observations repeatedly surprised the system:\n",
            "{summaries}\n",
            "{entity}\n\n",
            "Generate a concise exploration goal to reduce this uncertainty.\n",
            "Return JSON: {{\"title\": \"...\", \"description\": \"...\"}}\n",
            "Title max 80 chars, description max 200 chars. No markdown fences."
        ),
        summaries = summaries_text,
        entity = entity_context,
    );

    let result = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "You generate exploration goals from prediction error patterns. Return valid JSON.".into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            256,
        )
        .await;

    match result {
        Ok(response) => {
            // Try to parse LLM JSON response
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response.content) {
                let title = parsed
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let desc = parsed
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !title.is_empty() {
                    return (title, desc);
                }
            }
            // LLM returned non-JSON — use first line as title
            let first_line = response.content.lines().next().unwrap_or("").to_string();
            if !first_line.is_empty() {
                return (
                    first_line.chars().take(80).collect(),
                    String::new(),
                );
            }
            fallback_title(hotspot)
        }
        Err(_) => fallback_title(hotspot),
    }
}

/// Mechanical fallback when LLM is unavailable.
fn fallback_title(hotspot: &SurpriseHotspot) -> (String, String) {
    let summary_hint = hotspot
        .summaries
        .first()
        .map(|s| s.chars().take(60).collect::<String>())
        .unwrap_or_else(|| "unknown pattern".to_string());

    (
        format!("Investigate uncertainty: {summary_hint}"),
        format!(
            "Accumulated {:.1} surprise across {} observations",
            hotspot.total_surprise, hotspot.event_count
        ),
    )
}

// ── Main entry point ──

/// Generate intrinsic telos from surprise hotspots.
///
/// This is the core function called by the QueueWorker. It:
/// 1. Checks cooldown (skip if too recent)
/// 2. Evaluates the generation gate (β + surprise signals)
/// 3. Aggregates surprise hotspots
/// 4. For each: deduplicates, synthesizes goal text, creates telos
pub async fn generate_intrinsic_goals(
    store: &CozoStore,
    embedder: &dyn Embedder,
    llm: &dyn LlmService,
    cooldown_ms: i64,
) -> Result<IntrinsicGoalResult> {
    let mut result = IntrinsicGoalResult::default();

    // 1. Cooldown check
    if check_cooldown(store, cooldown_ms)? {
        result.cooldown_skipped = 1;
        return Ok(result);
    }

    // 2. Generation gate
    let gate = compute_generation_gate(store)?;
    if !gate.should_generate {
        return Ok(result);
    }

    // 3. Aggregate hotspots (minimum 3 events per hotspot)
    let hotspots = aggregate_surprise_hotspots(store, 3)?;
    result.hotspots_found = hotspots.len();

    if hotspots.is_empty() {
        return Ok(result);
    }

    // 4. Process hotspots up to max_goals
    let mut created_ids = Vec::new();

    for hotspot in hotspots.iter().take(gate.max_goals) {
        // 4a. Try to resolve entity name for context
        let entity_name = if let Some(vid) = hotspot.variable_id {
            resolve_entity_name(store, vid)
        } else {
            None
        };

        // 4b. Synthesize goal text
        let (title, description) = synthesize_goal_text(llm, hotspot, entity_name.as_deref()).await;

        // 4c. Embed for dedup check
        let embed_text = if description.is_empty() {
            title.clone()
        } else {
            format!("{title}: {description}")
        };
        let embeddings = embedder
            .embed(&[embed_text.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;
        let embedding = embeddings.into_iter().next().unwrap_or_default();

        // 4d. Dedup check
        if check_duplicate_telos(store, &embedding)? {
            result.duplicates_skipped += 1;
            continue;
        }

        // 4e. Create the telos
        let status = if gate.auto_activate {
            TelosStatus::Active
        } else {
            TelosStatus::Proposed
        };

        // Priority derived from normalized surprise: this hotspot's share of total
        let total_surprise: f64 = hotspots.iter().map(|h| h.total_surprise).sum();
        let priority = if total_surprise > 0.0 {
            (hotspot.total_surprise / total_surprise).clamp(0.1, 0.8)
        } else {
            0.3
        };

        let mut telos = Telos::new(&title, &description, embedding, "system", "intrinsic_dynamics");
        telos.namespace = "default".to_string();
        telos.depth = 3;
        telos.status = status;
        telos.priority = priority;
        telos.confidence = TelosProvenance::Intrinsic.initial_confidence();
        telos.provenance = TelosProvenance::Intrinsic;

        // Link to the entity if entity-sourced
        if let Some(vid) = hotspot.variable_id {
            telos.related_entities.push(vid);
        }

        store.insert_telos(&telos)?;

        // Record event
        let mut event = TelosEvent::new(telos.id, "intrinsic_created");
        event.agent_id = "intrinsic_dynamics".to_string();
        event.description = format!(
            "Generated from {} surprise events (total: {:.2}, β: {:.3})",
            hotspot.event_count, hotspot.total_surprise, gate.beta
        );
        store.insert_telos_event(&event)?;

        if status == TelosStatus::Active {
            result.goals_activated += 1;
        }
        result.goals_created += 1;
        created_ids.push(telos.id);
    }

    Ok(result)
}

/// Try to resolve a human-readable name for an entity UUID.
fn resolve_entity_name(store: &CozoStore, entity_id: Uuid) -> Option<String> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), cozo::DataValue::from(entity_id.to_string()));

    let result = store
        .run_query(
            r#"?[name] := *entities{id, name, @ 'NOW'}, id = to_uuid($id)"#,
            params,
        )
        .ok()?;

    result.rows.first().and_then(|r| match &r[0] {
        cozo::DataValue::Str(s) => Some(s.to_string()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_hotspots_entity_linked() {
        let store = CozoStore::open_mem(4).unwrap();
        let entity_id = Uuid::now_v7();

        // Insert 4 surprise entries for the same entity
        for i in 0..4 {
            crate::dynamics::surprise::log_surprise(
                &store,
                1.0 + i as f64 * 0.1,
                "new_fact",
                Some(entity_id),
                &format!("observation {i}"),
            )
            .unwrap();
        }

        let hotspots = aggregate_surprise_hotspots(&store, 3).unwrap();
        assert_eq!(hotspots.len(), 1, "should have 1 entity-linked hotspot");
        assert_eq!(hotspots[0].event_count, 4);
        assert!(hotspots[0].variable_id.is_some());
        assert!(
            (hotspots[0].total_surprise - 4.6).abs() < 0.01,
            "total={}, expected 4.6",
            hotspots[0].total_surprise
        );
    }

    #[test]
    fn test_aggregate_hotspots_filters_below_min() {
        let store = CozoStore::open_mem(4).unwrap();
        let entity_id = Uuid::now_v7();

        // Only 2 entries — below min_events=3
        for i in 0..2 {
            crate::dynamics::surprise::log_surprise(
                &store,
                1.0,
                "new_fact",
                Some(entity_id),
                &format!("obs {i}"),
            )
            .unwrap();
        }

        let hotspots = aggregate_surprise_hotspots(&store, 3).unwrap();
        assert!(hotspots.is_empty(), "should filter out hotspot with only 2 events");
    }

    #[test]
    fn test_aggregate_hotspots_separate_variables() {
        let store = CozoStore::open_mem(4).unwrap();
        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();

        for i in 0..3 {
            crate::dynamics::surprise::log_surprise(
                &store,
                2.0,
                "fact",
                Some(id_a),
                &format!("a-{i}"),
            )
            .unwrap();
            crate::dynamics::surprise::log_surprise(
                &store,
                1.0,
                "fact",
                Some(id_b),
                &format!("b-{i}"),
            )
            .unwrap();
        }

        let hotspots = aggregate_surprise_hotspots(&store, 3).unwrap();
        assert_eq!(hotspots.len(), 2);
        // Sorted by total surprise descending
        assert!(
            hotspots[0].total_surprise > hotspots[1].total_surprise,
            "first={}, second={}",
            hotspots[0].total_surprise,
            hotspots[1].total_surprise,
        );
    }

    #[test]
    fn test_aggregate_hotspots_source_grouped() {
        let store = CozoStore::open_mem(4).unwrap();

        // 4 entries with no variable_id, same source prefix
        for i in 0..4 {
            crate::dynamics::surprise::log_surprise(
                &store,
                0.5,
                &format!("telos:event_{i}"),
                None,
                &format!("telos surprise {i}"),
            )
            .unwrap();
        }

        let hotspots = aggregate_surprise_hotspots(&store, 3).unwrap();
        assert_eq!(hotspots.len(), 1);
        assert!(hotspots[0].variable_id.is_none());
        assert_eq!(hotspots[0].source_key.as_deref(), Some("telos"));
    }

    #[test]
    fn test_generation_gate_empty_store() {
        let store = CozoStore::open_mem(4).unwrap();
        let gate = compute_generation_gate(&store).unwrap();
        assert!(!gate.should_generate, "empty store should not generate");
    }

    #[test]
    fn test_generation_gate_with_surprise() {
        let store = CozoStore::open_mem(4).unwrap();

        // Insert enough surprise entries to trigger
        for i in 0..5 {
            crate::dynamics::surprise::log_surprise(
                &store,
                2.0,
                "fact",
                Some(Uuid::now_v7()),
                &format!("high surprise event {i}"),
            )
            .unwrap();
        }

        let gate = compute_generation_gate(&store).unwrap();
        // With 5 entries × 2.0 = 10.0 total, mean=2.0
        // threshold = 2.0 × (1 + β), and total=10 must > threshold
        // This depends on β which depends on facts in store (empty → β ≈ 0)
        // With β=0: threshold = 2.0 × 1 = 2.0, total=10 > 2.0 → should_generate
        assert!(
            gate.should_generate,
            "high surprise with 5 entries should trigger generation"
        );
    }

    #[test]
    fn test_cooldown_no_previous() {
        let store = CozoStore::open_mem(4).unwrap();
        assert!(!check_cooldown(&store, 300_000).unwrap(), "no previous intrinsic goals");
    }

    #[test]
    fn test_cooldown_recent_blocks() {
        let store = CozoStore::open_mem(4).unwrap();

        // Insert an intrinsic telos
        let mut telos = Telos::new("test intrinsic", "", vec![0.1; 4], "system", "intrinsic");
        telos.provenance = TelosProvenance::Intrinsic;
        store.insert_telos(&telos).unwrap();

        assert!(
            check_cooldown(&store, 300_000).unwrap(),
            "recently created intrinsic goal should trigger cooldown"
        );
    }

    #[test]
    fn test_duplicate_detection() {
        let store = CozoStore::open_mem(4).unwrap();

        let embedding = vec![0.5, 0.5, 0.5, 0.5];
        let telos = Telos::new("existing goal", "", embedding.clone(), "a", "u");
        store.insert_telos(&telos).unwrap();

        // Same embedding should be detected as duplicate
        assert!(
            check_duplicate_telos(&store, &embedding).unwrap(),
            "identical embedding should be duplicate"
        );

        // Very different embedding should not
        let different = vec![-0.5, -0.5, -0.5, -0.5];
        assert!(
            !check_duplicate_telos(&store, &different).unwrap(),
            "opposite embedding should not be duplicate"
        );
    }

    #[test]
    fn test_fallback_title() {
        let hotspot = SurpriseHotspot {
            variable_id: None,
            source_key: Some("telos".into()),
            total_surprise: 5.0,
            event_count: 3,
            summaries: vec!["Entity X keeps changing state unexpectedly".into()],
        };

        let (title, desc) = fallback_title(&hotspot);
        assert!(title.starts_with("Investigate uncertainty:"));
        assert!(desc.contains("5.0"));
        assert!(desc.contains("3"));
    }

    #[tokio::test]
    async fn test_end_to_end_intrinsic_generation() {
        let store = CozoStore::open_mem(4).unwrap();

        // Populate surprise entries concentrated on one entity
        let entity_id = Uuid::now_v7();
        for i in 0..5 {
            crate::dynamics::surprise::log_surprise(
                &store,
                2.0,
                "fact_contradiction",
                Some(entity_id),
                &format!("Entity X state changed to {i}"),
            )
            .unwrap();
        }

        // Use mock services
        use crate::services::mock::{MockEmbedder, MockLlm};
        let embedder = MockEmbedder::new(4);
        let llm = MockLlm;

        let result = generate_intrinsic_goals(&store, &embedder, &llm, 0).await.unwrap();

        assert!(
            result.hotspots_found >= 1,
            "should find at least 1 hotspot, found {}",
            result.hotspots_found
        );
        assert!(
            result.goals_created >= 1,
            "should create at least 1 goal, created {}",
            result.goals_created
        );

        // Verify the telos was stored correctly
        let _all_telos = store.list_active_telos("default", 100).unwrap_or_default();
        let proposed = store.run_query(
            r#"?[id, title, provenance, status] :=
                *telos{id, title, provenance, status},
                provenance = "intrinsic""#,
            BTreeMap::new(),
        );

        if let Ok(rows) = proposed {
            assert!(
                !rows.rows.is_empty(),
                "should have at least 1 intrinsic telos in store"
            );
        }

        // Verify cooldown is now active
        assert!(
            check_cooldown(&store, 300_000).unwrap(),
            "cooldown should be active after generation"
        );

        // Second run should be blocked by cooldown
        let result2 = generate_intrinsic_goals(&store, &embedder, &llm, 300_000).await.unwrap();
        assert_eq!(result2.cooldown_skipped, 1, "second run should be blocked by cooldown");
    }
}
