//! Counterfactual causal attribution.
//!
//! When a task fails, we ask: "Would the task have succeeded if memory X
//! had been absent?" by finding similar tasks that did NOT use memory X
//! and checking their success rate.

use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::Embedder;
use crate::skills::storage;
use crate::store::CozoStore;

/// A single causal attribution: how much did a memory/fact contribute to failure?
#[derive(Debug, Clone)]
pub struct Attribution {
    pub memory_id: Uuid,
    pub causal_impact: f64,
    pub counterfactual_success_rate: f64,
    pub baseline_success_rate: f64,
    pub sample_size: usize,
    /// Impact from do-operator interventional analysis (simulating memory removal).
    pub interventional_impact: f64,
}

/// Result of causal attribution analysis on a failed task.
#[derive(Debug)]
pub struct AttributionResult {
    pub task_id: Uuid,
    pub attributions: Vec<Attribution>,
    pub total_tasks_analyzed: usize,
}

/// Perform counterfactual causal attribution on a failed task.
///
/// Steps:
/// 1. Get the failed task's recall context (what memories were used)
/// 2. Find similar tasks via HNSW on task_outcomes
/// 3. For each memory in the context, compute counterfactual:
///    P(success | do(memory = absent)) vs baseline P(success)
/// 4. Rank by causal impact (difference from baseline)
pub async fn attribute_failure(
    store: &CozoStore,
    embedder: &Arc<dyn Embedder>,
    task_id: Uuid,
    task_description: &str,
) -> Result<AttributionResult> {
    // Step 1: Get the recall context
    let recall_contexts = storage::get_recall_contexts(store, task_id)?;
    if recall_contexts.is_empty() {
        return Ok(AttributionResult {
            task_id,
            attributions: vec![],
            total_tasks_analyzed: 0,
        });
    }

    // Collect all memory IDs used in this task
    let used_memory_ids: Vec<Uuid> = recall_contexts
        .iter()
        .flat_map(|rc| rc.memory_ids.clone())
        .collect();

    if used_memory_ids.is_empty() {
        return Ok(AttributionResult {
            task_id,
            attributions: vec![],
            total_tasks_analyzed: 0,
        });
    }

    // Step 2: Find similar tasks
    let embeddings = embedder
        .embed(&[task_description])
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    if embeddings.is_empty() {
        return Ok(AttributionResult {
            task_id,
            attributions: vec![],
            total_tasks_analyzed: 0,
        });
    }

    let similar_tasks = storage::find_similar_tasks(store, &embeddings[0], 50, 0.5)?;

    if similar_tasks.is_empty() {
        return Ok(AttributionResult {
            task_id,
            attributions: vec![],
            total_tasks_analyzed: 0,
        });
    }

    // Compute baseline success rate across similar tasks
    let total_similar = similar_tasks.len();
    let baseline_successes = similar_tasks
        .iter()
        .filter(|(to, _)| to.outcome == "success")
        .count();
    let baseline_success_rate = if total_similar > 0 {
        baseline_successes as f64 / total_similar as f64
    } else {
        0.0
    };

    // Step 3: For each memory, compute counterfactual
    // Use d-separation to skip memories that are conditionally independent from the task
    let mut attributions = Vec::new();

    for memory_id in &used_memory_ids {
        // Gate: if this memory is d-separated from the task (conditioned on other used memories),
        // it's conditionally independent and unlikely to have causal impact — skip it.
        let conditioning: Vec<Uuid> = used_memory_ids
            .iter()
            .filter(|id| *id != memory_id)
            .copied()
            .collect();
        if let Ok(true) = super::d_separation::is_d_separated(
            store,
            *memory_id,
            task_id,
            &conditioning,
        ) {
            continue; // Conditionally independent — skip costly counterfactual
        }
        // Find similar tasks that did NOT use this memory
        let contexts_for_similar: Vec<(bool, bool)> = similar_tasks
            .iter()
            .map(|(to, _)| {
                let is_success = to.outcome == "success";
                // Check if this similar task used this memory
                let contexts = storage::get_recall_contexts(store, to.task_id).unwrap_or_default();
                let used_this_memory = contexts.iter().any(|rc| rc.memory_ids.contains(memory_id));
                (is_success, used_this_memory)
            })
            .collect();

        // Counterfactual: tasks that did NOT use this memory
        let without: Vec<&(bool, bool)> = contexts_for_similar.iter().filter(|(_, used)| !used).collect();
        let without_successes = without.iter().filter(|(success, _)| *success).count();
        let counterfactual_success_rate = if without.is_empty() {
            baseline_success_rate // no data → use baseline
        } else {
            without_successes as f64 / without.len() as f64
        };

        // Causal impact: difference between counterfactual and baseline
        // Positive impact = removing this memory would INCREASE success rate
        // (meaning this memory HURT performance)
        let causal_impact = counterfactual_success_rate - baseline_success_rate;

        attributions.push(Attribution {
            memory_id: *memory_id,
            causal_impact,
            counterfactual_success_rate,
            baseline_success_rate,
            sample_size: without.len(),
            interventional_impact: 0.0, // will be filled below
        });
    }

    // Step 4: Interventional analysis via do-operator
    // For each memory, simulate removing it (setting confidence to 0.0)
    // and measure the propagated impact through the knowledge graph
    for attr in &mut attributions {
        let do_result = super::do_operator::do_intervention(
            store,
            attr.memory_id,
            0.0, // Simulate removing the memory
            3,   // max 3 hops of propagation
        );

        if let Ok(result) = do_result {
            // Interventional impact: average confidence drop across affected nodes
            if !result.propagated_effects.is_empty() {
                let total_drop: f64 = result
                    .propagated_effects
                    .iter()
                    .map(|e| (e.original_value - e.new_value).abs())
                    .sum();
                attr.interventional_impact =
                    total_drop / result.propagated_effects.len() as f64;
            }
        }

        // Blend observational and interventional impacts
        // final_impact = observational * 0.4 + interventional * 0.6
        attr.causal_impact =
            attr.causal_impact * 0.4 + attr.interventional_impact * 0.6;
    }

    // Sort by blended causal impact (most impactful first)
    attributions.sort_by(|a, b| {
        b.causal_impact
            .abs()
            .partial_cmp(&a.causal_impact.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Accumulate causal edges from attribution results (Source 1)
    accumulate_attribution_edges(store, task_id, &attributions);

    Ok(AttributionResult {
        task_id,
        attributions,
        total_tasks_analyzed: total_similar,
    })
}

/// Apply attribution results: update confidence of implicated memories.
pub fn apply_attribution(
    store: &CozoStore,
    result: &AttributionResult,
    impact_threshold: f64,
) -> Result<usize> {
    let mut updated = 0;

    for attr in &result.attributions {
        // Only adjust if causal impact is significant and we have enough data
        if attr.causal_impact.abs() < impact_threshold || attr.sample_size < 3 {
            continue;
        }

        // If removing this memory would have helped (positive impact),
        // reduce its confidence
        if attr.causal_impact > 0.0 {
            let current = store.run_query(
                &format!(
                    r#"?[confidence] :=
                        *memories{{id, confidence, @ 'NOW'}},
                        id = to_uuid("{}")"#,
                    attr.memory_id
                ),
                std::collections::BTreeMap::new(),
            )?;

            if let Some(row) = current.rows.first() {
                let old_conf = row[0].get_float().unwrap_or(1.0);
                let new_conf = (old_conf - attr.causal_impact * 0.3).max(0.1);

                let mut params = std::collections::BTreeMap::new();
                params.insert("id".into(), cozo::DataValue::from(attr.memory_id.to_string()));
                params.insert("confidence".into(), cozo::DataValue::from(new_conf));

                // This is a simplified update — in production we'd read the full
                // row and write back with Validity
                store.run_script(
                    r#"?[id, valid_at, confidence] <- [[$id, 'ASSERT', $confidence]]
                    :put memories {id, valid_at => confidence}"#,
                    params,
                )?;
                updated += 1;
            }
        }
    }

    Ok(updated)
}

/// Store attribution results as causal edges (Source 1: Attribution).
///
/// Each attribution with significant causal impact becomes a directed edge
/// from the implicated memory to the failed task. Strength is derived from
/// the normalized causal impact.
fn accumulate_attribution_edges(
    store: &CozoStore,
    task_id: Uuid,
    attributions: &[Attribution],
) {
    let now = crate::types::memory::now_ms();

    for attr in attributions {
        // Only store edges with meaningful impact
        if attr.causal_impact.abs() < 0.05 {
            continue;
        }

        // Normalize impact to [0, 1] range for causal strength
        let strength = attr.causal_impact.abs().min(1.0);

        let edge = super::graph::CausalEdge {
            cause_id: attr.memory_id,
            effect_id: task_id,
            causal_strength: strength,
            observations: 1,
            last_observed: now,
            mechanism: super::graph::CausalMechanism::RecallAttribution,
            confidence: 0.5, // moderate initial confidence
            namespace: String::new(),
        };

        if let Err(e) = super::graph::accumulate_causal_edge(store, &edge) {
            eprintln!("Failed to accumulate causal edge from attribution: {e}");
        }
    }
}
