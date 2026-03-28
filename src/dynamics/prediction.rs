//! Predictive Generation — closes the Active Inference loop.
//!
//! Full AIF: Predict → Observe → Prediction error → Update beliefs → Predict again
//!
//! Four generators:
//! 1. **PPM-C Sequence** — variable-order Markov prediction of next task type
//! 2. **ETS Trend** — exponential smoothing extrapolation of telos progress
//! 3. **BOCPD Regime Change** — Bayesian online changepoint detection on surprise stream
//! 4. **Episodic Pattern** — LLM-assisted predictions (low frequency, high cost)
//!
//! Predictions are matched against observations in `tell()` to compute prediction error,
//! which feeds back into free energy.

use std::collections::BTreeMap;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::store::CozoStore;
use crate::types::memory::now_ms;

use super::ppm::PpmModel;

// ── Types ──────────────────────────────────────────────────────────────

/// A prediction about future observations.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub id: Uuid,
    pub kind: PredictionKind,
    pub content: String,
    /// Embedding of the predicted content (for matching against observations).
    pub embedding: Vec<f32>,
    /// When the prediction was made (ms since epoch).
    pub predicted_at: i64,
    /// When the prediction should have been observed by (ms since epoch).
    pub expected_by: i64,
    /// Predicted probability / confidence.
    pub confidence: f64,
    /// Prediction interval from ETS (if applicable).
    pub confidence_interval: Option<(f64, f64)>,
    /// Which generator produced this prediction.
    pub source: PredictionSource,
    /// IDs of memories/facts that informed this prediction.
    pub context_ids: Vec<Uuid>,
    /// Namespace this prediction applies to.
    pub namespace: String,
    /// Whether this prediction has been resolved.
    pub resolved: bool,
    /// Resolution details (populated when resolved).
    pub resolution: Option<PredictionResolution>,
}

/// What kind of prediction this is.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PredictionKind {
    /// PPM-C sequence prediction of next task type.
    NextTask,
    /// Expected interaction / follow-up pattern.
    FollowUp,
    /// ETS-extrapolated goal progress.
    TelosProgress,
    /// Causal prediction about entity state (Initiative 5).
    EntityState,
    /// BOCPD changepoint detected — regime shift.
    RegimeChange,
}

impl PredictionKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::NextTask => "next_task",
            Self::FollowUp => "follow_up",
            Self::TelosProgress => "telos_progress",
            Self::EntityState => "entity_state",
            Self::RegimeChange => "regime_change",
        }
    }
}

/// Which generator produced the prediction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PredictionSource {
    /// PPM-C variable-order Markov.
    SequencePPM,
    /// augurs AutoETS extrapolation.
    TrendETS,
    /// LLM-assisted episodic pattern.
    EpisodicPattern,
    /// Causal model (Initiative 5, future).
    CausalModel,
    /// BOCPD changepoint detection.
    ChangePointDetection,
}

impl PredictionSource {
    pub fn as_str(&self) -> &str {
        match self {
            Self::SequencePPM => "sequence_ppm",
            Self::TrendETS => "trend_ets",
            Self::EpisodicPattern => "episodic_pattern",
            Self::CausalModel => "causal_model",
            Self::ChangePointDetection => "changepoint",
        }
    }
}

/// How a prediction was resolved.
#[derive(Debug, Clone)]
pub struct PredictionResolution {
    /// Whether the prediction matched an observation.
    pub matched: bool,
    /// The memory ID that matched (if any).
    pub actual_observation_id: Option<Uuid>,
    /// Prediction error: 0.0 = perfect match, 1.0 = total miss.
    pub prediction_error: f64,
    /// When the prediction was resolved (ms since epoch).
    pub resolved_at: i64,
}

/// Result from a prediction generation cycle.
#[derive(Debug, Clone)]
pub struct PredictionCycleResult {
    /// New predictions generated.
    pub predictions_generated: usize,
    /// Expired predictions resolved as misses.
    pub expired_resolved: usize,
    /// Current regime stability (from BOCPD).
    pub regime_stable: bool,
}

// ── Generator 1: PPM-C Sequence Prediction ─────────────────────────────

/// Load or create the PPM model for an agent, generate next-task predictions.
pub fn generate_sequence_predictions(
    store: &CozoStore,
    agent_id: &str,
    namespace: &str,
    max_depth: usize,
) -> Result<Vec<Prediction>> {
    // Load existing PPM model or create fresh
    let _loaded = load_ppm_model(store, agent_id)?;

    // Fetch recent task types for this agent (ordered by time)
    let result = store.run_query(
        r#"?[task_type, task_id] :=
            *task_outcomes{task_id, agent_id, task_type},
            agent_id = $agent_id,
            task_type != ""
        :sort task_id"#,
        {
            let mut p = BTreeMap::new();
            p.insert("agent_id".into(), cozo::DataValue::from(agent_id));
            p
        },
    )?;

    let task_types: Vec<String> = result.rows.iter()
        .filter_map(|row| row[0].get_str().map(|s| s.to_string()))
        .collect();

    if task_types.is_empty() {
        return Ok(Vec::new());
    }

    // Re-train the model on the full sequence
    // (In production, we'd do incremental updates, but for correctness
    //  we rebuild from scratch — the PPM tree is small and fast)
    let mut model = PpmModel::new(max_depth);
    let mut context: Vec<String> = Vec::new();
    for task_type in &task_types {
        model.update(&context, task_type);
        context.push(task_type.clone());
        if context.len() > max_depth {
            context.remove(0);
        }
    }

    // Generate predictions from the current context
    let predictions_raw = model.predict(&context);

    // Save updated model
    save_ppm_model(store, agent_id, &model)?;

    // Convert to Prediction structs
    let now = now_ms();
    let hour_ms: i64 = 3_600_000;

    let predictions = predictions_raw
        .into_iter()
        .take(3) // top 3 predictions
        .filter(|(_, conf)| *conf > 0.1) // minimum confidence threshold
        .map(|(task_type, confidence)| Prediction {
            id: Uuid::now_v7(),
            kind: PredictionKind::NextTask,
            content: format!("next_task:{}", task_type),
            embedding: Vec::new(), // sequence predictions don't need embeddings
            predicted_at: now,
            expected_by: now + hour_ms, // expect within 1 hour
            confidence,
            confidence_interval: None,
            source: PredictionSource::SequencePPM,
            context_ids: Vec::new(),
            namespace: namespace.to_string(),
            resolved: false,
            resolution: None,
        })
        .collect();

    Ok(predictions)
}

// ── Generator 2: ETS Trend Extrapolation ───────────────────────────────

/// Generate telos progress predictions using AutoETS.
pub fn generate_telos_predictions(
    store: &CozoStore,
    namespace: &str,
) -> Result<Vec<Prediction>> {
    use augurs::ets::AutoETS;
    use augurs::{Fit, Predict};

    let active_telos = store.list_active_telos(namespace, 50)?;
    let mut predictions = Vec::new();
    let now = now_ms();
    let hour_ms: i64 = 3_600_000;

    for telos in &active_telos {
        // Fetch progress history from telos_events
        let result = store.run_query(
            r#"?[impact, ts] :=
                *telos_events{telos_id, event_type, impact, ts},
                telos_id = to_uuid($telos_id),
                event_type = "progress"
            :sort ts"#,
            {
                let mut p = BTreeMap::new();
                p.insert("telos_id".into(), cozo::DataValue::from(telos.id.to_string()));
                p
            },
        )?;

        // Need at least 3 data points for ETS
        let progress_values: Vec<f64> = result.rows.iter()
            .filter_map(|row| row[0].get_float())
            .collect();

        if progress_values.len() < 3 {
            continue;
        }

        // Fit AutoETS and predict next step
        let unfit = AutoETS::non_seasonal();
        let fitted = match unfit.fit(&progress_values) {
            Ok(m) => m,
            Err(_) => continue, // Skip if ETS can't fit
        };

        let forecast = match fitted.predict(1, Some(0.95)) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let point = forecast.point[0];
        let (ci_lower, ci_upper) = forecast.intervals.as_ref()
            .map(|iv| (iv.lower[0], iv.upper[0]))
            .unwrap_or((point - 0.1, point + 0.1));

        // Confidence from interval width: narrow = high confidence
        let interval_width = (ci_upper - ci_lower).abs();
        let confidence = f64::clamp(1.0 - interval_width, 0.1, 0.99);

        // Detect unexpected stall: if predicted progress > current but barely moved
        let expected_progress = f64::clamp(point, 0.0, 1.0);
        let is_stall = expected_progress > telos.progress + 0.05
            && telos.progress < ci_lower;

        let content = if is_stall {
            format!(
                "telos_stall:{}:expected_progress={:.2},actual={:.2}",
                telos.id, expected_progress, telos.progress
            )
        } else {
            format!(
                "telos_progress:{}:predicted={:.2}",
                telos.id, expected_progress
            )
        };

        predictions.push(Prediction {
            id: Uuid::now_v7(),
            kind: PredictionKind::TelosProgress,
            content,
            embedding: Vec::new(),
            predicted_at: now,
            expected_by: now + 2 * hour_ms, // expect within 2 hours
            confidence,
            confidence_interval: Some((ci_lower, ci_upper)),
            source: PredictionSource::TrendETS,
            context_ids: vec![telos.id],
            namespace: namespace.to_string(),
            resolved: false,
            resolution: None,
        });
    }

    Ok(predictions)
}

// ── Generator 3: BOCPD Regime Change Detection ─────────────────────────

/// Feed surprise values to the BOCPD detector and check for changepoints.
pub fn detect_regime_changes(
    store: &CozoStore,
    namespace: &str,
) -> Result<(Vec<Prediction>, bool)> {
    use changepoint::{BocpdLike, BocpdTruncated};
    use changepoint::rv::prelude::NormalGamma;

    // Fetch recent surprise values
    let result = store.run_query(
        r#"?[surprise, ts] :=
            *surprise_log{surprise, ts, resolved},
            resolved = false
        :sort ts
        :limit 500"#,
        BTreeMap::new(),
    )?;

    let surprise_values: Vec<f64> = result.rows.iter()
        .filter_map(|row| row[0].get_float())
        .collect();

    if surprise_values.len() < 10 {
        return Ok((Vec::new(), true)); // too few points, assume stable
    }

    // Initialize BOCPD with Normal-Gamma prior
    let prior = NormalGamma::new_unchecked(0.0, 1.0, 1.0, 0.01);
    let mut detector = BocpdTruncated::new(250.0, prior);

    let mut regime_stable = true;
    let mut predictions = Vec::new();
    let now = now_ms();

    // Feed all observations; check last few for changepoints
    let check_window = surprise_values.len().saturating_sub(5);
    for (i, &obs) in surprise_values.iter().enumerate() {
        let run_lengths = detector.step(&obs);

        // Only check the last few observations for changepoints
        if i >= check_window {
            // A changepoint is detected when the probability of a short run length
            // (< 5) exceeds 0.5 — meaning the model thinks data recently changed
            let short_run_prob: f64 = run_lengths.iter().take(5).sum();
            if short_run_prob > 0.5 {
                regime_stable = false;
                if i == surprise_values.len() - 1 {
                    // Only create a prediction for the most recent changepoint
                    predictions.push(Prediction {
                        id: Uuid::now_v7(),
                        kind: PredictionKind::RegimeChange,
                        content: format!(
                            "regime_change:surprise_shift:p={:.2}",
                            short_run_prob
                        ),
                        embedding: Vec::new(),
                        predicted_at: now,
                        expected_by: now + 3_600_000, // informational
                        confidence: short_run_prob,
                        confidence_interval: None,
                        source: PredictionSource::ChangePointDetection,
                        context_ids: Vec::new(),
                        namespace: namespace.to_string(),
                        resolved: false,
                        resolution: None,
                    });
                }
            }
        }
    }

    Ok((predictions, regime_stable))
}

// ── Generator 4: Episodic Pattern Prediction (LLM-assisted) ────────────

/// Generate predictions by asking the LLM to extrapolate from recent episodes
/// and active goals. Runs less frequently than the algorithmic generators
/// (every 5th prediction cycle) because it's the most expensive.
pub async fn generate_episodic_predictions(
    store: &CozoStore,
    llm: &dyn crate::services::traits::LlmService,
    embedder: &dyn crate::services::traits::Embedder,
    namespace: &str,
) -> Result<Vec<Prediction>> {
    // Gather recent episodes (last 10)
    let episodes = store.run_query(
        r#"?[id, summary, outcome] :=
            *episodes{id, summary, outcome, @ 'NOW'},
            summary != ""
        :sort -id
        :limit 10"#,
        BTreeMap::new(),
    );

    let episode_summaries: Vec<String> = episodes
        .map(|r| {
            r.rows.iter()
                .filter_map(|row| {
                    let summary = row[1].get_str()?.to_string();
                    let outcome = row[2].get_str().unwrap_or("ongoing");
                    Some(format!("- [{}] {}", outcome, summary))
                })
                .collect()
        })
        .unwrap_or_default();

    // Gather active telos
    let active_telos = store.list_active_telos(namespace, 10)?;
    let telos_lines: Vec<String> = active_telos.iter()
        .map(|t| format!("- {} ({:.0}% complete)", t.title, t.progress * 100.0))
        .collect();

    if episode_summaries.is_empty() && telos_lines.is_empty() {
        return Ok(Vec::new());
    }

    // Ask LLM for 1-3 predictions
    let prompt = format!(
        concat!(
            "Based on these recent episodes and active goals, predict 1-3 things ",
            "that are likely to happen next. Each prediction should be specific and ",
            "falsifiable.\n\n",
            "Recent episodes:\n{episodes}\n\n",
            "Active goals:\n{goals}\n\n",
            "Return a JSON array of objects, each with:\n",
            "- \"prediction\": one-sentence prediction\n",
            "- \"confidence\": float 0.0-1.0\n",
            "- \"expected_within_hours\": integer\n",
            "No markdown fences. Just the JSON array."
        ),
        episodes = if episode_summaries.is_empty() {
            "None yet.".to_string()
        } else {
            episode_summaries.join("\n")
        },
        goals = if telos_lines.is_empty() {
            "None active.".to_string()
        } else {
            telos_lines.join("\n")
        },
    );

    use crate::services::traits::Message;
    let response = llm.complete(
        &[
            Message {
                role: "system".into(),
                content: "You predict near-future events based on patterns. Return valid JSON only.".into(),
            },
            Message {
                role: "user".into(),
                content: prompt,
            },
        ],
        512,
    ).await.map_err(|e| MemoriaError::Llm(e.to_string()))?;

    // Parse LLM response
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&response.content)
        .unwrap_or_default();

    let now = now_ms();
    let hour_ms: i64 = 3_600_000;
    let mut predictions = Vec::new();

    for item in parsed.iter().take(3) {
        let text = item.get("prediction")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let confidence = item.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5)
            .clamp(0.1, 0.95);
        let hours = item.get("expected_within_hours")
            .and_then(|v| v.as_i64())
            .unwrap_or(4);

        if text.is_empty() {
            continue;
        }

        // Embed the prediction for matching
        let embedding = embedder.embed(&[text])
            .await
            .map_err(|e| MemoriaError::Embedding(e.to_string()))?
            .into_iter()
            .next()
            .unwrap_or_default();

        predictions.push(Prediction {
            id: Uuid::now_v7(),
            kind: PredictionKind::FollowUp,
            content: format!("episodic:{}", text),
            embedding,
            predicted_at: now,
            expected_by: now + hours * hour_ms,
            confidence,
            confidence_interval: None,
            source: PredictionSource::EpisodicPattern,
            context_ids: Vec::new(),
            namespace: namespace.to_string(),
            resolved: false,
            resolution: None,
        });
    }

    Ok(predictions)
}

// ── Prediction Storage ─────────────────────────────────────────────────

/// Store a batch of predictions in CozoDB.
pub fn store_predictions(store: &CozoStore, predictions: &[Prediction]) -> Result<()> {
    for pred in predictions {
        let mut params = BTreeMap::new();
        params.insert("id".into(), cozo::DataValue::from(pred.id.to_string()));
        params.insert("kind".into(), cozo::DataValue::from(pred.kind.as_str()));
        params.insert("content".into(), cozo::DataValue::from(pred.content.as_str()));
        params.insert("predicted_at".into(), cozo::DataValue::from(pred.predicted_at));
        params.insert("expected_by".into(), cozo::DataValue::from(pred.expected_by));
        params.insert("confidence".into(), cozo::DataValue::from(pred.confidence));
        params.insert(
            "ci_lower".into(),
            cozo::DataValue::from(pred.confidence_interval.map_or(0.0, |ci| ci.0)),
        );
        params.insert(
            "ci_upper".into(),
            cozo::DataValue::from(pred.confidence_interval.map_or(1.0, |ci| ci.1)),
        );
        params.insert("source".into(), cozo::DataValue::from(pred.source.as_str()));
        params.insert(
            "context_ids".into(),
            cozo::DataValue::from(
                serde_json::to_string(&pred.context_ids)
                    .unwrap_or_else(|_| "[]".to_string()),
            ),
        );
        params.insert("namespace".into(), cozo::DataValue::from(pred.namespace.as_str()));

        store.run_script(
            r#"?[id, valid_at, kind, content, predicted_at, expected_by,
               confidence, ci_lower, ci_upper, source, context_ids, namespace] <- [
                [to_uuid($id), 'ASSERT', $kind, $content, $predicted_at, $expected_by,
                 $confidence, $ci_lower, $ci_upper, $source, $context_ids, $namespace]
            ]
            :put predictions {
                id, valid_at =>
                kind, content, predicted_at, expected_by,
                confidence, ci_lower, ci_upper, source, context_ids, namespace
            }"#,
            params,
        )?;
    }

    Ok(())
}

/// Get all pending (unresolved, not expired) predictions for a namespace.
pub fn get_pending_predictions(
    store: &CozoStore,
    namespace: &str,
    now: i64,
) -> Result<Vec<Prediction>> {
    let mut params = BTreeMap::new();
    params.insert("namespace".into(), cozo::DataValue::from(namespace));
    params.insert("now".into(), cozo::DataValue::from(now));

    let result = store.run_script(
        r#"?[id, kind, content, predicted_at, expected_by, confidence,
           ci_lower, ci_upper, source, context_ids, namespace] :=
            *predictions{id, kind, content, predicted_at, expected_by,
                         confidence, ci_lower, ci_upper, source, context_ids,
                         namespace, resolved, @ 'NOW'},
            namespace = $namespace,
            resolved = false,
            expected_by >= $now"#,
        params,
    )?;

    parse_prediction_rows(&result)
}

/// Get expired unresolved predictions (for scoring as misses).
pub fn get_expired_unresolved(
    store: &CozoStore,
    namespace: &str,
    now: i64,
) -> Result<Vec<Prediction>> {
    let mut params = BTreeMap::new();
    params.insert("namespace".into(), cozo::DataValue::from(namespace));
    params.insert("now".into(), cozo::DataValue::from(now));

    let result = store.run_script(
        r#"?[id, kind, content, predicted_at, expected_by, confidence,
           ci_lower, ci_upper, source, context_ids, namespace] :=
            *predictions{id, kind, content, predicted_at, expected_by,
                         confidence, ci_lower, ci_upper, source, context_ids,
                         namespace, resolved, @ 'NOW'},
            namespace = $namespace,
            resolved = false,
            expected_by < $now"#,
        params,
    )?;

    parse_prediction_rows(&result)
}

/// Resolve a prediction as matched or missed.
pub fn resolve_prediction(
    store: &CozoStore,
    prediction_id: Uuid,
    matched: bool,
    actual_memory_id: Option<Uuid>,
    prediction_error: f64,
) -> Result<()> {
    let now = now_ms();

    // Read existing prediction to carry forward non-updated columns
    let mut read_params = BTreeMap::new();
    read_params.insert("id".into(), cozo::DataValue::from(prediction_id.to_string()));
    let existing = store.run_script(
        r#"?[kind, content, predicted_at, expected_by, confidence,
           ci_lower, ci_upper, source, context_ids, namespace] :=
            *predictions{id, kind, content, predicted_at, expected_by, confidence,
                         ci_lower, ci_upper, source, context_ids, namespace, @ 'NOW'},
            id = to_uuid($id)"#,
        read_params,
    )?;

    if existing.rows.is_empty() {
        return Err(MemoriaError::Store(format!(
            "prediction {} not found for resolution", prediction_id
        )));
    }
    let row = &existing.rows[0];

    let mut params = BTreeMap::new();
    params.insert("id".into(), cozo::DataValue::from(prediction_id.to_string()));
    // Carry forward original columns
    params.insert("kind".into(), row[0].clone());
    params.insert("content".into(), row[1].clone());
    params.insert("predicted_at".into(), row[2].clone());
    params.insert("expected_by".into(), row[3].clone());
    params.insert("confidence".into(), row[4].clone());
    params.insert("ci_lower".into(), row[5].clone());
    params.insert("ci_upper".into(), row[6].clone());
    params.insert("source".into(), row[7].clone());
    params.insert("context_ids".into(), row[8].clone());
    params.insert("namespace".into(), row[9].clone());
    // Resolution columns
    params.insert("matched".into(), cozo::DataValue::from(matched));
    params.insert("prediction_error".into(), cozo::DataValue::from(prediction_error));
    params.insert(
        "actual_memory_id".into(),
        cozo::DataValue::from(
            actual_memory_id.map_or(String::new(), |id| id.to_string()),
        ),
    );
    params.insert("resolved_at".into(), cozo::DataValue::from(now));

    store.run_script(
        r#"?[id, valid_at, kind, content, predicted_at, expected_by,
           confidence, ci_lower, ci_upper, source, context_ids, namespace,
           resolved, matched, prediction_error, actual_memory_id, resolved_at] <- [
            [to_uuid($id), 'ASSERT',
             $kind, $content, $predicted_at, $expected_by,
             $confidence, $ci_lower, $ci_upper, $source, $context_ids, $namespace,
             true, $matched, $prediction_error, $actual_memory_id, $resolved_at]
        ]
        :put predictions {
            id, valid_at =>
            kind, content, predicted_at, expected_by,
            confidence, ci_lower, ci_upper, source, context_ids, namespace,
            resolved, matched, prediction_error, actual_memory_id, resolved_at
        }"#,
        params,
    )?;

    Ok(())
}

/// Get recently resolved predictions for free energy computation.
pub fn get_recent_resolved(
    store: &CozoStore,
    namespace: &str,
) -> Result<Vec<(f64, f64)>> {
    let mut params = BTreeMap::new();
    params.insert("namespace".into(), cozo::DataValue::from(namespace));

    let result = store.run_script(
        r#"?[id, confidence, prediction_error] :=
            *predictions{id, confidence, prediction_error, namespace, resolved},
            namespace = $namespace,
            resolved = true"#,
        params,
    )?;

    Ok(result.rows.iter()
        .filter_map(|row| {
            // row[0] = id (for uniqueness), row[1] = confidence, row[2] = prediction_error
            let conf = row[1].get_float()?;
            let err = row[2].get_float()?;
            Some((conf, err))
        })
        .collect())
}

/// Compute rolling prediction accuracy for a namespace.
pub fn prediction_accuracy(store: &CozoStore, namespace: &str) -> Result<f64> {
    let resolved = get_recent_resolved(store, namespace)?;
    if resolved.is_empty() {
        return Ok(0.5); // prior: 50% accuracy
    }

    let avg_error: f64 = resolved.iter().map(|(_, err)| err).sum::<f64>() / resolved.len() as f64;
    Ok(1.0 - avg_error)
}

// ── PPM Model Persistence ──────────────────────────────────────────────

fn load_ppm_model(store: &CozoStore, agent_id: &str) -> Result<Option<PpmModel>> {
    let mut params = BTreeMap::new();
    params.insert("agent_id".into(), cozo::DataValue::from(agent_id));

    let result = store.run_script(
        r#"?[tree_blob] :=
            *ppm_tree{agent_id, tree_blob, @ 'NOW'},
            agent_id = $agent_id"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    let blob = match &result.rows[0][0] {
        cozo::DataValue::Bytes(b) => b.to_vec(),
        _ => return Ok(None),
    };

    PpmModel::from_bytes(&blob)
        .map(Some)
        .map_err(|e| MemoriaError::Store(format!("deserializing PPM model: {e}")))
}

fn save_ppm_model(store: &CozoStore, agent_id: &str, model: &PpmModel) -> Result<()> {
    let blob = model.to_bytes()
        .map_err(|e| MemoriaError::Store(format!("serializing PPM model: {e}")))?;

    let now = now_ms();
    let mut params = BTreeMap::new();
    params.insert("agent_id".into(), cozo::DataValue::from(agent_id));
    params.insert("tree_blob".into(), cozo::DataValue::Bytes(blob));
    params.insert("max_depth".into(), cozo::DataValue::from(model.max_depth as i64));
    params.insert("alphabet_size".into(), cozo::DataValue::from(model.alphabet_size() as i64));
    params.insert("updated_at".into(), cozo::DataValue::from(now));

    store.run_script(
        r#"?[agent_id, valid_at, tree_blob, max_depth, alphabet_size, updated_at] <- [
            [$agent_id, 'ASSERT', $tree_blob, $max_depth, $alphabet_size, $updated_at]
        ]
        :put ppm_tree {agent_id, valid_at => tree_blob, max_depth, alphabet_size, updated_at}"#,
        params,
    )?;

    Ok(())
}

// ── Match Predictions Against Observations ─────────────────────────────

/// Result of matching an observation against pending predictions.
#[derive(Debug, Clone)]
pub struct PredictionMatchResult {
    /// Total prediction surprise delta (negative = expected, positive = unexpected).
    pub surprise_delta: f64,
    /// Number of predictions that matched this observation.
    pub matches: usize,
    /// Number of expired predictions resolved as misses.
    pub expirations: usize,
}

/// Match a new observation against pending predictions.
///
/// Called from `tell()` after storing the observation. Checks:
/// 1. Pending predictions with content similarity above threshold
/// 2. Expired unresolved predictions (absence of expected = surprise)
///
/// For NextTask predictions, uses exact string matching on task type.
/// For other predictions, uses embedding cosine similarity.
pub fn match_observation_against_predictions(
    store: &CozoStore,
    observation_content: &str,
    observation_embedding: &[f32],
    observation_id: Uuid,
    namespace: &str,
) -> Result<PredictionMatchResult> {
    let now = now_ms();
    let mut surprise_delta = 0.0f64;
    let mut match_count = 0usize;

    // 1. Check pending predictions for matches
    let pending = get_pending_predictions(store, namespace, now)?;
    for pred in &pending {
        let matched = match pred.kind {
            PredictionKind::NextTask => {
                // Exact match: "next_task:debug" vs observation containing "debug"
                if let Some(predicted_type) = pred.content.strip_prefix("next_task:") {
                    observation_content.contains(predicted_type)
                } else {
                    false
                }
            }
            _ => {
                // Embedding similarity for other prediction types
                if !pred.embedding.is_empty() && !observation_embedding.is_empty() {
                    let sim = cosine_similarity(observation_embedding, &pred.embedding);
                    sim > 0.6
                } else {
                    // Fall back to content containment for predictions without embeddings
                    let pred_key = pred.content.split(':').nth(1).unwrap_or(&pred.content);
                    observation_content.contains(pred_key)
                }
            }
        };

        if matched {
            let error = match pred.kind {
                PredictionKind::NextTask => 0.0, // exact match = perfect
                _ => {
                    if !pred.embedding.is_empty() && !observation_embedding.is_empty() {
                        1.0 - cosine_similarity(observation_embedding, &pred.embedding)
                    } else {
                        0.2 // partial match assumed
                    }
                }
            };

            resolve_prediction(store, pred.id, true, Some(observation_id), error)?;
            surprise_delta -= pred.confidence * (1.0 - error);
            match_count += 1;
        }
    }

    // 2. Resolve expired unmatched predictions (absence of expected = surprise)
    let expired = get_expired_unresolved(store, namespace, now)?;
    let expiration_count = expired.len();
    for pred in &expired {
        resolve_prediction(store, pred.id, false, None, 1.0)?;
        surprise_delta += pred.confidence; // absence of expected = surprise
    }

    Ok(PredictionMatchResult {
        surprise_delta,
        matches: match_count,
        expirations: expiration_count,
    })
}

// ── Helper: Parse prediction rows from CozoDB query result ─────────────

fn parse_prediction_rows(result: &cozo::NamedRows) -> Result<Vec<Prediction>> {
    let mut predictions = Vec::new();
    for row in &result.rows {
        let id = crate::store::cozo::parse_uuid_pub(&row[0])
            .map_err(|e| MemoriaError::Store(format!("bad prediction id: {e}")))?;

        let kind = match row[1].get_str().unwrap_or("") {
            "next_task" => PredictionKind::NextTask,
            "follow_up" => PredictionKind::FollowUp,
            "telos_progress" => PredictionKind::TelosProgress,
            "entity_state" => PredictionKind::EntityState,
            "regime_change" => PredictionKind::RegimeChange,
            _ => PredictionKind::NextTask,
        };

        let source = match row[8].get_str().unwrap_or("") {
            "sequence_ppm" => PredictionSource::SequencePPM,
            "trend_ets" => PredictionSource::TrendETS,
            "episodic_pattern" => PredictionSource::EpisodicPattern,
            "causal_model" => PredictionSource::CausalModel,
            "changepoint" => PredictionSource::ChangePointDetection,
            _ => PredictionSource::SequencePPM,
        };

        predictions.push(Prediction {
            id,
            kind,
            content: row[2].get_str().unwrap_or("").to_string(),
            embedding: Vec::new(), // embeddings stored separately via HNSW
            predicted_at: row[3].get_int().unwrap_or(0),
            expected_by: row[4].get_int().unwrap_or(0),
            confidence: row[5].get_float().unwrap_or(0.5),
            confidence_interval: {
                let lower = row[6].get_float().unwrap_or(0.0);
                let upper = row[7].get_float().unwrap_or(1.0);
                if lower == 0.0 && upper == 1.0 {
                    None
                } else {
                    Some((lower, upper))
                }
            },
            source,
            context_ids: Vec::new(), // parsed from JSON if needed
            namespace: row[10].get_str().unwrap_or("").to_string(),
            resolved: false,
            resolution: None,
        });
    }

    Ok(predictions)
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::CozoStore;

    #[test]
    fn test_store_and_retrieve_predictions() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = now_ms();

        let pred = Prediction {
            id: Uuid::now_v7(),
            kind: PredictionKind::NextTask,
            content: "next_task:debug".into(),
            embedding: Vec::new(),
            predicted_at: now,
            expected_by: now + 3_600_000,
            confidence: 0.8,
            confidence_interval: None,
            source: PredictionSource::SequencePPM,
            context_ids: Vec::new(),
            namespace: "default".into(),
            resolved: false,
            resolution: None,
        };

        store_predictions(&store, &[pred.clone()]).unwrap();

        let pending = get_pending_predictions(&store, "default", now).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].content, "next_task:debug");
        assert_eq!(pending[0].confidence, 0.8);
    }

    #[test]
    fn test_resolve_prediction() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = now_ms();
        let pred_id = Uuid::now_v7();

        let pred = Prediction {
            id: pred_id,
            kind: PredictionKind::NextTask,
            content: "next_task:test".into(),
            embedding: Vec::new(),
            predicted_at: now,
            expected_by: now + 3_600_000,
            confidence: 0.7,
            confidence_interval: None,
            source: PredictionSource::SequencePPM,
            context_ids: Vec::new(),
            namespace: "default".into(),
            resolved: false,
            resolution: None,
        };

        store_predictions(&store, &[pred]).unwrap();
        resolve_prediction(&store, pred_id, true, None, 0.1).unwrap();

        // Should no longer appear in pending
        let pending = get_pending_predictions(&store, "default", now).unwrap();
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn test_expired_predictions() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = now_ms();

        let pred = Prediction {
            id: Uuid::now_v7(),
            kind: PredictionKind::NextTask,
            content: "next_task:deploy".into(),
            embedding: Vec::new(),
            predicted_at: now - 7_200_000, // 2 hours ago
            expected_by: now - 3_600_000,  // expired 1 hour ago
            confidence: 0.9,
            confidence_interval: None,
            source: PredictionSource::SequencePPM,
            context_ids: Vec::new(),
            namespace: "default".into(),
            resolved: false,
            resolution: None,
        };

        store_predictions(&store, &[pred]).unwrap();

        let expired = get_expired_unresolved(&store, "default", now).unwrap();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].content, "next_task:deploy");
    }

    #[test]
    fn test_prediction_accuracy_default() {
        let store = CozoStore::open_mem(4).unwrap();
        let acc = prediction_accuracy(&store, "default").unwrap();
        assert!((acc - 0.5).abs() < 0.01, "default accuracy should be 0.5");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}
