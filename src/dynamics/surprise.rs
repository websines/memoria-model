//! Precision-weighted surprise computation.
//!
//! Surprise measures how much a new observation violates existing beliefs.
//! It drives all downstream dynamics: reflection, reconsolidation, compression.
//!
//! Formula: surprise = prediction_error × observation_precision
//!
//! High-precision observation contradicting low-precision belief → high surprise
//! Low-precision observation contradicting high-precision belief → low surprise

use std::collections::BTreeMap;

use cozo::DataValue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::fact::Fact;

/// An incoming observation to compare against existing beliefs.
#[derive(Debug, Clone)]
pub struct Observation {
    pub content: String,
    pub predicate: Option<String>,
    pub object_value: Option<String>,
    pub confidence: f64,
    pub provenance: String,
    pub source: String,
}

/// Result of surprise computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseResult {
    /// Raw surprise value (prediction_error × observation_precision).
    pub surprise: f64,
    /// Kalman-like gain: how much the belief should shift toward the observation.
    pub gain: f64,
    /// Whether this surprise should trigger memory reconsolidation.
    pub should_reconsolidate: bool,
    /// Whether this surprise should be attributed to specific causes.
    pub should_attribute: bool,
}

impl Default for SurpriseResult {
    fn default() -> Self {
        Self {
            surprise: 0.0,
            gain: 0.0,
            should_reconsolidate: false,
            should_attribute: false,
        }
    }
}

/// Weight factor for different provenance types.
/// Higher = more trusted source.
pub fn provenance_weight(provenance: &str) -> f64 {
    match provenance {
        "direct" => 1.0,
        "user_stated" => 0.95,
        "agent_reported" => 0.8,
        "extracted" => 0.6,
        "inferred" => 0.3,
        _ => 0.5,
    }
}

/// Compute surprise from a new observation against an existing belief (fact).
///
/// Uses precision-weighted prediction error via the canonical AIF precision formulas:
/// - belief_precision = confidence × ln(max(reinforcement_count, 1) + 1)
/// - obs_precision = confidence × provenance_weight
/// - gain = obs_precision / (belief_precision + obs_precision)  (Kalman-like)
/// - surprise = prediction_error × obs_precision
pub fn compute_surprise(belief: &Fact, observation: &Observation) -> SurpriseResult {
    let belief_precision =
        crate::aif::belief_update::precision(belief.confidence, belief.reinforcement_count);
    let obs_precision = crate::aif::belief_update::observation_precision(
        observation.confidence,
        provenance_weight(&observation.provenance),
    );

    let gain = obs_precision / (belief_precision + obs_precision);

    let prediction_error = if contradicts(belief, observation) {
        1.0
    } else {
        0.0
    };

    let surprise = prediction_error * obs_precision;

    SurpriseResult {
        surprise,
        gain,
        should_reconsolidate: surprise > 0.0 && gain > 0.3,
        should_attribute: surprise > 2.0,
    }
}

/// Check if an observation contradicts an existing fact.
///
/// A contradiction occurs when both refer to the same predicate
/// but have different object values.
fn contradicts(belief: &Fact, observation: &Observation) -> bool {
    // Must have matching predicates to be comparable
    let Some(obs_pred) = &observation.predicate else {
        return false;
    };
    if belief.predicate != *obs_pred {
        return false;
    }

    // Check if object values differ
    match (&belief.object_value, &observation.object_value) {
        (Some(belief_val), Some(obs_val)) => {
            belief_val.to_lowercase() != obs_val.to_lowercase()
        }
        _ => false,
    }
}

/// Log a surprise event to the surprise_log relation in CozoDB.
pub fn log_surprise(
    store: &CozoStore,
    surprise: f64,
    source: &str,
    variable_id: Option<Uuid>,
    observation_summary: &str,
) -> Result<()> {
    let mut params = BTreeMap::new();
    params.insert("surprise".into(), DataValue::from(surprise));
    params.insert("source".into(), DataValue::from(source));
    params.insert(
        "observation_summary".into(),
        DataValue::from(observation_summary),
    );

    let id = Uuid::now_v7();
    let now = crate::types::memory::now_ms();
    params.insert("id".into(), DataValue::from(id.to_string()));
    params.insert("ts".into(), DataValue::from(now));

    if let Some(vid) = variable_id {
        params.insert("variable_id".into(), DataValue::from(vid.to_string()));
    } else {
        params.insert("variable_id".into(), DataValue::Null);
    }
    params.insert("factor_id".into(), DataValue::Null);

    store.run_script(
        concat!(
            "?[id, ts, surprise, source, variable_id, factor_id, ",
            "observation_summary, resolved] <- ",
            "[[$id, $ts, $surprise, $source, $variable_id, ",
            "$factor_id, $observation_summary, false]] ",
            ":put surprise_log {id, ts => surprise, source, variable_id, ",
            "factor_id, observation_summary, resolved}",
        ),
        params,
    )?;

    Ok(())
}

/// Get the total unresolved surprise accumulated since the last reflection.
pub fn accumulated_unresolved_surprise(store: &CozoStore) -> Result<f64> {
    let result = store.run_query(
        r#"?[sum(surprise)] := *surprise_log{surprise, resolved}, resolved = false"#,
        BTreeMap::new(),
    )?;

    if result.rows.is_empty() {
        return Ok(0.0);
    }

    Ok(result.rows[0][0].get_float().unwrap_or(0.0))
}

/// Mark all unresolved surprise entries as resolved.
pub fn resolve_all_surprise(store: &CozoStore) -> Result<usize> {
    // First get all unresolved entries
    let result = store.run_query(
        r#"?[id, ts, surprise, source, variable_id, factor_id,
            observation_summary] :=
            *surprise_log{id, ts, surprise, source, variable_id,
                         factor_id, observation_summary, resolved},
            resolved = false"#,
        BTreeMap::new(),
    )?;

    let count = result.rows.len();
    if count == 0 {
        return Ok(0);
    }

    // Update each to resolved=true
    for row in &result.rows {
        let mut params = BTreeMap::new();
        params.insert("id".into(), row[0].clone());
        params.insert("ts".into(), row[1].clone());
        params.insert("surprise".into(), row[2].clone());
        params.insert("source".into(), row[3].clone());
        params.insert("variable_id".into(), row[4].clone());
        params.insert("factor_id".into(), row[5].clone());
        params.insert("observation_summary".into(), row[6].clone());

        store.run_script(
            concat!(
                "?[id, ts, surprise, source, variable_id, factor_id, ",
                "observation_summary, resolved] <- ",
                "[[$id, $ts, $surprise, $source, $variable_id, ",
                "$factor_id, $observation_summary, true]] ",
                ":put surprise_log {id, ts => surprise, source, variable_id, ",
                "factor_id, observation_summary, resolved}",
            ),
            params,
        )?;
    }

    Ok(count)
}

// ── Telos surprise ──

/// Compute surprise for a telos event.
///
/// Telos surprise captures unexpected changes to goals:
/// - Dependency failure (high surprise, proportional to priority)
/// - Deadline change (moderate surprise)
/// - Key entity state change (variable surprise)
/// - Unexpected completion/failure
///
/// The surprise is weighted by the telos priority and urgency,
/// then logged to the surprise_log for consolidation.
pub fn compute_telos_surprise(
    store: &CozoStore,
    telos_id: Uuid,
    event_type: &str,
    impact: f64,
) -> Result<f64> {
    // Get the telos to compute weighted surprise
    let telos = store.get_telos(telos_id)?
        .ok_or_else(|| crate::error::MemoriaError::NotFound(telos_id))?;

    // Base surprise from event type
    let base_surprise = match event_type {
        "dependency_failed" => 1.5,
        "deadline_changed" => 0.8,
        "entity_state_change" => 1.0,
        "unexpected_completion" => 1.2,
        "unexpected_failure" => 1.8,
        "blocked" => 1.0,
        "stalled" => 0.5,
        _ => impact.abs().min(2.0),
    };

    // Weight by priority and urgency: important/urgent goals generate more surprise
    let weight = telos.priority * (1.0 + telos.urgency);
    let surprise = base_surprise * weight;

    // Log to surprise_log
    let summary = format!(
        "Telos '{}' event: {} (impact: {:.2})",
        telos.title, event_type, impact
    );
    log_surprise(
        store,
        surprise,
        &format!("telos:{}", event_type),
        Some(telos_id),
        &summary,
    )?;

    Ok(surprise)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_belief(predicate: &str, value: &str, confidence: f64, rc: i64) -> Fact {
        let mut f = Fact::with_value(Uuid::now_v7(), predicate, value);
        f.confidence = confidence;
        f.reinforcement_count = rc;
        f
    }

    fn make_observation(predicate: &str, value: &str, confidence: f64, prov: &str) -> Observation {
        Observation {
            content: format!("{predicate}: {value}"),
            predicate: Some(predicate.to_string()),
            object_value: Some(value.to_string()),
            confidence,
            provenance: prov.to_string(),
            source: "test".to_string(),
        }
    }

    #[test]
    fn test_no_surprise_when_agreeing() {
        let belief = make_belief("role", "engineer", 1.0, 3);
        let obs = make_observation("role", "engineer", 1.0, "direct");

        let result = compute_surprise(&belief, &obs);
        assert_eq!(result.surprise, 0.0);
        assert!(!result.should_reconsolidate);
    }

    #[test]
    fn test_high_surprise_contradicting_weak_belief() {
        let belief = make_belief("role", "engineer", 0.5, 1);
        let obs = make_observation("role", "manager", 1.0, "direct");

        let result = compute_surprise(&belief, &obs);
        assert!(result.surprise > 0.0);
        assert!(result.gain > 0.3, "gain={} should trigger reconsolidation", result.gain);
        assert!(result.should_reconsolidate);
    }

    #[test]
    fn test_low_surprise_contradicting_strong_belief() {
        let belief = make_belief("role", "engineer", 1.0, 10);
        let obs = make_observation("role", "manager", 0.3, "inferred");

        let result = compute_surprise(&belief, &obs);
        assert!(result.surprise > 0.0);
        // Low-confidence inferred observation against strong belief → low gain
        assert!(result.gain < 0.3, "gain={} should NOT trigger reconsolidation", result.gain);
        assert!(!result.should_reconsolidate);
    }

    #[test]
    fn test_different_predicates_no_contradiction() {
        let belief = make_belief("role", "engineer", 1.0, 1);
        let obs = make_observation("location", "NYC", 1.0, "direct");

        let result = compute_surprise(&belief, &obs);
        assert_eq!(result.surprise, 0.0);
    }

    #[test]
    fn test_provenance_weights() {
        assert_eq!(provenance_weight("direct"), 1.0);
        assert!(provenance_weight("inferred") < provenance_weight("extracted"));
        assert!(provenance_weight("extracted") < provenance_weight("direct"));
    }

    #[test]
    fn test_telos_surprise_priority_weighting() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut t_high = crate::types::telos::Telos::new("Critical goal", "", vec![0.1; 4], "a", "u");
        t_high.priority = 0.9;
        t_high.urgency = 0.8;
        store.insert_telos(&t_high).unwrap();

        let mut t_low = crate::types::telos::Telos::new("Low goal", "", vec![0.1; 4], "a", "u");
        t_low.priority = 0.2;
        t_low.urgency = 0.1;
        store.insert_telos(&t_low).unwrap();

        let s_high = compute_telos_surprise(&store, t_high.id, "dependency_failed", 1.0).unwrap();
        let s_low = compute_telos_surprise(&store, t_low.id, "dependency_failed", 1.0).unwrap();

        assert!(
            s_high > s_low,
            "high priority surprise ({s_high}) should > low ({s_low})"
        );

        // Check it was logged
        let total = accumulated_unresolved_surprise(&store).unwrap();
        assert!(total > 0.0);
    }

    #[test]
    fn test_surprise_log_roundtrip() {
        let store = CozoStore::open_mem(4).unwrap();

        log_surprise(&store, 1.5, "new_fact", None, "Alice is now a manager").unwrap();
        log_surprise(&store, 0.8, "contradiction", None, "Bob left Acme").unwrap();

        let total = accumulated_unresolved_surprise(&store).unwrap();
        assert!((total - 2.3).abs() < 0.001, "total={total}, expected 2.3");

        let resolved = resolve_all_surprise(&store).unwrap();
        assert_eq!(resolved, 2);

        let after = accumulated_unresolved_surprise(&store).unwrap();
        assert_eq!(after, 0.0);
    }
}
