//! Bethe Free Energy computation over the entire knowledge store.
//!
//! Free energy `F = factor_energy - variable_entropy` where:
//! - `factor_energy = Σ -ln(precision_i + ε)` — measures how well factors predict observations
//! - `variable_entropy = Σ H(conf_i)` — measures uncertainty in beliefs
//! - `β = variable_entropy / (factor_energy + variable_entropy + ε)` — auto-tunes exploration

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::aif::belief_update::precision;
use crate::error::Result;
use crate::store::CozoStore;

const EPSILON: f64 = 1e-10;

/// Result of computing the Bethe Free Energy over the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyState {
    /// Total free energy: factor_energy - variable_entropy.
    pub free_energy: f64,
    /// Accuracy term: -Σ ln(precision + ε). Lower = better model fit.
    pub accuracy: f64,
    /// Complexity term: variable entropy. Lower = more certain beliefs.
    pub complexity: f64,
    /// Auto-tuned exploration parameter β ∈ [0, 1].
    /// High β → more exploration; low β → more exploitation.
    pub beta: f64,
    /// Total Shannon entropy across all variable nodes (fact confidences).
    pub variable_entropy: f64,
    /// Total factor energy across all factor nodes.
    pub factor_energy: f64,
}

impl Default for FreeEnergyState {
    fn default() -> Self {
        Self {
            free_energy: 0.0,
            accuracy: 0.0,
            complexity: 0.0,
            beta: 1.0, // maximum exploration when no data
            variable_entropy: 0.0,
            factor_energy: 0.0,
        }
    }
}

/// Extract success_rate from a CozoDB performance JSON column.
/// Falls back to 0.5 (prior for unknown skills) if missing/unparseable.
fn extract_success_rate(val: &cozo::DataValue) -> f64 {
    // Try to parse as JSON and extract success_rate
    let json_str = match val {
        cozo::DataValue::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    };
    // Try direct JSON parse
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
        if let Some(sr) = parsed.get("success_rate").and_then(|v| v.as_f64()) {
            return sr;
        }
    }
    // CozoDB Json debug format: json("{\"success_rate\":0.8,...}")
    if json_str.starts_with("json(") {
        let inner = json_str
            .trim_start_matches("json(\"")
            .trim_end_matches("\")")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&inner) {
            if let Some(sr) = parsed.get("success_rate").and_then(|v| v.as_f64()) {
                return sr;
            }
        }
    }
    0.5 // prior for untested/unknown skills
}

/// Binary Shannon entropy: H(c) = -c*ln(c) - (1-c)*ln(1-c).
/// Returns 0 for c ∈ {0, 1}, max ≈ 0.693 at c = 0.5.
fn binary_entropy(c: f64) -> f64 {
    if c <= 0.0 || c >= 1.0 {
        return 0.0;
    }
    -(c * c.ln() + (1.0 - c) * (1.0 - c).ln())
}

/// Compute the Bethe Free Energy across the entire knowledge store.
///
/// Aggregates:
/// 1. **Facts** — each fact contributes to both factor energy and variable entropy
/// 2. **Skills** — each skill contributes to factor energy (precision from confidence + version)
/// 3. **Entities** — each entity contributes to factor energy (precision from confidence + mention_count)
///
/// All aggregation is done in Rust (not CozoDB) to avoid CozoDB's aggregation-in-output gotcha.
pub fn compute_bethe_free_energy(store: &CozoStore) -> Result<FreeEnergyState> {
    let mut factor_energy = 0.0f64;
    let mut variable_entropy = 0.0f64;

    // 1. Facts: contribute both factor energy and variable entropy
    let facts_result = store.run_query(
        r#"?[confidence, reinforcement_count] := *facts{confidence, reinforcement_count, @ 'NOW'}"#,
        BTreeMap::new(),
    )?;

    for row in &facts_result.rows {
        let conf = row[0].get_float().unwrap_or(0.5);
        let rc = row[1].get_int().unwrap_or(1);

        let prec = precision(conf, rc);
        factor_energy += -(prec + EPSILON).ln();
        variable_entropy += binary_entropy(conf);
    }

    // 2. Skills: contribute to factor energy using confidence × success_rate (§15.2)
    let skills_result = store.run_query(
        r#"?[confidence, performance] := *skills{confidence, performance, @ 'NOW'}"#,
        BTreeMap::new(),
    )?;

    for row in &skills_result.rows {
        let conf = row[0].get_float().unwrap_or(0.5);
        // Extract success_rate from performance JSON
        let success_rate = extract_success_rate(&row[1]);
        let skill_precision = conf * success_rate;
        factor_energy += -(skill_precision + EPSILON).ln();
    }

    // 3. Entities: contribute to factor energy
    let entities_result = store.run_query(
        r#"?[confidence, mention_count] := *entities{confidence, mention_count, @ 'NOW'}"#,
        BTreeMap::new(),
    )?;

    for row in &entities_result.rows {
        let conf = row[0].get_float().unwrap_or(0.5);
        let mc = row[1].get_int().unwrap_or(1);

        let prec = precision(conf, mc);
        factor_energy += -(prec + EPSILON).ln();
    }

    // 4. Telos preference factors
    // Each active telos contributes: factor_energy += -ln(preference_precision) × goal_distance
    // Where goal_distance = 1.0 - progress, preference_precision = priority × confidence
    // This makes the system "want" to reduce goal distance — unachieved important goals
    // increase free energy, driving the system toward goal-directed action.
    let telos_result = store.run_query(
        r#"?[priority, confidence, progress] :=
            *telos{priority, confidence, progress, status},
            status = "active""#,
        BTreeMap::new(),
    )?;

    for row in &telos_result.rows {
        let priority = row[0].get_float().unwrap_or(0.5);
        let confidence = row[1].get_float().unwrap_or(0.5);
        let progress = row[2].get_float().unwrap_or(0.0);

        let preference_precision = priority * confidence;
        let goal_distance = 1.0 - progress;

        // Only contribute when goal is not yet achieved
        if goal_distance > EPSILON {
            factor_energy += -(preference_precision + EPSILON).ln() * goal_distance;
        }
    }

    // 5. Prediction accuracy: recently resolved predictions contribute to factor energy.
    //    Poor predictions → higher free energy → drives system to improve.
    let predictions_result = store.run_query(
        r#"?[confidence, prediction_error] :=
            *predictions{confidence, prediction_error, resolved, @ 'NOW'},
            resolved = true"#,
        BTreeMap::new(),
    );

    if let Ok(pred_rows) = predictions_result {
        if !pred_rows.rows.is_empty() {
            let avg_error: f64 = pred_rows.rows.iter()
                .filter_map(|row| {
                    let err = row[1].get_float()?;
                    Some(err)
                })
                .sum::<f64>() / pred_rows.rows.len() as f64;

            let prediction_precision = 1.0 - avg_error;
            factor_energy += -(prediction_precision.max(EPSILON)).ln();
        }
    }

    // 6. Embedding projection loss: poor projection → higher free energy.
    //    Untrained or high-loss projection contributes more; well-trained contributes less.
    //    This links projection quality to the meta-learning feedback loop.
    let proj_result = store.run_query(
        r#"?[last_loss, train_count] := *embedding_projection{last_loss, train_count}"#,
        BTreeMap::new(),
    );
    if let Ok(proj_rows) = proj_result {
        if let Some(row) = proj_rows.rows.first() {
            let last_loss = row[0].get_float().unwrap_or(1.0);
            let train_count = row[1].get_int().unwrap_or(0);
            if train_count > 0 {
                let projection_precision = (1.0 - last_loss.min(1.0)).max(EPSILON);
                factor_energy += -(projection_precision).ln();
            }
        }
    }

    // Compute derived values
    let free_energy = factor_energy - variable_entropy;
    let total = factor_energy + variable_entropy + EPSILON;
    let beta = (variable_entropy / total).clamp(0.0, 1.0);

    Ok(FreeEnergyState {
        free_energy,
        accuracy: factor_energy,
        complexity: variable_entropy,
        beta,
        variable_entropy,
        factor_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_entropy_edge_cases() {
        assert_eq!(binary_entropy(0.0), 0.0);
        assert_eq!(binary_entropy(1.0), 0.0);
        let h_half = binary_entropy(0.5);
        assert!((h_half - 2.0f64.ln()).abs() < 0.001, "H(0.5) ≈ ln(2) ≈ 0.693, got {h_half}");
    }

    #[test]
    fn binary_entropy_symmetric() {
        let h30 = binary_entropy(0.3);
        let h70 = binary_entropy(0.7);
        assert!((h30 - h70).abs() < 1e-10, "H(c) should be symmetric around 0.5");
    }

    #[test]
    fn free_energy_empty_store() {
        let store = CozoStore::open_mem(4).unwrap();
        let state = compute_bethe_free_energy(&store).unwrap();

        assert_eq!(state.factor_energy, 0.0);
        assert_eq!(state.variable_entropy, 0.0);
        assert_eq!(state.free_energy, 0.0);
        assert!((state.beta - 0.0).abs() < 0.01, "no entropy → β ≈ 0, got {}", state.beta);
    }

    #[test]
    fn free_energy_includes_telos_preferences() {
        let store = CozoStore::open_mem(4).unwrap();

        // Measure baseline
        let baseline = compute_bethe_free_energy(&store).unwrap();

        // Add an active telos with high priority, 0% progress
        let mut telos = crate::types::telos::Telos::new("Ship Q3", "", vec![0.1; 4], "a", "u");
        telos.priority = 0.9;
        telos.confidence = 1.0;
        telos.progress = 0.0;
        telos.status = crate::types::telos::TelosStatus::Active;
        store.insert_telos(&telos).unwrap();

        let with_telos = compute_bethe_free_energy(&store).unwrap();

        // Unachieved high-priority goal should increase factor energy
        assert!(
            with_telos.factor_energy > baseline.factor_energy,
            "telos should increase factor energy: {} > {}",
            with_telos.factor_energy,
            baseline.factor_energy
        );

        // Nearly-complete goal should contribute less
        let mut telos2 = crate::types::telos::Telos::new("Almost done", "", vec![0.1; 4], "a", "u");
        telos2.priority = 0.9;
        telos2.confidence = 1.0;
        telos2.progress = 0.95;
        telos2.status = crate::types::telos::TelosStatus::Active;
        store.insert_telos(&telos2).unwrap();

        let with_near_complete = compute_bethe_free_energy(&store).unwrap();
        // Factor energy should increase only slightly (5% remaining goal distance)
        let delta_first = with_telos.factor_energy - baseline.factor_energy;
        let delta_second = with_near_complete.factor_energy - with_telos.factor_energy;
        assert!(
            delta_first > delta_second,
            "full goal ({delta_first}) should add more energy than near-complete ({delta_second})"
        );
    }

    #[test]
    fn free_energy_increases_with_facts() {
        let store = CozoStore::open_mem(4).unwrap();

        // Insert a fact
        let mut params = BTreeMap::new();
        params.insert("id".into(), cozo::DataValue::from(uuid::Uuid::now_v7().to_string()));
        params.insert("subj".into(), cozo::DataValue::from(uuid::Uuid::now_v7().to_string()));
        params.insert("pred".into(), cozo::DataValue::from("works_at"));
        params.insert("conf".into(), cozo::DataValue::from(0.8));
        params.insert("rc".into(), cozo::DataValue::from(3i64));

        store.run_script(
            r#"?[id, valid_at, subject_entity, predicate, confidence, reinforcement_count] <- [
                [to_uuid($id), 'ASSERT', to_uuid($subj), $pred, $conf, $rc]
            ]
            :put facts {id, valid_at => subject_entity, predicate, confidence, reinforcement_count}"#,
            params,
        ).unwrap();

        let state = compute_bethe_free_energy(&store).unwrap();
        assert!(state.factor_energy != 0.0, "should have factor energy from the fact");
        assert!(state.variable_entropy > 0.0, "conf=0.8 has nonzero entropy");
    }
}
