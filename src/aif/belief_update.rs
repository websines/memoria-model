//! Canonical precision formulas and Kalman-like belief updates.
//!
//! These are the single source of truth for precision computation throughout Memoria.
//! All other modules (surprise, scoring, free energy) delegate here.

use serde::{Deserialize, Serialize};

/// Precision of an existing belief based on confidence and reinforcement count.
///
/// Higher confidence and more reinforcement → higher precision → harder to update.
///
/// Formula: `confidence × ln(max(reinforcement_count, 1) + 1)`
pub fn precision(confidence: f64, reinforcement_count: i64) -> f64 {
    confidence * ((reinforcement_count.max(1) as f64) + 1.0).ln()
}

/// Precision of an incoming observation based on confidence and provenance weight.
///
/// Formula: `confidence × provenance_weight`
pub fn observation_precision(confidence: f64, provenance_weight: f64) -> f64 {
    confidence * provenance_weight
}

/// Result of a Kalman-like belief update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefUpdateResult {
    /// Updated confidence after incorporating the observation.
    pub new_confidence: f64,
    /// Kalman gain: how much the belief shifted toward the observation.
    pub gain: f64,
    /// Precision of the prior belief.
    pub belief_precision: f64,
    /// Precision of the observation.
    pub observation_precision: f64,
    /// Whether the update is large enough to warrant reconsolidation.
    pub should_reconsolidate: bool,
}

/// Perform a Kalman-like belief update given prior belief parameters and an observation.
///
/// The gain determines how much weight to give the observation vs the prior:
/// - `gain = obs_precision / (belief_precision + obs_precision)`
/// - `new_confidence = (1 - gain) × belief_conf + gain × obs_conf`
///
/// Reconsolidation is triggered when `gain > reconsolidation_threshold`.
pub fn belief_update(
    belief_confidence: f64,
    belief_reinforcement_count: i64,
    obs_confidence: f64,
    obs_provenance_weight: f64,
    reconsolidation_threshold: f64,
) -> BeliefUpdateResult {
    let bp = precision(belief_confidence, belief_reinforcement_count);
    let op = observation_precision(obs_confidence, obs_provenance_weight);

    let total = bp + op;
    let gain = if total > 0.0 { op / total } else { 0.5 };

    let new_confidence = (1.0 - gain) * belief_confidence + gain * obs_confidence;

    BeliefUpdateResult {
        new_confidence: new_confidence.clamp(0.0, 1.0),
        gain,
        belief_precision: bp,
        observation_precision: op,
        should_reconsolidate: gain > reconsolidation_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_increases_with_reinforcement() {
        let p1 = precision(0.8, 1);
        let p5 = precision(0.8, 5);
        let p20 = precision(0.8, 20);
        assert!(p5 > p1, "more reinforcement → higher precision");
        assert!(p20 > p5);
    }

    #[test]
    fn precision_scales_with_confidence() {
        let low = precision(0.3, 5);
        let high = precision(0.9, 5);
        assert!(high > low, "higher confidence → higher precision");
    }

    #[test]
    fn observation_precision_basic() {
        assert!((observation_precision(1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((observation_precision(0.5, 0.6) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn belief_update_high_precision_belief_resists_change() {
        let result = belief_update(0.9, 10, 0.3, 0.3, 0.3);
        assert!(result.gain < 0.1, "strong belief resists weak observation, gain={}", result.gain);
        assert!(result.new_confidence > 0.8);
        assert!(!result.should_reconsolidate);
    }

    #[test]
    fn belief_update_weak_belief_shifts_toward_strong_observation() {
        let result = belief_update(0.3, 1, 1.0, 1.0, 0.3);
        assert!(result.gain > 0.3, "weak belief should shift, gain={}", result.gain);
        assert!(result.new_confidence > 0.3);
        assert!(result.should_reconsolidate);
    }

    #[test]
    fn belief_update_clamps_to_valid_range() {
        let result = belief_update(0.0, 0, 1.0, 1.0, 0.3);
        assert!(result.new_confidence >= 0.0 && result.new_confidence <= 1.0);
    }

    #[test]
    fn belief_update_equal_precision_splits_evenly() {
        // belief_precision = precision(0.8, 1) = 0.8 * ln(2) ≈ 0.5545
        // To get obs_precision ≈ 0.5545, we need obs_conf * prov_weight ≈ 0.5545
        // Use obs_conf=0.8, prov_weight = precision(0.8,1) / 0.8 = ln(2) ≈ 0.693
        let bp = precision(0.8, 1);
        let prov_w = bp / 0.8; // so obs_precision = 0.8 * prov_w = bp
        let result = belief_update(0.8, 1, 0.8, prov_w, 0.3);
        assert!((result.gain - 0.5).abs() < 0.01, "equal precision → gain ≈ 0.5, got {}", result.gain);
    }
}
