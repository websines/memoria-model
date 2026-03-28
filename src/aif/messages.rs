//! Precision-weighted factor messages for scoring fusion.
//!
//! Replaces the uniform-weight log-product in FactorScorer with precision-weighted
//! messages. Each factor sends a message with both a value and a precision weight.
//! Fusion is a precision-weighted average: `Σ(precision_i × message_i) / Σ(precision_i)`.

use serde::{Deserialize, Serialize};

use crate::types::memory::CandidateMemory;

const EPSILON: f64 = 1e-10;

/// Types of scoring factors in the recall pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorType {
    /// Embedding similarity (cosine distance → similarity).
    Similarity,
    /// Recency-weighted access count (activation).
    Activation,
    /// Co-access weight to current context (Hebbian).
    Hebbian,
    /// Graph centrality importance (PageRank).
    PageRank,
    /// Hard rule — blocks or allows based on rule evaluation.
    Rule,
    /// Soft rule — nudges score without blocking.
    SoftRule,
    /// Telos alignment — boosts memories relevant to active goals.
    TelosAlignment,
}

/// A message from a factor node to a variable node in the factor graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMessage {
    /// Which factor produced this message.
    pub factor_type: FactorType,
    /// The message value (log-domain).
    pub message_value: f64,
    /// Precision (weight) of this message. Higher → more influence on posterior.
    pub precision: f64,
}

impl FactorMessage {
    /// Create a hard-rule block message that sends -∞ to block the posterior.
    pub fn hard_rule_block() -> Self {
        Self {
            factor_type: FactorType::Rule,
            message_value: f64::NEG_INFINITY,
            precision: f64::INFINITY,
        }
    }

    /// Create a new factor message.
    pub fn new(factor_type: FactorType, message_value: f64, precision: f64) -> Self {
        Self {
            factor_type,
            message_value,
            precision,
        }
    }
}

/// Fuse multiple factor messages into a single posterior score.
///
/// Uses precision-weighted averaging: `Σ(precision_i × message_i) / Σ(precision_i)`.
///
/// If any message is -∞ (hard block), returns -∞ immediately.
/// If no messages, returns 0.0 (uniform prior).
pub fn fuse_messages(messages: &[FactorMessage]) -> f64 {
    if messages.is_empty() {
        return 0.0;
    }

    // Check for hard blocks
    if messages.iter().any(|m| m.message_value == f64::NEG_INFINITY) {
        return f64::NEG_INFINITY;
    }

    let total_precision: f64 = messages.iter().map(|m| m.precision).sum();

    if total_precision <= 0.0 {
        // Fallback: uniform weighting (sum of messages)
        return messages.iter().map(|m| m.message_value).sum();
    }

    let weighted_sum: f64 = messages
        .iter()
        .map(|m| m.precision * m.message_value)
        .sum();

    weighted_sum / total_precision
}

/// Compute recall factor messages for a candidate memory.
///
/// Converts the raw scoring signals on a `CandidateMemory` into precision-weighted
/// factor messages. The precision of each message is derived from the signal's
/// reliability/magnitude.
pub fn compute_recall_messages(candidate: &CandidateMemory) -> Vec<FactorMessage> {
    let mut messages = Vec::with_capacity(4);

    // Factor 1: Embedding similarity
    let sim = (1.0 - candidate.distance).max(0.0);
    let sim_log = (sim + EPSILON).ln();
    // Precision for similarity: higher similarity → higher precision
    let sim_precision = if sim > 0.0 { sim } else { EPSILON };
    messages.push(FactorMessage::new(FactorType::Similarity, sim_log, sim_precision));

    // Factor 2: Activation
    if let Some(activation) = candidate.activation {
        let act_log = (activation + EPSILON).ln();
        // Precision: proportional to activation magnitude, capped at 1.0
        let act_precision = activation.min(1.0).max(EPSILON);
        messages.push(FactorMessage::new(FactorType::Activation, act_log, act_precision));
    }

    // Factor 3: Hebbian association
    if let Some(weight) = candidate.hebbian_weight {
        let heb_log = (weight + EPSILON).ln();
        // Precision: proportional to weight magnitude, capped at 1.0
        let heb_precision = weight.min(1.0).max(EPSILON);
        messages.push(FactorMessage::new(FactorType::Hebbian, heb_log, heb_precision));
    }

    // Factor 4: PageRank
    if let Some(pagerank) = candidate.pagerank {
        let pr_log = (pagerank + EPSILON).ln();
        // Precision: proportional to pagerank, capped at 1.0
        let pr_precision = pagerank.min(1.0).max(EPSILON);
        messages.push(FactorMessage::new(FactorType::PageRank, pr_log, pr_precision));
    }

    // Factor 5: Memory precision (from candidate.precision field if available)
    if let Some(mem_precision) = candidate.precision {
        // This is the belief precision from the memory's confidence + version.
        // It acts as a prior-strength signal: well-established memories score higher.
        let prior_log = (mem_precision + EPSILON).ln();
        messages.push(FactorMessage::new(
            FactorType::SoftRule,
            prior_log,
            mem_precision.min(1.0).max(EPSILON),
        ));
    }

    // Factor 6: Telos alignment — goal relevance boost
    if let Some(boost) = candidate.telos_boost {
        let telos_log = (boost + EPSILON).ln();
        let telos_precision = boost.clamp(EPSILON, 1.0);
        messages.push(FactorMessage::new(
            FactorType::TelosAlignment,
            telos_log,
            telos_precision,
        ));
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::Memory;

    #[test]
    fn fuse_empty_returns_zero() {
        assert_eq!(fuse_messages(&[]), 0.0);
    }

    #[test]
    fn fuse_hard_block_returns_neg_inf() {
        let messages = vec![
            FactorMessage::new(FactorType::Similarity, -1.0, 0.5),
            FactorMessage::hard_rule_block(),
        ];
        assert_eq!(fuse_messages(&messages), f64::NEG_INFINITY);
    }

    #[test]
    fn fuse_precision_weighted() {
        let messages = vec![
            FactorMessage::new(FactorType::Similarity, -1.0, 0.8),
            FactorMessage::new(FactorType::Activation, -2.0, 0.2),
        ];
        let result = fuse_messages(&messages);
        // Expected: (0.8 × -1.0 + 0.2 × -2.0) / (0.8 + 0.2) = -1.2 / 1.0 = -1.2
        assert!((result - (-1.2)).abs() < 1e-10, "got {result}");
    }

    #[test]
    fn fuse_high_precision_dominates() {
        let messages = vec![
            FactorMessage::new(FactorType::Similarity, -1.0, 100.0),
            FactorMessage::new(FactorType::Activation, -10.0, 0.01),
        ];
        let result = fuse_messages(&messages);
        // High-precision similarity should dominate → result ≈ -1.0
        assert!((result - (-1.0)).abs() < 0.1, "high-precision message should dominate, got {result}");
    }

    #[test]
    fn compute_recall_messages_similarity_only() {
        let candidate = CandidateMemory {
            memory: Memory::new("test", "hello", vec![0.0; 4]),
            distance: 0.2,
            activation: None,
            hebbian_weight: None,
            pagerank: None,
            precision: None,
            telos_boost: None,
        };
        let messages = compute_recall_messages(&candidate);
        assert_eq!(messages.len(), 1, "only similarity factor");
        assert_eq!(messages[0].factor_type, FactorType::Similarity);
    }

    #[test]
    fn compute_recall_messages_all_factors() {
        let candidate = CandidateMemory {
            memory: Memory::new("test", "hello", vec![0.0; 4]),
            distance: 0.1,
            activation: Some(0.5),
            hebbian_weight: Some(0.3),
            pagerank: Some(0.8),
            precision: Some(0.7),
            telos_boost: None,
        };
        let messages = compute_recall_messages(&candidate);
        assert_eq!(messages.len(), 5, "all 5 factors should produce messages");
    }
}
