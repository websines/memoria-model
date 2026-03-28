//! PPM-C (Prediction by Partial Matching, method C) context tree.
//!
//! Variable-order Markov model for discrete sequence prediction.
//! Replaces the fixed-order bigram model in `causal::sequence_mining`.
//!
//! PPM-C uses escape probabilities based on the count of distinct symbols
//! seen in each context (Method C), providing well-calibrated predictions
//! that adapt context depth automatically via back-off.
//!
//! Reference: Begleiter, El-Yaniv & Yona (JAIR 2004) —
//! "On Prediction Using Variable Order Markov Models"

use std::collections::HashMap;

/// A node in the PPM-C context tree.
///
/// Each node stores symbol counts for the symbols that have followed
/// this context, plus children for longer contexts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PpmNode {
    /// Symbol → count at this context depth.
    counts: HashMap<String, u64>,
    /// Total observations at this node.
    total: u64,
    /// Count of distinct symbols seen (the "escape count" for Method C).
    escape_count: u64,
    /// Children indexed by the most recent context symbol.
    children: HashMap<String, PpmNode>,
}

/// The full PPM-C model for one agent/stream.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PpmModel {
    /// Root node (order-0 context).
    root: PpmNode,
    /// Maximum context depth (typically 5-8).
    pub max_depth: usize,
    /// Number of update calls.
    pub update_count: u64,
}

impl PpmNode {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
            escape_count: 0,
            children: HashMap::new(),
        }
    }

    /// Predict next symbol given context (longest match with back-off).
    ///
    /// `context` is ordered oldest-first: `[t-3, t-2, t-1]`.
    /// We traverse deepest-first: start at context[len-1] (most recent),
    /// then back off to shorter contexts if needed.
    ///
    /// PPM-C blending: try the longest context child first. Its prediction
    /// gets the non-escape probability mass. This node's base distribution
    /// serves as the escape (fallback) for symbols the child hasn't seen.
    ///
    /// Returns distribution over event types (probabilities sum to ~1).
    fn predict(&self, context: &[String], depth: usize) -> HashMap<String, f64> {
        if depth == 0 || context.is_empty() {
            return self.base_distribution();
        }

        let ctx_symbol = &context[context.len() - 1];
        if let Some(child) = self.children.get(ctx_symbol) {
            if child.total > 0 {
                // Child has observed this context — use its distribution
                // as primary, with escape to this node's distribution for
                // symbols the child hasn't seen.
                let fallback = self.predict(&context[..context.len() - 1], depth - 1);
                return child.blend_with_escape(&fallback);
            }
        }

        // Context not found — escape to shorter context
        self.predict(&context[..context.len() - 1], depth - 1)
    }

    /// Base (order-0) distribution: empirical frequencies at this node.
    fn base_distribution(&self) -> HashMap<String, f64> {
        if self.total == 0 {
            return HashMap::new();
        }
        self.counts
            .iter()
            .map(|(sym, &count)| (sym.clone(), count as f64 / self.total as f64))
            .collect()
    }

    /// Blend child prediction with escape probability (PPM Method C).
    ///
    /// In Method C, the escape probability for a context is:
    ///   P_esc = escape_count / (total + escape_count)
    /// The probability mass assigned to symbols seen in this context is:
    ///   P_seen = total / (total + escape_count)
    /// Back-off: symbols NOT seen in this context get their probability
    /// from the shorter-context prediction, scaled by P_esc.
    fn blend_with_escape(&self, shorter_pred: &HashMap<String, f64>) -> HashMap<String, f64> {
        if self.total == 0 {
            return shorter_pred.clone();
        }

        let denominator = self.total + self.escape_count;
        let p_esc = self.escape_count as f64 / denominator as f64;
        let p_seen_scale = self.total as f64 / denominator as f64;

        let mut dist = HashMap::new();

        // Symbols observed in this context: use direct count, scaled
        for (sym, &count) in &self.counts {
            let p = (count as f64 / self.total as f64) * p_seen_scale;
            dist.insert(sym.clone(), p);
        }

        // Symbols NOT seen here: inherit from shorter context, scaled by escape
        for (sym, &p) in shorter_pred {
            if !self.counts.contains_key(sym) {
                dist.insert(sym.clone(), p * p_esc);
            }
        }

        dist
    }

    /// Update the tree with an observed symbol after the given context.
    ///
    /// `context` is oldest-first. We update all context depths from 0 to depth.
    fn update(&mut self, context: &[String], symbol: &str, depth: usize) {
        // Update this node (current depth)
        let is_new = !self.counts.contains_key(symbol);
        *self.counts.entry(symbol.to_string()).or_insert(0) += 1;
        self.total += 1;
        if is_new {
            self.escape_count += 1;
        }

        // Recurse into deeper context if possible
        if depth > 0 && !context.is_empty() {
            let ctx_symbol = &context[context.len() - 1];
            let child = self
                .children
                .entry(ctx_symbol.clone())
                .or_insert_with(PpmNode::new);
            child.update(&context[..context.len() - 1], symbol, depth - 1);
        }
    }

    /// Count total nodes in the tree (for diagnostics).
    fn node_count(&self) -> usize {
        1 + self.children.values().map(|c| c.node_count()).sum::<usize>()
    }
}

impl PpmModel {
    /// Create a new PPM-C model with the given maximum context depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            root: PpmNode::new(),
            max_depth,
            update_count: 0,
        }
    }

    /// Predict the next symbol given recent context.
    ///
    /// `context` should be the most recent `max_depth` symbols, oldest-first.
    /// Returns (symbol, probability) pairs sorted by probability descending.
    pub fn predict(&self, context: &[String]) -> Vec<(String, f64)> {
        let effective_ctx = if context.len() > self.max_depth {
            &context[context.len() - self.max_depth..]
        } else {
            context
        };

        let dist = self.root.predict(effective_ctx, self.max_depth);
        let mut predictions: Vec<(String, f64)> = dist.into_iter().collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions
    }

    /// Update the model with an observed transition: context → symbol.
    ///
    /// `context` should be the recent history oldest-first.
    pub fn update(&mut self, context: &[String], symbol: &str) {
        let effective_ctx = if context.len() > self.max_depth {
            &context[context.len() - self.max_depth..]
        } else {
            context
        };
        self.root.update(effective_ctx, symbol, self.max_depth);
        self.update_count += 1;
    }

    /// Number of distinct symbols in the root (alphabet size).
    pub fn alphabet_size(&self) -> usize {
        self.root.counts.len()
    }

    /// Total nodes in the tree (for diagnostics).
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Serialize the model to bytes for CozoDB storage.
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_model_returns_empty() {
        let model = PpmModel::new(5);
        let preds = model.predict(&[]);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_single_symbol_predicts_itself() {
        let mut model = PpmModel::new(3);
        model.update(&[], "debug");
        model.update(&["debug".into()], "test");
        model.update(&["debug".into()], "test");
        model.update(&["debug".into()], "deploy");

        let preds = model.predict(&["debug".into()]);
        assert!(!preds.is_empty());
        // "test" should be top (2 out of 3 after "debug")
        assert_eq!(preds[0].0, "test");
    }

    #[test]
    fn test_context_depth() {
        let mut model = PpmModel::new(3);

        // Pattern: a → b → c → a → b → c → a → b → c
        let sequence = ["a", "b", "c", "a", "b", "c", "a", "b", "c"];
        let mut context: Vec<String> = Vec::new();

        for sym in &sequence {
            model.update(&context, sym);
            context.push(sym.to_string());
            if context.len() > 3 {
                context.remove(0);
            }
        }

        // After "b", "c" should be most likely
        let preds = model.predict(&["a".into(), "b".into()]);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, "c", "after [a, b], 'c' should be top prediction");
    }

    #[test]
    fn test_probabilities_sum_roughly_to_one() {
        let mut model = PpmModel::new(3);
        for sym in &["a", "b", "c", "a", "b", "a", "c", "b", "a"] {
            let ctx: Vec<String> = Vec::new();
            model.update(&ctx, sym);
        }

        let preds = model.predict(&[]);
        let sum: f64 = preds.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "probabilities should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut model = PpmModel::new(5);
        model.update(&[], "debug");
        model.update(&["debug".into()], "test");

        let bytes = model.to_bytes().unwrap();
        let restored = PpmModel::from_bytes(&bytes).unwrap();
        assert_eq!(restored.max_depth, 5);
        assert_eq!(restored.update_count, 2);
        assert_eq!(restored.alphabet_size(), model.alphabet_size());
    }

    #[test]
    fn test_backoff_to_shorter_context() {
        let mut model = PpmModel::new(3);
        // Train with context [a] → b (3 times)
        for _ in 0..3 {
            model.update(&["a".into()], "b");
        }

        // Query with unseen context [x, a] — should back off to [a] context
        let preds = model.predict(&["x".into(), "a".into()]);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, "b");
    }
}
