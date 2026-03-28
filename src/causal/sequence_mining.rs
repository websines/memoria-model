//! Sequential pattern mining from task history.
//!
//! Mines recurring task sequences from `task_outcomes` to discover temporal
//! patterns like "after debugging tasks, users typically do refactoring" or
//! "deployment always follows testing in this agent's workflow".
//!
//! Uses n-gram frequency counting on task_type sequences to find common patterns.

use std::collections::{BTreeMap, HashMap};

use cozo::DataValue;

use crate::error::Result;
use crate::store::CozoStore;

/// A sequential pattern discovered from task history.
#[derive(Debug, Clone)]
pub struct SequentialPattern {
    /// The task types forming the pattern, in order.
    pub sequence: Vec<String>,
    /// How many times this sequence was observed.
    pub frequency: usize,
    /// Confidence: frequency / total possible occurrences of prefix.
    pub confidence: f64,
    /// The typical agent(s) that exhibit this pattern.
    pub agents: Vec<String>,
}

/// Mine sequential patterns from task history.
///
/// Extracts n-grams (length 2..=max_n) from task_type sequences ordered by time,
/// grouped by agent. Returns patterns above `min_frequency` threshold.
pub fn mine_sequential_patterns(
    store: &CozoStore,
    max_n: usize,
    min_frequency: usize,
) -> Result<Vec<SequentialPattern>> {
    // Fetch task outcomes ordered by time per agent
    let result = store.run_query(
        r#"?[agent_id, task_type, task_id] :=
            *task_outcomes{task_id, agent_id, task_type},
            task_type != ""
        :sort agent_id, task_id"#,
        BTreeMap::new(),
    )?;

    // Group by agent into ordered task_type sequences
    let mut agent_sequences: HashMap<String, Vec<String>> = HashMap::new();
    for row in &result.rows {
        let agent = row[0].get_str().unwrap_or("").to_string();
        let task_type = row[1].get_str().unwrap_or("").to_string();
        if !task_type.is_empty() {
            agent_sequences.entry(agent).or_default().push(task_type);
        }
    }

    // Count n-gram frequencies across all agents
    let mut ngram_counts: HashMap<Vec<String>, usize> = HashMap::new();
    let mut ngram_agents: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    let mut prefix_counts: HashMap<Vec<String>, usize> = HashMap::new();

    for (agent, sequence) in &agent_sequences {
        for n in 2..=max_n.min(sequence.len()) {
            for window in sequence.windows(n) {
                let ngram = window.to_vec();
                *ngram_counts.entry(ngram.clone()).or_insert(0) += 1;
                ngram_agents
                    .entry(ngram.clone())
                    .or_default()
                    .push(agent.clone());

                // Count prefix (n-1 gram) for confidence computation
                let prefix = window[..n - 1].to_vec();
                *prefix_counts.entry(prefix).or_insert(0) += 1;
            }
        }
    }

    // Filter by min_frequency and compute confidence
    let mut patterns: Vec<SequentialPattern> = ngram_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_frequency)
        .map(|(sequence, frequency)| {
            let prefix = sequence[..sequence.len() - 1].to_vec();
            let prefix_count = prefix_counts.get(&prefix).copied().unwrap_or(1);
            let confidence = frequency as f64 / prefix_count as f64;

            let mut agents = ngram_agents.remove(&sequence).unwrap_or_default();
            agents.sort();
            agents.dedup();

            SequentialPattern {
                sequence,
                frequency,
                confidence,
                agents,
            }
        })
        .collect();

    // Sort by frequency descending
    patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));

    Ok(patterns)
}

/// Predict the next likely task type given a recent task history.
///
/// Uses bigram frequencies to predict what typically follows the last task type.
/// Returns (predicted_task_type, confidence) pairs sorted by confidence.
pub fn predict_next_task(
    store: &CozoStore,
    recent_task_type: &str,
) -> Result<Vec<(String, f64)>> {
    let result = store.run_query(
        r#"?[agent_id, task_type, task_id] :=
            *task_outcomes{task_id, agent_id, task_type},
            task_type != ""
        :sort agent_id, task_id"#,
        BTreeMap::new(),
    )?;

    // Build bigram model: count transitions from each task_type to next
    let mut transitions: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut prev_agent = String::new();
    let mut prev_type = String::new();

    for row in &result.rows {
        let agent = row[0].get_str().unwrap_or("").to_string();
        let task_type = row[1].get_str().unwrap_or("").to_string();

        if agent == prev_agent && !prev_type.is_empty() {
            *transitions
                .entry(prev_type.clone())
                .or_default()
                .entry(task_type.clone())
                .or_insert(0) += 1;
        }

        prev_agent = agent;
        prev_type = task_type;
    }

    // Look up transitions from the given task type
    let Some(next_counts) = transitions.get(recent_task_type) else {
        return Ok(Vec::new());
    };

    let total: usize = next_counts.values().sum();
    let mut predictions: Vec<(String, f64)> = next_counts
        .iter()
        .map(|(task_type, count)| (task_type.clone(), *count as f64 / total as f64))
        .collect();

    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(predictions)
}

/// Monotonic counter for deterministic task ordering.
static TASK_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Insert a task outcome.
///
/// Uses a monotonic counter-based UUID for deterministic ordering.
/// The counter fills the UUID's lower bits to ensure lexicographic sort
/// matches insertion order.
pub fn record_task_outcome(
    store: &CozoStore,
    task_type: &str,
    outcome: &str,
    agent_id: &str,
) -> Result<()> {
    let seq = TASK_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Build a UUID where the first 8 hex digits are the sequence number,
    // ensuring lexicographic sort = insertion order
    let task_id = uuid::Uuid::parse_str(&format!(
        "{:08x}-0000-7000-8000-000000000000",
        seq
    ))
    .unwrap_or_else(|_| uuid::Uuid::now_v7());
    let mut params = BTreeMap::new();
    params.insert("task_id".into(), DataValue::from(task_id.to_string()));
    params.insert("outcome".into(), DataValue::from(outcome));
    params.insert("task_type".into(), DataValue::from(task_type));
    params.insert("agent_id".into(), DataValue::from(agent_id));

    store.run_script(
        r#"?[task_id, valid_at, outcome, task_type, agent_id] <- [
            [to_uuid($task_id), 'ASSERT', $outcome, $task_type, $agent_id]
        ]
        :put task_outcomes {task_id, valid_at => outcome, task_type, agent_id}"#,
        params,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mine_empty_history() {
        let store = CozoStore::open_mem(4).unwrap();
        let patterns = mine_sequential_patterns(&store, 3, 2).unwrap();
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_mine_with_repeated_sequence() {
        let store = CozoStore::open_mem(4).unwrap();

        // Record a repeated sequence: debug → refactor → test (3 times)
        for _ in 0..3 {
            record_task_outcome(&store, "debug", "success", "agent-1").unwrap();
            record_task_outcome(&store, "refactor", "success", "agent-1").unwrap();
            record_task_outcome(&store, "test", "success", "agent-1").unwrap();
        }

        let patterns = mine_sequential_patterns(&store, 3, 2).unwrap();

        // Should find at least the bigram "debug → refactor" with frequency >= 2
        let debug_refactor = patterns
            .iter()
            .find(|p| p.sequence == vec!["debug", "refactor"]);
        assert!(
            debug_refactor.is_some(),
            "should find debug→refactor pattern, got: {:?}",
            patterns
        );
        assert!(debug_refactor.unwrap().frequency >= 2);
    }

    #[test]
    fn test_predict_next_task() {
        let store = CozoStore::open_mem(4).unwrap();

        // Build history: debug→test (3x), debug→deploy (1x)
        for _ in 0..3 {
            record_task_outcome(&store, "debug", "success", "a1").unwrap();
            record_task_outcome(&store, "test", "success", "a1").unwrap();
        }
        record_task_outcome(&store, "debug", "success", "a1").unwrap();
        record_task_outcome(&store, "deploy", "success", "a1").unwrap();

        let predictions = predict_next_task(&store, "debug").unwrap();
        assert!(!predictions.is_empty());
        // "test" should be the top prediction (3/4 = 75%)
        assert_eq!(predictions[0].0, "test");
        assert!(predictions[0].1 > 0.5);
    }
}
