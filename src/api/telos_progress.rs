//! Telos progress estimation — composite scoring from criteria + children + LLM.
//!
//! Progress is computed from three signals:
//! 1. **Criteria-based**: fraction of success criteria marked as met
//! 2. **Children-based**: weighted average of child telos progress
//! 3. **LLM cold path**: when both are insufficient, the LLM estimates progress
//!
//! The composite blends whichever signals are available.

use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::{LlmService, Message};
use crate::store::CozoStore;
use crate::types::telos::Telos;

/// Result of a progress estimation.
#[derive(Debug, Clone)]
pub struct ProgressEstimate {
    /// The estimated progress [0.0, 1.0].
    pub progress: f64,
    /// How the estimate was computed.
    pub method: ProgressMethod,
    /// Confidence in the estimate [0.0, 1.0].
    pub confidence: f64,
}

/// How progress was estimated.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgressMethod {
    /// Purely from success criteria (deterministic).
    Criteria,
    /// Weighted average of child telos progress.
    Children,
    /// Blend of criteria and children signals.
    Composite,
    /// LLM-estimated (cold path).
    Llm,
    /// No estimation possible (no signals available).
    None,
}

/// Estimate progress for a telos using deterministic signals.
///
/// Blends criteria-based and children-based progress using available signals.
/// Returns `ProgressMethod::None` if neither signal is available (caller
/// should use `estimate_progress_llm` as fallback).
pub fn estimate_progress(store: &CozoStore, telos_id: Uuid) -> Result<ProgressEstimate> {
    let telos = store
        .get_telos(telos_id)?
        .ok_or_else(|| MemoriaError::NotFound(telos_id))?;

    let criteria_signal = criteria_progress(&telos);
    let children_signal = children_progress(store, telos_id)?;

    match (criteria_signal, children_signal) {
        (Some(cp), Some(chp)) => {
            // Composite: criteria are more concrete, weight them higher
            let criteria_weight = 0.6;
            let children_weight = 0.4;
            let progress = cp * criteria_weight + chp * children_weight;
            Ok(ProgressEstimate {
                progress,
                method: ProgressMethod::Composite,
                confidence: 0.85,
            })
        }
        (Some(cp), None) => Ok(ProgressEstimate {
            progress: cp,
            method: ProgressMethod::Criteria,
            confidence: 0.9,
        }),
        (None, Some(chp)) => Ok(ProgressEstimate {
            progress: chp,
            method: ProgressMethod::Children,
            confidence: 0.75,
        }),
        (None, None) => Ok(ProgressEstimate {
            progress: telos.progress, // Keep existing
            method: ProgressMethod::None,
            confidence: 0.0,
        }),
    }
}

/// Estimate progress using the LLM (cold path).
///
/// Used when deterministic signals are insufficient. The LLM reviews
/// the goal, its events, and any available context to estimate progress.
pub async fn estimate_progress_llm(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    telos_id: Uuid,
) -> Result<ProgressEstimate> {
    let telos = store
        .get_telos(telos_id)?
        .ok_or_else(|| MemoriaError::NotFound(telos_id))?;

    // Gather recent events for context
    let events = store.get_telos_events(telos_id, 20)?;
    let event_summary: String = events
        .iter()
        .map(|e| format!("- [{}] {}: {}", e.event_type, e.agent_id, e.description))
        .collect::<Vec<_>>()
        .join("\n");

    let criteria_summary = if telos.success_criteria.is_empty() {
        "No success criteria defined.".to_string()
    } else {
        telos
            .success_criteria
            .iter()
            .map(|c| {
                format!(
                    "- [{}] {}",
                    if c.met { "MET" } else { "UNMET" },
                    c.description
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let prompt = format!(
        concat!(
            "Estimate the progress of this goal as a number between 0.0 and 1.0.\n\n",
            "Goal: {title}\n",
            "Description: {description}\n",
            "Current status: {status}\n",
            "Current progress: {progress:.0}%\n\n",
            "Success criteria:\n{criteria}\n\n",
            "Recent events:\n{events}\n\n",
            "Return ONLY a JSON object with:\n",
            "- \"progress\": number 0.0-1.0\n",
            "- \"reasoning\": brief explanation\n",
            "No markdown fences."
        ),
        title = telos.title,
        description = if telos.description.is_empty() {
            &telos.title
        } else {
            &telos.description
        },
        status = telos.status.as_str(),
        progress = telos.progress * 100.0,
        criteria = criteria_summary,
        events = if event_summary.is_empty() {
            "No events recorded.".to_string()
        } else {
            event_summary
        },
    );

    let llm_response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "You estimate goal progress. Return only valid JSON.".into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            512,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    // Parse LLM response
    let progress = parse_progress_response(&llm_response.content).unwrap_or(telos.progress);

    Ok(ProgressEstimate {
        progress,
        method: ProgressMethod::Llm,
        confidence: 0.5, // LLM estimates are less reliable
    })
}

/// Compute progress from success criteria.
fn criteria_progress(telos: &Telos) -> Option<f64> {
    if telos.success_criteria.is_empty() {
        return None;
    }
    let total = telos.success_criteria.len() as f64;
    let met = telos.success_criteria.iter().filter(|c| c.met).count() as f64;
    Some(met / total)
}

/// Compute progress from children telos, weighted by priority.
fn children_progress(store: &CozoStore, telos_id: Uuid) -> Result<Option<f64>> {
    let children = store.get_children_telos(telos_id)?;
    if children.is_empty() {
        return Ok(None);
    }

    let total_priority: f64 = children.iter().map(|c| c.priority).sum();
    if total_priority <= 0.0 {
        // Equal weighting fallback
        let avg = children.iter().map(|c| c.progress).sum::<f64>() / children.len() as f64;
        return Ok(Some(avg));
    }

    let weighted = children
        .iter()
        .map(|c| c.progress * c.priority)
        .sum::<f64>()
        / total_priority;

    Ok(Some(weighted))
}

/// Parse LLM progress response.
fn parse_progress_response(response: &str) -> Option<f64> {
    let trimmed = response.trim();

    // Try as JSON object
    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(p) = obj.get("progress").and_then(|v| v.as_f64()) {
            return Some(p.clamp(0.0, 1.0));
        }
    }

    // Try stripping markdown fences
    let stripped = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(stripped) {
        if let Some(p) = obj.get("progress").and_then(|v| v.as_f64()) {
            return Some(p.clamp(0.0, 1.0));
        }
    }

    // Try parsing as plain number
    if let Ok(p) = trimmed.parse::<f64>() {
        return Some(p.clamp(0.0, 1.0));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::telos::{SuccessCriterion, Telos};

    #[test]
    fn test_criteria_progress() {
        let mut t = Telos::new("test", "", vec![0.1; 4], "a", "u");
        assert!(criteria_progress(&t).is_none());

        t.success_criteria = vec![
            SuccessCriterion {
                id: "a".into(),
                description: "First".into(),
                met: true,
            },
            SuccessCriterion {
                id: "b".into(),
                description: "Second".into(),
                met: false,
            },
        ];
        let p = criteria_progress(&t).unwrap();
        assert!((p - 0.5).abs() < 0.01);

        t.success_criteria[1].met = true;
        let p = criteria_progress(&t).unwrap();
        assert!((p - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_children_progress() {
        let store = CozoStore::open_mem(4).unwrap();

        let parent = Telos::new("parent", "", vec![0.1; 4], "a", "u");
        store.insert_telos(&parent).unwrap();

        // No children
        assert!(children_progress(&store, parent.id).unwrap().is_none());

        // Add children with different priorities and progress
        let mut c1 = Telos::new("child1", "", vec![0.1; 4], "a", "u");
        c1.parent = Some(parent.id);
        c1.priority = 0.7;
        c1.progress = 1.0; // complete
        store.insert_telos(&c1).unwrap();

        let mut c2 = Telos::new("child2", "", vec![0.1; 4], "a", "u");
        c2.parent = Some(parent.id);
        c2.priority = 0.3;
        c2.progress = 0.0; // not started
        store.insert_telos(&c2).unwrap();

        let p = children_progress(&store, parent.id).unwrap().unwrap();
        // (1.0 * 0.7 + 0.0 * 0.3) / (0.7 + 0.3) = 0.7
        assert!((p - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_estimate_progress_composite() {
        let store = CozoStore::open_mem(4).unwrap();

        let mut parent = Telos::new("parent", "", vec![0.1; 4], "a", "u");
        parent.success_criteria = vec![
            SuccessCriterion {
                id: "a".into(),
                description: "First".into(),
                met: true,
            },
            SuccessCriterion {
                id: "b".into(),
                description: "Second".into(),
                met: false,
            },
        ];
        store.insert_telos(&parent).unwrap();

        let mut child = Telos::new("child", "", vec![0.1; 4], "a", "u");
        child.parent = Some(parent.id);
        child.priority = 1.0;
        child.progress = 0.8;
        store.insert_telos(&child).unwrap();

        let est = estimate_progress(&store, parent.id).unwrap();
        assert_eq!(est.method, ProgressMethod::Composite);
        // 0.5 * 0.6 + 0.8 * 0.4 = 0.3 + 0.32 = 0.62
        assert!((est.progress - 0.62).abs() < 0.01);
    }

    #[test]
    fn test_parse_progress_response() {
        assert!((parse_progress_response(r#"{"progress": 0.75, "reasoning": "good"}"#).unwrap() - 0.75).abs() < 0.01);
        assert!((parse_progress_response("0.42").unwrap() - 0.42).abs() < 0.01);
        assert!((parse_progress_response("```json\n{\"progress\": 0.9}\n```").unwrap() - 0.9).abs() < 0.01);
        assert!(parse_progress_response("invalid").is_none());
        // Clamping
        assert!((parse_progress_response(r#"{"progress": 1.5}"#).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_progress_no_signals() {
        let store = CozoStore::open_mem(4).unwrap();
        let mut t = Telos::new("bare", "", vec![0.1; 4], "a", "u");
        t.progress = 0.33;
        store.insert_telos(&t).unwrap();

        let est = estimate_progress(&store, t.id).unwrap();
        assert_eq!(est.method, ProgressMethod::None);
        assert!((est.progress - 0.33).abs() < 0.01); // preserves existing
    }
}
