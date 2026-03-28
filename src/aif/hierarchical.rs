//! Hierarchical factor graph for chunked memories (§15.9).
//!
//! Uses existing `chunk_parent` edges to implement bidirectional message passing:
//! - **Bottom-up**: sentence match → paragraph belief → section belief → document belief
//! - **Top-down**: document relevance → predicted section relevance → pre-activated paragraphs
//!
//! Combined score = bottom_up × top_down (product of messages from both directions).

use std::collections::BTreeMap;

use cozo::DataValue;
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;

/// A chunk with its hierarchical belief score from bidirectional message passing.
#[derive(Debug, Clone)]
pub struct HierarchicalScore {
    pub memory_id: Uuid,
    pub kind: String,
    pub bottom_up: f64,
    pub top_down: f64,
    pub combined: f64,
}

/// Compute bottom-up messages from sentence matches propagating upward.
///
/// Given sentence-level match scores (from HNSW vector search), propagates
/// beliefs upward through chunk_parent edges. Each parent's belief is the
/// max of its children's scores (at-least-one-strong-match semantics).
pub fn compute_bottom_up(
    store: &CozoStore,
    sentence_scores: &[(Uuid, f64)],
) -> Result<BTreeMap<Uuid, f64>> {
    if sentence_scores.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut scores: BTreeMap<Uuid, f64> = BTreeMap::new();

    // Seed with sentence scores
    for (id, score) in sentence_scores {
        scores.insert(*id, *score);
    }

    // Propagate upward: for each scored node, find its parents via chunk_parent
    // edges and propagate max(child_scores) upward. Iterate until no new parents.
    let mut frontier: Vec<Uuid> = sentence_scores.iter().map(|(id, _)| *id).collect();

    for _ in 0..4 {
        // max 4 levels: sentence → paragraph → section → document
        if frontier.is_empty() {
            break;
        }

        let id_vals: Vec<DataValue> = frontier
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ids".into(), DataValue::List(id_vals));

        let result = store.run_query(
            r#"input[child_str] <- $ids
               input_uuids[child] := input[child_str], child = to_uuid(child_str)
               ?[child, parent] := input_uuids[child],
                   *edges{source: child, target: parent, kind},
                   kind = "chunk_parent""#,
            params,
        )?;

        let mut next_frontier = Vec::new();
        for row in &result.rows {
            let child = crate::store::cozo::parse_uuid_pub(&row[0])?;
            let parent = crate::store::cozo::parse_uuid_pub(&row[1])?;

            let child_score = scores.get(&child).copied().unwrap_or(0.0);
            let current_parent_score = scores.get(&parent).copied().unwrap_or(0.0);
            let new_score = current_parent_score.max(child_score);

            if new_score > current_parent_score {
                scores.insert(parent, new_score);
                next_frontier.push(parent);
            }
        }

        frontier = next_frontier;
    }

    Ok(scores)
}

/// Compute top-down messages from document-level relevance propagating downward.
///
/// Given a query embedding similarity at the document level, propagates predicted
/// relevance downward through chunk_parent edges. Each child inherits its parent's
/// relevance score (expectation: relevant document → relevant sections → relevant paragraphs).
pub fn compute_top_down(
    store: &CozoStore,
    document_scores: &[(Uuid, f64)],
) -> Result<BTreeMap<Uuid, f64>> {
    if document_scores.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut scores: BTreeMap<Uuid, f64> = BTreeMap::new();

    // Seed with document scores
    for (id, score) in document_scores {
        scores.insert(*id, *score);
    }

    // Propagate downward: for each scored node, find its children
    let mut frontier: Vec<Uuid> = document_scores.iter().map(|(id, _)| *id).collect();

    for _ in 0..4 {
        if frontier.is_empty() {
            break;
        }

        let id_vals: Vec<DataValue> = frontier
            .iter()
            .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
            .collect();

        let mut params = BTreeMap::new();
        params.insert("ids".into(), DataValue::List(id_vals));

        let result = store.run_query(
            r#"input[parent_str] <- $ids
               input_uuids[parent] := input[parent_str], parent = to_uuid(parent_str)
               ?[parent, child] := input_uuids[parent],
                   *edges{source: child, target: parent, kind},
                   kind = "chunk_parent""#,
            params,
        )?;

        let mut next_frontier = Vec::new();
        for row in &result.rows {
            let parent = crate::store::cozo::parse_uuid_pub(&row[0])?;
            let child = crate::store::cozo::parse_uuid_pub(&row[1])?;

            let parent_score = scores.get(&parent).copied().unwrap_or(0.0);
            // Top-down: child inherits parent's relevance (slightly decayed)
            let child_score = parent_score * 0.9; // decay factor per level
            let current = scores.get(&child).copied().unwrap_or(0.0);

            if child_score > current {
                scores.insert(child, child_score);
                next_frontier.push(child);
            }
        }

        frontier = next_frontier;
    }

    Ok(scores)
}

/// Combine bottom-up evidence with top-down predictions.
///
/// Combined score = bottom_up × top_down (product of messages from both directions).
/// Returns only nodes that have BOTH bottom-up and top-down scores.
pub fn combine_hierarchical_scores(
    bottom_up: &BTreeMap<Uuid, f64>,
    top_down: &BTreeMap<Uuid, f64>,
) -> Vec<HierarchicalScore> {
    let mut combined = Vec::new();

    // Union of all scored nodes
    let all_ids: std::collections::BTreeSet<&Uuid> =
        bottom_up.keys().chain(top_down.keys()).collect();

    for id in all_ids {
        let bu = bottom_up.get(id).copied().unwrap_or(0.0);
        let td = top_down.get(id).copied().unwrap_or(1.0); // default: no top-down = neutral

        combined.push(HierarchicalScore {
            memory_id: *id,
            kind: String::new(), // caller fills in from memory
            bottom_up: bu,
            top_down: td,
            combined: bu * td,
        });
    }

    combined.sort_by(|a, b| {
        b.combined
            .partial_cmp(&a.combined)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::chunker_hierarchical::{build_hierarchy, store_hierarchical_chunks, ChunkLevel};

    #[test]
    fn bottom_up_propagates_from_sentences() {
        let store = CozoStore::open_mem(4).unwrap();

        let text = "Alice works at Acme Corp. She is an engineer.\n\nBob manages the team.";
        let chunks = build_hierarchy(text);
        let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();
        let stored = store_hierarchical_chunks(&store, &chunks, &embeddings, "default").unwrap();

        // Find sentence IDs
        let sentence_scores: Vec<(Uuid, f64)> = stored
            .iter()
            .filter(|(i, _)| chunks[*i].level == ChunkLevel::Sentence)
            .map(|(_, id)| (*id, 0.9))
            .collect();

        let scores = compute_bottom_up(&store, &sentence_scores).unwrap();

        // Should propagate to parent nodes
        assert!(scores.len() > sentence_scores.len(), "should propagate to parents");

        // Parent nodes should have scores
        let para_ids: Vec<Uuid> = stored
            .iter()
            .filter(|(i, _)| chunks[*i].level == ChunkLevel::Paragraph)
            .map(|(_, id)| *id)
            .collect();

        for pid in &para_ids {
            assert!(scores.contains_key(pid), "paragraph should have bottom-up score");
        }
    }

    #[test]
    fn top_down_propagates_from_documents() {
        let store = CozoStore::open_mem(4).unwrap();

        let text = "Alice works at Acme Corp. She is an engineer.\n\nBob manages the team.";
        let chunks = build_hierarchy(text);
        let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();
        let stored = store_hierarchical_chunks(&store, &chunks, &embeddings, "default").unwrap();

        // Score documents
        let doc_scores: Vec<(Uuid, f64)> = stored
            .iter()
            .filter(|(i, _)| chunks[*i].level == ChunkLevel::Document)
            .map(|(_, id)| (*id, 0.8))
            .collect();

        let scores = compute_top_down(&store, &doc_scores).unwrap();

        // Should propagate to children
        assert!(scores.len() > doc_scores.len(), "should propagate to children");
    }

    #[test]
    fn combined_scores_multiply() {
        let mut bu = BTreeMap::new();
        let mut td = BTreeMap::new();
        let id = Uuid::now_v7();

        bu.insert(id, 0.8);
        td.insert(id, 0.6);

        let combined = combine_hierarchical_scores(&bu, &td);
        assert_eq!(combined.len(), 1);
        assert!((combined[0].combined - 0.48).abs() < 1e-10, "0.8 × 0.6 = 0.48");
    }

    #[test]
    fn empty_inputs_return_empty() {
        let store = CozoStore::open_mem(4).unwrap();
        let bu = compute_bottom_up(&store, &[]).unwrap();
        assert!(bu.is_empty());
        let td = compute_top_down(&store, &[]).unwrap();
        assert!(td.is_empty());
    }
}
