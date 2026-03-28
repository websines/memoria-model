//! Confidence propagation — cascade belief updates through source chains (§15.8).
//!
//! When a belief changes (via reconsolidation or observation), the change should
//! propagate downstream through the `source_ids` provenance chain. A fact derived
//! from a now-suspect source should have its confidence reduced proportionally.
//!
//! This implements the `confidence_propagation` queue task type.

use std::collections::BTreeMap;

use cozo::DataValue;
use uuid::Uuid;

use crate::aif::belief_update;
use crate::error::Result;
use crate::store::CozoStore;

/// Result of a confidence propagation pass.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Number of memories whose confidence was updated.
    pub memories_updated: usize,
    /// Number of facts whose confidence was updated.
    pub facts_updated: usize,
    /// Total number of nodes visited in the propagation.
    pub nodes_visited: usize,
}

/// Propagate a confidence change from a source memory to all derived memories.
///
/// When a memory's confidence changes (e.g., it was reconsolidated and its
/// confidence dropped from 0.9 to 0.4), all memories whose `source_ids` include
/// this memory should have their confidence adjusted proportionally.
///
/// The adjustment uses a Kalman-like update: the derived memory's confidence
/// shifts toward the new source confidence, weighted by how much the derived
/// memory depends on this source (1 / number_of_sources).
pub fn propagate_confidence(
    store: &CozoStore,
    source_id: Uuid,
    new_confidence: f64,
) -> Result<PropagationResult> {
    let mut memories_updated = 0usize;
    let mut facts_updated = 0usize;
    let mut nodes_visited = 0usize;

    // Find all memories that list source_id in their source_ids
    let mut params = BTreeMap::new();
    params.insert("source_id".into(), DataValue::from(source_id.to_string()));

    // Query memories where source_id appears in source_ids array
    let result = store.run_query(
        r#"?[id, confidence, source_ids] :=
            *memories{id, confidence, source_ids},
            src_id = to_uuid($source_id),
            src_id in source_ids"#,
        params.clone(),
    )?;

    for row in &result.rows {
        let mem_id = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let old_conf = row[1].get_float().unwrap_or(1.0);
        let source_count = match &row[2] {
            DataValue::List(list) => list.len().max(1),
            _ => 1,
        };

        nodes_visited += 1;

        // Influence weight = 1 / number_of_sources
        // This memory is only partially dependent on the changed source
        let influence = 1.0 / source_count as f64;

        // New confidence = old × (1 - influence) + source_conf × influence
        let adjusted = old_conf * (1.0 - influence) + new_confidence * influence;
        let adjusted = adjusted.clamp(0.0, 1.0);

        // Only update if the change is significant (> 1%)
        if (adjusted - old_conf).abs() > 0.01 {
            update_memory_confidence(store, mem_id, adjusted)?;
            memories_updated += 1;
        }
    }

    // Also propagate to facts derived from entities mentioned in this memory
    let fact_result = store.run_query(
        r#"?[fact_id, fact_conf] :=
            *entity_mentions{memory_id, entity_id},
            memory_id = to_uuid($source_id),
            *facts{id: fact_id, subject_entity: entity_id, confidence: fact_conf, @ 'NOW'}"#,
        params,
    )?;

    for row in &fact_result.rows {
        let fact_id = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let old_fact_conf = row[1].get_float().unwrap_or(1.0);

        nodes_visited += 1;

        // Facts are influenced by their source memory's confidence
        // Use belief_update to compute the appropriate shift
        let result = belief_update::belief_update(
            old_fact_conf,
            1, // treat as singly-reinforced for propagation
            new_confidence,
            0.5, // medium provenance weight for propagation
            0.3,
        );

        if (result.new_confidence - old_fact_conf).abs() > 0.01 {
            update_fact_confidence(store, fact_id, result.new_confidence)?;
            facts_updated += 1;
        }
    }

    Ok(PropagationResult {
        memories_updated,
        facts_updated,
        nodes_visited,
    })
}

/// Update a memory's confidence via Validity versioning.
fn update_memory_confidence(store: &CozoStore, memory_id: Uuid, new_confidence: f64) -> Result<()> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(memory_id.to_string()));
    params.insert("conf".into(), DataValue::from(new_confidence));

    // Read current memory, update confidence, write back with new Validity
    let current = store.run_query(
        r#"?[id, kind, content, embedding, fields, namespace, pinned, expires_at,
            version, created_at, confidence, provenance, source_ids] :=
            *memories{id, kind, content, embedding, fields, namespace, pinned,
                      expires_at, version, created_at, confidence, provenance, source_ids},
            id = to_uuid($id)"#,
        params.clone(),
    )?;

    if current.rows.is_empty() {
        return Ok(());
    }

    let row = &current.rows[0];
    let mut write_params = BTreeMap::new();
    for (i, col) in ["id", "kind", "content", "embedding", "fields", "namespace",
                      "pinned", "expires_at", "version", "created_at", "confidence",
                      "provenance", "source_ids"].iter().enumerate() {
        if *col == "confidence" {
            write_params.insert((*col).into(), DataValue::from(new_confidence));
        } else {
            write_params.insert((*col).into(), row[i].clone());
        }
    }

    store.run_script(
        r#"?[id, valid_at, kind, content, embedding, fields, namespace, pinned,
            expires_at, version, created_at, confidence, provenance, source_ids] <- [[
            $id, 'ASSERT', $kind, $content, $embedding, $fields, $namespace, $pinned,
            $expires_at, $version, $created_at, $confidence, $provenance, $source_ids
        ]]
        :put memories {id, valid_at => kind, content, embedding, fields, namespace,
            pinned, expires_at, version, created_at, confidence, provenance, source_ids}"#,
        write_params,
    )?;

    Ok(())
}

/// Update a fact's confidence via Validity versioning.
fn update_fact_confidence(store: &CozoStore, fact_id: Uuid, new_confidence: f64) -> Result<()> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(fact_id.to_string()));

    let current = store.run_query(
        r#"?[id, subject_entity, predicate, object_entity, object_value,
            namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] :=
            *facts{id, subject_entity, predicate, object_entity, object_value,
                   namespace, temporal_status, confidence, provenance, source_ids,
                   reinforcement_count, @ 'NOW'},
            id = to_uuid($id)"#,
        params,
    )?;

    if current.rows.is_empty() {
        return Ok(());
    }

    let row = &current.rows[0];
    let mut write_params = BTreeMap::new();
    let cols = ["id", "subject_entity", "predicate", "object_entity", "object_value",
                "namespace", "temporal_status", "confidence", "provenance", "source_ids", "reinforcement_count"];
    for (i, col) in cols.iter().enumerate() {
        if *col == "confidence" {
            write_params.insert((*col).into(), DataValue::from(new_confidence));
        } else {
            write_params.insert((*col).into(), row[i].clone());
        }
    }

    store.run_script(
        r#"?[id, valid_at, subject_entity, predicate, object_entity, object_value,
            namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count] <- [[
            $id, 'ASSERT', $subject_entity, $predicate, $object_entity, $object_value,
            $namespace, $temporal_status, $confidence, $provenance, $source_ids, $reinforcement_count
        ]]
        :put facts {id, valid_at => subject_entity, predicate, object_entity, object_value,
            namespace, temporal_status, confidence, provenance, source_ids, reinforcement_count}"#,
        write_params,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::Memory;

    #[test]
    fn propagate_with_no_derived_memories() {
        let store = CozoStore::open_mem(4).unwrap();
        let source_id = Uuid::now_v7();

        // No memories reference this source
        let result = propagate_confidence(&store, source_id, 0.5).unwrap();
        assert_eq!(result.memories_updated, 0);
        assert_eq!(result.facts_updated, 0);
    }

    #[test]
    fn propagate_updates_derived_memory() {
        let store = CozoStore::open_mem(4).unwrap();

        // Create source memory
        let source = Memory::new("test", "source fact", vec![0.0; 4]);
        let source_id = source.id;
        store.insert_memory(&source).unwrap();

        // Create derived memory with source_ids pointing to source
        let mut derived = Memory::new("test", "derived from source", vec![0.0; 4]);
        derived.confidence = 0.9;
        derived.source_ids = vec![source_id];
        let derived_id = derived.id;
        store.insert_memory(&derived).unwrap();

        // Propagate a confidence drop
        let result = propagate_confidence(&store, source_id, 0.2).unwrap();
        assert_eq!(result.memories_updated, 1, "derived memory should be updated");

        // Verify the derived memory's confidence changed
        let updated = store.get_memory(derived_id).unwrap().unwrap();
        assert!(
            updated.confidence < 0.9,
            "derived confidence should decrease, got {}",
            updated.confidence
        );
    }

    #[test]
    fn propagate_multi_source_dilutes_influence() {
        let store = CozoStore::open_mem(4).unwrap();

        let src1 = Memory::new("test", "source 1", vec![0.0; 4]);
        let src2 = Memory::new("test", "source 2", vec![0.0; 4]);
        store.insert_memory(&src1).unwrap();
        store.insert_memory(&src2).unwrap();

        // Derived from TWO sources
        let mut derived = Memory::new("test", "derived from both", vec![0.0; 4]);
        derived.confidence = 0.9;
        derived.source_ids = vec![src1.id, src2.id];
        let derived_id = derived.id;
        store.insert_memory(&derived).unwrap();

        // Drop src1's confidence to 0
        let result = propagate_confidence(&store, src1.id, 0.0).unwrap();
        assert_eq!(result.memories_updated, 1);

        // With 2 sources, influence = 0.5, so confidence should drop by ~half
        let updated = store.get_memory(derived_id).unwrap().unwrap();
        assert!(
            updated.confidence > 0.3 && updated.confidence < 0.7,
            "multi-source should dilute influence, got {}",
            updated.confidence
        );
    }
}
