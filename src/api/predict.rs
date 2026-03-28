//! Predictive graph traversal — forward inference on the knowledge graph.
//!
//! Instead of just embed-and-search, predict which entities and facts are
//! likely relevant by traversing edges outward from entities mentioned in
//! the query. This pre-activates connected knowledge before retrieval.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::memory::CandidateMemory;

/// Result of predictive graph traversal.
#[derive(Debug)]
pub struct PredictionResult {
    /// Memory IDs predicted as relevant, with prediction confidence.
    pub predicted_memories: Vec<CandidateMemory>,
    /// Entity IDs that were traversed.
    pub traversed_entities: Vec<Uuid>,
    /// Number of hops used in traversal.
    pub max_depth_reached: usize,
}

/// Perform predictive graph traversal from entities mentioned in a query.
///
/// 1. Extract entity names from the query text (heuristic)
/// 2. Look up matching entities in the knowledge graph
/// 3. BFS outward through edges (entity_mentions, facts, edges) up to `max_depth`
/// 4. Collect memories connected to traversed entities
/// 5. Return predicted-relevant memories sorted by proximity (closer = more relevant)
pub fn predict_relevant_memories(
    store: &CozoStore,
    query_text: &str,
    namespace: &str,
    max_depth: usize,
    max_results: usize,
) -> Result<PredictionResult> {
    // Step 1: Find seed entities from query text
    let seed_entities = find_seed_entities(store, query_text, namespace)?;

    if seed_entities.is_empty() {
        return Ok(PredictionResult {
            predicted_memories: Vec::new(),
            traversed_entities: Vec::new(),
            max_depth_reached: 0,
        });
    }

    // Step 2: BFS through the knowledge graph from seed entities
    let mut visited_entities: HashSet<Uuid> = HashSet::new();
    let mut entity_distances: HashMap<Uuid, usize> = HashMap::new();
    let mut frontier: Vec<(Uuid, usize)> = Vec::new();
    let mut max_depth_reached = 0;

    for &entity_id in &seed_entities {
        visited_entities.insert(entity_id);
        entity_distances.insert(entity_id, 0);
        frontier.push((entity_id, 0));
    }

    while let Some((entity_id, depth)) = frontier.pop() {
        if depth >= max_depth {
            continue;
        }
        max_depth_reached = max_depth_reached.max(depth);

        // Find connected entities via facts
        let connected = find_connected_entities(store, entity_id)?;
        for neighbor_id in connected {
            if visited_entities.insert(neighbor_id) {
                let new_depth = depth + 1;
                entity_distances.insert(neighbor_id, new_depth);
                frontier.push((neighbor_id, new_depth));
            }
        }
    }

    // Step 3: Collect memories linked to traversed entities
    let traversed: Vec<Uuid> = visited_entities.iter().copied().collect();
    let mut memory_distances: HashMap<Uuid, usize> = HashMap::new();

    for &entity_id in &traversed {
        let entity_dist = entity_distances.get(&entity_id).copied().unwrap_or(0);
        let linked_mems = find_memories_for_entity(store, entity_id)?;
        for mem_id in linked_mems {
            memory_distances
                .entry(mem_id)
                .and_modify(|d| *d = (*d).min(entity_dist))
                .or_insert(entity_dist);
        }
    }

    // Step 4: Build candidate memories sorted by proximity
    let mut mem_ids_sorted: Vec<(Uuid, usize)> = memory_distances.into_iter().collect();
    mem_ids_sorted.sort_by_key(|&(_, dist)| dist);
    mem_ids_sorted.truncate(max_results);

    let ids: Vec<Uuid> = mem_ids_sorted.iter().map(|(id, _)| *id).collect();
    let memories = store.get_memories_by_ids(&ids)?;

    let predicted_memories: Vec<CandidateMemory> = memories
        .into_iter()
        .map(|mem| {
            let dist = mem_ids_sorted
                .iter()
                .find(|(id, _)| *id == mem.id)
                .map(|(_, d)| *d)
                .unwrap_or(0);
            CandidateMemory {
                // Distance inversely proportional to graph proximity
                distance: dist as f64 * 0.1,
                memory: mem,
                activation: None,
                hebbian_weight: None,
                pagerank: None,
                precision: None,
                telos_boost: None,
            }
        })
        .collect();

    Ok(PredictionResult {
        predicted_memories,
        traversed_entities: traversed,
        max_depth_reached,
    })
}

/// Find seed entities by matching words in the query against entity names.
fn find_seed_entities(
    store: &CozoStore,
    query_text: &str,
    namespace: &str,
) -> Result<Vec<Uuid>> {
    let mut seeds = Vec::new();

    // Try common entity extraction patterns
    let patterns = ["who is ", "what is ", "tell me about ", "about "];
    let lower = query_text.to_lowercase();

    for prefix in &patterns {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let name = rest.trim_end_matches('?').trim_end_matches('.').trim();
            if !name.is_empty() {
                // Try exact match (title-cased)
                let title = crate::api::query_planner::title_case(name);
                if let Some(entity) = store.find_entity_by_name(&title, namespace)? {
                    seeds.push(entity.id);
                }
                // Also try as-is
                if let Some(entity) = store.find_entity_by_name(name, namespace)? {
                    if !seeds.contains(&entity.id) {
                        seeds.push(entity.id);
                    }
                }
            }
        }
    }

    // Also try each capitalized word as a potential entity name
    for word in query_text.split_whitespace() {
        if word.len() > 1 && word.chars().next().map_or(false, |c| c.is_uppercase()) {
            let clean = word.trim_end_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty() {
                if let Some(entity) = store.find_entity_by_name(clean, namespace)? {
                    if !seeds.contains(&entity.id) {
                        seeds.push(entity.id);
                    }
                }
            }
        }
    }

    Ok(seeds)
}

/// Find entities connected to the given entity via facts (subject→object, object→subject).
fn find_connected_entities(store: &CozoStore, entity_id: Uuid) -> Result<Vec<Uuid>> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(entity_id.to_string()));

    let result = store.run_query(
        r#"
        outgoing[target] :=
            *facts{subject_entity, object_entity},
            subject_entity = to_uuid($id),
            object_entity != null,
            target = object_entity
        incoming[source] :=
            *facts{subject_entity, object_entity},
            object_entity = to_uuid($id),
            source = subject_entity
        ?[connected] := outgoing[connected]
        ?[connected] := incoming[connected]
        "#,
        params,
    )?;

    let mut entities = Vec::new();
    for row in &result.rows {
        if let Ok(id) = crate::store::cozo::parse_uuid_pub(&row[0]) {
            entities.push(id);
        }
    }

    Ok(entities)
}

/// Find memory IDs linked to an entity via entity_mentions.
fn find_memories_for_entity(store: &CozoStore, entity_id: Uuid) -> Result<Vec<Uuid>> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("entity_id".into(), DataValue::from(entity_id.to_string()));

    let result = store.run_query(
        r#"?[memory_id] :=
            *entity_mentions{entity_id, memory_id},
            entity_id = to_uuid($entity_id)"#,
        params,
    )?;

    let mut mems = Vec::new();
    for row in &result.rows {
        if let Ok(id) = crate::store::cozo::parse_uuid_pub(&row[0]) {
            mems.push(id);
        }
    }

    Ok(mems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_empty_store() {
        let store = CozoStore::open_mem(4).unwrap();
        let result = predict_relevant_memories(&store, "Who is Alice?", "test", 3, 10).unwrap();
        assert!(result.predicted_memories.is_empty());
        assert!(result.traversed_entities.is_empty());
    }

    #[test]
    fn test_find_seed_entities() {
        let store = CozoStore::open_mem(4).unwrap();
        // No entities in store → no seeds
        let seeds = find_seed_entities(&store, "Who is Alice?", "test").unwrap();
        assert!(seeds.is_empty());
    }
}
