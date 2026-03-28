//! Pearl's do-operator — interventional reasoning via graph surgery.
//!
//! `do(X = x)` differs from observing `X = x`:
//! - **Observation**: condition on X=x, all edges intact, both upstream and downstream affected.
//! - **Intervention**: cut all incoming edges to X, clamp X=x, propagate only outward.
//!
//! This enables counterfactual reasoning like "What would happen to entity B's confidence
//! if we *set* entity A's confidence to 0.9?" — without the confounding of A's upstream causes.

use cozo::DataValue;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;

/// Result of a do-operator intervention.
#[derive(Debug, Clone)]
pub struct DoResult {
    /// The variable that was intervened on.
    pub target_id: Uuid,
    /// The value it was clamped to.
    pub clamped_value: f64,
    /// Nodes affected by forward propagation, with their new values.
    pub propagated_effects: Vec<PropagatedEffect>,
    /// Number of incoming edges that were "cut" (conceptually).
    pub edges_cut: usize,
}

/// A downstream effect of the intervention.
#[derive(Debug, Clone)]
pub struct PropagatedEffect {
    pub node_id: Uuid,
    pub original_value: f64,
    pub new_value: f64,
    pub edge_kind: String,
    pub distance: usize, // hops from intervention target
}

/// Perform a do-operator intervention: `do(target = value)`.
///
/// Steps (Pearl's 3-step procedure):
/// 1. **Graph surgery**: identify and conceptually remove all incoming edges to `target`.
/// 2. **Clamp**: set the target variable to `value`.
/// 3. **Forward propagation**: traverse outgoing edges from target, computing
///    downstream effects using edge weights as influence factors.
///
/// This is a read-only simulation — no database state is modified.
/// The intervention is computed over the knowledge graph's `edges` relation.
pub fn do_intervention(
    store: &CozoStore,
    target: Uuid,
    value: f64,
    max_depth: usize,
) -> Result<DoResult> {
    // Step 1: Count incoming edges (conceptually "cut")
    let edges_cut = count_incoming_edges(store, target)?;

    // Step 2: Get the target's current confidence (from entities or facts)
    let _original_value = get_node_confidence(store, target)?;

    // Step 3: Forward propagation — BFS from target through outgoing edges
    let mut effects = Vec::new();
    let mut frontier = vec![(target, value, 0usize)]; // (node, value, depth)
    let mut visited = std::collections::HashSet::new();
    visited.insert(target);

    while let Some((current_node, current_value, depth)) = frontier.pop() {
        if depth >= max_depth {
            continue;
        }

        // Get outgoing edges from current node
        let outgoing = get_outgoing_edges(store, current_node)?;

        for (neighbor, weight, kind) in outgoing {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);

            let neighbor_original = get_node_confidence(store, neighbor)?;
            // Propagated value: linear influence via edge weight
            // new = original * (1 - weight) + current_value * weight
            let propagated = neighbor_original * (1.0 - weight) + current_value * weight;

            effects.push(PropagatedEffect {
                node_id: neighbor,
                original_value: neighbor_original,
                new_value: propagated,
                edge_kind: kind,
                distance: depth + 1,
            });

            frontier.push((neighbor, propagated, depth + 1));
        }
    }

    Ok(DoResult {
        target_id: target,
        clamped_value: value,
        propagated_effects: effects,
        edges_cut,
    })
}

/// Counterfactual query: "What would Y be if we did do(X = x)?"
///
/// Returns the propagated value of `query_node` after intervening on `target`.
/// Returns `None` if `query_node` is not reachable from `target`.
pub fn counterfactual_query(
    store: &CozoStore,
    target: Uuid,
    value: f64,
    query_node: Uuid,
    max_depth: usize,
) -> Result<Option<f64>> {
    let result = do_intervention(store, target, value, max_depth)?;
    Ok(result
        .propagated_effects
        .iter()
        .find(|e| e.node_id == query_node)
        .map(|e| e.new_value))
}

/// Compare observational vs interventional reasoning.
///
/// Returns `(observational, interventional)` — the difference reveals confounding.
pub fn compare_observe_vs_intervene(
    store: &CozoStore,
    target: Uuid,
    value: f64,
    query_node: Uuid,
    max_depth: usize,
) -> Result<(f64, f64)> {
    // Observational: just get the current value (no surgery)
    let observational = get_node_confidence(store, query_node)?;

    // Interventional: do-operator with graph surgery
    let interventional = counterfactual_query(store, target, value, query_node, max_depth)?
        .unwrap_or(observational);

    Ok((observational, interventional))
}

/// Perform a structural do-operator intervention using the causal edge graph.
///
/// Unlike `do_intervention` which uses the generic `edges` relation,
/// this function loads the structural causal model from `causal_edges`
/// into a petgraph DiGraph and performs proper graph surgery:
///
/// 1. Load causal graph from CozoDB
/// 2. Remove all incoming edges to `target` (graph surgery)
/// 3. BFS forward-propagation through the mutilated graph
///
/// Falls back to `do_intervention` if no causal graph exists.
pub fn do_intervention_structural(
    store: &CozoStore,
    target: Uuid,
    value: f64,
    namespace: &str,
    max_depth: usize,
) -> Result<DoResult> {
    // Load the structural causal model
    let cg = super::graph::load_causal_graph(store, namespace, 0.1)?;

    if cg.edge_count() == 0 {
        // No structural model available — fall back to legacy do-operator
        return do_intervention(store, target, value, max_depth);
    }

    let target_node = match cg.get_node(&target) {
        Some(n) => n,
        None => {
            // Target not in causal graph — no structural effects
            return Ok(DoResult {
                target_id: target,
                clamped_value: value,
                propagated_effects: vec![],
                edges_cut: 0,
            });
        }
    };

    // Graph surgery: create a mutilated graph without incoming edges to target
    let mut mutilated = cg.graph.clone();
    let incoming: Vec<_> = mutilated
        .edges_directed(target_node, Direction::Incoming)
        .map(|e| e.id())
        .collect();
    let edges_cut = incoming.len();
    for edge_id in incoming {
        mutilated.remove_edge(edge_id);
    }

    // Forward-propagation through the mutilated graph via BFS
    let mut effects = Vec::new();
    let mut frontier = vec![(target_node, value, 0usize)];
    let mut visited = std::collections::HashSet::new();
    visited.insert(target_node);

    while let Some((current_node, current_value, depth)) = frontier.pop() {
        if depth >= max_depth {
            continue;
        }

        for edge in mutilated.edges_directed(current_node, Direction::Outgoing) {
            let neighbor = edge.target();
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);

            let causal_strength = *edge.weight();
            let neighbor_id = mutilated[neighbor];

            // Get original confidence for this node
            let original = get_node_confidence(store, neighbor_id)?;

            // Propagated value: linear influence via causal strength
            let propagated = original * (1.0 - causal_strength) + current_value * causal_strength;

            effects.push(PropagatedEffect {
                node_id: neighbor_id,
                original_value: original,
                new_value: propagated,
                edge_kind: "causal".to_string(),
                distance: depth + 1,
            });

            frontier.push((neighbor, propagated, depth + 1));
        }
    }

    Ok(DoResult {
        target_id: target,
        clamped_value: value,
        propagated_effects: effects,
        edges_cut,
    })
}

// ── Internal helpers ──

fn count_incoming_edges(store: &CozoStore, target: Uuid) -> Result<usize> {
    let mut params = BTreeMap::new();
    params.insert("target".into(), DataValue::from(target.to_string()));

    let result = store.run_query(
        r#"?[count(source)] :=
            *edges{source, target, kind: _k, @ 'NOW'},
            target = to_uuid($target)"#,
        params,
    )?;

    Ok(result
        .rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0) as usize)
}

fn get_node_confidence(store: &CozoStore, node: Uuid) -> Result<f64> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(node.to_string()));

    // Try entities first
    let result = store.run_query(
        r#"?[confidence] :=
            *entities{id, confidence, @ 'NOW'},
            id = to_uuid($id)"#,
        params.clone(),
    )?;

    if let Some(row) = result.rows.first() {
        return Ok(row[0].get_float().unwrap_or(1.0));
    }

    // Try memories
    let result = store.run_query(
        r#"?[confidence] :=
            *memories{id, confidence},
            id = to_uuid($id)"#,
        params,
    )?;

    Ok(result
        .rows
        .first()
        .and_then(|r| r[0].get_float())
        .unwrap_or(1.0))
}

fn get_outgoing_edges(
    store: &CozoStore,
    source: Uuid,
) -> Result<Vec<(Uuid, f64, String)>> {
    let mut params = BTreeMap::new();
    params.insert("source".into(), DataValue::from(source.to_string()));

    let result = store.run_query(
        r#"?[target, weight, kind] :=
            *edges{source, target, kind, weight, @ 'NOW'},
            source = to_uuid($source)"#,
        params,
    )?;

    let mut edges = Vec::new();
    for row in &result.rows {
        let target = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let weight = row[1].get_float().unwrap_or(1.0);
        let kind = row[2].get_str().unwrap_or("").to_string();
        edges.push((target, weight, kind));
    }

    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_do_intervention_no_edges() {
        let store = CozoStore::open_mem(4).unwrap();
        let target = Uuid::now_v7();

        let result = do_intervention(&store, target, 0.5, 3).unwrap();
        assert_eq!(result.target_id, target);
        assert_eq!(result.clamped_value, 0.5);
        assert_eq!(result.edges_cut, 0);
        assert!(result.propagated_effects.is_empty());
    }

    #[test]
    fn test_do_intervention_with_chain() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();

        // Create chain: a → b → c with weight 0.5
        store.insert_edge(a, b, "causal", 0.5).unwrap();
        store.insert_edge(b, c, "causal", 0.5).unwrap();
        // Also add incoming edge to a (will be "cut")
        let upstream = Uuid::now_v7();
        store.insert_edge(upstream, a, "upstream", 1.0).unwrap();

        let result = do_intervention(&store, a, 0.9, 3).unwrap();
        assert_eq!(result.edges_cut, 1); // upstream → a was cut
        assert_eq!(result.propagated_effects.len(), 2); // b and c affected

        let b_effect = result.propagated_effects.iter().find(|e| e.node_id == b).unwrap();
        assert_eq!(b_effect.distance, 1);
        // b_new = b_orig * (1 - 0.5) + 0.9 * 0.5 = 1.0 * 0.5 + 0.45 = 0.95
        assert!((b_effect.new_value - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_counterfactual_query() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();

        store.insert_edge(a, b, "causal", 0.5).unwrap();

        let result = counterfactual_query(&store, a, 0.0, b, 3).unwrap();
        assert!(result.is_some());
        // b_new = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert!((result.unwrap() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_counterfactual_unreachable() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        // No edge between a and b

        let result = counterfactual_query(&store, a, 0.0, b, 3).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_compare_observe_vs_intervene() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();

        store.insert_edge(a, b, "causal", 0.5).unwrap();

        let (obs, inter) = compare_observe_vs_intervene(&store, a, 0.0, b, 3).unwrap();
        // Observational: b's current confidence (default 1.0)
        assert!((obs - 1.0).abs() < 0.01);
        // Interventional: propagated via do(a=0.0)
        assert!((inter - 0.5).abs() < 0.01);
    }
}
