//! d-Separation checks for conditional independence in the provenance graph.
//!
//! Used by causal attribution to prune dependent variables before computing
//! counterfactuals. If two memories are d-separated given a conditioning set,
//! they are conditionally independent and can be analyzed independently.

use cozo::DataValue;
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;

/// Check if two nodes are d-separated in the provenance graph,
/// given a conditioning set.
///
/// Simplified implementation: checks if there's a path between node_a and
/// node_b through the edges graph that doesn't go through any node in
/// the conditioning set.
pub fn is_d_separated(
    store: &CozoStore,
    node_a: Uuid,
    node_b: Uuid,
    conditioning_set: &[Uuid],
) -> Result<bool> {
    if conditioning_set.is_empty() {
        // Without conditioning, check if there's any path
        return Ok(!has_path(store, node_a, node_b)?);
    }

    // Build conditioning set for query
    let cond_vals: Vec<DataValue> = conditioning_set
        .iter()
        .map(|id| DataValue::List(vec![DataValue::from(id.to_string())]))
        .collect();

    let mut params = BTreeMap::new();
    params.insert("source".into(), DataValue::from(node_a.to_string()));
    params.insert("target".into(), DataValue::from(node_b.to_string()));
    params.insert("blocked".into(), DataValue::List(cond_vals));

    // Try to find a path that doesn't go through any blocked node
    let result = store.run_query(
        r#"blocked_raw[id] <- $blocked
        blocked[uid] := blocked_raw[id], uid = to_uuid(id)

        reachable[node] := node = to_uuid($source)
        reachable[next] :=
            reachable[current],
            *edges{source: current, target: next, kind: _k, @ 'NOW'},
            not blocked[next]
        reachable[next] :=
            reachable[current],
            *edges{source: next, target: current, kind: _k, @ 'NOW'},
            not blocked[next]

        ?[node] :=
            reachable[node],
            node = to_uuid($target)"#,
        params,
    )?;

    // d-separated means NO path exists (when conditioning blocks all paths)
    Ok(result.rows.is_empty())
}

/// Check if there's any path between two nodes in the edges graph.
fn has_path(store: &CozoStore, from: Uuid, to: Uuid) -> Result<bool> {
    let mut params = BTreeMap::new();
    params.insert("source".into(), DataValue::from(from.to_string()));
    params.insert("target".into(), DataValue::from(to.to_string()));

    let result = store.run_query(
        r#"reachable[node] := node = to_uuid($source)
        reachable[next] :=
            reachable[current],
            *edges{source: current, target: next, kind: _k, @ 'NOW'}
        reachable[next] :=
            reachable[current],
            *edges{source: next, target: current, kind: _k, @ 'NOW'}

        ?[node] :=
            reachable[node],
            node = to_uuid($target)"#,
        params,
    )?;

    Ok(!result.rows.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_separated_no_edges() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();

        // No edges → d-separated (no path)
        assert!(is_d_separated(&store, a, b, &[]).unwrap());
    }

    #[test]
    fn test_d_separated_with_edge() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();

        // Create edges: a → c → b
        store.insert_edge(a, c, "provenance", 1.0).unwrap();
        store.insert_edge(c, b, "provenance", 1.0).unwrap();

        // Not d-separated (path exists a → c → b)
        assert!(!is_d_separated(&store, a, b, &[]).unwrap());

        // d-separated when conditioning on c (blocks the path)
        assert!(is_d_separated(&store, a, b, &[c]).unwrap());
    }
}
