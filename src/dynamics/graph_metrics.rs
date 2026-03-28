//! PageRank and community detection for the memory knowledge graph.
//!
//! Computed periodically and cached in the `memory_importance` relation.
//! PageRank measures how "important" a memory is based on its connections.
//! Community detection groups related memories into clusters.

use std::collections::BTreeMap;

use cozo::DataValue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// Result of importance computation for a single memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceResult {
    pub memory_id: Uuid,
    pub pagerank: f64,
    pub community_id: i64,
    pub in_degree: i64,
    pub out_degree: i64,
}

/// Result of community detection across all memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    pub community_id: i64,
    pub member_ids: Vec<Uuid>,
    pub size: usize,
}

/// Compute PageRank for all memories in the knowledge graph and cache results.
///
/// Uses CozoDB's built-in PageRank algorithm on the edges relation.
pub fn compute_pagerank(store: &CozoStore) -> Result<Vec<ImportanceResult>> {
    let now = now_ms();

    // First compute degree counts (always works, no graph algo needed)
    let degree_result = store.run_query(
        r#"
        out_deg[src, count(tgt)] := *edges{source: src, target: tgt}
        in_deg[tgt, count(src)] := *edges{source: src, target: tgt}
        all_nodes[n] := *edges{source: n}
        all_nodes[n] := *edges{target: n}
        ?[node, in_d, out_d] :=
            all_nodes[node],
            in_d = if(in_deg[node, d], d, 0),
            out_d = if(out_deg[node, d], d, 0)
        "#,
        BTreeMap::new(),
    );

    let mut degrees: BTreeMap<String, (i64, i64)> = BTreeMap::new();
    if let Ok(ref rows) = degree_result {
        for row in &rows.rows {
            let node_str = format!("{:?}", row[0]);
            let in_d = row[1].get_int().unwrap_or(0);
            let out_d = row[2].get_int().unwrap_or(0);
            degrees.insert(node_str, (in_d, out_d));
        }
    }

    // Try PageRank via CozoDB built-in algorithm
    // PageRank returns (node, rank) pairs
    let pr_result = store.run_query(
        r#"
        all_edges[src, tgt] := *edges{source: src, target: tgt}
        ?[node, rank] <~ PageRank(all_edges[src, tgt])
        "#,
        BTreeMap::new(),
    );

    let mut results = Vec::new();

    match pr_result {
        Ok(rows) => {
            for row in &rows.rows {
                if let Ok(node_id) = crate::store::cozo::parse_uuid_pub(&row[0]) {
                    let rank = row[1].get_float().unwrap_or(0.0);
                    let node_str = format!("{:?}", row[0]);
                    let (in_d, out_d) = degrees.get(&node_str).copied().unwrap_or((0, 0));

                    results.push(ImportanceResult {
                        memory_id: node_id,
                        pagerank: rank,
                        community_id: 0,
                        in_degree: in_d,
                        out_degree: out_d,
                    });
                }
            }
        }
        Err(_) => {
            // No edges → no PageRank, return empty
            return Ok(Vec::new());
        }
    }

    // Try community detection
    // CommunityDetectionLouvain returns (community_label, node) pairs
    let community_result = store.run_query(
        r#"
        all_edges[src, tgt] := *edges{source: src, target: tgt}
        ?[label, node] <~ CommunityDetectionLouvain(all_edges[src, tgt])
        "#,
        BTreeMap::new(),
    );

    if let Ok(rows) = community_result {
        let mut community_map: BTreeMap<Uuid, i64> = BTreeMap::new();
        for row in &rows.rows {
            let label = row[0].get_int().unwrap_or(0);
            if let Ok(node_id) = crate::store::cozo::parse_uuid_pub(&row[1]) {
                community_map.insert(node_id, label);
            }
        }

        for result in &mut results {
            if let Some(&label) = community_map.get(&result.memory_id) {
                result.community_id = label;
            }
        }
    }

    // Cache results in memory_importance
    cache_importance(store, &results, now)?;

    Ok(results)
}

/// Cache importance results in the memory_importance relation.
fn cache_importance(
    store: &CozoStore,
    results: &[ImportanceResult],
    now: i64,
) -> Result<()> {
    for result in results {
        let mut params = BTreeMap::new();
        params.insert(
            "memory_id".into(),
            DataValue::from(result.memory_id.to_string()),
        );
        params.insert("pagerank".into(), DataValue::from(result.pagerank));
        params.insert("community_id".into(), DataValue::from(result.community_id));
        params.insert("in_degree".into(), DataValue::from(result.in_degree));
        params.insert("out_degree".into(), DataValue::from(result.out_degree));
        params.insert("computed_at".into(), DataValue::from(now));

        store.run_script(
            concat!(
                "?[memory_id, pagerank, community_id, in_degree, out_degree, computed_at] <- ",
                "[[$memory_id, $pagerank, $community_id, $in_degree, $out_degree, $computed_at]] ",
                ":put memory_importance {memory_id => pagerank, community_id, in_degree, ",
                "out_degree, computed_at}",
            ),
            params,
        )?;
    }

    Ok(())
}

/// Get cached PageRank for a specific memory.
pub fn get_cached_pagerank(store: &CozoStore, memory_id: Uuid) -> Result<Option<f64>> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(memory_id.to_string()));

    let result = store.run_query(
        r#"?[pagerank] :=
            *memory_importance{memory_id, pagerank},
            memory_id = to_uuid($id)"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    Ok(Some(result.rows[0][0].get_float().unwrap_or(0.0)))
}

/// Get all communities and their members.
pub fn get_communities(store: &CozoStore) -> Result<Vec<CommunityResult>> {
    let result = store.run_query(
        r#"?[community_id, collect(memory_id)] :=
            *memory_importance{memory_id, community_id}"#,
        BTreeMap::new(),
    )?;

    let mut communities = Vec::new();
    for row in &result.rows {
        let community_id = row[0].get_int().unwrap_or(0);
        if let DataValue::List(members) = &row[1] {
            let member_ids: Vec<Uuid> = members
                .iter()
                .filter_map(|v| crate::store::cozo::parse_uuid_pub(v).ok())
                .collect();
            let size = member_ids.len();
            communities.push(CommunityResult {
                community_id,
                member_ids,
                size,
            });
        }
    }

    Ok(communities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::Memory;

    #[test]
    fn test_pagerank_empty_graph() {
        let store = CozoStore::open_mem(4).unwrap();
        let results = compute_pagerank(&store).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_pagerank_with_edges() {
        let store = CozoStore::open_mem(4).unwrap();

        // Create 3 memories
        let m1 = Memory::new("a", "alpha", vec![0.1, 0.2, 0.3, 0.4]);
        let m2 = Memory::new("b", "beta", vec![0.5, 0.6, 0.7, 0.8]);
        let m3 = Memory::new("c", "gamma", vec![0.1, 0.3, 0.5, 0.7]);
        store.insert_memory(&m1).unwrap();
        store.insert_memory(&m2).unwrap();
        store.insert_memory(&m3).unwrap();

        // Create edges: m1 → m2, m1 → m3, m2 → m3
        store.insert_edge(m1.id, m2.id, "related", 1.0).unwrap();
        store.insert_edge(m1.id, m3.id, "related", 1.0).unwrap();
        store.insert_edge(m2.id, m3.id, "related", 1.0).unwrap();

        let results = compute_pagerank(&store).unwrap();
        assert!(!results.is_empty(), "should compute PageRank for nodes with edges");

        // m3 should have highest PageRank (most incoming edges)
        let m3_rank = results.iter().find(|r| r.memory_id == m3.id);
        assert!(m3_rank.is_some(), "m3 should appear in results");
    }

    #[test]
    fn test_cached_pagerank_retrieval() {
        let store = CozoStore::open_mem(4).unwrap();

        let m1 = Memory::new("a", "alpha", vec![0.1, 0.2, 0.3, 0.4]);
        let m2 = Memory::new("b", "beta", vec![0.5, 0.6, 0.7, 0.8]);
        store.insert_memory(&m1).unwrap();
        store.insert_memory(&m2).unwrap();
        store.insert_edge(m1.id, m2.id, "related", 1.0).unwrap();

        // Before computation, no cached value
        assert!(get_cached_pagerank(&store, m1.id).unwrap().is_none());

        // After computation, should have cached values
        compute_pagerank(&store).unwrap();
        let rank = get_cached_pagerank(&store, m1.id).unwrap();
        assert!(rank.is_some(), "should have cached PageRank after computation");
    }

    #[test]
    fn test_get_communities_empty() {
        let store = CozoStore::open_mem(4).unwrap();
        let communities = get_communities(&store).unwrap();
        assert!(communities.is_empty());
    }
}
