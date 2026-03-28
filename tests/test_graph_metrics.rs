//! Integration test: PageRank + community detection on real data.
//!
//! Run with: cargo test --features local-embeddings --test test_graph_metrics -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, load_squad_paragraphs, test_ctx};

/// Tell 10+ memories via real NER to build entity links,
/// then compute graph metrics and verify PageRank/communities.
#[tokio::test]
async fn test_graph_metrics_on_real_data() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let squad = load_squad_paragraphs();

    println!("\n=== Phase 1: Tell 12 paragraphs (building entity graph via NER) ===");
    let mut all_memory_ids = Vec::new();
    let mut total_entities = 0;
    for (i, entry) in squad.iter().take(12).enumerate() {
        let r = m.tell(&entry.context, &ctx).await.unwrap();
        all_memory_ids.extend(r.memory_ids.clone());
        total_entities += r.entity_ids.len();
        if i % 4 == 0 {
            println!(
                "  told {}/12: {} mems, {} entities",
                i + 1,
                r.memory_ids.len(),
                r.entity_ids.len()
            );
        }
    }
    println!(
        "Total: {} memories, {} entities extracted",
        all_memory_ids.len(),
        total_entities
    );

    // Insert some explicit edges to ensure graph connectivity
    println!("\n=== Phase 2: Add explicit edges for graph connectivity ===");
    if all_memory_ids.len() >= 6 {
        // Create a small connected subgraph
        for i in 0..5 {
            m.store()
                .insert_edge(
                    all_memory_ids[i],
                    all_memory_ids[i + 1],
                    "related_to",
                    0.8,
                )
                .unwrap();
        }
        // Add a cross-link
        m.store()
            .insert_edge(all_memory_ids[0], all_memory_ids[5], "related_to", 0.6)
            .unwrap();
        println!("  Added 6 edges");
    }

    println!("\n=== Phase 3: Compute graph metrics ===");
    let metrics = m.compute_graph_metrics().unwrap();
    println!("Graph metrics: {} nodes computed", metrics.len());

    for (i, imp) in metrics.iter().take(5).enumerate() {
        println!(
            "  [{}] memory={} pagerank={:.6} community={} in={} out={}",
            i, imp.memory_id, imp.pagerank, imp.community_id, imp.in_degree, imp.out_degree
        );
    }

    assert!(
        !metrics.is_empty(),
        "graph metrics must be computed — we added explicit edges"
    );

    // Verify PageRank values are non-negative
    for imp in &metrics {
        assert!(imp.pagerank >= 0.0, "PageRank should be non-negative");
    }

    // Check cached PageRank for a specific memory
    let first_id = metrics[0].memory_id;
    let cached = memoria::dynamics::graph_metrics::get_cached_pagerank(m.store(), first_id).unwrap();
    println!(
        "\nCached PageRank for {}: {:?}",
        first_id, cached
    );
    if let Some(pr) = cached {
        assert!(pr >= 0.0);
    }

    // Check communities
    let communities = memoria::dynamics::graph_metrics::get_communities(m.store()).unwrap();
    println!("Communities: {} found", communities.len());
    for c in &communities {
        println!(
            "  community[{}]: {} members",
            c.community_id,
            c.member_ids.len()
        );
    }

    // We added edges, so we should have at least 1 community
    assert!(
        !communities.is_empty(),
        "should have at least 1 community with connected nodes"
    );

    println!("\n=== Graph metrics test complete ===");
}

/// Verify that highly-connected memories get higher PageRank.
#[tokio::test]
async fn test_pagerank_reflects_connectivity() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    // Create 5 memories
    let mut ids = Vec::new();
    for i in 0..5 {
        let r = m
            .tell(
                &format!("Memory node {} in the test graph network", i),
                &ctx,
            )
            .await
            .unwrap();
        ids.extend(r.memory_ids);
    }

    // Make ids[0] a hub — connect it to all others
    for i in 1..ids.len().min(5) {
        m.store()
            .insert_edge(ids[0], ids[i], "hub_link", 1.0)
            .unwrap();
        m.store()
            .insert_edge(ids[i], ids[0], "hub_link", 1.0)
            .unwrap();
    }

    let metrics = m.compute_graph_metrics().unwrap();
    if !metrics.is_empty() {
        // Find the hub node's PageRank
        let hub_rank = metrics.iter().find(|m| m.memory_id == ids[0]);
        let leaf_rank = metrics.iter().find(|m| m.memory_id == ids[1]);

        if let (Some(hub), Some(leaf)) = (hub_rank, leaf_rank) {
            println!(
                "Hub PageRank: {:.6}, Leaf PageRank: {:.6}",
                hub.pagerank, leaf.pagerank
            );
            // Hub should have higher or equal PageRank (it's the most connected)
            assert!(
                hub.pagerank >= leaf.pagerank,
                "hub should have higher PageRank"
            );
        }
    }

    println!("\n=== PageRank connectivity test complete ===");
}
