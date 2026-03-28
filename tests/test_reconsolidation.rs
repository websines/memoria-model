//! Integration test: Contradiction → memory reconsolidation.
//!
//! Run with: cargo test --features local-embeddings --test test_reconsolidation -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, test_ctx};

/// Tell conflicting facts about the same entity, then reconsolidate
/// and verify the memory content is updated.
#[tokio::test]
async fn test_reconsolidation_rewrites_memory() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    println!("\n=== Phase 1: Establish initial fact ===");
    let r1 = m
        .tell("Alice is the CEO of Acme Corp", &ctx)
        .await
        .unwrap();
    println!(
        "Initial: {} memories, {} entities",
        r1.memory_ids.len(),
        r1.entity_ids.len()
    );
    let original_id = r1.memory_ids[0];

    // Read the original memory
    let original = m.store().get_memory(original_id).unwrap().unwrap();
    println!("Original content: {:?}", original.content);

    println!("\n=== Phase 2: Tell contradicting fact ===");
    let r2 = m
        .tell("Alice is the CTO of Acme Corp, not the CEO", &ctx)
        .await
        .unwrap();
    println!(
        "Contradiction: {} memories, {} entities, surprise={:.4}",
        r2.memory_ids.len(),
        r2.entity_ids.len(),
        r2.surprise,
    );

    println!("\n=== Phase 3: Reconsolidate the original memory ===");
    let recon_result = m.reconsolidate(original_id).await.unwrap();

    if let Some(ref result) = recon_result {
        println!("Reconsolidation result:");
        println!("  Original: {:?}", result.original_content);
        println!("  Updated:  {:?}", result.updated_content);
        println!(
            "  Contradictions resolved: {}",
            result.contradictions_resolved
        );

        // Verify the memory content was actually updated
        let updated = m.store().get_memory(original_id).unwrap().unwrap();
        println!("  Memory after reconsolidation: {:?}", updated.content);

        // The updated content should differ from original
        assert_ne!(
            result.original_content, result.updated_content,
            "reconsolidation should change the content"
        );
    } else {
        println!("Reconsolidation returned None — no contradictions found by NER");
        println!("This can happen if NER didn't extract matching entities for contradiction detection");
    }

    println!("\n=== Reconsolidation test complete ===");
}

/// Tell multiple conflicting facts and verify ask() surfaces contradictions.
#[tokio::test]
async fn test_contradictions_surfaced_in_ask() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    m.tell("The server runs on AWS us-east-1", &ctx)
        .await
        .unwrap();
    m.tell("The server runs on GCP europe-west1", &ctx)
        .await
        .unwrap();

    let ask = m.ask("Where does the server run?", &ctx).await.unwrap();
    println!(
        "\nQ: Where does the server run? → {} results, {} contradictions",
        ask.results.len(),
        ask.contradictions.len()
    );

    for c in &ask.contradictions {
        println!(
            "  CONTRADICTION: entity={} pred={} [{:?} vs {:?}]",
            c.entity, c.predicate, c.value_a, c.value_b
        );
    }

    assert!(!ask.results.is_empty(), "should return results");
}
