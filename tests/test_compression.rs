//! Integration test: Memory compression + abstraction promotion.
//!
//! Run with: cargo test --features local-embeddings --test test_compression -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, load_squad_paragraphs, test_ctx};

/// Tell 20+ memories in the same namespace, build co-activation via asks,
/// then compress and verify summary memories are created.
#[tokio::test]
async fn test_compression_creates_summaries() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let squad = load_squad_paragraphs();
    let batch = &squad[..20]; // first 20 entries

    println!("\n=== Phase 1: Tell 20 paragraphs ===");
    let mut all_ids = Vec::new();
    for (i, entry) in batch.iter().enumerate() {
        match m.tell(&entry.context, &ctx).await {
            Ok(r) => {
                all_ids.extend(r.memory_ids.clone());
                if i % 5 == 0 {
                    println!("  told {}/20: {} memories", i + 1, r.memory_ids.len());
                }
            }
            Err(e) => {
                println!("  tell {}/20 failed (transient): {}", i + 1, e);
            }
        }
    }
    let pre_count = m.store().count_memories().unwrap();
    println!("Total memories before compression: {}", pre_count);

    println!("\n=== Phase 2: Ask questions to build co-activation ===");
    // Ask questions from the squad set to build Hebbian co-activation
    for entry in batch.iter().take(10) {
        let ask = m.ask(&entry.question, &ctx).await.unwrap();
        println!(
            "  Q: {} → {} results",
            &entry.question[..entry.question.len().min(50)],
            ask.results.len()
        );
    }

    println!("\n=== Phase 3: Compress ===");
    // Use min_cluster_size=2 to ensure we get some clusters even with limited co-activation
    let results = m.compress("test", 2).await.unwrap();
    println!("Compression results: {} clusters compressed", results.len());

    for (i, cr) in results.iter().enumerate() {
        println!(
            "  cluster[{}]: {} sources → summary: {:?}",
            i,
            cr.source_ids.len(),
            &cr.summary[..cr.summary.len().min(80)]
        );
    }

    assert!(
        !results.is_empty(),
        "compression must produce clusters — co_activations should be populated by Hebbian learning"
    );

    // Verify summary memories exist
    let post_count = m.store().count_memories().unwrap();
    println!(
        "\nMemories after compression: {} (was {})",
        post_count, pre_count
    );
    assert!(
        post_count > pre_count,
        "compression should create summary memories"
    );

    // Verify summaries are retrievable via ask
    let summary_ask = m
        .ask("What topics are covered in the knowledge base?", &ctx)
        .await
        .unwrap();
    println!(
        "Post-compression ask: {} results",
        summary_ask.results.len()
    );
    assert!(
        !summary_ask.results.is_empty(),
        "should find results after compression"
    );

    println!("\n=== Compression test complete ===");
}

/// Verify that tick() detects compression threshold.
#[tokio::test]
async fn test_tick_triggers_compression_check() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    // Tell a few memories (under threshold)
    for i in 0..5 {
        let _ = m
            .tell(
                &format!("Test memory number {} about various topics", i),
                &ctx,
            )
            .await;
    }

    // Tick should run without error (compression not triggered since count < 100)
    let tick_result = m.tick().unwrap();
    println!(
        "Tick result: compression_enqueued={}, reflection_enqueued={}",
        tick_result.compression_enqueued, tick_result.reflection_enqueued
    );

    // No task queue wired, so nothing can be enqueued
    assert!(
        !tick_result.compression_enqueued,
        "no queue means no enqueue"
    );
}
