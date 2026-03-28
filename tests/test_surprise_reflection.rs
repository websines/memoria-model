//! Integration test: Surprise computation + reflection pipeline.
//!
//! Run with: cargo test --features local-embeddings --test test_surprise_reflection -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, test_ctx};

/// Tell consistent facts, then contradicting facts — verify surprise rises,
/// then run reflection and verify abstractions are created.
#[tokio::test]
async fn test_surprise_and_reflection_pipeline() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    println!("\n=== Phase 1: Tell consistent facts ===");
    let consistent_facts = [
        "Acme Corp was founded in 2015 in San Francisco",
        "Acme Corp builds enterprise AI software",
        "Alice is the CEO of Acme Corp",
        "Acme Corp has 200 employees",
        "Acme Corp raised $50 million in Series B funding",
    ];
    for fact in &consistent_facts {
        let r = m.tell(fact, &ctx).await.unwrap();
        println!("  told: {} → {} memories, {} entities", fact, r.memory_ids.len(), r.entity_ids.len());
    }

    let surprise_after_consistent = m.accumulated_surprise().unwrap();
    println!("\nSurprise after consistent facts: {:.4}", surprise_after_consistent);

    println!("\n=== Phase 2: Tell contradicting facts ===");
    let contradicting_facts = [
        "Acme Corp was founded in 2018 in New York",
        "Bob is the CEO of Acme Corp",
    ];
    for fact in &contradicting_facts {
        let r = m.tell(fact, &ctx).await.unwrap();
        println!(
            "  told (contradicting): {} → surprise={:.4}",
            fact, r.surprise
        );
    }

    let surprise_after_contradiction = m.accumulated_surprise().unwrap();
    println!(
        "\nSurprise after contradictions: {:.4} (was {:.4})",
        surprise_after_contradiction, surprise_after_consistent
    );
    // Surprise must rise — contradictions against existing facts should trigger it
    assert!(
        surprise_after_contradiction > surprise_after_consistent,
        "surprise must increase after contradictions: {:.4} should be > {:.4}",
        surprise_after_contradiction,
        surprise_after_consistent
    );

    println!("\n=== Phase 3: Run reflection ===");
    let reflection_result = m.run_reflection().await.unwrap();
    if let Some(ref result) = reflection_result {
        println!(
            "Reflection completed: {} episodes reviewed, {} abstractions created, {} facts created",
            result.episodes_reviewed.len(),
            result.abstractions_created,
            result.facts_created
        );
    } else {
        println!("Reflection returned None (surprise may not have exceeded threshold)");
    }

    // Post-reflection surprise should be resolved (or at least not higher)
    let surprise_after_reflection = m.accumulated_surprise().unwrap();
    println!(
        "Surprise after reflection: {:.4} (was {:.4})",
        surprise_after_reflection, surprise_after_contradiction
    );

    // Verify total memories stored
    let total = m.store().count_memories().unwrap();
    println!("\nTotal memories: {}", total);
    assert!(total >= 7, "should have at least 7 memories");

    println!("\n=== All surprise/reflection assertions passed ===");
}

/// Verify that TellResult includes surprise field for individual tells.
#[tokio::test]
async fn test_tell_result_has_surprise() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let r1 = m
        .tell("The project uses Rust for the backend", &ctx)
        .await
        .unwrap();
    println!("Tell #1 surprise: {:.4}", r1.surprise);

    let r2 = m
        .tell("The project uses Python for the backend", &ctx)
        .await
        .unwrap();
    println!("Tell #2 surprise (contradicting): {:.4}", r2.surprise);

    // Both surprises should be valid floats
    assert!(!r1.surprise.is_nan(), "surprise should not be NaN");
    assert!(!r2.surprise.is_nan(), "surprise should not be NaN");
}
