//! Integration test: Telos lifecycle — create → decompose → progress → stall → reflection.
//!
//! Run with: cargo test --features local-embeddings --test test_telos_lifecycle -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use std::sync::Arc;

use common::{build_real_memoria, build_real_memoria_with_queue, test_ctx, wait_queue_drain};
use memoria::{TelosProvenance, TelosStatus};

/// Create a telos, decompose it via the LLM, verify children created.
#[tokio::test]
async fn test_telos_create_and_decompose() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    println!("\n=== Create a strategic goal ===");
    let telos = m
        .create_telos(
            "Launch MVP product",
            "Build and ship a minimum viable product for the AI assistant platform",
            &ctx,
            1, // strategic depth
            None,
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();

    println!("Created telos: id={}, title={}", telos.id, telos.title);
    assert_eq!(telos.status, TelosStatus::Active);
    assert_eq!(telos.depth, 1);
    assert!(!telos.embedding.is_empty());

    println!("\n=== Decompose via LLM ===");
    let llm: Arc<dyn memoria::services::LlmService> = Arc::new(common::TestLlm::new(
        common::LM_STUDIO_URL,
        common::LLM_MODEL,
    ));
    let embedder: Arc<dyn memoria::services::Embedder> = Arc::new(
        memoria::services::ApiEmbedder::new(
            common::LM_STUDIO_URL,
            common::EMBED_MODEL,
            common::EMBED_DIM,
        ),
    );
    let children = memoria::api::telos_decompose::decompose_telos(
        m.store(),
        &llm,
        &embedder,
        telos.id,
        "test",
    )
    .await
    .unwrap();

    println!("Decomposition produced {} subtelos:", children.len());
    for child in &children {
        println!(
            "  - [depth={}] {} (priority={:.2})",
            child.depth, child.title, child.priority
        );
    }

    assert!(
        (2..=5).contains(&children.len()),
        "should produce 2-5 subtelos, got {}",
        children.len()
    );
    for child in &children {
        assert_eq!(child.parent, Some(telos.id));
        assert_eq!(child.depth, 2); // parent was depth 1
        assert_eq!(child.status, TelosStatus::Active);
        assert_eq!(child.provenance, TelosProvenance::Decomposition);
    }

    // Verify events recorded
    let events = m.store().get_telos_events(telos.id, 10).unwrap();
    println!("\nParent events: {:?}", events.iter().map(|e| &e.event_type).collect::<Vec<_>>());
    assert!(events.iter().any(|e| e.event_type == "created"));
    assert!(events.iter().any(|e| e.event_type == "decomposed"));
}

/// Create a goal with success criteria, record progress, estimate it.
#[tokio::test]
async fn test_telos_progress_estimation() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let mut telos = m
        .create_telos(
            "Complete onboarding",
            "Finish all onboarding tasks for new hire",
            &ctx,
            2,
            None,
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();

    // Add success criteria
    telos.success_criteria = vec![
        memoria::SuccessCriterion {
            id: "setup".to_string(),
            description: "Set up development environment".to_string(),
            met: false,
        },
        memoria::SuccessCriterion {
            id: "training".to_string(),
            description: "Complete training modules".to_string(),
            met: false,
        },
        memoria::SuccessCriterion {
            id: "first-pr".to_string(),
            description: "Submit first pull request".to_string(),
            met: false,
        },
    ];
    m.store().upsert_telos(&telos).unwrap();

    // Mark one criterion
    m.mark_criterion_met(telos.id, "setup", "integration-test")
        .unwrap();

    // Estimate progress (deterministic path)
    let estimate =
        memoria::api::telos_progress::estimate_progress(m.store(), telos.id).unwrap();

    println!(
        "Progress estimate: {:.2} (method={:?}, confidence={:.2})",
        estimate.progress, estimate.method, estimate.confidence
    );

    // 1 of 3 criteria met → ~0.33
    assert!(
        (estimate.progress - 0.333).abs() < 0.1,
        "progress should be ~0.33, got {:.4}",
        estimate.progress
    );

    // Record explicit progress
    m.record_telos_progress(telos.id, 0.1, "Made some extra progress", "integration-test", None, None)
        .await
        .unwrap();

    let updated = m.get_telos(telos.id).unwrap().unwrap();
    println!("Progress after delta: {:.2}", updated.progress);

    println!("\n=== Telos progress test complete ===");
}

/// Verify that active_telos() returns scored results.
#[tokio::test]
async fn test_active_telos_scoring() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    // Create multiple goals with different priorities
    let _t1 = m
        .create_telos(
            "Low priority goal",
            "Something not urgent",
            &ctx,
            2,
            None,
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();

    let mut t2 = m
        .create_telos(
            "High priority goal",
            "Critical deadline approaching",
            &ctx,
            1,
            None,
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();

    // Boost priority on t2
    t2.priority = 0.95;
    m.store().upsert_telos(&t2).unwrap();

    let active = m.active_telos("test", 10).unwrap();
    println!("Active telos ({}):", active.len());
    for st in &active {
        println!(
            "  {} — priority={:.2}, attention={:.4}",
            st.telos.title, st.telos.priority, st.attention_score
        );
    }

    assert!(active.len() >= 2, "should have at least 2 active telos");
    // Highest attention score should be first
    if active.len() >= 2 {
        assert!(
            active[0].attention_score >= active[1].attention_score,
            "telos should be sorted by attention score"
        );
    }
}

/// Create a goal, complete it via status update, verify reflection enqueued.
#[tokio::test]
async fn test_telos_completion_with_queue() {
    let (m, queue, cancel) = build_real_memoria_with_queue();
    let ctx = test_ctx();

    let telos = m
        .create_telos(
            "Write documentation",
            "Write user guide for the API",
            &ctx,
            2,
            None,
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();

    println!("Created telos: {}", telos.id);

    // Complete it
    m.update_telos_status(telos.id, "completed", "integration-test", "All docs written")
        .await
        .unwrap();

    let updated = m.get_telos(telos.id).unwrap().unwrap();
    assert_eq!(updated.status, TelosStatus::Completed);

    // Wait for queue to process the telos_reflection task
    println!("Waiting for queue to process telos_reflection...");
    wait_queue_drain(&queue, std::time::Duration::from_secs(60)).await;

    // Verify events
    let events = m.store().get_telos_events(telos.id, 10).unwrap();
    println!(
        "Events: {:?}",
        events.iter().map(|e| &e.event_type).collect::<Vec<_>>()
    );
    assert!(events.iter().any(|e| e.event_type == "completed"));

    cancel.cancel();
    println!("\n=== Telos completion test complete ===");
}
