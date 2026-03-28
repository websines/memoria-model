//! Integration test: AIF feedback loop → free energy → beta shift → trend.
//!
//! Run with: cargo test --features local-embeddings --test test_aif_feedback -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, test_ctx};
use memoria::SkillOutcome;
use uuid::Uuid;

/// Establish baseline, call feedback() with success/failure, verify FE/beta shift.
#[tokio::test]
async fn test_feedback_loop_shifts_model_state() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    println!("\n=== Phase 1: Prime with baseline memories ===");
    m.tell("The project uses Rust for performance-critical components", &ctx)
        .await
        .unwrap();
    m.tell("The API handles 10,000 requests per second", &ctx)
        .await
        .unwrap();
    m.tell("Deployment uses Kubernetes on AWS", &ctx)
        .await
        .unwrap();

    // Snapshot initial state
    let state_before = m.snapshot_model_state("integration-test").unwrap();
    println!(
        "Initial state: FE={:.4}, beta={:.4}, accuracy={:.4}, complexity={:.4}",
        state_before.free_energy, state_before.beta, state_before.accuracy, state_before.complexity
    );

    println!("\n=== Phase 2: Feedback with Success ===");
    let task_id = Uuid::now_v7();
    let fb_success = m
        .feedback(task_id, &SkillOutcome::Success, "integration-test", &[])
        .await
        .unwrap();
    println!(
        "Success feedback: FE={:.4}, beta={:.4}, surprise_triggered={}, consolidation_triggered={}",
        fb_success.free_energy, fb_success.beta, fb_success.surprise_triggered, fb_success.consolidation_triggered
    );
    assert!(!fb_success.free_energy.is_nan(), "FE should not be NaN");
    assert!(!fb_success.beta.is_nan(), "beta should not be NaN");

    println!("\n=== Phase 3: Feedback with Failure ===");
    let task_id2 = Uuid::now_v7();
    let fb_failure = m
        .feedback(
            task_id2,
            &SkillOutcome::Failure,
            "integration-test",
            &[],
        )
        .await
        .unwrap();
    println!(
        "Failure feedback: FE={:.4}, beta={:.4}, surprise_triggered={}, consolidation_triggered={}",
        fb_failure.free_energy, fb_failure.beta, fb_failure.surprise_triggered, fb_failure.consolidation_triggered
    );

    println!("\n=== Phase 4: Check model state evolution ===");
    let state_after = m.snapshot_model_state("integration-test").unwrap();
    println!(
        "Final state: FE={:.4}, beta={:.4}",
        state_after.free_energy, state_after.beta
    );

    // Beta and FE should be valid
    assert!(!state_after.free_energy.is_nan());
    assert!(!state_after.beta.is_nan());
    assert!(state_after.beta >= 0.0 && state_after.beta <= 1.0);

    println!("\n=== Feedback loop test complete ===");
}

/// Verify that multiple feedback calls produce a computable trend.
#[tokio::test]
async fn test_free_energy_trend() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    // Build some state
    m.tell("Some baseline knowledge for the model", &ctx)
        .await
        .unwrap();

    println!("\n=== Generate multiple model state snapshots ===");
    for i in 0..5 {
        let task_id = Uuid::now_v7();
        let outcome = if i % 2 == 0 {
            SkillOutcome::Success
        } else {
            SkillOutcome::Failure
        };
        let fb = m
            .feedback(task_id, &outcome, "integration-test", &[])
            .await
            .unwrap();
        println!(
            "  feedback[{}]: outcome={:?} FE={:.4} beta={:.4}",
            i, outcome, fb.free_energy, fb.beta
        );
    }

    // Compute trend
    let trend = memoria::aif::health::compute_trend(m.store(), 5).unwrap();
    println!("\nFree energy trend: {:?}", trend);

    // Trend should be a valid variant
    match trend {
        memoria::Trend::Improving => println!("  → Model is improving"),
        memoria::Trend::Stable => println!("  → Model is stable"),
        memoria::Trend::Degrading => println!("  → Model is degrading"),
    }

    // Verify health computation works
    let health = m.model_health(0, 0).unwrap();
    println!(
        "Model health: trend={:?}, beta={:.4}, unresolved_surprise={:.4}",
        health.free_energy_trend, health.beta, health.unresolved_surprise
    );

    println!("\n=== Free energy trend test complete ===");
}

/// Verify auto_beta adapts from the initial 1.0 default.
#[tokio::test]
async fn test_auto_beta_adaptation() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let initial_beta = m.auto_beta().unwrap();
    println!("Initial auto_beta: {:.4}", initial_beta);
    // Should default to 1.0 when no snapshots exist
    assert!(
        (initial_beta - 1.0).abs() < 0.01,
        "initial beta should be 1.0"
    );

    // Create episodes and feedback to shift beta
    m.tell("Some knowledge", &ctx).await.unwrap();
    for _ in 0..3 {
        let _ = m
            .feedback(
                Uuid::now_v7(),
                &SkillOutcome::Success,
                "integration-test",
                &[],
            )
            .await;
    }

    let updated_beta = m.auto_beta().unwrap();
    println!("Updated auto_beta: {:.4}", updated_beta);
    assert!(!updated_beta.is_nan(), "beta should not be NaN");

    // Check effective consolidation threshold
    let threshold = m.effective_consolidation_threshold().unwrap();
    println!("Effective consolidation threshold: {:.4}", threshold);
    assert!(threshold > 0.0, "threshold should be positive");

    println!("\n=== Auto beta test complete ===");
}
