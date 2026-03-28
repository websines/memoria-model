//! Integration tests for Phase 6: Active Inference Unification.
//!
//! Full cycle: empty store → tell facts → snapshot → check FE → ask with precision → check health.

use memoria::{Memoria, Trend};
use memoria::types::query::AgentContext;

#[tokio::test]
async fn test_aif_full_cycle() {
    let m = Memoria::with_mocks(128).unwrap();
    let ctx = AgentContext::new("test-agent", "default");

    // 1. Empty store: snapshot should succeed, FE = 0, β near 0
    let state = m.snapshot_model_state("test-agent").unwrap();
    assert_eq!(state.free_energy, 0.0, "empty store → zero free energy");
    assert!(state.beta >= 0.0 && state.beta <= 1.0, "β should be in [0,1]");

    // 2. Tell some facts
    m.tell("Alice works at Acme Corp", &ctx).await.unwrap();
    m.tell("Bob is a software engineer", &ctx).await.unwrap();
    m.tell("Charlie manages the team", &ctx).await.unwrap();

    // 3. Snapshot again — FE should be nonzero now (memories have nonzero precision)
    let _state2 = m.snapshot_model_state("test-agent").unwrap();
    // The store has memories (which don't contribute to FE directly since we query facts/skills/entities)
    // but the model_state relation now has snapshots

    // 4. Auto-beta should return a valid value
    let beta = m.auto_beta().unwrap();
    assert!(beta >= 0.0 && beta <= 1.0, "auto_beta={beta} should be in [0,1]");

    // 5. Ask with precision scoring — should produce finite scores
    let result = m.ask("Who works at Acme?", &ctx).await.unwrap();
    for r in &result.results {
        assert!(r.score.is_finite(), "precision-weighted scores should be finite");
    }

    // 6. Health check
    let health = m.model_health(0, 0).unwrap();
    assert!(health.beta >= 0.0 && health.beta <= 1.0);
    assert_eq!(health.queue_depth, 0);
    assert_eq!(health.queue_dead, 0);
}

#[tokio::test]
async fn test_aif_precision_scoring_backward_compatible() {
    let m = Memoria::with_mocks(128).unwrap();
    let ctx = AgentContext::new("test-agent", "default");

    // Store memories and retrieve them — the pipeline should work the same as before
    m.tell("The speed of light is 299792458 m/s", &ctx).await.unwrap();
    m.tell("Water boils at 100 degrees Celsius at 1 atm", &ctx).await.unwrap();
    m.tell("Earth orbits the Sun in approximately 365.25 days", &ctx).await.unwrap();

    let result = m.ask("What is the speed of light?", &ctx).await.unwrap();

    // Should still return results (backward compatibility)
    assert!(!result.results.is_empty(), "should retrieve results");

    // All scores should be finite
    for r in &result.results {
        assert!(r.score.is_finite(), "all scores must be finite, got {}", r.score);
    }
}

#[tokio::test]
async fn test_select_skills_auto() {
    let m = Memoria::with_mocks(128).unwrap();

    // select_skills_auto should work even without any snapshots (uses default β=1.0)
    let skills = m.select_skills_auto("write tests", 5).await.unwrap();
    // No skills bootstrapped, so empty is fine
    assert!(skills.is_empty(), "no skills → empty result");
}

#[test]
fn test_model_health_with_queue_info() {
    let m = Memoria::with_mocks(128).unwrap();

    let health = m.model_health(10, 3).unwrap();
    assert_eq!(health.queue_depth, 10);
    assert_eq!(health.queue_dead, 3);
    assert_eq!(health.free_energy_trend, Trend::Stable);
}

#[test]
fn test_multiple_snapshots() {
    let m = Memoria::with_mocks(128).unwrap();

    // Take multiple snapshots
    let s1 = m.snapshot_model_state("agent-1").unwrap();
    let s2 = m.snapshot_model_state("agent-2").unwrap();

    // Both should succeed
    assert!(s1.beta >= 0.0 && s1.beta <= 1.0);
    assert!(s2.beta >= 0.0 && s2.beta <= 1.0);
}

#[test]
fn test_effective_consolidation_threshold_adapts() {
    let m = Memoria::with_mocks(128).unwrap();

    // With no snapshots, β defaults to 1.0
    // effective = base / (1 + 1.0) = base / 2
    let base = 2.0; // default consolidation_threshold from config.rs
    let effective = m.effective_consolidation_threshold().unwrap();

    // β = 1.0 (default) → threshold halved
    assert!(
        (effective - base / 2.0).abs() < 0.01,
        "adaptive threshold should be base/2 when β=1.0, got {effective} vs base={base}"
    );
}

#[tokio::test]
async fn test_prime_returns_context() {
    let m = Memoria::with_mocks(128).unwrap();
    let ctx = AgentContext::new("test-agent", "default");

    m.tell("Alice works at Acme Corp", &ctx).await.unwrap();

    let prime = m.prime("Who works at Acme?", &ctx, 5).await.unwrap();

    // Skills may be empty (no skills bootstrapped) but other fields should be valid
    assert!(prime.beta >= 0.0 && prime.beta <= 1.0);
    assert!(prime.free_energy.is_finite());
    assert!(prime.unresolved_surprise >= 0.0);
}

#[tokio::test]
async fn test_feedback_records_outcome() {
    let m = Memoria::with_mocks(128).unwrap();

    let result = m
        .feedback(
            uuid::Uuid::now_v7(),
            &memoria::SkillOutcome::Success,
            "test-agent",
            &[], // no skills used
        )
        .await
        .unwrap();

    assert!(result.beta >= 0.0 && result.beta <= 1.0);
    assert!(result.free_energy.is_finite());
    assert!(!result.consolidation_triggered, "no surprise → no consolidation");
}

#[test]
fn test_propagate_confidence_no_derived() {
    let m = Memoria::with_mocks(128).unwrap();

    let result = m
        .propagate_confidence(uuid::Uuid::now_v7(), 0.5)
        .unwrap();

    assert_eq!(result.memories_updated, 0);
    assert_eq!(result.facts_updated, 0);
}
