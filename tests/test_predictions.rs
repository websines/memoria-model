//! Integration tests for Predictive Generation (Initiative 2).
//!
//! Tests the full prediction lifecycle:
//! - PPM-C sequence predictions from task history
//! - Prediction storage, retrieval, and resolution
//! - Prediction matching against observations in tell()
//! - Prediction accuracy tracking
//! - Free energy integration (prediction error → higher free energy)

use std::sync::Arc;

use memoria::runtime::Memoria;

/// Helper: create a Memoria instance with mock services.
fn setup() -> Memoria {
    let mut m = Memoria::with_mocks(4).unwrap();
    let queue = Arc::new(memoria::queue::TaskQueue::new(
        memoria::store::CozoStore::open_mem(4).unwrap(),
    ));
    m.set_task_queue(queue);
    m
}

/// PPM-C generates predictions from task outcome history.
#[test]
fn test_ppm_sequence_predictions() {
    let m = setup();

    // Record a repeated task pattern: debug → test → deploy (×3)
    use memoria::causal::sequence_mining::record_task_outcome;
    for _ in 0..3 {
        record_task_outcome(m.store(), "debug", "success", "agent-1").unwrap();
        record_task_outcome(m.store(), "test", "success", "agent-1").unwrap();
        record_task_outcome(m.store(), "deploy", "success", "agent-1").unwrap();
    }

    // Generate PPM-C predictions
    let predictions = memoria::dynamics::prediction::generate_sequence_predictions(
        m.store(), "agent-1", "default", 5,
    ).unwrap();

    assert!(!predictions.is_empty(), "should generate at least one prediction");

    // The top prediction after the last "deploy" should be "debug" (the cycle restarts)
    let top = &predictions[0];
    assert_eq!(top.kind, memoria::dynamics::PredictionKind::NextTask);
    assert!(top.confidence > 0.1, "confidence should be meaningful");

    // Store and retrieve
    memoria::dynamics::prediction::store_predictions(m.store(), &predictions).unwrap();
    let now = memoria::types::memory::now_ms();
    let pending = memoria::dynamics::prediction::get_pending_predictions(
        m.store(), "default", now,
    ).unwrap();
    assert_eq!(pending.len(), predictions.len());
}

/// Predictions resolved as matches reduce surprise; expired ones add surprise.
#[test]
fn test_prediction_resolution_lifecycle() {
    let m = setup();
    let now = memoria::types::memory::now_ms();

    // Create a prediction that will match
    let match_pred = memoria::dynamics::Prediction {
        id: uuid::Uuid::now_v7(),
        kind: memoria::dynamics::PredictionKind::NextTask,
        content: "next_task:debug".into(),
        embedding: Vec::new(),
        predicted_at: now,
        expected_by: now + 3_600_000,
        confidence: 0.8,
        confidence_interval: None,
        source: memoria::dynamics::PredictionSource::SequencePPM,
        context_ids: Vec::new(),
        namespace: "default".into(),
        resolved: false,
        resolution: None,
    };

    // Create a prediction that will expire
    let expire_pred = memoria::dynamics::Prediction {
        id: uuid::Uuid::now_v7(),
        kind: memoria::dynamics::PredictionKind::NextTask,
        content: "next_task:deploy".into(),
        embedding: Vec::new(),
        predicted_at: now - 7_200_000,
        expected_by: now - 3_600_000, // already expired
        confidence: 0.9,
        confidence_interval: None,
        source: memoria::dynamics::PredictionSource::SequencePPM,
        context_ids: Vec::new(),
        namespace: "default".into(),
        resolved: false,
        resolution: None,
    };

    memoria::dynamics::prediction::store_predictions(
        m.store(), &[match_pred.clone(), expire_pred.clone()],
    ).unwrap();

    // Match the first prediction
    let obs_id = uuid::Uuid::now_v7();
    let result = memoria::dynamics::prediction::match_observation_against_predictions(
        m.store(), "debug task completed", &[], obs_id, "default",
    ).unwrap();

    assert_eq!(result.matches, 1, "should match 'next_task:debug'");
    assert_eq!(result.expirations, 1, "should expire the deploy prediction");
    // Match reduces surprise, expiration increases it
    // Net: -0.8 (match) + 0.9 (expiry) = +0.1
    assert!(result.surprise_delta > 0.0, "net surprise should be positive due to expiry");
}

/// Prediction accuracy reflects resolution outcomes.
#[test]
fn test_prediction_accuracy_tracking() {
    let m = setup();
    let now = memoria::types::memory::now_ms();

    // Create and resolve some predictions with known errors
    for i in 0..5 {
        let pred = memoria::dynamics::Prediction {
            id: uuid::Uuid::now_v7(),
            kind: memoria::dynamics::PredictionKind::NextTask,
            content: format!("next_task:task_{i}"),
            embedding: Vec::new(),
            predicted_at: now,
            expected_by: now + 3_600_000,
            confidence: 0.8,
            confidence_interval: None,
            source: memoria::dynamics::PredictionSource::SequencePPM,
            context_ids: Vec::new(),
            namespace: "default".into(),
            resolved: false,
            resolution: None,
        };
        memoria::dynamics::prediction::store_predictions(m.store(), &[pred.clone()]).unwrap();

        // Resolve: 3 correct (error=0.1), 2 wrong (error=0.9)
        let error = if i < 3 { 0.1 } else { 0.9 };
        memoria::dynamics::prediction::resolve_prediction(
            m.store(), pred.id, error < 0.5, None, error,
        ).unwrap();
    }

    let accuracy = memoria::dynamics::prediction::prediction_accuracy(
        m.store(), "default",
    ).unwrap();

    // Expected: 1.0 - avg(0.1, 0.1, 0.1, 0.9, 0.9) = 1.0 - 0.42 = 0.58
    assert!(accuracy > 0.5, "accuracy should be > 0.5, got {accuracy}");
    assert!(accuracy < 0.7, "accuracy should be < 0.7, got {accuracy}");
}

/// PPM-C model serialization roundtrip through CozoDB.
#[test]
fn test_ppm_persistence() {
    let m = setup();

    // Record some task history
    use memoria::causal::sequence_mining::record_task_outcome;
    for _ in 0..5 {
        record_task_outcome(m.store(), "code", "success", "agent-x").unwrap();
        record_task_outcome(m.store(), "review", "success", "agent-x").unwrap();
    }

    // Generate predictions (this saves the PPM model)
    let preds1 = memoria::dynamics::prediction::generate_sequence_predictions(
        m.store(), "agent-x", "default", 5,
    ).unwrap();
    assert!(!preds1.is_empty());

    // Generate again (this loads the saved model and rebuilds)
    let preds2 = memoria::dynamics::prediction::generate_sequence_predictions(
        m.store(), "agent-x", "default", 5,
    ).unwrap();

    // Both runs should produce the same predictions
    assert_eq!(preds1.len(), preds2.len());
    assert_eq!(preds1[0].content, preds2[0].content);
}

/// Free energy increases when predictions have high error.
#[test]
fn test_prediction_error_increases_free_energy() {
    let m = setup();
    let now = memoria::types::memory::now_ms();

    // Baseline free energy (no predictions)
    let baseline = memoria::aif::compute_bethe_free_energy(m.store()).unwrap();

    // Add resolved predictions with high error
    for _ in 0..5 {
        let pred = memoria::dynamics::Prediction {
            id: uuid::Uuid::now_v7(),
            kind: memoria::dynamics::PredictionKind::NextTask,
            content: "next_task:wrong".into(),
            embedding: Vec::new(),
            predicted_at: now,
            expected_by: now + 3_600_000,
            confidence: 0.9,
            confidence_interval: None,
            source: memoria::dynamics::PredictionSource::SequencePPM,
            context_ids: Vec::new(),
            namespace: "default".into(),
            resolved: false,
            resolution: None,
        };
        memoria::dynamics::prediction::store_predictions(m.store(), &[pred.clone()]).unwrap();
        memoria::dynamics::prediction::resolve_prediction(
            m.store(), pred.id, false, None, 1.0, // total miss
        ).unwrap();
    }

    let with_errors = memoria::aif::compute_bethe_free_energy(m.store()).unwrap();

    assert!(
        with_errors.factor_energy > baseline.factor_energy,
        "high prediction error should increase factor energy: {} > {}",
        with_errors.factor_energy, baseline.factor_energy,
    );
}
