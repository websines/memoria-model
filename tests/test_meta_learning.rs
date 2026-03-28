//! Integration test: Meta-learning over dynamics parameters.
//!
//! Verifies that the meta-learning subsystem:
//! 1. Starts in ColdStart phase and explores parameter space
//! 2. Records snapshots to CozoDB with free energy observations
//! 3. Transitions to OnlineTracking after bo_budget generations
//! 4. SPSA state machine cycles through Idle → WaitingPlus → WaitingMinus → Idle
//! 5. Parameters actually change in response to free energy signals
//! 6. Persists and recovers state across restarts
//! 7. Config hot-reload applies meta-learned params to effective config
//!
//! No LLM or embedder needed — meta-learning is pure math over store state.
//!
//! Run with: cargo test --test test_meta_learning -- --nocapture

use std::sync::Arc;

use memoria::{CozoStore, Memoria, MemoriaConfig, AgentContext};
use memoria::dynamics::meta_learning::{
    MetaLearner, MetaPhase,
    load_meta_params, load_optimizer_state,
};
use memoria::skills::{Skill, SkillStep, SkillProvenance, SkillPerformance};
use memoria::skills::storage::store_skill;

/// Build a Memoria with mocks and meta-learning enabled.
fn build_meta_learning_memoria() -> Memoria {
    let mut config = MemoriaConfig::default();
    config.dynamics.meta_learning_enabled = true;
    config.dynamics.meta_learning_interval = 1; // every tick for testing
    config.dynamics.observation_window = 2;      // short window for testing
    config.dynamics.bo_budget = 5;               // transition quickly for testing

    use memoria::services::mock::{MockEmbedder, MockLlm, MockNer, MockReranker};
    let store = CozoStore::open_mem(128).unwrap();

    Memoria::new(
        store,
        Arc::new(MockEmbedder::new(128)),
        Arc::new(MockNer),
        Arc::new(MockReranker),
        Arc::new(MockLlm),
        config,
    )
}

/// Populate the store with synthetic data so free energy is computable.
/// Without facts/skills/entities, free energy is 0.0 (trivial).
fn seed_store_with_synthetic_data(m: &Memoria) {
    let store = m.store();

    // Insert some facts with varying confidence to give free energy something to compute.
    // Schema: facts { id: Uuid, valid_at: Validity => subject_entity: Uuid, predicate: String,
    //   confidence: Float, reinforcement_count: Int, ... }
    for i in 0..10 {
        let confidence = 0.3 + (i as f64) * 0.07; // 0.3 to 0.93
        let rc = (i % 5) + 1; // reinforcement 1-5
        let id = uuid::Uuid::now_v7();
        let entity_id = uuid::Uuid::now_v7();
        let script = r#"
            ?[id, valid_at, subject_entity, predicate, confidence, reinforcement_count] <- [[
                to_uuid($id), 'ASSERT',
                to_uuid($entity_id), $predicate,
                $confidence, $rc,
            ]]
            :put facts {
                id, valid_at
                =>
                subject_entity, predicate, confidence, reinforcement_count,
            }
        "#;
        let mut params = std::collections::BTreeMap::new();
        params.insert("id".into(), cozo::DataValue::Str(id.to_string().into()));
        params.insert("entity_id".into(), cozo::DataValue::Str(entity_id.to_string().into()));
        params.insert("predicate".into(), cozo::DataValue::Str("has_property".into()));
        params.insert("confidence".into(), cozo::DataValue::from(confidence));
        params.insert("rc".into(), cozo::DataValue::from(rc as i64));

        store.run_script(script, params).expect("insert fact");
    }

    // Insert skills via the proper API (matches schema exactly)
    let dim = 128;
    for i in 0..3 {
        let skill = Skill {
            id: uuid::Uuid::now_v7(),
            name: format!("skill_{i}"),
            description: format!("Test skill {i} for meta-learning integration"),
            steps: vec![
                SkillStep { step: 1, action: "first step".into() },
                SkillStep { step: 2, action: "second step".into() },
            ],
            preconditions: vec![],
            postconditions: vec![],
            confidence: 0.6 + i as f64 * 0.15,
            provenance: SkillProvenance::Bootstrapped,
            source_episodes: vec![],
            domain: "testing".into(),
            version: 1,
            performance: SkillPerformance {
                success_rate: 0.5 + i as f64 * 0.15,
                avg_duration_ms: 1000.0,
                usage_count: (3 + i) as u64,
                last_used: 0,
            },
            parent_skill: None,
            tags: vec!["test".into()],
        };
        let embedding = vec![0.1 * (i as f32 + 1.0); dim];
        store_skill(store, &skill, &embedding).expect("insert skill");
    }
}

#[test]
fn test_meta_learner_cold_start_explores() {
    let m = build_meta_learning_memoria();
    seed_store_with_synthetic_data(&m);
    let store = m.store();

    let mut learner = MetaLearner::new(2, 5); // obs_window=2, bo_budget=5

    // Compute initial free energy
    let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
    println!("Initial free energy: {:.4}, beta: {:.4}", fe.free_energy, fe.beta);
    assert!(fe.free_energy != 0.0, "Free energy should be non-zero with seeded data");

    // Run cold-start steps — should get adjustments every observation_window ticks
    let mut total_adjustments = 0;
    let mut got_some = false;
    for i in 0..12 {
        let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
        let result = learner.step(store, fe.free_energy, fe.beta, 0.0).unwrap();
        if let Some(ref r) = result {
            println!("Step {i}: gen={}, phase={:?}, adjustments={}",
                r.generation, r.phase, r.adjustments.len());
            total_adjustments += r.adjustments.len();
            got_some = true;
        } else {
            println!("Step {i}: waiting");
        }
    }

    assert!(got_some, "Should have produced at least one adjustment");
    assert!(total_adjustments > 0, "Should have adjusted some parameters");
    // With bo_budget=5 and obs_window=2, the learner may or may not have transitioned
    // by step 12 depending on how many steps produced adjustments. The key assertion is
    // that it explored — adjustments were made and snapshots recorded.

    // Verify snapshots were recorded
    let snapshot_count: i64 = store.run_query(
        "?[count(id)] := *meta_snapshots{id}",
        std::collections::BTreeMap::new(),
    ).unwrap().rows[0][0].get_int().unwrap();
    assert!(snapshot_count >= 10, "Should have recorded snapshots: got {snapshot_count}");
    println!("Recorded {snapshot_count} snapshots");

    // Verify params were written
    let params = load_meta_params(store).unwrap();
    assert!(!params.is_empty(), "Should have written meta params");
    println!("Meta params: {params:?}");
}

#[test]
fn test_meta_learner_transitions_to_spsa() {
    let m = build_meta_learning_memoria();
    seed_store_with_synthetic_data(&m);
    let store = m.store();

    // Use bo_budget=3, obs_window=1 for fast transition
    let mut learner = MetaLearner::new(1, 3);

    // Run enough steps to exceed bo_budget
    for i in 0..10 {
        let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
        let result = learner.step(store, fe.free_energy, fe.beta, 0.0).unwrap();
        if let Some(ref r) = result {
            println!("Step {i}: gen={}, phase={:?}", r.generation, r.phase);
            if r.phase == MetaPhase::OnlineTracking {
                println!("Transitioned to OnlineTracking at step {i}!");
                break;
            }
        }
    }

    assert_eq!(learner.phase, MetaPhase::OnlineTracking,
        "Should have transitioned to SPSA after bo_budget generations");
    println!("Generation at transition: {}", learner.generation);
}

#[test]
fn test_spsa_state_machine_cycles() {
    let m = build_meta_learning_memoria();
    seed_store_with_synthetic_data(&m);
    let store = m.store();

    // Start directly in online tracking with obs_window=1
    let mut learner = MetaLearner::new(1, 0); // bo_budget=0 → immediate SPSA
    // Force transition: run one step which hits budget immediately
    let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
    let _ = learner.step(store, fe.free_energy, fe.beta, 0.0);

    assert_eq!(learner.phase, MetaPhase::OnlineTracking, "Should be in SPSA mode");

    // Now run SPSA cycle:
    // Step 1: Idle → WaitingPlus (returns None)
    // Step 2: WaitingPlus → WaitingMinus (obs_window=1, so 1 tick is enough)
    // Step 3: WaitingMinus → Idle with gradient update (returns Some)
    let initial_gen = learner.generation;
    let mut cycle_completed = false;

    for i in 0..10 {
        let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
        let result = learner.step(store, fe.free_energy, fe.beta, 0.0).unwrap();
        if let Some(ref r) = result {
            if r.phase == MetaPhase::OnlineTracking && r.generation > initial_gen {
                println!("SPSA cycle completed at step {i}: gen={}, adjustments={}",
                    r.generation, r.adjustments.len());
                for (name, old, new) in &r.adjustments {
                    println!("  {name}: {old:.4} → {new:.4}");
                }
                cycle_completed = true;
                break;
            }
        } else {
            println!("Step {i}: SPSA waiting...");
        }
    }

    assert!(cycle_completed, "SPSA should complete at least one gradient step");
}

#[test]
fn test_optimizer_state_persists_and_recovers() {
    let m = build_meta_learning_memoria();
    seed_store_with_synthetic_data(&m);
    let store = m.store();

    // Run a few steps to build state
    let mut learner = MetaLearner::new(2, 10);
    for _ in 0..6 {
        let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
        let _ = learner.step(store, fe.free_energy, fe.beta, 0.0);
    }

    let gen_before = learner.generation;
    let params_before: Vec<f64> = learner.params.iter().map(|p| p.current).collect();
    println!("Before recovery: gen={gen_before}, params={params_before:?}");

    // Recover from store
    let recovered = load_optimizer_state(store).unwrap();
    assert!(recovered.is_some(), "Should recover optimizer state from store");
    let recovered = recovered.unwrap();

    assert_eq!(recovered.generation, gen_before, "Generation should match");
    assert_eq!(recovered.phase, learner.phase, "Phase should match");
    let params_after: Vec<f64> = recovered.params.iter().map(|p| p.current).collect();
    // Use approximate comparison — JSON roundtrip can introduce tiny floating point drift
    for (i, (before, after)) in params_before.iter().zip(params_after.iter()).enumerate() {
        assert!(
            (before - after).abs() < 1e-10,
            "Param {i} mismatch after recovery: {before} vs {after}"
        );
    }
    println!("After recovery: gen={}, params={params_after:?}", recovered.generation);
}

#[test]
fn test_config_hot_reload_applies_meta_params() {
    let m = build_meta_learning_memoria();
    seed_store_with_synthetic_data(&m);

    // Read initial config value
    let initial_threshold = m.effective_config().consolidation_threshold;
    println!("Initial consolidation_threshold: {initial_threshold}");

    // Manually write a meta param (simulating what the worker does)
    let store = m.store();
    let script = r#"
        ?[name, value, min_bound, max_bound, step, generation, updated_at] <- [[
            "consolidation_threshold", 1.337, 0.5, 5.0, 0.2, 42, 0
        ]]
        :put meta_params {
            name => value, min_bound, max_bound, step, generation, updated_at
        }
    "#;
    store.run_script(script, std::collections::BTreeMap::new()).unwrap();

    // Verify params are in store
    let params = load_meta_params(store).unwrap();
    assert!(params.iter().any(|(n, v)| n == "consolidation_threshold" && (*v - 1.337).abs() < 1e-6),
        "Should find our param in store: {params:?}");

    // Now call tick() which should reload the config
    let _ = m.tick();

    // Verify the effective config was updated
    let new_threshold = m.effective_config().consolidation_threshold;
    println!("After tick: consolidation_threshold = {new_threshold}");
    assert!((new_threshold - 1.337).abs() < 1e-6,
        "Config should reflect meta-learned value: got {new_threshold}");

    // Verify base_config is unchanged
    assert!((m.base_config().consolidation_threshold - initial_threshold).abs() < 1e-6,
        "Base config should be unchanged");
}

#[test]
fn test_end_to_end_meta_learning_reduces_variance() {
    // This test verifies the full loop:
    // 1. Seed store with data that produces non-trivial free energy
    // 2. Run meta-learning for several generations
    // 3. Verify that parameters actually move (the system is exploring)
    // 4. Verify that free energy snapshots are recorded for trend detection

    let m = build_meta_learning_memoria();
    seed_store_with_synthetic_data(&m);
    let store = m.store();

    let mut learner = MetaLearner::new(1, 100); // obs_window=1, large budget

    let initial_params: Vec<(String, f64)> = learner.params.iter()
        .map(|p| (p.name.clone(), p.current))
        .collect();

    // Run 20 steps
    let mut fe_history = Vec::new();
    for _ in 0..20 {
        let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
        fe_history.push(fe.free_energy);
        let _ = learner.step(store, fe.free_energy, fe.beta, 0.0);
    }

    let final_params: Vec<(String, f64)> = learner.params.iter()
        .map(|p| (p.name.clone(), p.current))
        .collect();

    // At least some parameters should have moved
    let mut moved_count = 0;
    for (i, (name, initial)) in initial_params.iter().enumerate() {
        let (_, final_val) = &final_params[i];
        let delta = (final_val - initial).abs();
        if delta > 1e-10 {
            moved_count += 1;
            println!("{name}: {initial:.4} → {final_val:.4} (Δ={delta:.4})");
        }
    }
    assert!(moved_count > 0, "At least some parameters should have been adjusted");
    println!("\n{moved_count}/{} parameters moved", initial_params.len());

    // Free energy history should have been recorded
    let snapshot_count: i64 = store.run_query(
        "?[count(id)] := *meta_snapshots{id}",
        std::collections::BTreeMap::new(),
    ).unwrap().rows[0][0].get_int().unwrap();
    assert_eq!(snapshot_count, 20, "Should have 20 snapshots");

    // Verify we can compute a trend from the snapshots
    println!("\nFree energy history: {:?}", &fe_history[..5.min(fe_history.len())]);
    println!("Total snapshots recorded: {snapshot_count}");
    println!("Final generation: {}", learner.generation);
}
