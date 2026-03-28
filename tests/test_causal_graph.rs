//! Integration tests for Initiative 5: Structural Causal Models.
//!
//! Tests cover:
//! 1. Causal edge accumulation with Bayesian updates
//! 2. Bayes-Ball d-separation on various graph structures
//! 3. NOTEARS structure learning on synthetic data
//! 4. Structural do-operator intervention
//! 5. Causal ancestors and confounder detection
//! 6. Edge confidence adjustment (NOTEARS validation)

use memoria::CozoStore;
use uuid::Uuid;

// ── Helpers ──

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn make_edge(
    cause: Uuid,
    effect: Uuid,
    strength: f64,
    mechanism: memoria::CausalMechanism,
) -> memoria::CausalEdge {
    memoria::CausalEdge {
        cause_id: cause,
        effect_id: effect,
        causal_strength: strength,
        observations: 1,
        last_observed: now_ms(),
        mechanism,
        confidence: 0.5,
        namespace: String::new(),
    }
}

// ── 1. Edge Accumulation ──

#[test]
fn test_edge_accumulation_creates_and_updates() {
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();

    // First observation: strength 0.9
    let edge = make_edge(a, b, 0.9, memoria::CausalMechanism::RecallAttribution);
    memoria::causal::graph::accumulate_causal_edge(&store, &edge).unwrap();

    let edges = memoria::causal::graph::load_all_edges(&store, "").unwrap();
    assert_eq!(edges.len(), 1);
    assert!((edges[0].causal_strength - 0.9).abs() < 0.01);
    assert_eq!(edges[0].observations, 1);

    // Second observation: strength 0.5
    let edge2 = make_edge(a, b, 0.5, memoria::CausalMechanism::RecallAttribution);
    memoria::causal::graph::accumulate_causal_edge(&store, &edge2).unwrap();

    let edges = memoria::causal::graph::load_all_edges(&store, "").unwrap();
    assert_eq!(edges.len(), 1);
    // Bayesian update: (0.9 * 1 + 0.5) / 2 = 0.7
    assert!(
        (edges[0].causal_strength - 0.7).abs() < 0.01,
        "expected ~0.7, got {}",
        edges[0].causal_strength
    );
    assert_eq!(edges[0].observations, 2);

    // Third observation: strength 0.7
    let edge3 = make_edge(a, b, 0.7, memoria::CausalMechanism::NotearsDiscovered);
    memoria::causal::graph::accumulate_causal_edge(&store, &edge3).unwrap();

    let edges = memoria::causal::graph::load_all_edges(&store, "").unwrap();
    assert_eq!(edges.len(), 1);
    // Bayesian update: (0.7 * 2 + 0.7) / 3 = 0.7
    assert!(
        (edges[0].causal_strength - 0.7).abs() < 0.01,
        "expected ~0.7, got {}",
        edges[0].causal_strength
    );
    assert_eq!(edges[0].observations, 3);
}

#[test]
fn test_multiple_edges_independent() {
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();

    // a→b and a→c are independent edges
    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &make_edge(a, b, 0.8, memoria::CausalMechanism::RecallAttribution),
    )
    .unwrap();

    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &make_edge(a, c, 0.3, memoria::CausalMechanism::LlmProposed),
    )
    .unwrap();

    assert_eq!(memoria::causal::graph::count_edges(&store, "").unwrap(), 2);

    let edges = memoria::causal::graph::load_all_edges(&store, "").unwrap();
    let ab = edges.iter().find(|e| e.effect_id == b).unwrap();
    let ac = edges.iter().find(|e| e.effect_id == c).unwrap();
    assert!((ab.causal_strength - 0.8).abs() < 0.01);
    assert!((ac.causal_strength - 0.3).abs() < 0.01);
}

// ── 2. Bayes-Ball d-Separation ──

#[test]
fn test_bayes_ball_chain() {
    // A → B → C
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();

    for (cause, effect) in [(a, b), (b, c)] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(cause, effect, 0.7, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    let cg = memoria::causal::graph::load_causal_graph(&store, "", 0.0).unwrap();

    // Unconditional: A and C are NOT d-separated (path exists)
    assert!(!memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[]));

    // Conditioned on B: A and C ARE d-separated
    assert!(memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[b]));
}

#[test]
fn test_bayes_ball_fork() {
    // A ← B → C (B is common cause)
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();

    for effect in [a, c] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(b, effect, 0.7, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    let cg = memoria::causal::graph::load_causal_graph(&store, "", 0.0).unwrap();

    // Unconditional: A and C are NOT d-separated (confounded by B)
    assert!(!memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[]));

    // Conditioned on B: A and C ARE d-separated
    assert!(memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[b]));
}

#[test]
fn test_bayes_ball_collider() {
    // A → B ← C (B is collider)
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();

    for cause in [a, c] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(cause, b, 0.7, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    let cg = memoria::causal::graph::load_causal_graph(&store, "", 0.0).unwrap();

    // Unconditional: A and C ARE d-separated (collider blocks)
    assert!(memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[]));

    // Conditioned on B: A and C are NOT d-separated (explaining away)
    assert!(!memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[b]));
}

#[test]
fn test_bayes_ball_collider_descendant() {
    // A → B ← C, B → D : conditioning on D activates collider B
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();
    let d = Uuid::now_v7();

    for (cause, effect) in [(a, b), (c, b), (b, d)] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(cause, effect, 0.7, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    let cg = memoria::causal::graph::load_causal_graph(&store, "", 0.0).unwrap();

    // Unconditional: A and C are d-separated
    assert!(memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[]));

    // Conditioned on D (descendant of collider B): A and C are NOT d-separated
    assert!(!memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[d]));
}

#[test]
fn test_bayes_ball_disconnected() {
    // A → B, C → D : A and C are always d-separated
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();
    let d = Uuid::now_v7();

    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &make_edge(a, b, 0.7, memoria::CausalMechanism::RecallAttribution),
    )
    .unwrap();
    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &make_edge(c, d, 0.7, memoria::CausalMechanism::RecallAttribution),
    )
    .unwrap();

    let cg = memoria::causal::graph::load_causal_graph(&store, "", 0.0).unwrap();

    assert!(memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, c, &[]));
    assert!(memoria::causal::bayes_ball::d_separated_by_uuid(&cg, a, d, &[]));
}

// ── 3. NOTEARS ──

#[test]
fn test_notears_identity() {
    use ndarray::Array2;

    // Zero W should have h(W) = 0
    let _w: Array2<f64> = Array2::zeros((3, 3));
    let exp = memoria::causal::notears::NotearsConfig::default();
    // Just verify the config can be created and has reasonable defaults
    assert!(exp.lambda_l1 > 0.0);
    assert!(exp.max_outer_iter > 0);
}

#[test]
fn test_notears_simple_linear() {
    use ndarray::Array2;

    // Generate data from X1 → X2 (linear relationship)
    let n = 100;
    let mut data = Array2::zeros((n, 2));
    for i in 0..n {
        let x1 = (i as f64) / n as f64;
        let x2 = 0.9 * x1 + 0.05 * ((i * 7 % 11) as f64 / 11.0);
        data[[i, 0]] = x1;
        data[[i, 1]] = x2;
    }

    let config = memoria::NotearsConfig {
        lambda_l1: 0.01,
        max_outer_iter: 10,
        max_inner_iter: 200,
        lr: 0.0005,
        weight_threshold: 0.05,
        ..Default::default()
    };

    let result = memoria::causal::notears::notears(&data, &config);

    // Should find at least one edge
    assert!(
        result.edge_count >= 1,
        "linear data should yield at least 1 edge, got {}",
        result.edge_count
    );

    // No self-loops
    assert_eq!(result.adjacency[[0, 0]], 0.0);
    assert_eq!(result.adjacency[[1, 1]], 0.0);
}

// ── 4. Structural Do-Operator ──

#[test]
fn test_structural_do_intervention_chain() {
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();

    // Build causal chain: a → b → c with known strengths
    for (cause, effect, strength) in [(a, b, 0.8), (b, c, 0.6)] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(cause, effect, strength, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    // Add upstream edge to a (should be cut by intervention)
    let upstream = Uuid::now_v7();
    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &make_edge(upstream, a, 0.9, memoria::CausalMechanism::RecallAttribution),
    )
    .unwrap();

    // Intervene: do(a = 0.0)
    let result =
        memoria::causal::do_operator::do_intervention_structural(&store, a, 0.0, "", 5).unwrap();

    // Upstream edge to a should be cut
    assert_eq!(result.edges_cut, 1);

    // b and c should be affected
    assert_eq!(result.propagated_effects.len(), 2);

    // b effect: original * (1 - 0.8) + 0.0 * 0.8 = 1.0 * 0.2 = 0.2
    let b_effect = result
        .propagated_effects
        .iter()
        .find(|e| e.node_id == b)
        .unwrap();
    assert!(
        (b_effect.new_value - 0.2).abs() < 0.01,
        "expected b ≈ 0.2, got {}",
        b_effect.new_value
    );
    assert_eq!(b_effect.distance, 1);
}

#[test]
fn test_structural_do_fallback() {
    // When no causal graph exists, should fall back to legacy do-operator
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();

    // No causal edges, just use legacy edges relation
    let result =
        memoria::causal::do_operator::do_intervention_structural(&store, a, 0.5, "", 3).unwrap();

    // Should succeed (falls back to do_intervention)
    assert_eq!(result.target_id, a);
    assert_eq!(result.clamped_value, 0.5);
}

// ── 5. Causal Ancestors & Confounders ──

#[test]
fn test_causal_ancestors_transitive() {
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();

    // a → b → c
    for (cause, effect) in [(a, b), (b, c)] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(cause, effect, 0.8, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    let ancestors = memoria::causal::graph::get_causal_ancestors(&store, c, "", 10).unwrap();

    // Both a (indirect) and b (direct) should be ancestors
    let ancestor_ids: Vec<Uuid> = ancestors.iter().map(|(id, _)| *id).collect();
    assert!(ancestor_ids.contains(&b), "b should be ancestor of c");
    // a is transitive ancestor (a→b→c)
    assert!(ancestor_ids.contains(&a), "a should be transitive ancestor of c");
}

#[test]
fn test_confounders() {
    let store = CozoStore::open_mem(4).unwrap();
    let confounder = Uuid::now_v7();
    let x = Uuid::now_v7();
    let y = Uuid::now_v7();

    // confounder → x, confounder → y
    for effect in [x, y] {
        memoria::causal::graph::accumulate_causal_edge(
            &store,
            &make_edge(confounder, effect, 0.85, memoria::CausalMechanism::RecallAttribution),
        )
        .unwrap();
    }

    let confounders =
        memoria::causal::graph::get_common_confounders(&store, x, y, "").unwrap();
    assert_eq!(confounders.len(), 1);
    assert_eq!(confounders[0].0, confounder);
    assert!((confounders[0].1 - 0.85).abs() < 0.01);
    assert!((confounders[0].2 - 0.85).abs() < 0.01);
}

// ── 6. Edge Confidence Adjustment ──

#[test]
fn test_confidence_boost_and_reduce() {
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();

    // Start with moderate confidence
    let edge = memoria::CausalEdge {
        cause_id: a,
        effect_id: b,
        causal_strength: 0.6,
        observations: 1,
        last_observed: now_ms(),
        mechanism: memoria::CausalMechanism::LlmProposed,
        confidence: 0.3,
        namespace: String::new(),
    };
    memoria::causal::graph::accumulate_causal_edge(&store, &edge).unwrap();

    // Boost: NOTEARS confirms
    memoria::causal::graph::adjust_edge_confidence(&store, a, b, true).unwrap();
    let edges = memoria::causal::graph::load_all_edges(&store, "").unwrap();
    let boosted = edges[0].confidence;
    assert!(boosted > 0.3, "confidence should increase after boost");

    // Reduce: NOTEARS contradicts
    memoria::causal::graph::adjust_edge_confidence(&store, a, b, false).unwrap();
    let edges = memoria::causal::graph::load_all_edges(&store, "").unwrap();
    let reduced = edges[0].confidence;
    assert!(reduced < boosted, "confidence should decrease after contradiction");
    assert!(reduced > 0.0, "confidence should not go below 0");
}

// ── 7. Load Graph with Confidence Filtering ──

#[test]
fn test_load_graph_filters_by_confidence() {
    let store = CozoStore::open_mem(4).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let c = Uuid::now_v7();
    let d = Uuid::now_v7();

    // High confidence edge
    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &memoria::CausalEdge {
            cause_id: a,
            effect_id: b,
            causal_strength: 0.8,
            observations: 1,
            last_observed: now_ms(),
            mechanism: memoria::CausalMechanism::RecallAttribution,
            confidence: 0.8,
            namespace: String::new(),
        },
    )
    .unwrap();

    // Low confidence edge
    memoria::causal::graph::accumulate_causal_edge(
        &store,
        &memoria::CausalEdge {
            cause_id: c,
            effect_id: d,
            causal_strength: 0.3,
            observations: 1,
            last_observed: now_ms(),
            mechanism: memoria::CausalMechanism::LlmProposed,
            confidence: 0.2,
            namespace: String::new(),
        },
    )
    .unwrap();

    // Load with high confidence threshold — only first edge should appear
    let cg_high = memoria::causal::graph::load_causal_graph(&store, "", 0.5).unwrap();
    assert_eq!(cg_high.edge_count(), 1);

    // Load with low threshold — both edges appear
    let cg_low = memoria::causal::graph::load_causal_graph(&store, "", 0.0).unwrap();
    assert_eq!(cg_low.edge_count(), 2);
}

// ── 8. Mechanism Enum ──

#[test]
fn test_mechanism_variants() {
    use memoria::CausalMechanism;

    for mech in [
        CausalMechanism::RecallAttribution,
        CausalMechanism::LlmProposed,
        CausalMechanism::NotearsDiscovered,
        CausalMechanism::EnterpriseShared,
    ] {
        let s = mech.as_str();
        assert!(!s.is_empty());
        assert_eq!(CausalMechanism::from_str(s), mech);
    }
}
