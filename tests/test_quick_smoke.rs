//! Quick smoke tests — minimal data, real services, strict assertions.
//!
//! Each test uses 1-3 LLM calls max. Validates core dynamics actually work
//! without waiting 20+ minutes for the big integration tests.
//!
//! Run with: cargo test --features local-embeddings --test test_quick_smoke -- --nocapture --test-threads=1

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, test_ctx, TestResultGuard};

// ── Surprise: contradiction detection ────────────────────────────

#[tokio::test]
async fn smoke_surprise_rises_on_contradiction() {
    let mut guard = TestResultGuard::new("smoke_surprise_rises_on_contradiction");
    let m = build_real_memoria();
    let ctx = test_ctx();

    let r1 = m.tell("Acme Corp was founded in 2015", &ctx).await.unwrap();
    println!("tell #1: {} memories, {} entities, surprise={:.4}",
        r1.memory_ids.len(), r1.entity_ids.len(), r1.surprise);

    let surprise_before = m.accumulated_surprise().unwrap();

    let r2 = m.tell("Acme Corp was founded in 2020", &ctx).await.unwrap();
    println!("tell #2 (contradiction): surprise={:.4}", r2.surprise);

    let surprise_after = m.accumulated_surprise().unwrap();
    println!("accumulated: before={:.4} after={:.4}", surprise_before, surprise_after);

    guard.set_details(&format!("surprise: {:.4} → {:.4}", surprise_before, surprise_after));
    assert!(
        surprise_after > surprise_before,
        "surprise must rise on contradiction: {:.4} should be > {:.4}",
        surprise_after, surprise_before
    );
    guard.mark_passed();
}

#[tokio::test]
async fn smoke_tell_returns_nonzero_surprise_on_contradiction() {
    let mut guard = TestResultGuard::new("smoke_tell_returns_nonzero_surprise_on_contradiction");
    let m = build_real_memoria();
    let ctx = test_ctx();

    m.tell("Alice joined Google in 2019", &ctx).await.unwrap();
    let r2 = m.tell("Alice joined Microsoft in 2019", &ctx).await.unwrap();
    println!("contradiction tell surprise: {:.4}", r2.surprise);

    guard.set_details(&format!("surprise={:.4}", r2.surprise));
    assert!(
        r2.surprise > 0.0,
        "contradicting tell must have surprise > 0, got {:.4}",
        r2.surprise
    );
    guard.mark_passed();
}

// ── Compression: manually seed co-activations, then compress ─────
// 3 tells + 1 compress (no asks needed — we manually insert co-activations)

#[tokio::test]
async fn smoke_compression() {
    let mut guard = TestResultGuard::new("smoke_compression");
    let m = build_real_memoria();
    let ctx = test_ctx();

    // Tell 3 related facts
    let mut ids = Vec::new();
    for f in &[
        "Rust uses ownership for memory safety",
        "Rust compiles to native code via LLVM",
        "Rust prevents data races at compile time",
    ] {
        let r = m.tell(f, &ctx).await.unwrap();
        ids.extend(r.memory_ids);
    }
    println!("stored {} memory ids", ids.len());

    // Manually seed co-activations (skip ask() calls to save LLM round-trips)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            m.store().upsert_co_activation(ids[i], ids[j], now).unwrap();
        }
    }
    println!("seeded co-activations for {} pairs", ids.len() * (ids.len() - 1) / 2);

    let results = m.compress("test", 2).await.unwrap();
    println!("compression: {} clusters", results.len());
    for (i, cr) in results.iter().enumerate() {
        println!("  cluster[{}]: {} sources → {}",
            i, cr.source_ids.len(), &cr.summary[..cr.summary.len().min(60)]);
    }

    guard.set_details(&format!("{} clusters", results.len()));
    assert!(
        !results.is_empty(),
        "compression must produce clusters from co-activated memories"
    );
    guard.mark_passed();
}

// ── Skill bootstrap + select ─────────────────────────────────────
// 1 LLM call (bootstrap) + 1 embedding call (select)

#[tokio::test]
async fn smoke_skill_bootstrap() {
    let mut guard = TestResultGuard::new("smoke_skill_bootstrap");
    let m = build_real_memoria();
    let ctx = test_ctx();

    let md = "## Debug API Errors\n1. Check logs\n2. Reproduce with curl\n3. Fix and verify";

    let result = m.bootstrap_skills(md, &ctx).await.unwrap();
    println!("bootstrapped: {} skills", result.skills_created);
    assert!(result.skills_created >= 1, "must bootstrap at least 1 skill");

    // Select must not crash (was UUID parse error before HNSW offset fix)
    let selected = m.select_skills("debug API errors", 1.0, 5).await.unwrap();
    println!("selected: {} skills", selected.len());

    guard.set_details(&format!("created={} selected={}", result.skills_created, selected.len()));
    guard.mark_passed();
}

// ── Skill crystallization ────────────────────────────────────────
// 3 episodes × 1 tell each = 3 LLM calls + 1 crystallize call

#[tokio::test]
async fn smoke_skill_crystallization() {
    let mut guard = TestResultGuard::new("smoke_skill_crystallization");
    let m = build_real_memoria();
    let ctx = test_ctx();

    let tasks = [
        ("Parse CSV data file", "success", "Parsed structured data file, extracted fields, validated types, produced summary statistics"),
        ("Parse JSON config file", "success", "Parsed structured data file, extracted fields, validated schema, produced cleaned output"),
        ("Parse XML settings file", "success", "Parsed structured data file, extracted fields, validated format, produced configuration map"),
        ("Parse YAML deployment file", "success", "Parsed structured data file, extracted fields, validated structure, produced service configs"),
        ("Parse TOML config file", "success", "Parsed structured data file, extracted fields, validated types, produced settings object"),
    ];
    for (task, outcome, summary) in &tasks {
        let ep = m.create_episode(&ctx.agent_id, "task", Default::default()).unwrap();
        m.tell(&format!("Task: {}", task), &ctx).await.unwrap();
        m.close_episode(ep.id, outcome, summary).await.unwrap();
    }

    let before = m.count_skills().unwrap();
    let result = m.crystallize_skills().await.unwrap();
    let after = m.count_skills().unwrap();
    println!("skills: {} → {} (new={}, reinforced={})",
        before, after, result.new_skill_ids.len(), result.reinforced_skill_ids.len());

    guard.set_details(&format!("{} new skills", result.new_skill_ids.len()));
    assert!(
        !result.new_skill_ids.is_empty(),
        "crystallization must produce skills from 5 similar episodes"
    );
    guard.mark_passed();
}

// ── Graph metrics ────────────────────────────────────────────────
// 2 tells + manual edges (graph metrics is pure Datalog, no LLM)

#[tokio::test]
async fn smoke_graph_metrics() {
    let mut guard = TestResultGuard::new("smoke_graph_metrics");
    let m = build_real_memoria();
    let ctx = test_ctx();

    let mut ids = Vec::new();
    for text in &["Node A data", "Node B data"] {
        let r = m.tell(text, &ctx).await.unwrap();
        ids.extend(r.memory_ids);
    }

    if ids.len() >= 2 {
        m.store().insert_edge(ids[0], ids[1], "related", 0.8).unwrap();
        m.store().insert_edge(ids[1], ids[0], "related", 0.8).unwrap();
    }

    let metrics = m.compute_graph_metrics().unwrap();
    println!("graph metrics: {} nodes", metrics.len());

    guard.set_details(&format!("{} nodes", metrics.len()));
    assert!(!metrics.is_empty(), "must compute graph metrics with explicit edges");
    guard.mark_passed();
}

// ── Hebbian weight growth ────────────────────────────────────────
// 2 tells + 1 ask (minimum: need co-retrieved memories)

#[tokio::test]
async fn smoke_hebbian() {
    let mut guard = TestResultGuard::new("smoke_hebbian");
    let m = build_real_memoria();
    let ctx = test_ctx();

    m.tell("Python is great for data science", &ctx).await.unwrap();
    m.tell("Python has numpy and pandas", &ctx).await.unwrap();

    // One ask triggers hebbian learning on co-retrieved memories
    let ask = m.ask("What is Python?", &ctx).await.unwrap();
    let ids: Vec<_> = ask.results.iter().map(|r| r.memory.id).collect();
    println!("ask returned {} results", ids.len());

    if ids.len() >= 2 {
        let weights = m.store()
            .get_association_weights(&ids[..1], &ids[1..])
            .unwrap();
        let max_w = weights.iter().map(|(_, w)| *w).fold(0.0_f64, f64::max);
        println!("max weight: {:.4}", max_w);
        guard.set_details(&format!("max_weight={:.4}", max_w));
        assert!(max_w > 0.0, "weights must be positive after co-retrieval");
    } else {
        guard.set_details("fewer than 2 results — skipped weight check");
    }

    guard.mark_passed();
}

// ── Reflection ───────────────────────────────────────────────────
// 2 tells (1 fact + 1 contradiction) + reflection call

#[tokio::test]
async fn smoke_reflection() {
    let mut guard = TestResultGuard::new("smoke_reflection");
    let m = build_real_memoria();
    let ctx = test_ctx();

    m.tell("Acme Corp was founded in San Francisco", &ctx).await.unwrap();
    m.tell("Acme Corp was founded in New York", &ctx).await.unwrap();

    let surprise = m.accumulated_surprise().unwrap();
    println!("surprise before reflection: {:.4}", surprise);

    let reflection = m.run_reflection().await.unwrap();
    println!("reflection: {:?}", reflection.as_ref().map(|r|
        format!("{} eps, {} abstractions", r.episodes_reviewed.len(), r.abstractions_created)));

    guard.set_details(&format!("surprise={:.4} reflected={}", surprise, reflection.is_some()));
    assert!(surprise > 0.0, "surprise must be > 0 after contradiction");
    guard.mark_passed();
}
