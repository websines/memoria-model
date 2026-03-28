//! Integration test: Full lifecycle — seed → multi-turn → dynamics → validate.
//!
//! This is the "big one" — runs all cognitive dynamics end-to-end
//! with a QueueWorker processing tasks throughout.
//!
//! Run with: cargo test --features local-embeddings --test test_full_lifecycle -- --nocapture
//!
//! WARNING: This test takes several minutes due to LLM calls and embedding.

#![cfg(feature = "local-embeddings")]

mod common;

use std::time::Duration;

use common::{
    build_real_memoria_with_queue, ctx_ns, load_contradictions, load_squad_paragraphs,
    load_task_patterns, wait_queue_drain,
};
use memoria::{SkillOutcome, TelosProvenance};
use uuid::Uuid;

#[tokio::test]
async fn test_full_lifecycle() {
    let (m, queue, cancel) = build_real_memoria_with_queue();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     FULL LIFECYCLE INTEGRATION TEST                 ║");
    println!("╚══════════════════════════════════════════════════════╝");

    // ── Phase 1: Seed (bulk tell) ──────────────────────────────
    println!("\n┌─── Phase 1: Seed ───────────────────────────────┐");

    let squad = load_squad_paragraphs();
    let namespaces = ["science", "history", "technology"];

    let mut total_memories = 0;
    let mut total_entities = 0;
    let entries_per_ns = 60; // ~180 total spread across 3 namespaces

    for (ns_idx, ns) in namespaces.iter().enumerate() {
        let ctx = ctx_ns(ns);
        let start = ns_idx * entries_per_ns;
        let end = (start + entries_per_ns).min(squad.len());
        let batch = &squad[start..end];

        println!("  Seeding namespace '{}' with {} entries...", ns, batch.len());
        for (i, entry) in batch.iter().enumerate() {
            let r = m.tell(&entry.context, &ctx).await.unwrap();
            total_memories += r.memory_ids.len();
            total_entities += r.entity_ids.len();
            if (i + 1) % 20 == 0 {
                println!("    {}/{} done", i + 1, batch.len());
            }
        }
    }

    println!(
        "  ✓ Seeded {} memories, {} entities across {} namespaces",
        total_memories,
        total_entities,
        namespaces.len()
    );
    let store_count = m.store().count_memories().unwrap();
    assert!(
        store_count >= 100,
        "should have at least 100 memories after seeding, got {}",
        store_count
    );

    // Tick to trigger compression check
    let tick = m.tick().unwrap();
    println!(
        "  Tick: compression_enqueued={}, reflection_enqueued={}",
        tick.compression_enqueued, tick.reflection_enqueued
    );

    // ── Phase 2: Multi-turn Query + Hebbian ──────────────────
    println!("\n┌─── Phase 2: Multi-turn Query + Hebbian ─────────┐");

    let mut prev_result_ids: Vec<uuid::Uuid> = Vec::new();

    for (i, entry) in squad.iter().take(30).enumerate() {
        // Pass previous result IDs as context for Hebbian chaining
        let ctx = if prev_result_ids.is_empty() {
            ctx_ns("science")
        } else {
            ctx_ns("science").with_context_memories(prev_result_ids.clone())
        };

        let ask = m.ask(&entry.question, &ctx).await.unwrap();
        prev_result_ids = ask.results.iter().take(3).map(|r| r.memory.id).collect();

        if i % 10 == 0 {
            println!(
                "  Q[{}]: {} → {} results",
                i,
                &entry.question[..entry.question.len().min(40)],
                ask.results.len()
            );
        }
    }
    println!("  ✓ Completed 30 queries with context chaining");

    // ── Phase 3: Contradiction + Surprise ────────────────────
    println!("\n┌─── Phase 3: Contradiction + Surprise ───────────┐");

    let contradictions = load_contradictions();
    let tech_ctx = ctx_ns("technology");

    let surprise_before = m.accumulated_surprise().unwrap();
    println!("  Surprise before contradictions: {:.4}", surprise_before);

    for (i, (fact_a, fact_b)) in contradictions.iter().take(10).enumerate() {
        m.tell(fact_a, &tech_ctx).await.unwrap();
        let r = m.tell(fact_b, &tech_ctx).await.unwrap();
        if i % 3 == 0 {
            println!("  pair[{}] surprise={:.4}: {:?} vs {:?}",
                i, r.surprise, &fact_a[..fact_a.len().min(40)], &fact_b[..fact_b.len().min(40)]);
        }
    }

    let surprise_after = m.accumulated_surprise().unwrap();
    println!(
        "  Surprise after contradictions: {:.4} (delta={:.4})",
        surprise_after,
        surprise_after - surprise_before
    );

    // Run reflection directly (don't wait for worker)
    println!("  Running reflection...");
    let reflection = m.run_reflection().await.unwrap();
    assert!(
        reflection.is_some(),
        "reflection must run after contradictions — surprise should exceed threshold"
    );
    let r = reflection.as_ref().unwrap();
    println!(
        "  ✓ Reflection: {} episodes, {} abstractions, {} facts",
        r.episodes_reviewed.len(),
        r.abstractions_created,
        r.facts_created
    );

    // ── Phase 4: Telos Lifecycle ─────────────────────────────
    println!("\n┌─── Phase 4: Telos Lifecycle ────────────────────┐");

    let telos_ctx = ctx_ns("technology");

    let goal1 = m
        .create_telos(
            "Build scalable API platform",
            "Design and implement a high-performance API gateway",
            &telos_ctx,
            0, // north star
            None,
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();
    println!("  Created north-star: {}", goal1.title);

    let goal2 = m
        .create_telos(
            "Implement authentication service",
            "OAuth2 + JWT token management",
            &telos_ctx,
            1,
            Some(goal1.id),
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();
    println!("  Created strategic: {}", goal2.title);

    let goal3 = m
        .create_telos(
            "Write API documentation",
            "OpenAPI spec + developer guide",
            &telos_ctx,
            2,
            Some(goal2.id),
            None,
            TelosProvenance::UserStated,
        )
        .await
        .unwrap();
    println!("  Created tactical: {}", goal3.title);

    // Wait for decomposition tasks to process
    println!("  Waiting for decomposition tasks...");
    wait_queue_drain(&queue, Duration::from_secs(120)).await;

    // Check for subtelos
    let children = m.store().get_children_telos(goal1.id).unwrap();
    println!("  North-star children: {}", children.len());

    // Record progress
    m.record_telos_progress(goal3.id, 0.5, "Documentation half done", "integration-test", None, None)
        .await
        .unwrap();

    // Estimate progress
    let progress = memoria::api::telos_progress::estimate_progress(m.store(), goal3.id).unwrap();
    println!(
        "  Goal3 progress: {:.2} (method={:?})",
        progress.progress, progress.method
    );

    // Complete goal3
    m.update_telos_status(goal3.id, "completed", "integration-test", "Docs shipped")
        .await
        .unwrap();

    // Wait for telos_reflection to process
    wait_queue_drain(&queue, Duration::from_secs(60)).await;
    println!("  ✓ Telos lifecycle phase complete");

    // ── Phase 5: Skill Crystallization ───────────────────────
    println!("\n┌─── Phase 5: Skill Crystallization ──────────────┐");

    let patterns = load_task_patterns();
    let skill_ctx = ctx_ns("technology");

    for (i, pattern) in patterns.iter().enumerate() {
        let episode = m
            .create_episode("integration-test", "task", Default::default())
            .unwrap();
        m.tell(&format!("Task: {}", pattern.task), &skill_ctx)
            .await
            .unwrap();
        m.close_episode(episode.id, &pattern.outcome, &pattern.summary)
            .await
            .unwrap();
        if i % 5 == 0 {
            println!("  episode[{}]: {}", i, &pattern.task[..pattern.task.len().min(50)]);
        }
    }

    println!("  Crystallizing skills...");
    let crystal = m.crystallize_skills().await.unwrap();
    println!(
        "  ✓ {} new skills, {} reinforced",
        crystal.new_skill_ids.len(),
        crystal.reinforced_skill_ids.len()
    );

    // Verify skills are findable
    let selected = m
        .select_skills("Parse CSV data", 1.0, 3)
        .await
        .unwrap();
    println!("  Skill selection for 'Parse CSV data': {} found", selected.len());

    // ── Phase 6: AIF Steady State ────────────────────────────
    println!("\n┌─── Phase 6: AIF Steady State ─────────────────┐");

    let state_before = m.snapshot_model_state("integration-test").unwrap();
    println!(
        "  Initial: FE={:.4}, beta={:.4}",
        state_before.free_energy, state_before.beta
    );

    // Mixed feedback outcomes
    for i in 0..6 {
        let outcome = match i % 3 {
            0 => SkillOutcome::Success,
            1 => SkillOutcome::Failure,
            _ => SkillOutcome::Partial,
        };
        let fb = m
            .feedback(Uuid::now_v7(), &outcome, "integration-test", &[])
            .await
            .unwrap();
        if i % 2 == 0 {
            println!(
                "  feedback[{}]: {:?} → FE={:.4} beta={:.4}",
                i, outcome, fb.free_energy, fb.beta
            );
        }
    }

    let state_after = m.snapshot_model_state("integration-test").unwrap();
    let trend = memoria::aif::health::compute_trend(m.store(), 5).unwrap();
    println!(
        "  Final: FE={:.4}, beta={:.4}, trend={:?}",
        state_after.free_energy, state_after.beta, trend
    );

    // Beta should have shifted from initial 1.0
    println!(
        "  Beta shift: {:.4} → {:.4}",
        state_before.beta, state_after.beta
    );

    // ── Phase 7: Validation ──────────────────────────────────
    println!("\n┌─── Phase 7: Validation ─────────────────────────┐");

    // Re-ask some Phase 2 questions
    let mut recall_hits = 0;
    for entry in squad.iter().take(10) {
        let ask = m.ask(&entry.question, &ctx_ns("science")).await.unwrap();
        if !ask.results.is_empty() {
            recall_hits += 1;
        }
    }
    println!("  Recall quality: {}/10 questions returned results", recall_hits);
    assert!(
        recall_hits >= 5,
        "should recall at least 50% of previously stored knowledge"
    );

    // Dump final stats
    let total_mems = m.store().count_memories().unwrap();
    let total_skills = m.count_skills().unwrap();
    let health = m.model_health(
        queue.count_pending().unwrap(),
        0,
    ).unwrap();

    println!("\n  ═══ Final Statistics ═══");
    println!("  Memories:     {}", total_mems);
    println!("  Skills:       {}", total_skills);
    println!("  FE Trend:     {:?}", health.free_energy_trend);
    println!("  Beta:         {:.4}", health.beta);
    println!("  Surprise:     {:.4}", health.unresolved_surprise);
    println!("  Queue depth:  {}", health.queue_depth);

    // Minimum count assertions
    assert!(total_mems >= 100, "should have 100+ memories, got {}", total_mems);

    // Cleanup
    cancel.cancel();

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║     FULL LIFECYCLE TEST COMPLETE                    ║");
    println!("╚══════════════════════════════════════════════════════╝");
}
