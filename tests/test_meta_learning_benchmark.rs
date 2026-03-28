//! Benchmark: Meta-learning with real multi-session conversation data.
//!
//! Full loop: real NER (GLiNER), real embedder (LM Studio), real LLM (qwen3.5),
//! real QueueWorker processing reflection/compression/skills between batches.
//! Meta-learning parameters control when dynamics fire, and the dynamics
//! actually run and change the store state, closing the feedback loop.
//!
//! Data: 53 sessions from LongMemEval (ICLR 2025).
//!
//! Run with:
//!   cargo test --features local-embeddings --test test_meta_learning_benchmark -- --nocapture
//!
//! Requires:
//!   - LM Studio at http://100.127.90.120:1234/v1
//!   - GLiNER NER at http://localhost:9100

#![cfg(feature = "local-embeddings")]

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{TestLlm, RERANKER, LM_STUDIO_URL, NER_URL, EMBED_MODEL, EMBED_DIM, LLM_MODEL, wait_queue_drain};
use memoria::dynamics::meta_learning::{MetaLearner, MetaPhase};
use memoria::{AgentContext, CozoStore, Memoria, MemoriaConfig, TaskQueue, QueueWorker};
use memoria::pipeline::verifier::RelationVerifier;
use memoria::services::{ApiEmbedder, GlinerSidecar};

#[derive(serde::Deserialize)]
struct LongMemEntry {
    question: String,
    answer: serde_json::Value,
    #[allow(dead_code)]
    question_date: String,
    haystack_sessions: Vec<Vec<Turn>>,
    haystack_dates: Vec<String>,
    #[allow(dead_code)]
    haystack_session_ids: Vec<serde_json::Value>,
    #[allow(dead_code)]
    answer_session_ids: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize, Clone)]
struct Turn {
    role: String,
    content: String,
    #[allow(dead_code)]
    #[serde(default)]
    has_answer: Option<bool>,
}

fn load_longmemeval_entry() -> LongMemEntry {
    let data = include_str!("../tests/fixtures/longmemeval_single.json");
    serde_json::from_str(data).expect("parse longmemeval entry")
}

/// Build Memoria with ALL real services + TaskQueue + QueueWorker.
/// Meta-learning enabled. Worker processes reflection, compression, skills.
fn build_full_stack_memoria() -> (Arc<Memoria>, Arc<TaskQueue>, tokio_util::sync::CancellationToken) {
    let mut config = MemoriaConfig::default();
    config.dynamics.meta_learning_enabled = true;
    config.dynamics.meta_learning_interval = 1;
    config.dynamics.observation_window = 2;
    config.dynamics.bo_budget = 8;
    // Lower compression threshold so it actually triggers during the benchmark
    config.dynamics.compression_memory_threshold = 30;

    let store = CozoStore::open_mem(EMBED_DIM).unwrap();
    let embedder: Arc<dyn memoria::services::Embedder> =
        Arc::new(ApiEmbedder::new(LM_STUDIO_URL, EMBED_MODEL, EMBED_DIM));
    let ner: Arc<dyn memoria::services::NerService> =
        Arc::new(GlinerSidecar::new(NER_URL));
    let reranker = RERANKER.clone();
    let llm: Arc<dyn memoria::services::LlmService> =
        Arc::new(TestLlm::new(LM_STUDIO_URL, LLM_MODEL));

    // TaskQueue gets its own store (queue metadata lives separately)
    // Worker accesses Memoria's store for dynamics operations
    let queue_store = CozoStore::open_mem(4).unwrap();
    let queue = Arc::new(TaskQueue::new(queue_store));

    let mut m = Memoria::new(store, embedder.clone(), ner, reranker, llm.clone(), config);
    m.set_task_queue(Arc::clone(&queue));
    let m = Arc::new(m);

    // Spawn QueueWorker — processes reflection, compression, skills, meta-learning, etc.
    let cancel = tokio_util::sync::CancellationToken::new();
    let verifier = Arc::new(RelationVerifier::new(llm.clone()));
    let worker = QueueWorker::with_services(
        Arc::clone(&queue),
        verifier,
        m.store().clone(),
        embedder,
        llm,
        Duration::from_millis(500),
    );
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        worker.run(cancel_clone).await;
    });

    (m, queue, cancel)
}

/// Format user turns from a session for tell().
/// Only feed user turns — assistant responses are generic LLM output,
/// user turns contain the actual personal information to memorize.
/// Cap total length to avoid overloading the embedder/LLM.
fn format_user_turns(turns: &[Turn], date: &str) -> String {
    let mut text = format!("[Conversation on {}]\n", date);
    let mut total_len = text.len();
    for turn in turns {
        if turn.role == "user" && total_len < 2000 {
            let content = if turn.content.len() > 300 {
                format!("{}...", &turn.content[..300])
            } else {
                turn.content.clone()
            };
            let line = format!("User said: {}\n", content);
            total_len += line.len();
            text.push_str(&line);
        }
    }
    text
}

#[tokio::test]
async fn benchmark_meta_learning_full_loop() {
    let t0 = Instant::now();

    // ── Setup ──
    let entry = load_longmemeval_entry();
    let (m, queue, cancel) = build_full_stack_memoria();
    let ctx = AgentContext::new("benchmark", "longmemeval");
    let store = m.store();

    let answer_string = entry.answer.to_string();
    let expected = entry.answer.as_str().unwrap_or(&answer_string);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Meta-Learning Benchmark — FULL LOOP                   ║");
    println!("║  Real NER + Real Embedder + Real LLM + QueueWorker     ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Q: {}", entry.question);
    println!("║  A: {expected}");
    println!("║  Sessions: {}", entry.haystack_sessions.len());
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Initialize meta-learner ──
    let mut learner = MetaLearner::new(2, 8);
    let initial_params: Vec<(String, f64)> = learner.params.iter()
        .map(|p| (p.name.clone(), p.current))
        .collect();

    // ── Feed sessions, let dynamics run, meta-learn ──
    let batch_size = 2;
    let max_sessions = 10;
    let sessions = &entry.haystack_sessions[..max_sessions.min(entry.haystack_sessions.len())];
    let dates = &entry.haystack_dates[..max_sessions.min(entry.haystack_dates.len())];
    let total_batches = (sessions.len() + batch_size - 1) / batch_size;

    let mut fe_history: Vec<f64> = Vec::new();
    let mut beta_history: Vec<f64> = Vec::new();
    let mut fact_counts: Vec<usize> = Vec::new();
    let mut memory_counts: Vec<usize> = Vec::new();

    for batch_idx in 0..total_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(sessions.len());
        let batch_t0 = Instant::now();

        // 1. Feed sessions via tell() — real embeddings + real NER extract entities/facts
        let mut batch_memories = 0;
        for i in start..end {
            let text = format_user_turns(&sessions[i], &dates[i]);
            match m.tell(&text, &ctx).await {
                Ok(result) => {
                    batch_memories += result.memory_ids.len();
                }
                Err(e) => {
                    eprintln!("  tell() error session {i}: {e}");
                }
            }
        }

        // 2. Run tick() — checks thresholds using META-LEARNED params, enqueues tasks
        let _ = m.tick();

        // 3. Let the QueueWorker process what it can in a fixed window.
        //    Don't block forever — some tasks (reflection) call the LLM and can be slow.
        //    We give it 15s per batch. Unfinished tasks carry over to next batch.
        tokio::time::sleep(Duration::from_secs(10)).await;
        // Drain any quick remaining tasks
        let drain_deadline = Instant::now() + Duration::from_secs(5);
        while queue.count_pending().unwrap_or(0) > 0 && Instant::now() < drain_deadline {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // 4. Measure free energy AFTER dynamics have run
        let fe = memoria::aif::compute_bethe_free_energy(store).unwrap();
        let surprise = memoria::dynamics::surprise::accumulated_unresolved_surprise(store)
            .unwrap_or(0.0);

        // Count facts (extracted by real NER + stored by tell pipeline)
        let fact_count = store.run_query(
            "?[count(id)] := *facts{id, @ 'NOW'}",
            std::collections::BTreeMap::new(),
        ).map(|r| r.rows.first()
            .and_then(|row| row[0].get_int())
            .unwrap_or(0) as usize
        ).unwrap_or(0);

        fe_history.push(fe.free_energy);
        beta_history.push(fe.beta);
        fact_counts.push(fact_count);
        memory_counts.push(store.count_memories().unwrap_or(0));

        // 5. Meta-learning step — observes free energy that was SHAPED by dynamics
        let meta_result = learner.step(store, fe.free_energy, fe.beta, surprise).unwrap();

        let batch_ms = batch_t0.elapsed().as_millis();
        let phase_char = match learner.phase {
            MetaPhase::ColdStart => "C",
            MetaPhase::OnlineTracking => "S",
        };
        let adj_str = if let Some(ref r) = meta_result {
            format!("{} adj", r.adjustments.len())
        } else {
            "wait".into()
        };

        println!(
            "  Batch {:2}/{} | sess {:2}-{:2} | +{:2} mem ({:3} total) | {:3} facts | FE={:>8.4} β={:.4} surp={:.2} | [{phase_char}] gen={:2} {adj_str} | {}ms",
            batch_idx, total_batches,
            start, end - 1,
            batch_memories,
            memory_counts.last().unwrap(),
            fact_count,
            fe.free_energy, fe.beta, surprise,
            learner.generation,
            batch_ms,
        );
    }

    // ── Recall test ──
    println!();
    println!("─── Recall Test ───────────────────────────────────────────");
    println!("  Q: {}", entry.question);

    let ask_result = m.ask(&entry.question, &ctx).await.unwrap();

    let mut found_evidence = false;
    for (i, scored) in ask_result.results.iter().take(5).enumerate() {
        let preview: String = scored.memory.content.chars().take(120).collect();
        let contains_answer = scored.memory.content.to_lowercase()
            .contains(&expected.to_lowercase());
        let marker = if contains_answer { " ← FOUND" } else { "" };
        if contains_answer { found_evidence = true; }
        println!("  #{}: score={:.4} conf={:.2} | {}...{marker}",
            i + 1, scored.score, scored.confidence, preview);
    }
    println!("  Expected: {expected}");
    println!("  Retrieved: {}", if found_evidence { "YES" } else { "NO" });

    // ── Stop worker ──
    cancel.cancel();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // ── Results ──
    println!();
    println!("═══ META-LEARNING RESULTS ═══════════════════════════════");

    let final_params: Vec<(String, f64)> = learner.params.iter()
        .map(|p| (p.name.clone(), p.current))
        .collect();

    println!();
    println!("  Parameter Adjustments:");
    let mut moved = 0;
    for (i, (name, initial)) in initial_params.iter().enumerate() {
        let (_, final_val) = &final_params[i];
        let delta = final_val - initial;
        let pct = if initial.abs() > 1e-10 { (delta / initial) * 100.0 } else { 0.0 };
        if delta.abs() > 1e-10 {
            moved += 1;
            let dir = if delta > 0.0 { "↑" } else { "↓" };
            println!("    {name:35} {initial:>14.4} → {final_val:>14.4}  {dir} {pct:+.1}%");
        } else {
            println!("    {name:35} {initial:>14.4}    (unchanged)");
        }
    }
    println!("  Moved: {moved}/{}", initial_params.len());

    println!();
    println!("  Free Energy Trajectory:");
    for (i, (fe, beta)) in fe_history.iter().zip(beta_history.iter()).enumerate() {
        let facts = fact_counts.get(i).copied().unwrap_or(0);
        let mems = memory_counts.get(i).copied().unwrap_or(0);
        let bar_len = ((fe.abs() / 30.0) * 40.0).min(40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    Batch {:2}: FE={:>8.4} β={:.4} | {facts:3}f {mems:3}m | {bar}",
            i, fe, beta);
    }

    let fe_first = *fe_history.first().unwrap_or(&0.0);
    let fe_last = *fe_history.last().unwrap_or(&0.0);
    println!();
    println!("  Summary:");
    println!("    FE:       {fe_first:.4} → {fe_last:.4}");
    println!("    β:        {:.4} → {:.4}", beta_history.first().unwrap_or(&0.0), beta_history.last().unwrap_or(&0.0));
    println!("    Facts:    {} → {}", fact_counts.first().unwrap_or(&0), fact_counts.last().unwrap_or(&0));
    println!("    Memories: {} → {}", memory_counts.first().unwrap_or(&0), memory_counts.last().unwrap_or(&0));
    println!("    Phase:    {:?} (gen {})", learner.phase, learner.generation);
    println!("    Time:     {:.1}s", t0.elapsed().as_secs_f64());

    // ── Assertions ──
    assert!(moved > 0, "Meta-learner should adjust parameters");
    assert!(*memory_counts.last().unwrap_or(&0) > 0, "Should store memories");
    assert!(*fact_counts.last().unwrap_or(&0) > 0, "Real NER should extract facts");
    // Free energy should be non-zero with real facts
    assert!(fe_last.abs() > 0.01, "Free energy should be non-trivial with real data");

    println!();
    println!("═══ BENCHMARK COMPLETE ═══════════════════════════════════");
}
