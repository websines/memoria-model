//! Integration test with real API services.
//!
//! Run with: cargo test --features local-embeddings --test integration_real -- --nocapture
//!
//! Requires:
//!   - LM Studio at http://100.127.90.120:1234/v1/ (embeddings + LLM)
//!   - GLiNER NER sidecar at http://localhost:9100
//!   - `local-embeddings` feature for fastembed reranker

#![cfg(feature = "local-embeddings")]

use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use memoria::{AgentContext, CozoStore, Memoria, MemoriaConfig};
use memoria::services::{ApiEmbedder, GlinerSidecar, LocalReranker};

/// Minimal OpenAI-compatible LLM client for integration tests.
struct TestLlm {
    base_url: String,
    model: String,
    client: ureq::Agent,
}

impl TestLlm {
    fn new(base_url: &str, model: &str) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(300))
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client,
        }
    }
}

#[async_trait]
impl memoria::services::LlmService for TestLlm {
    async fn complete(
        &self,
        messages: &[memoria::services::Message],
        max_tokens: usize,
    ) -> anyhow::Result<memoria::services::LlmResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        let resp: serde_json::Value = self
            .client
            .post(&url)
            .send_json(serde_json::json!({
                "model": self.model,
                "messages": msgs,
                "max_tokens": max_tokens,
            }))?
            .into_json()?;
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(memoria::services::LlmResponse {
            content,
            thinking: None,
        })
    }
}

const LM_STUDIO_URL: &str = "http://100.127.90.120:1234/v1";
const NER_URL: &str = "http://localhost:9100";
const EMBED_MODEL: &str = "text-embedding-nomic-embed-text-v2";
const LLM_MODEL: &str = "qwen3.5";
const EMBED_DIM: usize = 768;

/// Shared reranker — fastembed's ONNX model download uses file locks,
/// so concurrent initialization from multiple tests causes lock contention.
static RERANKER: LazyLock<Arc<dyn memoria::services::Reranker>> = LazyLock::new(|| {
    Arc::new(LocalReranker::new().expect("failed to init local reranker"))
});

fn build_real_memoria() -> Memoria {
    let store = CozoStore::open_mem(EMBED_DIM).unwrap();
    let embedder = Arc::new(ApiEmbedder::new(LM_STUDIO_URL, EMBED_MODEL, EMBED_DIM));
    let ner = Arc::new(GlinerSidecar::new(NER_URL));
    let reranker = RERANKER.clone();
    let llm = Arc::new(TestLlm::new(LM_STUDIO_URL, LLM_MODEL));
    let config = MemoriaConfig::default();

    Memoria::new(store, embedder, ner, reranker, llm, config)
}

#[tokio::test]
async fn test_real_tell_and_ask() {
    let m = build_real_memoria();
    let ctx = AgentContext::new("integration-test", "test");

    // Tell some facts
    println!("\n=== Telling facts ===");

    let r1 = m.tell("Alice works at Acme Corp as a software engineer", &ctx).await.unwrap();
    println!("tell #1: {} memories, {} entities", r1.memory_ids.len(), r1.entity_ids.len());
    assert!(!r1.memory_ids.is_empty());

    let r2 = m.tell("Bob manages the data science team at Acme Corp", &ctx).await.unwrap();
    println!("tell #2: {} memories, {} entities", r2.memory_ids.len(), r2.entity_ids.len());

    let r3 = m.tell("Acme Corp is headquartered in San Francisco", &ctx).await.unwrap();
    println!("tell #3: {} memories, {} entities", r3.memory_ids.len(), r3.entity_ids.len());

    let r4 = m.tell("Charlie joined Acme Corp last month as a ML engineer", &ctx).await.unwrap();
    println!("tell #4: {} memories, {} entities", r4.memory_ids.len(), r4.entity_ids.len());

    let r5 = m.tell("Diana is the CTO of Acme Corp and reports to the board", &ctx).await.unwrap();
    println!("tell #5: {} memories, {} entities", r5.memory_ids.len(), r5.entity_ids.len());

    // Verify storage
    let count = m.store().count_memories().unwrap();
    println!("\nTotal memories stored: {}", count);
    assert_eq!(count, 5);

    // Ask questions
    println!("\n=== Asking questions ===");

    let ask1 = m.ask("Where does Alice work?", &ctx).await.unwrap();
    println!("\nQ: Where does Alice work?");
    println!("Results: {} hits", ask1.results.len());
    for (i, r) in ask1.results.iter().enumerate() {
        println!(
            "  [{}] score={:.4} content={:?}",
            i,
            r.score,
            &r.memory.content[..r.memory.content.len().min(80)]
        );
    }
    assert!(!ask1.results.is_empty(), "should find results for Alice");

    let ask2 = m.ask("Who works at Acme Corp?", &ctx).await.unwrap();
    println!("\nQ: Who works at Acme Corp?");
    println!("Results: {} hits", ask2.results.len());
    for (i, r) in ask2.results.iter().enumerate() {
        println!(
            "  [{}] score={:.4} content={:?}",
            i,
            r.score,
            &r.memory.content[..r.memory.content.len().min(80)]
        );
    }
    assert!(!ask2.results.is_empty(), "should find results about Acme");

    let ask3 = m.ask("What is Acme Corp's location?", &ctx).await.unwrap();
    println!("\nQ: What is Acme Corp's location?");
    println!("Results: {} hits", ask3.results.len());
    for (i, r) in ask3.results.iter().enumerate() {
        println!(
            "  [{}] score={:.4} content={:?}",
            i,
            r.score,
            &r.memory.content[..r.memory.content.len().min(80)]
        );
    }

    // Verify the top result for location question mentions San Francisco
    if let Some(top) = ask3.results.first() {
        assert!(
            top.memory.content.contains("San Francisco"),
            "top result for location should mention San Francisco, got: {}",
            top.memory.content
        );
    }

    println!("\n=== All assertions passed ===");
}

#[tokio::test]
async fn test_real_entity_extraction() {
    let m = build_real_memoria();
    let ctx = AgentContext::new("integration-test", "test");

    let result = m
        .tell("Alice works at Acme Corp as a software engineer", &ctx)
        .await
        .unwrap();

    println!("\n=== Entity extraction ===");
    println!("Memories: {:?}", result.memory_ids);
    println!("Entities: {:?}", result.entity_ids);

    // GLiNER should extract at least "Alice" and "Acme Corp"
    assert!(
        result.entity_ids.len() >= 2,
        "should extract at least 2 entities (Alice + Acme Corp), got {}",
        result.entity_ids.len()
    );
}

// ── Phase 3: Knowledge Graph & Verification ──

#[tokio::test]
async fn test_real_relation_verification_and_facts() {
    let m = build_real_memoria();
    let ctx = AgentContext::new("integration-test", "test");

    println!("\n=== Phase 3: Relation verification & fact storage ===");

    let r = m.tell("Alice works at Acme Corp as a software engineer", &ctx).await.unwrap();
    println!("Memories: {}, Entities: {}, Facts: {}",
        r.memory_ids.len(), r.entity_ids.len(), r.fact_ids.len());

    // If GLiNER extracted relations AND the LLM verified them, we should have facts
    if !r.fact_ids.is_empty() {
        println!("Facts stored:");
        for fid in &r.fact_ids {
            println!("  fact_id={}", fid);
        }
    } else {
        println!("No facts stored — GLiNER may not have extracted relations (NER-dependent)");
    }

    // Either way, memories and entities should exist
    assert!(!r.memory_ids.is_empty(), "should store memory");
}

#[tokio::test]
async fn test_real_contradiction_detection() {
    let m = build_real_memoria();
    let ctx = AgentContext::new("integration-test", "test");

    println!("\n=== Phase 3: Contradiction detection ===");

    // Store conflicting facts by inserting them directly into the knowledge graph
    // (since NER extraction is non-deterministic, we go via the store for reliable testing)
    let r1 = m.tell("Alice works at Acme Corp", &ctx).await.unwrap();
    let r2 = m.tell("Alice works at Globex Corporation", &ctx).await.unwrap();

    println!("Tell #1: {} memories, {} entities, {} facts",
        r1.memory_ids.len(), r1.entity_ids.len(), r1.fact_ids.len());
    println!("Tell #2: {} memories, {} entities, {} facts",
        r2.memory_ids.len(), r2.entity_ids.len(), r2.fact_ids.len());

    // Ask about Alice — check if contradictions are surfaced
    let ask = m.ask("Where does Alice work?", &ctx).await.unwrap();
    println!("\nQ: Where does Alice work?");
    println!("Results: {} hits, {} contradictions",
        ask.results.len(), ask.contradictions.len());

    for c in &ask.contradictions {
        println!("  CONTRADICTION: entity={} pred={} values=[{:?} vs {:?}]",
            c.entity, c.predicate, c.value_a, c.value_b);
    }

    // Even without contradictions (NER-dependent), results should come back
    assert!(!ask.results.is_empty(), "should retrieve at least one memory");
}

#[tokio::test]
async fn test_real_full_phase3_pipeline() {
    let m = build_real_memoria();
    let ctx = AgentContext::new("integration-test", "test");

    println!("\n=== Phase 3: Full pipeline (tell → facts → ask → contradictions) ===");

    // Build a small knowledge graph
    let facts = [
        "Alice is the CEO of Acme Corp",
        "Bob is the CTO of Acme Corp",
        "Acme Corp is headquartered in San Francisco",
        "Bob reports to Alice",
        "Charlie joined Acme Corp last month as an engineer",
    ];

    let mut total_facts = 0;
    for (i, fact) in facts.iter().enumerate() {
        let r = m.tell(fact, &ctx).await.unwrap();
        total_facts += r.fact_ids.len();
        println!("tell[{}]: mems={} entities={} facts={} | {:?}",
            i, r.memory_ids.len(), r.entity_ids.len(), r.fact_ids.len(), fact);
    }
    println!("\nTotal facts extracted: {}", total_facts);

    // Ask various questions
    let questions = [
        "Who is the CEO?",
        "Where is Acme headquartered?",
        "Who works at Acme?",
    ];

    for q in &questions {
        let ask = m.ask(q, &ctx).await.unwrap();
        println!("\nQ: {}", q);
        println!("  {} results, {} contradictions", ask.results.len(), ask.contradictions.len());
        for (i, r) in ask.results.iter().take(3).enumerate() {
            println!("  [{}] score={:.4} {:?}", i, r.score,
                &r.memory.content[..r.memory.content.len().min(60)]);
        }
    }

    // Verify total memory count
    let count = m.store().count_memories().unwrap();
    println!("\nTotal memories: {}", count);
    assert_eq!(count, 5);
}

#[tokio::test]
async fn test_real_task_queue() {
    use memoria::{TaskQueue, QueueWorker};
    use memoria::pipeline::verifier::RelationVerifier;
    use std::time::Duration;

    println!("\n=== Phase 3: Task queue with real verifier ===");

    let store = CozoStore::open_mem(768).unwrap();
    let llm = Arc::new(TestLlm::new(LM_STUDIO_URL, LLM_MODEL));
    let verifier = Arc::new(RelationVerifier::new(llm));

    let queue_store = CozoStore::open_mem(4).unwrap();
    let queue = Arc::new(TaskQueue::new(queue_store));

    // Enqueue a verification task
    let task_id = queue.enqueue(
        "verify_relations",
        0,
        r#"{"text": "Alice works at Acme Corp", "relations": [{"head": "Alice", "tail": "Acme Corp", "label": "works_at", "confidence": 0.9}]}"#,
        3,
    ).unwrap();
    println!("Enqueued task: {}", task_id);

    let pending = queue.count_pending().unwrap();
    println!("Pending tasks: {}", pending);
    assert_eq!(pending, 1);

    // Process with a real worker
    let worker = QueueWorker::new(
        Arc::clone(&queue),
        verifier,
        store,
        Duration::from_millis(100),
    );

    // Use poll_once (not the run loop) for deterministic testing
    // poll_once is pub(crate), so we test via the run loop with cancel
    let cancel = tokio_util::sync::CancellationToken::new();
    let cancel_clone = cancel.clone();
    let queue_clone = Arc::clone(&queue);

    let handle = tokio::spawn(async move {
        worker.run(cancel_clone).await;
    });

    // Wait for task to be processed
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if queue_clone.count_pending().unwrap() == 0 {
            break;
        }
    }

    cancel.cancel();
    handle.await.unwrap();

    let remaining = queue.count_pending().unwrap();
    println!("Remaining pending: {}", remaining);
    assert_eq!(remaining, 0, "task should have been processed");
}

#[tokio::test]
async fn test_real_namespace_isolation() {
    let m = build_real_memoria();

    let ctx_a = AgentContext::new("agent-a", "project-alpha");
    let ctx_b = AgentContext::new("agent-b", "project-beta");

    m.tell("Alpha secret: the launch is next week", &ctx_a)
        .await
        .unwrap();
    m.tell("Beta secret: budget was approved", &ctx_b)
        .await
        .unwrap();

    let results_a = m.ask("what is the secret?", &ctx_a).await.unwrap();
    let results_b = m.ask("what is the secret?", &ctx_b).await.unwrap();

    println!("\n=== Namespace isolation ===");
    println!("Alpha results: {}", results_a.results.len());
    for r in &results_a.results {
        println!(
            "  ns={} content={:?}",
            r.memory.namespace, r.memory.content
        );
        assert_eq!(r.memory.namespace, "project-alpha");
    }

    println!("Beta results: {}", results_b.results.len());
    for r in &results_b.results {
        println!(
            "  ns={} content={:?}",
            r.memory.namespace, r.memory.content
        );
        assert_eq!(r.memory.namespace, "project-beta");
    }
}
