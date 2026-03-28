//! Shared test helpers for Memoria integration tests.
//!
//! Provides constants, builders, fixture loaders, and assertion helpers
//! used across all `test_*.rs` integration tests.

#![allow(dead_code)]

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use async_trait::async_trait;
use memoria::services::{ApiEmbedder, GlinerSidecar, LocalReranker};
use memoria::{
    AgentContext, CozoStore, Memoria, MemoriaConfig, QueueWorker, TaskQueue,
};
use memoria::pipeline::verifier::RelationVerifier;

// ── Service endpoints (from docs/internal/endpoints.txt) ──

pub const LM_STUDIO_URL: &str = "http://100.127.90.120:1234/v1";
pub const NER_URL: &str = "http://localhost:9100";
pub const EMBED_MODEL: &str = "text-embedding-nomic-embed-text-v2";
pub const LLM_MODEL: &str = "qwen3.5";
pub const EMBED_DIM: usize = 768;

// ── Test LLM (simple OpenAI-compatible client for tests) ──

/// Minimal OpenAI-compatible LLM client for integration tests.
/// Replaces the deleted `ApiLlm` — uses ureq (blocking) inside async.
pub struct TestLlm {
    base_url: String,
    model: String,
    client: ureq::Agent,
}

impl TestLlm {
    pub fn new(base_url: &str, model: &str) -> Self {
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

// ── Shared reranker (ONNX uses file locks — one instance per process) ──

pub static RERANKER: LazyLock<Arc<dyn memoria::services::Reranker>> = LazyLock::new(|| {
    Arc::new(LocalReranker::new().expect("failed to init local reranker"))
});

// ── Builder functions ──

/// Build a Memoria with real services (in-memory store, no task queue).
pub fn build_real_memoria() -> Memoria {
    let store = CozoStore::open_mem(EMBED_DIM).unwrap();
    let embedder = Arc::new(ApiEmbedder::new(LM_STUDIO_URL, EMBED_MODEL, EMBED_DIM));
    let ner = Arc::new(GlinerSidecar::new(NER_URL));
    let reranker = RERANKER.clone();
    let llm = Arc::new(TestLlm::new(LM_STUDIO_URL, LLM_MODEL));
    let config = MemoriaConfig::default();

    Memoria::new(store, embedder, ner, reranker, llm, config)
}

/// Build Memoria with a wired TaskQueue + spawned QueueWorker.
///
/// Returns `(Arc<Memoria>, Arc<TaskQueue>, CancellationToken)`.
/// Cancel the token to stop the worker.
pub fn build_real_memoria_with_queue() -> (Arc<Memoria>, Arc<TaskQueue>, CancellationToken) {
    let store = CozoStore::open_mem(EMBED_DIM).unwrap();
    let embedder: Arc<dyn memoria::services::Embedder> =
        Arc::new(ApiEmbedder::new(LM_STUDIO_URL, EMBED_MODEL, EMBED_DIM));
    let ner = Arc::new(GlinerSidecar::new(NER_URL));
    let reranker = RERANKER.clone();
    let llm: Arc<dyn memoria::services::LlmService> =
        Arc::new(TestLlm::new(LM_STUDIO_URL, LLM_MODEL));
    let config = MemoriaConfig::default();

    // Queue shares the same store
    let queue = Arc::new(TaskQueue::new(CozoStore::open_mem(4).unwrap()));

    let mut m = Memoria::new(store, embedder.clone(), ner, reranker, llm.clone(), config);
    m.set_task_queue(Arc::clone(&queue));
    let m = Arc::new(m);

    // Spawn worker
    let cancel = CancellationToken::new();
    let verifier = Arc::new(RelationVerifier::new(llm.clone()));
    let worker = QueueWorker::with_services(
        Arc::clone(&queue),
        verifier,
        m.store().clone(),
        embedder,
        llm,
        Duration::from_millis(200),
    );
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        worker.run(cancel_clone).await;
    });

    (m, queue, cancel)
}

/// Default test context.
pub fn test_ctx() -> AgentContext {
    AgentContext::new("integration-test", "test")
}

/// Create a context for a specific namespace.
pub fn ctx_ns(namespace: &str) -> AgentContext {
    AgentContext::new("integration-test", namespace)
}

// ── Fixture types ──

#[derive(Debug, Clone, Deserialize)]
pub struct SquadEntry {
    pub context: String,
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TaskPattern {
    pub task: String,
    pub outcome: String,
    pub summary: String,
}

// ── Fixture loaders ──

pub fn load_squad_paragraphs() -> Vec<SquadEntry> {
    let data = include_str!("../fixtures/squad_paragraphs.json");
    serde_json::from_str(data).expect("failed to parse squad_paragraphs.json")
}

pub fn load_contradictions() -> Vec<(String, String)> {
    let data = include_str!("../fixtures/contradictions.json");
    serde_json::from_str(data).expect("failed to parse contradictions.json")
}

pub fn load_task_patterns() -> Vec<TaskPattern> {
    let data = include_str!("../fixtures/task_patterns.json");
    serde_json::from_str(data).expect("failed to parse task_patterns.json")
}

// ── Assertion helpers ──

/// Count total memories in the store.
pub fn count_memories(m: &Memoria) -> usize {
    m.store().count_memories().unwrap()
}

/// Count memories in a specific namespace.
pub fn count_memories_ns(m: &Memoria, ns: &str) -> usize {
    m.store().count_memories_in_namespace(ns).unwrap()
}

/// Get accumulated unresolved surprise.
pub fn query_surprise(m: &Memoria) -> f64 {
    m.accumulated_surprise().unwrap()
}

/// Count skills in the store.
pub fn count_skills(m: &Memoria) -> usize {
    m.count_skills().unwrap()
}

/// Wait for queue to drain (up to `max_wait`).
pub async fn wait_queue_drain(queue: &TaskQueue, max_wait: Duration) {
    let deadline = tokio::time::Instant::now() + max_wait;
    loop {
        if queue.count_pending().unwrap() == 0 {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "Queue did not drain within {:?} — {} tasks remaining",
                max_wait,
                queue.count_pending().unwrap()
            );
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Tell multiple texts sequentially, returning all memory IDs.
pub async fn tell_many(m: &Memoria, texts: &[&str], ctx: &AgentContext) -> Vec<Uuid> {
    let mut ids = Vec::new();
    for text in texts {
        let r = m.tell(text, ctx).await.unwrap();
        ids.extend(r.memory_ids);
    }
    ids
}

// ── Test Result Guard ──

/// Drop-based logger that writes test results to `target/test_results.txt`.
///
/// Create at test start, call `mark_passed()` at end.
/// If the test panics before `mark_passed()`, Drop logs FAIL.
pub struct TestResultGuard {
    test_name: String,
    details: String,
    passed: bool,
}

impl TestResultGuard {
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            details: String::new(),
            passed: false,
        }
    }

    pub fn set_details(&mut self, details: &str) {
        self.details = details.to_string();
    }

    pub fn mark_passed(&mut self) {
        self.passed = true;
    }
}

impl Drop for TestResultGuard {
    fn drop(&mut self) {
        use std::io::Write;
        let status = if self.passed { "PASS" } else { "FAIL" };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let line = if self.details.is_empty() {
            format!("[{now}] {status} {}\n", self.test_name)
        } else {
            format!("[{now}] {status} {} — {}\n", self.test_name, self.details)
        };
        // Best-effort append to target/test_results.txt
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("target/test_results.txt")
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}
