use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ── NER Service ──

/// Entity/relation extraction result from NER (e.g., GLiNER2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub label: String,
    pub confidence: f64,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub head: String,
    pub tail: String,
    pub label: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct TextInput {
    pub text: String,
    pub id: Option<String>,
}

#[async_trait]
pub trait NerService: Send + Sync {
    async fn extract(
        &self,
        texts: &[TextInput],
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> anyhow::Result<Vec<ExtractionResult>>;
}

// ── Embedder ──

#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embedding dimension.
    fn dim(&self) -> usize;

    /// Embed a batch of texts into vectors.
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

// ── Reranker ──

#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f64,
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> anyhow::Result<Vec<RerankResult>>;
}

// ── LLM Service ──

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub thinking: Option<String>,
}

#[async_trait]
pub trait LlmService: Send + Sync {
    async fn complete(
        &self,
        messages: &[Message],
        max_tokens: usize,
    ) -> anyhow::Result<LlmResponse>;
}
