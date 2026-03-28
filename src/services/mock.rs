use async_trait::async_trait;

use super::traits::*;

// ── Mock Embedder ──

/// Returns random unit vectors of the configured dimension.
/// Deterministic when seeded — useful for testing.
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Simple deterministic pseudo-random based on text hash.
    fn pseudo_random_vec(&self, text: &str) -> Vec<f32> {
        let mut hash: u64 = 5381;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        let mut vec = Vec::with_capacity(self.dim);
        let mut state = hash;
        for _ in 0..self.dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            vec.push(val);
        }

        // Normalize to unit vector
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.pseudo_random_vec(t)).collect())
    }
}

// ── Mock NER ──

/// Returns empty extraction results — no entities or relations found.
pub struct MockNer;

#[async_trait]
impl NerService for MockNer {
    async fn extract(
        &self,
        texts: &[TextInput],
        _entity_labels: &[&str],
        _relation_labels: &[&str],
    ) -> anyhow::Result<Vec<ExtractionResult>> {
        Ok(texts
            .iter()
            .map(|_| ExtractionResult {
                entities: Vec::new(),
                relations: Vec::new(),
            })
            .collect())
    }
}

// ── Mock LLM ──

/// Echoes back the last user message content.
pub struct MockLlm;

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(
        &self,
        messages: &[Message],
        _max_tokens: usize,
    ) -> anyhow::Result<LlmResponse> {
        let content = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Detect verification prompts and return structured JSON
        if content.contains("verify") && content.contains("relation") {
            return Ok(LlmResponse {
                content: r#"[{"verified": true, "temporal_status": "current", "corrected_label": null, "confidence": 0.9, "reason": "Direct assertion in text"}]"#.to_string(),
                thinking: None,
            });
        }

        // Detect rule evaluation prompts
        if content.contains("violate the rule") {
            return Ok(LlmResponse {
                content: r#"{"result": "allow", "reason": "No violation detected"}"#.to_string(),
                thinking: None,
            });
        }

        Ok(LlmResponse {
            content: format!("[mock] {content}"),
            thinking: None,
        })
    }
}

// ── Mock Reranker ──

/// Returns documents in their original order with decreasing scores.
pub struct MockReranker;

#[async_trait]
impl Reranker for MockReranker {
    async fn rerank(
        &self,
        _query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> anyhow::Result<Vec<RerankResult>> {
        let n = top_n.min(documents.len());
        Ok((0..n)
            .map(|i| RerankResult {
                index: i,
                score: 1.0 - (i as f64 * 0.1),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_embedder_returns_correct_dimension() {
        let embedder = MockEmbedder::new(768);
        assert_eq!(embedder.dim(), 768);

        let vecs = embedder.embed(&["hello world"]).await.unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].len(), 768);
    }

    #[tokio::test]
    async fn mock_embedder_is_deterministic() {
        let embedder = MockEmbedder::new(128);
        let v1 = embedder.embed(&["test"]).await.unwrap();
        let v2 = embedder.embed(&["test"]).await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn mock_embedder_produces_unit_vectors() {
        let embedder = MockEmbedder::new(128);
        let vecs = embedder.embed(&["hello"]).await.unwrap();
        let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn mock_ner_returns_empty() {
        let ner = MockNer;
        let results = ner
            .extract(
                &[TextInput {
                    text: "Alice works at Acme".into(),
                    id: None,
                }],
                &["person"],
                &["works_at"],
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].entities.is_empty());
    }

    #[tokio::test]
    async fn mock_llm_echoes_user() {
        let llm = MockLlm;
        let resp = llm
            .complete(
                &[Message {
                    role: "user".into(),
                    content: "hello".into(),
                }],
                100,
            )
            .await
            .unwrap();
        assert_eq!(resp.content, "[mock] hello");
    }

    #[tokio::test]
    async fn mock_llm_handles_verification() {
        let llm = MockLlm;
        let resp = llm
            .complete(
                &[Message {
                    role: "user".into(),
                    content: "Please verify these relations extracted from text".into(),
                }],
                4096,
            )
            .await
            .unwrap();
        assert!(resp.content.contains("verified"));
    }

    #[tokio::test]
    async fn mock_reranker_returns_ordered() {
        let reranker = MockReranker;
        let results = reranker
            .rerank("query", &["a", "b", "c"], 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
    }
}
