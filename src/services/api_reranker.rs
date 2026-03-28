use async_trait::async_trait;
use serde::Deserialize;

use super::traits::{Reranker, RerankResult};

/// OpenAI-compatible reranker API client (e.g., Jina Reranker via LM Studio).
pub struct ApiReranker {
    base_url: String,
    model: String,
    client: ureq::Agent,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResponseItem>,
}

#[derive(Deserialize)]
struct RerankResponseItem {
    index: usize,
    relevance_score: f64,
}

impl ApiReranker {
    pub fn new(base_url: &str, model: &str) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(30))
            .build();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client,
        }
    }
}

#[async_trait]
impl Reranker for ApiReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> anyhow::Result<Vec<RerankResult>> {
        let url = format!("{}/rerank", self.base_url);

        let resp: RerankResponse = self
            .client
            .post(&url)
            .send_json(ureq::json!({
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }))?
            .into_json()?;

        Ok(resp
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                score: r.relevance_score,
            })
            .collect())
    }
}
