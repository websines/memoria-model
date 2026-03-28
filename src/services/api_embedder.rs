use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::traits::Embedder;

/// OpenAI-compatible embedding API client.
pub struct ApiEmbedder {
    base_url: String,
    model: String,
    dim: usize,
    client: ureq::Agent,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

impl ApiEmbedder {
    pub fn new(base_url: &str, model: &str, dim: usize) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(30))
            .build();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dim,
            client,
        }
    }
}

#[async_trait]
impl Embedder for ApiEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = EmbedRequest {
            model: self.model.clone(),
            input: texts.iter().map(|t| t.to_string()).collect(),
        };

        let resp: EmbedResponse = self
            .client
            .post(&url)
            .send_json(ureq::json!({
                "model": body.model,
                "input": body.input,
            }))?
            .into_json()?;

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}
