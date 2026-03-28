use async_trait::async_trait;
use std::sync::Arc;

use super::traits::*;

/// An embedder that tries a primary (API) embedder first,
/// falling back to a secondary (local) embedder on failure.
pub struct FallbackEmbedder {
    primary: Arc<dyn Embedder>,
    fallback: Arc<dyn Embedder>,
}

impl FallbackEmbedder {
    pub fn new(primary: Arc<dyn Embedder>, fallback: Arc<dyn Embedder>) -> Self {
        assert_eq!(
            primary.dim(),
            fallback.dim(),
            "primary and fallback embedders must have the same dimension"
        );
        Self { primary, fallback }
    }
}

#[async_trait]
impl Embedder for FallbackEmbedder {
    fn dim(&self) -> usize {
        self.primary.dim()
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        match self.primary.embed(texts).await {
            Ok(result) => Ok(result),
            Err(_) => self.fallback.embed(texts).await,
        }
    }
}

/// A reranker that tries a primary (API) reranker first,
/// falling back to a secondary (local) reranker on failure.
pub struct FallbackReranker {
    primary: Arc<dyn Reranker>,
    fallback: Arc<dyn Reranker>,
}

impl FallbackReranker {
    pub fn new(primary: Arc<dyn Reranker>, fallback: Arc<dyn Reranker>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl Reranker for FallbackReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> anyhow::Result<Vec<RerankResult>> {
        match self.primary.rerank(query, documents, top_n).await {
            Ok(result) => Ok(result),
            Err(_) => self.fallback.rerank(query, documents, top_n).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::mock::{MockEmbedder, MockReranker};

    /// An embedder that always fails — used to test fallback behavior.
    struct FailingEmbedder {
        dim: usize,
    }

    #[async_trait]
    impl Embedder for FailingEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }
        async fn embed(&self, _texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            anyhow::bail!("connection refused")
        }
    }

    struct FailingReranker;

    #[async_trait]
    impl Reranker for FailingReranker {
        async fn rerank(
            &self,
            _query: &str,
            _documents: &[&str],
            _top_n: usize,
        ) -> anyhow::Result<Vec<RerankResult>> {
            anyhow::bail!("connection refused")
        }
    }

    #[tokio::test]
    async fn fallback_embedder_uses_primary_when_available() {
        let primary = Arc::new(MockEmbedder::new(128));
        let fallback = Arc::new(MockEmbedder::new(128));
        let fb = FallbackEmbedder::new(primary, fallback);

        let result = fb.embed(&["test"]).await.unwrap();
        assert_eq!(result[0].len(), 128);
    }

    #[tokio::test]
    async fn fallback_embedder_falls_back_on_failure() {
        let primary = Arc::new(FailingEmbedder { dim: 128 });
        let fallback = Arc::new(MockEmbedder::new(128));
        let fb = FallbackEmbedder::new(primary, fallback);

        let result = fb.embed(&["test"]).await.unwrap();
        assert_eq!(result[0].len(), 128);
    }

    #[tokio::test]
    async fn fallback_reranker_falls_back_on_failure() {
        let primary = Arc::new(FailingReranker);
        let fallback = Arc::new(MockReranker);
        let fb = FallbackReranker::new(primary, fallback);

        let result = fb.rerank("query", &["a", "b"], 2).await.unwrap();
        assert_eq!(result.len(), 2);
    }
}
