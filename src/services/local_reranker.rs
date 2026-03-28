//! Local ONNX-based reranker using fastembed.
//!
//! Requires the `local-embeddings` feature flag.

#[cfg(feature = "local-embeddings")]
mod inner {
    use async_trait::async_trait;
    use fastembed::{TextRerank, RerankInitOptions, RerankerModel};
    use std::sync::Mutex;

    use crate::services::traits::{Reranker, RerankResult};

    /// Local cross-encoder reranker powered by fastembed (ONNX).
    ///
    /// Downloads the model on first use. Thread-safe via Mutex
    /// (fastembed's TextRerank requires &mut self for rerank).
    pub struct LocalReranker {
        model: Mutex<TextRerank>,
    }

    impl LocalReranker {
        /// Create a new local reranker with the default model (BGE Reranker Base).
        pub fn new() -> anyhow::Result<Self> {
            let model = TextRerank::try_new(
                RerankInitOptions::new(RerankerModel::BGERerankerBase)
                    .with_show_download_progress(true),
            )?;

            Ok(Self {
                model: Mutex::new(model),
            })
        }
    }

    #[async_trait]
    impl Reranker for LocalReranker {
        async fn rerank(
            &self,
            query: &str,
            documents: &[&str],
            top_n: usize,
        ) -> anyhow::Result<Vec<RerankResult>> {
            let docs: Vec<&str> = documents.to_vec();
            let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;

            let results = model.rerank(query, docs, false, None)?;

            Ok(results
                .into_iter()
                .take(top_n)
                .map(|r| RerankResult {
                    index: r.index,
                    score: r.score as f64,
                })
                .collect())
        }
    }
}

#[cfg(feature = "local-embeddings")]
pub use inner::LocalReranker;
