pub mod traits;
pub mod mock;
pub mod fallback;
pub mod api_embedder;
pub mod api_reranker;
pub mod gliner;
pub mod local_reranker;

pub use traits::*;
pub use mock::*;
pub use fallback::*;
pub use api_embedder::ApiEmbedder;
pub use api_reranker::ApiReranker;
pub use gliner::GlinerSidecar;

#[cfg(feature = "local-embeddings")]
pub use local_reranker::LocalReranker;
