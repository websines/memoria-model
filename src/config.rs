use serde::{Deserialize, Serialize};

/// Top-level Memoria configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoriaConfig {
    /// Path to the CozoDB RocksDB data directory.
    pub db_path: String,

    /// Service endpoints for external APIs.
    pub services: ServiceEndpoints,

    /// Default entity labels for NER extraction.
    #[serde(default = "default_entity_labels")]
    pub entity_labels: Vec<String>,

    /// Default relation labels for NER extraction.
    #[serde(default = "default_relation_labels")]
    pub relation_labels: Vec<String>,

    /// Surprise threshold that triggers cold-path consolidation.
    #[serde(default = "default_consolidation_threshold")]
    pub consolidation_threshold: f64,

    /// Maximum candidates to retrieve in broad HNSW search.
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,

    /// Maximum cosine distance for HNSW results.
    #[serde(default = "default_max_distance")]
    pub max_distance: f64,

    /// Time constant (tau) for activation decay in milliseconds.
    #[serde(default = "default_activation_tau")]
    pub activation_tau: f64,

    /// Phase 4: Memory dynamics configuration.
    #[serde(default)]
    pub dynamics: DynamicsConfig,

    /// Whether scope enforcement is enabled. When false, all agents have full access.
    #[serde(default = "default_false")]
    pub scope_enforcement_enabled: bool,

    /// Capacity of the memory event broadcast channel.
    #[serde(default = "default_event_channel_capacity")]
    pub event_channel_capacity: usize,

    /// Kernel rules configuration.
    #[serde(default)]
    pub kernel_rules: KernelRulesConfig,
}

/// Configuration for kernel-level rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KernelRulesConfig {
    /// Kernel rules to enforce on all operations.
    #[serde(default)]
    pub rules: Vec<crate::types::kernel::KernelRule>,
}

/// Configuration for Phase 4 memory dynamics subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsConfig {
    /// Minimum cluster size for compression (co-activation clusters).
    #[serde(default = "default_min_cluster_size")]
    pub min_cluster_size: usize,

    /// Number of episodes needed before an abstraction is promoted.
    #[serde(default = "default_promotion_threshold")]
    pub promotion_threshold: i64,

    /// Whether to run PageRank/community detection automatically.
    #[serde(default = "default_auto_graph_metrics")]
    pub auto_graph_metrics: bool,

    /// Whether β auto-tuning adapts consolidation threshold.
    /// When true, `effective_consolidation_threshold = base / (1 + β)`.
    /// High β (uncertain) → lower threshold → consolidate more aggressively.
    #[serde(default = "default_adaptive_beta")]
    pub adaptive_beta: bool,

    /// Memory count threshold per namespace before compression is auto-triggered.
    #[serde(default = "default_compression_memory_threshold")]
    pub compression_memory_threshold: usize,

    /// Whether to use hierarchical chunking in tell() instead of flat semantic chunking.
    /// Default: false (opt-in to preserve backward compatibility).
    #[serde(default)]
    pub use_hierarchical_chunking: bool,

    /// Weight for telos alignment in EFE skill selection.
    /// EFE = pragmatic + β × epistemic + γ × telos_alignment.
    /// Higher γ → skills more strongly pulled toward active goals.
    #[serde(default = "default_telos_gamma")]
    pub telos_gamma: f64,

    /// Interval in seconds for the background tick loop.
    /// The tick loop checks compression thresholds, runs scratchpad GC,
    /// and checks reflection thresholds.
    #[serde(default = "default_tick_interval_secs")]
    pub tick_interval_secs: u64,

    /// Whether the meta-learning subsystem is enabled.
    /// When true, hyperparameters are auto-tuned at each meta step.
    #[serde(default)]
    pub meta_learning_enabled: bool,

    /// Run a meta-learning step every N background ticks.
    #[serde(default = "default_meta_learning_interval")]
    pub meta_learning_interval: u64,

    /// Number of ticks to wait between consecutive SPSA measurements.
    #[serde(default = "default_observation_window")]
    pub observation_window: u64,

    /// Maximum number of Bayesian Optimisation evaluations before switching to SPSA.
    #[serde(default = "default_bo_budget")]
    pub bo_budget: u64,

    /// Whether the predictive generation subsystem is enabled.
    /// When true, PPM-C, ETS, and BOCPD generators run during tick().
    #[serde(default)]
    pub prediction_enabled: bool,

    /// Run prediction generation every N background ticks.
    #[serde(default = "default_prediction_interval")]
    pub prediction_interval: u64,

    /// Maximum context depth for the PPM-C sequence predictor.
    #[serde(default = "default_ppm_max_depth")]
    pub ppm_max_depth: usize,

    /// Minimum confidence threshold for emitting predictions.
    #[serde(default = "default_prediction_confidence_threshold")]
    pub prediction_confidence_threshold: f64,

    /// Whether the learned embedding projection subsystem is enabled.
    /// When true, ask() applies the projection to queries and training runs periodically.
    #[serde(default)]
    pub projection_enabled: bool,

    /// Minimum triplet count before projection training is triggered.
    #[serde(default = "default_projection_min_triplets")]
    pub projection_min_triplets: usize,

    /// Number of epochs per projection training session.
    #[serde(default = "default_projection_epochs")]
    pub projection_epochs: usize,

    /// Run projection training every N background ticks.
    #[serde(default = "default_projection_train_interval")]
    pub projection_train_interval: u64,

    /// Loss improvement threshold to trigger a projected index rebuild.
    /// If loss drops by more than this fraction, re-project all memories.
    #[serde(default = "default_projection_rebuild_threshold")]
    pub projection_rebuild_threshold: f64,

    /// Whether intrinsic goal generation from surprise hotspots is enabled.
    /// When true, the system autonomously creates exploratory telos from
    /// persistent surprise patterns, gated by β and free energy signals.
    #[serde(default)]
    pub intrinsic_goal_enabled: bool,

    /// Run intrinsic goal generation every N background ticks.
    #[serde(default = "default_intrinsic_goal_interval")]
    pub intrinsic_goal_interval: u64,
}

impl DynamicsConfig {
    /// Get the memory count threshold that triggers automatic compression.
    pub fn compression_threshold(&self) -> usize {
        self.compression_memory_threshold
    }
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: default_min_cluster_size(),
            promotion_threshold: default_promotion_threshold(),
            auto_graph_metrics: default_auto_graph_metrics(),
            adaptive_beta: default_adaptive_beta(),
            compression_memory_threshold: default_compression_memory_threshold(),
            use_hierarchical_chunking: false,
            telos_gamma: default_telos_gamma(),
            tick_interval_secs: default_tick_interval_secs(),
            meta_learning_enabled: false,
            meta_learning_interval: default_meta_learning_interval(),
            observation_window: default_observation_window(),
            bo_budget: default_bo_budget(),
            prediction_enabled: false,
            prediction_interval: default_prediction_interval(),
            ppm_max_depth: default_ppm_max_depth(),
            prediction_confidence_threshold: default_prediction_confidence_threshold(),
            projection_enabled: false,
            projection_min_triplets: default_projection_min_triplets(),
            projection_epochs: default_projection_epochs(),
            projection_train_interval: default_projection_train_interval(),
            projection_rebuild_threshold: default_projection_rebuild_threshold(),
            intrinsic_goal_enabled: false,
            intrinsic_goal_interval: default_intrinsic_goal_interval(),
        }
    }
}

/// Endpoints for external service APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoints {
    /// GLiNER2 NER service URL (e.g., "http://localhost:9100").
    pub ner_url: Option<String>,

    /// LLM API base URL (OpenAI-compatible /v1/messages).
    pub llm_url: Option<String>,

    /// LLM model name.
    #[serde(default)]
    pub llm_model: String,

    /// Embedding API base URL (/v1/embeddings).
    pub embedder_url: Option<String>,

    /// Embedding model name.
    #[serde(default)]
    pub embedder_model: String,

    /// Embedding dimension (resolved from embedder at runtime if not set).
    pub embedding_dim: Option<usize>,

    /// Reranker API base URL (if using API-based reranker).
    pub reranker_url: Option<String>,
}

impl Default for MemoriaConfig {
    fn default() -> Self {
        Self {
            db_path: "memoria_data".to_string(),
            services: ServiceEndpoints::default(),
            entity_labels: default_entity_labels(),
            relation_labels: default_relation_labels(),
            consolidation_threshold: default_consolidation_threshold(),
            max_candidates: default_max_candidates(),
            max_distance: default_max_distance(),
            activation_tau: default_activation_tau(),
            dynamics: DynamicsConfig::default(),
            scope_enforcement_enabled: default_false(),
            event_channel_capacity: default_event_channel_capacity(),
            kernel_rules: KernelRulesConfig::default(),
        }
    }
}

impl Default for ServiceEndpoints {
    fn default() -> Self {
        Self {
            ner_url: None,
            llm_url: None,
            llm_model: String::new(),
            embedder_url: None,
            embedder_model: String::new(),
            embedding_dim: None,
            reranker_url: None,
        }
    }
}

fn default_entity_labels() -> Vec<String> {
    ["person", "organization", "location", "project", "technology", "concept"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn default_relation_labels() -> Vec<String> {
    [
        "works_at",
        "manages",
        "knows",
        "uses",
        "part_of",
        "depends_on",
        "created_by",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn default_consolidation_threshold() -> f64 {
    2.0
}
fn default_max_candidates() -> usize {
    100
}
fn default_max_distance() -> f64 {
    0.8
}
fn default_activation_tau() -> f64 {
    86_400_000.0 // 1 day in ms
}
fn default_min_cluster_size() -> usize {
    3
}
fn default_promotion_threshold() -> i64 {
    3
}
fn default_auto_graph_metrics() -> bool {
    true
}
fn default_adaptive_beta() -> bool {
    true
}
fn default_false() -> bool {
    false
}
fn default_compression_memory_threshold() -> usize {
    100
}
fn default_event_channel_capacity() -> usize {
    1024
}
fn default_telos_gamma() -> f64 {
    0.3
}

fn default_tick_interval_secs() -> u64 {
    60
}
fn default_meta_learning_interval() -> u64 {
    10
}
fn default_observation_window() -> u64 {
    5
}
fn default_bo_budget() -> u64 {
    150
}
fn default_prediction_interval() -> u64 {
    5 // every 5 ticks
}
fn default_ppm_max_depth() -> usize {
    5
}
fn default_prediction_confidence_threshold() -> f64 {
    0.1
}
fn default_projection_min_triplets() -> usize {
    50
}
fn default_projection_epochs() -> usize {
    10
}
fn default_projection_train_interval() -> u64 {
    20
}
fn default_projection_rebuild_threshold() -> f64 {
    0.1
}
fn default_intrinsic_goal_interval() -> u64 {
    10 // every 10 ticks
}
