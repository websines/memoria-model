//! Phase 4: Memory Dynamics — surprise-driven reflection, reconsolidation,
//! compression, confidence decay, and graph metrics.
//!
//! All dynamics are triggered by surprise accumulation, not fixed timers.
//! The surprise system detects when new information conflicts with existing
//! beliefs using precision-weighted prediction error.

pub mod compression;
pub mod confidence;
pub mod deadline;
pub mod graph_metrics;
pub mod intrinsic;
pub mod meta_learning;
pub mod ppm;
pub mod prediction;
pub mod projection;
pub mod reconsolidation;
pub mod reflection;
pub mod surprise;
pub mod trust;

pub use compression::{CompressionLevel, CompressionResult};
pub use confidence::{effective_confidence, effective_fact_confidence, provenance_tau};
pub use graph_metrics::{CommunityResult, ImportanceResult};
pub use meta_learning::{MetaLearner, MetaPhase, MetaStepResult, TunableParam};
pub use ppm::PpmModel;
pub use prediction::{
    Prediction, PredictionCycleResult, PredictionKind, PredictionMatchResult,
    PredictionResolution, PredictionSource,
};
pub use projection::{EmbeddingProjection, ProjectionStats};
pub use reconsolidation::ReconsolidationResult;
pub use reflection::ReflectionResult;
pub use intrinsic::{IntrinsicGoalResult, SurpriseHotspot};
pub use surprise::{Observation, SurpriseResult};
pub use trust::TrustScore;
