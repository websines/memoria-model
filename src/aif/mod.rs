//! Active Inference Framework — unifies all subsystems under free energy minimization.
//!
//! This module provides:
//! - **Belief updates**: Canonical precision formulas and Kalman-like updates
//! - **Free energy**: Bethe Free Energy computation over the entire knowledge store
//! - **Model state**: Snapshots and auto-tuning of the exploration parameter β
//! - **Messages**: Precision-weighted factor messages for scoring fusion
//! - **Health**: Trend detection and system health monitoring
//! - **Hierarchical**: Bidirectional message passing over chunk hierarchies (§15.9)
//! - **Propagation**: Confidence propagation through source chains (§15.8)

pub mod belief_update;
pub mod free_energy;
pub mod health;
pub mod hierarchical;
pub mod messages;
pub mod model_state;
pub mod propagation;

pub use belief_update::{BeliefUpdateResult, belief_update, observation_precision, precision};
pub use free_energy::{FreeEnergyState, compute_bethe_free_energy};
pub use health::{ModelHealth, Trend, compute_health, compute_trend};
pub use hierarchical::{HierarchicalScore, combine_hierarchical_scores, compute_bottom_up, compute_top_down};
pub use messages::{FactorMessage, FactorType, compute_recall_messages, fuse_messages};
pub use model_state::{
    get_latest_beta, get_latest_model_state, snapshot_model_state,
};
pub use propagation::{PropagationResult, propagate_confidence};
