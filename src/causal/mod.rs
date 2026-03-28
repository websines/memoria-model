//! Causal reasoning — counterfactual attribution, structural causal models,
//! d-separation, do-operator, and sequence mining.
//!
//! ## Structural Causal Models (Initiative 5)
//!
//! The `graph` module provides persistent causal edges with Bayesian accumulation.
//! Edges are discovered from three sources:
//! - Attribution results (counterfactual analysis)
//! - LLM proposals during reflection
//! - NOTEARS structure learning from observation data
//!
//! The `bayes_ball` module implements correct d-separation on petgraph DAGs,
//! properly handling colliders (explaining-away) unlike the older Datalog
//! approach in `d_separation`.
//!
//! The `notears` module provides DAG structure learning via continuous
//! optimization with acyclicity constraints.

pub mod attribution;
pub mod bayes_ball;
pub mod d_separation;
pub mod do_operator;
pub mod graph;
pub mod notears;
pub mod sequence_mining;

pub use attribution::{Attribution, AttributionResult};
pub use do_operator::{DoResult, PropagatedEffect};
pub use graph::{CausalEdge, CausalGraph, CausalMechanism};
pub use notears::{NotearsConfig, NotearsResult};
pub use sequence_mining::SequentialPattern;
