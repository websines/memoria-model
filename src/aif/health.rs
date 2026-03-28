//! Health monitoring and trend detection for the Memoria system.
//!
//! Tracks free energy trends over time to detect model improvement or degradation.
//! Used by Orchestria to decide when to trigger maintenance operations.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::store::CozoStore;

/// Direction of free energy change over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    /// Free energy is decreasing → model is improving.
    Improving,
    /// Free energy is increasing → model is degrading.
    Degrading,
    /// Free energy is roughly stable.
    Stable,
}

/// Overall health of the Memoria model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealth {
    /// Direction of free energy change.
    pub free_energy_trend: Trend,
    /// Current exploration parameter.
    pub beta: f64,
    /// Total unresolved surprise waiting to be processed.
    pub unresolved_surprise: f64,
    /// Number of pending tasks in the queue.
    pub queue_depth: usize,
    /// Number of dead (permanently failed) tasks.
    pub queue_dead: usize,
}

/// Compute the trend of free energy over recent snapshots using linear regression.
///
/// `window_size` controls how many recent snapshots to consider.
/// Uses simple least-squares linear regression on the free energy values.
/// Reads historical snapshots (not just `@ 'NOW'`) to get multiple data points.
pub fn compute_trend(store: &CozoStore, window_size: usize) -> Result<Trend> {
    let window = window_size.max(2);

    // Use model_state history (sorted newest-first) for trend computation
    let snapshots = crate::aif::model_state::get_model_state_history(store, window)?;

    // Reverse to chronological order (oldest first) for regression
    let values: Vec<f64> = snapshots.iter().rev().map(|s| s.free_energy).collect();

    if values.len() < 2 {
        return Ok(Trend::Stable);
    }

    // Simple linear regression: slope of FE over time indices
    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..values.len()).map(|i| (i as f64) * (i as f64)).sum();

    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < 1e-10 {
        return Ok(Trend::Stable);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;

    // Threshold: slope magnitude must exceed 5% of mean FE to be significant
    let mean_fe = sum_y / n;
    let threshold = mean_fe.abs() * 0.05;

    if slope < -threshold {
        Ok(Trend::Improving) // FE decreasing → improving
    } else if slope > threshold {
        Ok(Trend::Degrading) // FE increasing → degrading
    } else {
        Ok(Trend::Stable)
    }
}

/// Compute the overall health of the Memoria system.
///
/// Combines free energy trend, β, unresolved surprise, and queue status.
pub fn compute_health(
    store: &CozoStore,
    queue_pending: usize,
    queue_dead: usize,
) -> Result<ModelHealth> {
    let trend = compute_trend(store, 10)?;

    let beta = crate::aif::model_state::get_latest_beta(store)?;
    let unresolved = crate::dynamics::surprise::accumulated_unresolved_surprise(store)?;

    Ok(ModelHealth {
        free_energy_trend: trend,
        beta,
        unresolved_surprise: unresolved,
        queue_depth: queue_pending,
        queue_dead,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trend_stable_with_no_snapshots() {
        let store = CozoStore::open_mem(4).unwrap();
        let trend = compute_trend(&store, 10).unwrap();
        assert_eq!(trend, Trend::Stable);
    }

    #[test]
    fn health_default_state() {
        let store = CozoStore::open_mem(4).unwrap();
        let health = compute_health(&store, 0, 0).unwrap();
        assert_eq!(health.free_energy_trend, Trend::Stable);
        assert_eq!(health.beta, 1.0);
        assert_eq!(health.unresolved_surprise, 0.0);
        assert_eq!(health.queue_depth, 0);
        assert_eq!(health.queue_dead, 0);
    }

    #[test]
    fn health_reflects_queue_state() {
        let store = CozoStore::open_mem(4).unwrap();
        let health = compute_health(&store, 5, 2).unwrap();
        assert_eq!(health.queue_depth, 5);
        assert_eq!(health.queue_dead, 2);
    }
}
