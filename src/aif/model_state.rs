//! Model state snapshots and β auto-tuning.
//!
//! Stores periodic snapshots of the system's free energy state in CozoDB.
//! The latest β is used by `select_skills_auto()` for exploration/exploitation tuning.

use std::collections::BTreeMap;

use cozo::DataValue;
use serde::{Deserialize, Serialize};

use crate::aif::free_energy::{compute_bethe_free_energy, FreeEnergyState};
use crate::error::Result;
use crate::store::CozoStore;

/// A snapshot of the model state stored in CozoDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStateSnapshot {
    pub free_energy: f64,
    pub accuracy: f64,
    pub complexity: f64,
    pub beta: f64,
    pub unresolved_surprise: f64,
    pub agent_id: String,
}

/// Compute the current free energy state and store a snapshot in CozoDB.
///
/// Returns the computed `FreeEnergyState` for immediate use.
pub fn snapshot_model_state(store: &CozoStore, agent_id: &str) -> Result<FreeEnergyState> {
    let state = compute_bethe_free_energy(store)?;

    // Get unresolved surprise
    let unresolved = crate::dynamics::surprise::accumulated_unresolved_surprise(store)?;

    let mut params = BTreeMap::new();
    params.insert("fe".into(), DataValue::from(state.free_energy));
    params.insert("acc".into(), DataValue::from(state.accuracy));
    params.insert("comp".into(), DataValue::from(state.complexity));
    params.insert("beta".into(), DataValue::from(state.beta));
    params.insert("surprise".into(), DataValue::from(unresolved));
    params.insert("agent_id".into(), DataValue::from(agent_id));

    store.run_script(
        r#"?[valid_at, free_energy, accuracy, complexity, beta, unresolved_surprise, agent_id] <- [
            ['ASSERT', $fe, $acc, $comp, $beta, $surprise, $agent_id]
        ]
        :put model_state {
            valid_at, free_energy, accuracy, complexity, beta, unresolved_surprise, agent_id
        }"#,
        params,
    )?;

    Ok(state)
}

/// Get the latest model state snapshot.
///
/// Uses `@ 'NOW'` to get the current valid snapshot. With the Validity-keyed schema,
/// this deterministically returns the most recent version.
pub fn get_latest_model_state(store: &CozoStore) -> Result<Option<ModelStateSnapshot>> {
    let result = store.run_query(
        r#"?[free_energy, accuracy, complexity, beta, unresolved_surprise, agent_id] :=
            *model_state{free_energy, accuracy, complexity, beta, unresolved_surprise, agent_id, @ 'NOW'}"#,
        BTreeMap::new(),
    )?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    let row = &result.rows[0];
    Ok(Some(ModelStateSnapshot {
        free_energy: row[0].get_float().unwrap_or(0.0),
        accuracy: row[1].get_float().unwrap_or(0.0),
        complexity: row[2].get_float().unwrap_or(0.0),
        beta: row[3].get_float().unwrap_or(1.0),
        unresolved_surprise: row[4].get_float().unwrap_or(0.0),
        agent_id: row[5].get_str().map(|s| s.to_string()).unwrap_or_default(),
    }))
}

/// Get multiple historical model state snapshots, ordered newest first.
///
/// Reads raw validity data (all versions) to support trend computation.
pub fn get_model_state_history(store: &CozoStore, limit: usize) -> Result<Vec<ModelStateSnapshot>> {
    let mut params = BTreeMap::new();
    params.insert("limit".into(), DataValue::from(limit as i64));

    // Read raw validity data — each 'ASSERT' created a new version
    // valid_at gives us the timestamp ordering
    let result = store.run_query(
        r#"?[valid_at, free_energy, accuracy, complexity, beta, unresolved_surprise, agent_id] :=
            *model_state{valid_at, free_energy, accuracy, complexity, beta, unresolved_surprise, agent_id}
        :sort -valid_at
        :limit $limit"#,
        params,
    )?;

    let mut snapshots = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        // row[0] is valid_at, skip it
        snapshots.push(ModelStateSnapshot {
            free_energy: row[1].get_float().unwrap_or(0.0),
            accuracy: row[2].get_float().unwrap_or(0.0),
            complexity: row[3].get_float().unwrap_or(0.0),
            beta: row[4].get_float().unwrap_or(1.0),
            unresolved_surprise: row[5].get_float().unwrap_or(0.0),
            agent_id: row[6].get_str().map(|s| s.to_string()).unwrap_or_default(),
        });
    }

    Ok(snapshots)
}

/// Get the latest β value, defaulting to 1.0 (maximum exploration) if no snapshots exist.
pub fn get_latest_beta(store: &CozoStore) -> Result<f64> {
    match get_latest_model_state(store)? {
        Some(state) => Ok(state.beta),
        None => Ok(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_and_retrieve() {
        let store = CozoStore::open_mem(4).unwrap();

        let state = snapshot_model_state(&store, "test-agent").unwrap();
        assert!(state.beta >= 0.0 && state.beta <= 1.0);

        let latest = get_latest_model_state(&store).unwrap();
        assert!(latest.is_some());
        let snap = latest.unwrap();
        assert_eq!(snap.agent_id, "test-agent");
        assert!((snap.free_energy - state.free_energy).abs() < 1e-6);
    }

    #[test]
    fn get_latest_beta_default() {
        let store = CozoStore::open_mem(4).unwrap();
        let beta = get_latest_beta(&store).unwrap();
        assert_eq!(beta, 1.0, "no snapshots → default β = 1.0");
    }

    #[test]
    fn get_latest_beta_after_snapshot() {
        let store = CozoStore::open_mem(4).unwrap();
        snapshot_model_state(&store, "agent").unwrap();
        let beta = get_latest_beta(&store).unwrap();
        assert!(beta >= 0.0 && beta <= 1.0);
    }
}
