//! Meta-learning over Memoria dynamics parameters.
//!
//! Uses a two-phase approach:
//! - **Phase 1 (ColdStart):** Collects observations and explores the parameter space.
//!   Each tick records (params, free_energy) pairs. After `bo_budget` observations,
//!   transitions to online tracking.
//! - **Phase 2 (OnlineTracking):** SPSA (Simultaneous Perturbation Stochastic Approximation)
//!   for continuous drift tracking. Needs only 2 evaluations per gradient step regardless
//!   of parameter dimensionality.
//!
//! The loss function is **free energy** — already computed by the AIF module.
//! Meta-learning tunes the dynamics parameters to minimize free energy over time.

use crate::error::{MemoriaError, Result};
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// A parameter eligible for meta-learning with its current value and bounds.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TunableParam {
    pub name: String,
    pub current: f64,
    pub min: f64,
    pub max: f64,
    /// Perturbation scale (SPSA uses this to scale the random direction).
    pub step: f64,
}

/// Which optimization phase the meta-learner is in.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MetaPhase {
    /// Exploration phase: collect observations, try diverse parameter settings.
    ColdStart,
    /// Online tracking: SPSA gradient descent for drift compensation.
    OnlineTracking,
}

/// Internal state of the SPSA algorithm between ticks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SpsaState {
    /// Ready to start a new step.
    Idle,
    /// Positive perturbation applied, waiting for observation window to elapse.
    WaitingPlus {
        saved_params: Vec<f64>,
        delta: Vec<f64>,
        applied_at_tick: u64,
    },
    /// Negative perturbation applied, waiting for observation window to elapse.
    WaitingMinus {
        saved_params: Vec<f64>,
        delta: Vec<f64>,
        y_plus: f64,
        applied_at_tick: u64,
    },
}

/// Result of one meta-learning step.
#[derive(Clone, Debug)]
pub struct MetaStepResult {
    /// Parameters that were adjusted: (name, old_value, new_value).
    pub adjustments: Vec<(String, f64, f64)>,
    /// Current free energy at time of step.
    pub free_energy: f64,
    /// Current generation (how many completed steps).
    pub generation: u64,
    /// Which phase produced this result.
    pub phase: MetaPhase,
}

/// The meta-learner: manages parameter tuning across ticks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MetaLearner {
    pub params: Vec<TunableParam>,
    pub phase: MetaPhase,
    pub generation: u64,
    pub tick_counter: u64,
    pub observation_window: u64,
    pub bo_budget: u64,
    spsa_state: SpsaState,
}

impl MetaLearner {
    /// Create a new meta-learner with default tunable parameters.
    pub fn new(observation_window: u64, bo_budget: u64) -> Self {
        Self {
            params: default_tunable_params(),
            phase: MetaPhase::ColdStart,
            generation: 0,
            tick_counter: 0,
            observation_window,
            bo_budget,
            spsa_state: SpsaState::Idle,
        }
    }

    /// Advance one meta-learning tick. Returns Some(result) if params changed.
    ///
    /// During ColdStart: records observation, explores with random perturbation.
    /// During OnlineTracking: runs SPSA state machine.
    pub fn step(
        &mut self,
        store: &CozoStore,
        current_free_energy: f64,
        current_beta: f64,
        current_surprise: f64,
    ) -> Result<Option<MetaStepResult>> {
        self.tick_counter += 1;

        // Always record the snapshot
        record_meta_snapshot(
            store,
            current_free_energy,
            current_beta,
            current_surprise,
            &self.params,
            self.generation,
            &self.phase,
        )?;

        match &self.phase {
            MetaPhase::ColdStart => {
                self.step_cold_start(store, current_free_energy)
            }
            MetaPhase::OnlineTracking => {
                self.step_spsa(store, current_free_energy)
            }
        }
    }

    /// Cold-start phase: explore parameter space with random perturbations.
    ///
    /// We don't use full Bayesian Optimization inline because egobox requires
    /// batch-style interaction. Instead, we use a simpler strategy:
    /// 1. Collect (params, free_energy) observations
    /// 2. Every `observation_window` ticks, apply a random perturbation
    /// 3. After `bo_budget` observations, find the best observed point and switch to SPSA
    fn step_cold_start(
        &mut self,
        store: &CozoStore,
        current_free_energy: f64,
    ) -> Result<Option<MetaStepResult>> {
        // Only perturb every `observation_window` ticks
        if self.tick_counter % self.observation_window != 0 {
            return Ok(None);
        }

        self.generation += 1;

        // Check if we should transition to online tracking
        if self.generation >= self.bo_budget {
            // Find best observed parameters from snapshots
            let best = find_best_observed_params(store)?;
            if let Some(best_params) = best {
                let adjustments = self.apply_param_values(&best_params);
                write_meta_params(store, &self.params, self.generation)?;
                self.phase = MetaPhase::OnlineTracking;
                self.spsa_state = SpsaState::Idle;
                save_optimizer_state(store, self)?;
                return Ok(Some(MetaStepResult {
                    adjustments,
                    free_energy: current_free_energy,
                    generation: self.generation,
                    phase: MetaPhase::OnlineTracking,
                }));
            }
        }

        // Random exploration: perturb each parameter by a random fraction of its step
        let adjustments = self.random_perturbation();
        write_meta_params(store, &self.params, self.generation)?;
        save_optimizer_state(store, self)?;

        Ok(Some(MetaStepResult {
            adjustments,
            free_energy: current_free_energy,
            generation: self.generation,
            phase: MetaPhase::ColdStart,
        }))
    }

    /// SPSA online tracking phase.
    ///
    /// State machine: Idle → WaitingPlus → WaitingMinus → Idle (with update).
    /// Each transition requires `observation_window` ticks to observe the effect.
    fn step_spsa(
        &mut self,
        store: &CozoStore,
        current_free_energy: f64,
    ) -> Result<Option<MetaStepResult>> {
        let state = self.spsa_state.clone();
        match state {
            SpsaState::Idle => {
                // Apply positive perturbation
                let saved: Vec<f64> = self.params.iter().map(|p| p.current).collect();
                let delta = bernoulli_perturbation(self.params.len());
                let c_k = spsa_c_k(self.generation);

                for (i, p) in self.params.iter_mut().enumerate() {
                    p.current = (p.current + c_k * delta[i] * p.step).clamp(p.min, p.max);
                }
                write_meta_params(store, &self.params, self.generation)?;

                self.spsa_state = SpsaState::WaitingPlus {
                    saved_params: saved,
                    delta,
                    applied_at_tick: self.tick_counter,
                };
                save_optimizer_state(store, self)?;
                Ok(None) // waiting for observation
            }
            SpsaState::WaitingPlus { saved_params, delta, applied_at_tick } => {
                // Wait for observation window
                if self.tick_counter - applied_at_tick < self.observation_window {
                    return Ok(None);
                }

                let y_plus = current_free_energy;

                // Apply negative perturbation
                let c_k = spsa_c_k(self.generation);
                for (i, p) in self.params.iter_mut().enumerate() {
                    p.current = (saved_params[i] - c_k * delta[i] * p.step).clamp(p.min, p.max);
                }
                write_meta_params(store, &self.params, self.generation)?;

                self.spsa_state = SpsaState::WaitingMinus {
                    saved_params,
                    delta,
                    y_plus,
                    applied_at_tick: self.tick_counter,
                };
                save_optimizer_state(store, self)?;
                Ok(None) // waiting for observation
            }
            SpsaState::WaitingMinus { saved_params, delta, y_plus, applied_at_tick } => {
                // Wait for observation window
                if self.tick_counter - applied_at_tick < self.observation_window {
                    return Ok(None);
                }

                let y_minus = current_free_energy;

                // Compute gradient estimate and update
                let a_k = spsa_a_k(self.generation);
                let c_k = spsa_c_k(self.generation);
                let g_hat = (y_plus - y_minus) / (2.0 * c_k);

                let mut adjustments = Vec::new();
                for (i, p) in self.params.iter_mut().enumerate() {
                    let old = saved_params[i];
                    let new_val = (old - a_k * g_hat * delta[i] * p.step).clamp(p.min, p.max);
                    if (new_val - old).abs() > 1e-10 {
                        adjustments.push((p.name.clone(), old, new_val));
                    }
                    p.current = new_val;
                }

                self.generation += 1;
                write_meta_params(store, &self.params, self.generation)?;
                self.spsa_state = SpsaState::Idle;
                save_optimizer_state(store, self)?;

                Ok(Some(MetaStepResult {
                    adjustments,
                    free_energy: (y_plus + y_minus) / 2.0,
                    generation: self.generation,
                    phase: MetaPhase::OnlineTracking,
                }))
            }
        }
    }

    /// Apply a random perturbation to each parameter (cold-start exploration).
    fn random_perturbation(&mut self) -> Vec<(String, f64, f64)> {
        let mut adjustments = Vec::new();
        for p in &mut self.params {
            let old = p.current;
            // Random perturbation: uniform in [-step, +step]
            let r: f64 = (random_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let new_val = (p.current + r * p.step).clamp(p.min, p.max);
            if (new_val - old).abs() > 1e-10 {
                adjustments.push((p.name.clone(), old, new_val));
            }
            p.current = new_val;
        }
        adjustments
    }

    /// Apply specific parameter values, returning the adjustments made.
    fn apply_param_values(&mut self, values: &[(String, f64)]) -> Vec<(String, f64, f64)> {
        let mut adjustments = Vec::new();
        for (name, value) in values {
            if let Some(p) = self.params.iter_mut().find(|p| p.name == *name) {
                let old = p.current;
                p.current = value.clamp(p.min, p.max);
                if (p.current - old).abs() > 1e-10 {
                    adjustments.push((p.name.clone(), old, p.current));
                }
            }
        }
        adjustments
    }
}

// ── SPSA gain sequences ──

/// Step-size sequence: a_k = a / (k + 1)^0.602
fn spsa_a_k(generation: u64) -> f64 {
    0.1 / (generation as f64 + 1.0).powf(0.602)
}

/// Perturbation magnitude: c_k = c / (k + 1)^0.101
fn spsa_c_k(generation: u64) -> f64 {
    0.05 / (generation as f64 + 1.0).powf(0.101)
}

/// Generate a Bernoulli ±1 perturbation vector.
fn bernoulli_perturbation(dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|_| if random_u64() % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// Simple random u64 using std (no external RNG crate needed).
fn random_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish()
}

// ── Default tunable parameters ──

/// The default set of parameters eligible for meta-learning.
pub fn default_tunable_params() -> Vec<TunableParam> {
    vec![
        TunableParam {
            name: "consolidation_threshold".into(),
            current: 2.0,
            min: 0.5,
            max: 5.0,
            step: 0.2,
        },
        TunableParam {
            name: "compression_memory_threshold".into(),
            current: 100.0,
            min: 20.0,
            max: 500.0,
            step: 20.0,
        },
        TunableParam {
            name: "telos_gamma".into(),
            current: 0.3,
            min: 0.0,
            max: 0.8,
            step: 0.05,
        },
        TunableParam {
            name: "activation_tau".into(),
            current: 86_400_000.0,
            min: 3_600_000.0,
            max: 604_800_000.0,
            step: 3_600_000.0,
        },
        TunableParam {
            name: "min_cluster_size".into(),
            current: 3.0,
            min: 2.0,
            max: 10.0,
            step: 1.0,
        },
        TunableParam {
            name: "max_distance".into(),
            current: 0.8,
            min: 0.4,
            max: 1.0,
            step: 0.05,
        },
        TunableParam {
            name: "promotion_threshold".into(),
            current: 3.0,
            min: 2.0,
            max: 8.0,
            step: 1.0,
        },
    ]
}

// ── CozoDB persistence ──

/// Record a meta-learning snapshot to the store.
fn record_meta_snapshot(
    store: &CozoStore,
    free_energy: f64,
    beta: f64,
    surprise: f64,
    params: &[TunableParam],
    generation: u64,
    phase: &MetaPhase,
) -> Result<()> {
    let params_json = serde_json::to_string(params)
        .map_err(|e| MemoriaError::Config(e.to_string()))?;
    let phase_str = match phase {
        MetaPhase::ColdStart => "cold_start",
        MetaPhase::OnlineTracking => "online_tracking",
    };
    let ts = now_ms();
    let id = uuid::Uuid::now_v7();
    let script = r#"
        ?[id, free_energy, beta, unresolved_surprise, params_json, generation, phase, ts] <- [[
            to_uuid($id),
            $free_energy,
            $beta,
            $surprise,
            $params_json,
            $generation,
            $phase,
            $ts,
        ]]
        :put meta_snapshots {
            id
            =>
            free_energy,
            beta,
            unresolved_surprise,
            params_json,
            generation,
            phase,
            ts,
        }
    "#;
    let mut params_map = std::collections::BTreeMap::new();
    params_map.insert("id".into(), cozo::DataValue::Str(id.to_string().into()));
    params_map.insert("free_energy".into(), cozo::DataValue::from(free_energy));
    params_map.insert("beta".into(), cozo::DataValue::from(beta));
    params_map.insert("surprise".into(), cozo::DataValue::from(surprise));
    params_map.insert("params_json".into(), cozo::DataValue::Str(params_json.into()));
    params_map.insert("generation".into(), cozo::DataValue::from(generation as i64));
    params_map.insert("phase".into(), cozo::DataValue::Str(phase_str.into()));
    params_map.insert("ts".into(), cozo::DataValue::from(ts));

    store.run_script(script, params_map)
        .map_err(|e| MemoriaError::Store(e.to_string()))?;
    Ok(())
}

/// Write current tunable parameter values to the store.
fn write_meta_params(
    store: &CozoStore,
    params: &[TunableParam],
    generation: u64,
) -> Result<()> {
    let ts = now_ms();
    for p in params {
        let script = r#"
            ?[name, value, min_bound, max_bound, step, generation, updated_at] <- [[
                $name, $value, $min, $max, $step, $generation, $ts,
            ]]
            :put meta_params {
                name
                =>
                value,
                min_bound,
                max_bound,
                step,
                generation,
                updated_at,
            }
        "#;
        let mut pm = std::collections::BTreeMap::new();
        pm.insert("name".into(), cozo::DataValue::Str(p.name.clone().into()));
        pm.insert("value".into(), cozo::DataValue::from(p.current));
        pm.insert("min".into(), cozo::DataValue::from(p.min));
        pm.insert("max".into(), cozo::DataValue::from(p.max));
        pm.insert("step".into(), cozo::DataValue::from(p.step));
        pm.insert("generation".into(), cozo::DataValue::from(generation as i64));
        pm.insert("ts".into(), cozo::DataValue::from(ts));

        store.run_script(script, pm)
            .map_err(|e| MemoriaError::Store(e.to_string()))?;
    }
    Ok(())
}

/// Load current tunable parameter values from the store.
pub fn load_meta_params(store: &CozoStore) -> Result<Vec<(String, f64)>> {
    let script = r#"
        ?[name, value] := *meta_params{name, value}
    "#;
    let result = store.run_query(script, std::collections::BTreeMap::new())
        .map_err(|e| MemoriaError::Store(e.to_string()))?;

    let mut params = Vec::new();
    for row in &result.rows {
        if let (Some(name), Some(value)) = (row[0].get_str(), row[1].get_float()) {
            params.push((name.to_string(), value));
        }
    }
    Ok(params)
}

/// Find the parameter values that produced the lowest free energy.
fn find_best_observed_params(store: &CozoStore) -> Result<Option<Vec<(String, f64)>>> {
    let script = r#"
        ?[params_json, free_energy] := *meta_snapshots{params_json, free_energy}
        :order free_energy
        :limit 1
    "#;
    let result = store.run_query(script, std::collections::BTreeMap::new())
        .map_err(|e| MemoriaError::Store(e.to_string()))?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    let params_json = result.rows[0][0].get_str()
        .ok_or_else(|| MemoriaError::Config("no params_json".into()))?;
    let params: Vec<TunableParam> = serde_json::from_str(params_json)
        .map_err(|e| MemoriaError::Config(e.to_string()))?;

    Ok(Some(
        params.iter().map(|p| (p.name.clone(), p.current)).collect()
    ))
}

/// Save the full optimizer state (for restart recovery).
fn save_optimizer_state(store: &CozoStore, learner: &MetaLearner) -> Result<()> {
    let state_json = serde_json::to_string(learner)
        .map_err(|e| MemoriaError::Config(e.to_string()))?;
    let phase_str = match learner.phase {
        MetaPhase::ColdStart => "cold_start",
        MetaPhase::OnlineTracking => "online_tracking",
    };
    let ts = now_ms();

    let script = r#"
        ?[id, phase, state_json, generation, updated_at] <- [[
            0, $phase, $state_json, $generation, $ts,
        ]]
        :put meta_optimizer {
            id
            =>
            phase,
            state_json,
            generation,
            updated_at,
        }
    "#;
    let mut pm = std::collections::BTreeMap::new();
    pm.insert("phase".into(), cozo::DataValue::Str(phase_str.into()));
    pm.insert("state_json".into(), cozo::DataValue::Str(state_json.into()));
    pm.insert("generation".into(), cozo::DataValue::from(learner.generation as i64));
    pm.insert("ts".into(), cozo::DataValue::from(ts));

    store.run_script(script, pm)
        .map_err(|e| MemoriaError::Store(e.to_string()))?;
    Ok(())
}

/// Load the optimizer state from the store (for restart recovery).
pub fn load_optimizer_state(store: &CozoStore) -> Result<Option<MetaLearner>> {
    let script = r#"
        ?[state_json] := *meta_optimizer{id: 0, state_json}
    "#;
    let result = store.run_query(script, std::collections::BTreeMap::new());

    match result {
        Ok(r) if !r.rows.is_empty() => {
            let json = r.rows[0][0].get_str()
                .ok_or_else(|| MemoriaError::Config("no state_json".into()))?;
            let learner: MetaLearner = serde_json::from_str(json)
                .map_err(|e| MemoriaError::Config(format!("deserialize MetaLearner: {e}")))?;
            Ok(Some(learner))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_have_valid_bounds() {
        let params = default_tunable_params();
        for p in &params {
            assert!(p.min < p.max, "{}: min >= max", p.name);
            assert!(p.current >= p.min, "{}: current < min", p.name);
            assert!(p.current <= p.max, "{}: current > max", p.name);
            assert!(p.step > 0.0, "{}: step <= 0", p.name);
        }
    }

    #[test]
    fn test_spsa_gain_sequences_decay() {
        let a0 = spsa_a_k(0);
        let a100 = spsa_a_k(100);
        assert!(a0 > a100, "a_k should decay over generations");

        let c0 = spsa_c_k(0);
        let c100 = spsa_c_k(100);
        assert!(c0 > c100, "c_k should decay over generations");
    }

    #[test]
    fn test_bernoulli_perturbation_is_plus_minus_one() {
        let delta = bernoulli_perturbation(10);
        assert_eq!(delta.len(), 10);
        for d in &delta {
            assert!((*d - 1.0).abs() < 1e-10 || (*d + 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_meta_learner_new() {
        let learner = MetaLearner::new(5, 150);
        assert_eq!(learner.phase, MetaPhase::ColdStart);
        assert_eq!(learner.generation, 0);
        assert_eq!(learner.params.len(), 7);
    }
}
