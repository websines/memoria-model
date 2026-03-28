//! Provenance-based confidence decay.
//!
//! Different provenance types decay at different rates:
//! - "direct" (user typed it): τ = 1 year
//! - "user_stated": τ = 6 months
//! - "extracted" (NER/LLM pulled it out): τ = 3 months
//! - "inferred" (derived from other facts): τ = 1 month
//!
//! For facts, reinforcement slows decay: effective_τ = τ × (1 + ln(rc))

use std::collections::BTreeMap;

use cozo::DataValue;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// Time constant (τ) in milliseconds for each provenance type.
pub fn provenance_tau(provenance: &str) -> f64 {
    match provenance {
        "direct" => 31_536_000_000.0,     // 1 year
        "user_stated" => 15_768_000_000.0, // 6 months
        "extracted" => 7_884_000_000.0,    // 3 months
        "inferred" => 2_628_000_000.0,     // 1 month
        _ => 7_884_000_000.0,              // default: 3 months
    }
}

/// Compute effective confidence for a memory at the current time.
///
/// confidence_eff = initial × exp(-age / τ)
pub fn effective_confidence(
    initial_confidence: f64,
    provenance: &str,
    created_at: i64,
    now: i64,
) -> f64 {
    let tau = provenance_tau(provenance);
    let age = (now - created_at).max(0) as f64;
    initial_confidence * (-age / tau).exp()
}

/// Compute effective confidence for a fact, where reinforcement slows decay.
///
/// effective_τ = τ × (1 + ln(rc))
/// confidence_eff = initial × exp(-age / effective_τ)
pub fn effective_fact_confidence(
    initial_confidence: f64,
    provenance: &str,
    reinforcement_count: i64,
    created_at: i64,
    now: i64,
) -> f64 {
    let tau = provenance_tau(provenance);
    let rc = reinforcement_count.max(1) as f64;
    let effective_tau = tau * (1.0 + rc.ln());
    let age = (now - created_at).max(0) as f64;
    initial_confidence * (-age / effective_tau).exp()
}

/// Resolve the provenance chain for a memory by walking `source_ids` recursively.
///
/// Returns (id, raw_confidence) pairs for each memory in the chain.
/// Uses cycle detection via HashSet and limits depth to 10 to prevent infinite loops.
pub fn resolve_provenance_chain(
    store: &CozoStore,
    memory_id: uuid::Uuid,
) -> Result<Vec<(uuid::Uuid, f64)>> {
    use std::collections::HashSet;

    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    let mut frontier = vec![memory_id];
    let max_depth = 10;
    let mut depth = 0;

    while !frontier.is_empty() && depth < max_depth {
        let mut next_frontier = Vec::new();
        for id in &frontier {
            if !visited.insert(*id) {
                continue;
            }
            if let Some(mem) = store.get_memory(*id)? {
                // Don't include the root memory itself in the chain
                if *id != memory_id {
                    chain.push((*id, mem.confidence));
                }
                for source_id in &mem.source_ids {
                    if !visited.contains(source_id) {
                        next_frontier.push(*source_id);
                    }
                }
            }
        }
        frontier = next_frontier;
        depth += 1;
    }

    Ok(chain)
}

/// Compute effective confidence with weakest-source semantics.
///
/// The final effective confidence is: `min(own_effective, weakest_in_chain)`
/// This prevents derived memories from exceeding their source's confidence.
pub fn effective_confidence_with_chain(
    own_confidence: f64,
    provenance: &str,
    created_at: i64,
    now: i64,
    chain_confidences: &[f64],
) -> f64 {
    let own_eff = effective_confidence(own_confidence, provenance, created_at, now);
    if chain_confidences.is_empty() {
        return own_eff;
    }
    let weakest = chain_confidences
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    own_eff.min(weakest)
}

// ── Telos confidence decay ──

/// Time constant (τ) in milliseconds for each telos provenance type.
///
/// Enterprise goals never decay (effectively infinite τ).
/// Attention count slows decay: effective_τ = τ × (1 + ln(max(attention_count, 1)))
pub fn telos_provenance_tau(provenance: &str) -> f64 {
    match provenance {
        "user_stated" => 31_536_000_000.0,    // 1 year
        "enterprise" => f64::INFINITY,          // never decays
        "decomposition" => 15_768_000_000.0,   // 6 months
        "agent_proposed" => 7_884_000_000.0,   // 3 months
        "inferred" => 2_628_000_000.0,         // 1 month
        _ => 7_884_000_000.0,                  // default: 3 months
    }
}

/// Compute effective confidence for a telos at the current time.
///
/// Uses exponential decay with provenance-specific τ, slowed by attention:
/// `effective_τ = τ × (1 + ln(max(attention_count, 1)))`
/// `confidence_eff = initial × exp(-age / effective_τ)`
///
/// Enterprise goals return `initial` unchanged (infinite τ).
pub fn effective_telos_confidence(
    initial_confidence: f64,
    provenance: &str,
    created_at: i64,
    now: i64,
    attention_count: i64,
) -> f64 {
    let tau = telos_provenance_tau(provenance);
    if tau.is_infinite() {
        return initial_confidence;
    }
    let ac = attention_count.max(1) as f64;
    let effective_tau = tau * (1.0 + ac.ln());
    let age = (now - created_at).max(0) as f64;
    initial_confidence * (-age / effective_tau).exp()
}

/// Compute effective confidence for all active telos in a namespace.
///
/// Returns (telos_id, effective_confidence) pairs.
pub fn compute_all_telos_confidences(
    store: &CozoStore,
    namespace: &str,
    now: Option<i64>,
) -> Result<Vec<(uuid::Uuid, f64)>> {
    let now = now.unwrap_or_else(now_ms);

    let mut params = BTreeMap::new();
    params.insert("namespace".into(), DataValue::from(namespace));

    let result = store.run_query(
        r#"?[id, confidence, provenance, created_at, attention_count] :=
            *telos{id, confidence, provenance, created_at, attention_count, status, namespace},
            namespace = $namespace,
            status != "completed",
            status != "failed",
            status != "abandoned""#,
        params,
    )?;

    let mut out = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        let id = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let conf = row[1].get_float().unwrap_or(1.0);
        let prov = row[2].get_str().unwrap_or("user_stated");
        let created = row[3].get_int().unwrap_or(0);
        let ac = row[4].get_int().unwrap_or(0);

        let eff = effective_telos_confidence(conf, prov, created, now, ac);
        out.push((id, eff));
    }

    Ok(out)
}

/// Compute effective confidence for all memories in the store via Datalog.
///
/// Returns (memory_id, effective_confidence) pairs.
pub fn compute_all_effective_confidences(
    store: &CozoStore,
    now: Option<i64>,
) -> Result<Vec<(uuid::Uuid, f64)>> {
    let now = now.unwrap_or_else(now_ms);

    let mut params = BTreeMap::new();
    params.insert("now".into(), DataValue::from(now));

    // Compute in Rust because CozoDB doesn't have a match/case expression for provenance_tau.
    // We fetch raw data and compute client-side.
    let result = store.run_query(
        r#"?[id, confidence, provenance, created_at] :=
            *memories{id, confidence, provenance, created_at}"#,
        params,
    )?;

    let mut out = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        let id = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let conf = row[1].get_float().unwrap_or(1.0);
        let prov = row[2].get_str().unwrap_or("direct");
        let created = row[3].get_int().unwrap_or(0);

        let eff = effective_confidence(conf, prov, created, now);
        out.push((id, eff));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_tau_ordering() {
        assert!(provenance_tau("direct") > provenance_tau("user_stated"));
        assert!(provenance_tau("user_stated") > provenance_tau("extracted"));
        assert!(provenance_tau("extracted") > provenance_tau("inferred"));
    }

    #[test]
    fn test_no_decay_at_creation() {
        let conf = effective_confidence(1.0, "direct", 1000, 1000);
        assert!((conf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_over_time() {
        let now = 1000 + 31_536_000_000; // 1 year later
        let conf = effective_confidence(1.0, "direct", 1000, now);
        // After 1 τ, should be ~0.368
        assert!((conf - (-1.0_f64).exp()).abs() < 0.01);
    }

    #[test]
    fn test_inferred_decays_faster() {
        let now = 1000 + 2_628_000_000; // 1 month later
        let direct = effective_confidence(1.0, "direct", 1000, now);
        let inferred = effective_confidence(1.0, "inferred", 1000, now);
        assert!(
            direct > inferred,
            "direct ({direct}) should decay slower than inferred ({inferred})"
        );
    }

    #[test]
    fn test_reinforcement_slows_fact_decay() {
        let now = 1000 + 7_884_000_000; // 3 months later
        let rc1 = effective_fact_confidence(1.0, "extracted", 1, 1000, now);
        let rc5 = effective_fact_confidence(1.0, "extracted", 5, 1000, now);
        assert!(
            rc5 > rc1,
            "higher reinforcement ({rc5}) should decay slower than lower ({rc1})"
        );
    }

    #[test]
    fn test_reinforcement_count_1_equals_basic() {
        let now = 1000 + 5_000_000_000;
        let basic = effective_confidence(1.0, "extracted", 1000, now);
        let fact = effective_fact_confidence(1.0, "extracted", 1, 1000, now);
        // rc=1 → ln(1)=0 → effective_tau = tau × 1 = tau → same as basic
        assert!((basic - fact).abs() < 1e-10);
    }

    // ── Telos confidence tests ──

    #[test]
    fn test_telos_provenance_tau_ordering() {
        assert!(telos_provenance_tau("enterprise").is_infinite());
        assert!(telos_provenance_tau("user_stated") > telos_provenance_tau("decomposition"));
        assert!(telos_provenance_tau("decomposition") > telos_provenance_tau("agent_proposed"));
        assert!(telos_provenance_tau("agent_proposed") > telos_provenance_tau("inferred"));
    }

    #[test]
    fn test_enterprise_telos_never_decays() {
        let now = 1000 + 100_000_000_000; // ~3 years later
        let conf = effective_telos_confidence(1.0, "enterprise", 1000, now, 0);
        assert!((conf - 1.0).abs() < 1e-10, "enterprise should never decay");
    }

    #[test]
    fn test_telos_no_decay_at_creation() {
        let conf = effective_telos_confidence(1.0, "user_stated", 1000, 1000, 1);
        assert!((conf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_telos_decay_over_time() {
        let now = 1000 + 31_536_000_000; // 1 year later
        let conf = effective_telos_confidence(1.0, "user_stated", 1000, now, 1);
        // user_stated τ = 1 year, ac=1 → ln(1)=0 → effective_τ = τ
        // After 1 τ: ~0.368
        assert!((conf - (-1.0_f64).exp()).abs() < 0.01);
    }

    #[test]
    fn test_telos_attention_slows_decay() {
        let now = 1000 + 7_884_000_000; // 3 months
        let low_attention = effective_telos_confidence(1.0, "agent_proposed", 1000, now, 1);
        let high_attention = effective_telos_confidence(1.0, "agent_proposed", 1000, now, 10);
        assert!(
            high_attention > low_attention,
            "high_attention={high_attention} should > low_attention={low_attention}"
        );
    }

    #[test]
    fn test_telos_inferred_decays_fastest() {
        let now = 1000 + 2_628_000_000; // 1 month
        let user = effective_telos_confidence(1.0, "user_stated", 1000, now, 1);
        let inferred = effective_telos_confidence(1.0, "inferred", 1000, now, 1);
        assert!(user > inferred);
    }
}
