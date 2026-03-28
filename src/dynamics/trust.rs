//! Cross-namespace trust computation for multi-agent scenarios.
//!
//! When agent B queries agent A's knowledge (via scope grants), the retrieved
//! memories need a trust multiplier based on:
//! 1. Provenance chain quality (how many hops from direct observation)
//! 2. Historical accuracy (success rate of tasks using that agent's memories)
//! 3. Scope grant relationship (explicitly trusted vs transitive)

use crate::error::Result;
use crate::store::CozoStore;

/// Cross-agent trust score with contributing factors.
#[derive(Debug, Clone)]
pub struct TrustScore {
    pub trust: f64,
    pub provenance_factor: f64,
    pub accuracy_factor: f64,
    pub grant_factor: f64,
}

/// Compute trust score for memories from a provider namespace viewed by a requester.
///
/// Returns a multiplier in [0.0, 1.0] to apply to cross-namespace memory confidence.
pub fn compute_cross_agent_trust(
    store: &CozoStore,
    requester_agent: &str,
    provider_namespace: &str,
) -> Result<TrustScore> {
    // 1. Grant factor: does a direct scope grant exist?
    let grant_factor = compute_grant_factor(store, requester_agent, provider_namespace)?;

    // 2. Provenance factor: average provenance quality of provider's memories
    let provenance_factor = compute_provenance_factor(store, provider_namespace)?;

    // 3. Accuracy factor: historical task success rate involving provider's memories
    let accuracy_factor = compute_accuracy_factor(store, provider_namespace)?;

    // Combined trust = weighted product
    let trust = (grant_factor * 0.4 + provenance_factor * 0.3 + accuracy_factor * 0.3)
        .clamp(0.0, 1.0);

    Ok(TrustScore {
        trust,
        provenance_factor,
        accuracy_factor,
        grant_factor,
    })
}

/// Check if a direct scope grant exists between requester and provider namespace.
fn compute_grant_factor(
    store: &CozoStore,
    requester_agent: &str,
    provider_namespace: &str,
) -> Result<f64> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("agent".into(), DataValue::from(requester_agent));
    params.insert("ns".into(), DataValue::from(provider_namespace));

    let result = store.run_query(
        r#"?[count(id)] := *scope_grants{id, agent_pattern, namespace_pattern},
            agent_pattern = $agent,
            namespace_pattern = $ns"#,
        params,
    )?;

    let count = result.rows.first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0);

    // Direct grant → high trust, no grant → low baseline
    Ok(if count > 0 { 0.9 } else { 0.3 })
}

/// Average provenance quality of memories in a namespace.
fn compute_provenance_factor(
    store: &CozoStore,
    namespace: &str,
) -> Result<f64> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("ns".into(), DataValue::from(namespace));

    // Count memories by provenance type
    let result = store.run_query(
        r#"?[provenance, count(id)] := *memories{id, namespace, provenance}, namespace = $ns"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Ok(0.5); // no data → neutral trust
    }

    let mut total_weight = 0.0;
    let mut total_count = 0i64;

    for row in &result.rows {
        let prov = row[0].get_str().unwrap_or("unknown");
        let count = row[1].get_int().unwrap_or(0);
        let weight = crate::dynamics::surprise::provenance_weight(prov);
        total_weight += weight * count as f64;
        total_count += count;
    }

    if total_count == 0 {
        return Ok(0.5);
    }

    Ok(total_weight / total_count as f64)
}

/// Historical accuracy of tasks that used memories from this namespace.
fn compute_accuracy_factor(
    store: &CozoStore,
    namespace: &str,
) -> Result<f64> {
    use cozo::DataValue;
    use std::collections::BTreeMap;

    let mut params = BTreeMap::new();
    params.insert("ns".into(), DataValue::from(namespace));

    // Look at task outcomes for this agent/namespace
    let result = store.run_query(
        r#"?[outcome, count(task_id)] := *task_outcomes{task_id, outcome, agent_id},
            agent_id = $ns"#,
        params,
    )?;

    let mut successes = 0i64;
    let mut total = 0i64;

    for row in &result.rows {
        let outcome = row[0].get_str().unwrap_or("");
        let count = row[1].get_int().unwrap_or(0);
        if outcome == "success" {
            successes += count;
        }
        total += count;
    }

    if total == 0 {
        return Ok(0.5); // no history → neutral
    }

    Ok(successes as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_default_no_data() {
        let store = CozoStore::open_mem(4).unwrap();
        let score = compute_cross_agent_trust(&store, "agent-b", "agent-a-ns").unwrap();
        // No grants, no memories, no history → baseline trust
        assert!(score.trust > 0.0 && score.trust < 1.0);
        assert_eq!(score.grant_factor, 0.3); // no grant
    }
}
