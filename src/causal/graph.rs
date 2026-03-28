//! Structural causal graph — persistent causal edges with Bayesian accumulation.
//!
//! Causal edges are accumulated from three sources:
//! 1. **RecallAttribution** — counterfactual attribution results
//! 2. **LlmProposed** — causal relationships proposed by LLM during reflection
//! 3. **NotearsDiscovered** — structure learning from observation data
//!
//! Each edge stores a Bayesian-updated causal strength that reflects
//! accumulated evidence. The graph can be loaded into petgraph for
//! efficient d-separation queries and structural interventions.

use std::collections::BTreeMap;

use cozo::DataValue;
use petgraph::graph::{DiGraph, NodeIndex};
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;

/// How a causal edge was discovered.
#[derive(Debug, Clone, PartialEq)]
pub enum CausalMechanism {
    /// From counterfactual attribution analysis.
    RecallAttribution,
    /// Proposed by LLM during reflection.
    LlmProposed,
    /// Discovered by NOTEARS structure learning.
    NotearsDiscovered,
    /// Shared across instances via NATS.
    EnterpriseShared,
}

impl CausalMechanism {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RecallAttribution => "recall_attribution",
            Self::LlmProposed => "llm_proposed",
            Self::NotearsDiscovered => "notears_discovered",
            Self::EnterpriseShared => "enterprise_shared",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "recall_attribution" => Self::RecallAttribution,
            "llm_proposed" => Self::LlmProposed,
            "notears_discovered" => Self::NotearsDiscovered,
            "enterprise_shared" => Self::EnterpriseShared,
            _ => Self::RecallAttribution,
        }
    }
}

/// A directed causal edge: cause → effect.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub cause_id: Uuid,
    pub effect_id: Uuid,
    pub causal_strength: f64,
    pub observations: u64,
    pub last_observed: i64,
    pub mechanism: CausalMechanism,
    pub confidence: f64,
    pub namespace: String,
}

/// A loaded causal graph with petgraph representation and UUID↔NodeIndex mapping.
pub struct CausalGraph {
    pub graph: DiGraph<Uuid, f64>,
    pub node_map: BTreeMap<Uuid, NodeIndex>,
}

impl CausalGraph {
    /// Look up or insert a node for a given UUID.
    pub fn get_or_insert(&mut self, id: Uuid) -> NodeIndex {
        *self.node_map.entry(id).or_insert_with(|| self.graph.add_node(id))
    }

    /// Get the NodeIndex for a UUID, if it exists.
    pub fn get_node(&self, id: &Uuid) -> Option<NodeIndex> {
        self.node_map.get(id).copied()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

/// Accumulate a causal edge using Bayesian update.
///
/// If the edge already exists, update its strength using:
///   posterior = (prior × observations + likelihood) / (observations + 1)
///   observations += 1
///
/// If the edge is new, insert with the provided strength as prior.
pub fn accumulate_causal_edge(store: &CozoStore, edge: &CausalEdge) -> Result<()> {
    let mut params = BTreeMap::new();
    params.insert("cause".into(), DataValue::from(edge.cause_id.to_string()));
    params.insert("effect".into(), DataValue::from(edge.effect_id.to_string()));

    // Check for existing edge
    let existing = store.run_script(
        r#"?[strength, obs, mechanism, confidence, namespace] :=
            *causal_edges{cause_id, effect_id, causal_strength: strength,
                          observations: obs, mechanism, confidence, namespace, @ 'NOW'},
            cause_id = to_uuid($cause),
            effect_id = to_uuid($effect)"#,
        params.clone(),
    )?;

    if let Some(row) = existing.rows.first() {
        // Bayesian update
        let prior = row[0].get_float().unwrap_or(0.5);
        let obs = row[1].get_int().unwrap_or(1) as u64;
        let old_confidence = row[3].get_float().unwrap_or(0.5);
        let namespace = row[4].get_str().unwrap_or("").to_string();

        let likelihood = edge.causal_strength;
        let posterior = (prior * obs as f64 + likelihood) / (obs as f64 + 1.0);
        let new_obs = obs + 1;

        // Confidence increases with observations, capped at 0.95
        let new_confidence = (old_confidence + (1.0 - old_confidence) * 0.1).min(0.95);

        let mut put_params = BTreeMap::new();
        put_params.insert("cause".into(), DataValue::from(edge.cause_id.to_string()));
        put_params.insert("effect".into(), DataValue::from(edge.effect_id.to_string()));
        put_params.insert("strength".into(), DataValue::from(posterior));
        put_params.insert("obs".into(), DataValue::from(new_obs as i64));
        put_params.insert("last_observed".into(), DataValue::from(edge.last_observed));
        put_params.insert(
            "mechanism".into(),
            DataValue::from(edge.mechanism.as_str()),
        );
        put_params.insert("confidence".into(), DataValue::from(new_confidence));
        put_params.insert("namespace".into(), DataValue::from(namespace.as_str()));

        store.run_script(
            r#"?[cause_id, effect_id, valid_at, causal_strength, observations,
               last_observed, mechanism, confidence, namespace] <-
              [[$cause, $effect, 'ASSERT', $strength, $obs,
                $last_observed, $mechanism, $confidence, $namespace]]
            :put causal_edges {cause_id, effect_id, valid_at =>
                 causal_strength, observations, last_observed, mechanism,
                 confidence, namespace}"#,
            put_params,
        )?;
    } else {
        // New edge — insert with initial values
        params.insert(
            "strength".into(),
            DataValue::from(edge.causal_strength),
        );
        params.insert("obs".into(), DataValue::from(edge.observations as i64));
        params.insert(
            "last_observed".into(),
            DataValue::from(edge.last_observed),
        );
        params.insert(
            "mechanism".into(),
            DataValue::from(edge.mechanism.as_str()),
        );
        params.insert("confidence".into(), DataValue::from(edge.confidence));
        params.insert(
            "namespace".into(),
            DataValue::from(edge.namespace.as_str()),
        );

        store.run_script(
            r#"?[cause_id, effect_id, valid_at, causal_strength, observations,
               last_observed, mechanism, confidence, namespace] <-
              [[$cause, $effect, 'ASSERT', $strength, $obs,
                $last_observed, $mechanism, $confidence, $namespace]]
            :put causal_edges {cause_id, effect_id, valid_at =>
                 causal_strength, observations, last_observed, mechanism,
                 confidence, namespace}"#,
            params,
        )?;
    }

    Ok(())
}

/// Load the causal graph from CozoDB into a petgraph DiGraph.
///
/// Only loads edges above `min_confidence` threshold.
/// Returns the graph and a UUID↔NodeIndex mapping.
pub fn load_causal_graph(
    store: &CozoStore,
    namespace: &str,
    min_confidence: f64,
) -> Result<CausalGraph> {
    let mut params = BTreeMap::new();
    params.insert("namespace".into(), DataValue::from(namespace));
    params.insert("min_conf".into(), DataValue::from(min_confidence));

    let result = store.run_script(
        r#"?[cause_id, effect_id, causal_strength] :=
            *causal_edges{cause_id, effect_id, causal_strength, confidence,
                          namespace, @ 'NOW'},
            confidence >= $min_conf,
            namespace = $namespace"#,
        params,
    )?;

    let mut cg = CausalGraph {
        graph: DiGraph::new(),
        node_map: BTreeMap::new(),
    };

    for row in &result.rows {
        let cause_id = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let effect_id = crate::store::cozo::parse_uuid_pub(&row[1])?;
        let strength = row[2].get_float().unwrap_or(0.5);

        let cause_idx = cg.get_or_insert(cause_id);
        let effect_idx = cg.get_or_insert(effect_id);
        cg.graph.add_edge(cause_idx, effect_idx, strength);
    }

    Ok(cg)
}

/// Load all causal edges as structs (for analysis, NOTEARS cross-referencing).
pub fn load_all_edges(store: &CozoStore, namespace: &str) -> Result<Vec<CausalEdge>> {
    let mut params = BTreeMap::new();
    params.insert("namespace".into(), DataValue::from(namespace));

    let result = store.run_script(
        r#"?[cause_id, effect_id, causal_strength, observations,
           last_observed, mechanism, confidence] :=
            *causal_edges{cause_id, effect_id, causal_strength, observations,
                          last_observed, mechanism, confidence, namespace, @ 'NOW'},
            namespace = $namespace"#,
        params,
    )?;

    let mut edges = Vec::new();
    for row in &result.rows {
        edges.push(CausalEdge {
            cause_id: crate::store::cozo::parse_uuid_pub(&row[0])?,
            effect_id: crate::store::cozo::parse_uuid_pub(&row[1])?,
            causal_strength: row[2].get_float().unwrap_or(0.5),
            observations: row[3].get_int().unwrap_or(1) as u64,
            last_observed: row[4].get_int().unwrap_or(0),
            mechanism: CausalMechanism::from_str(row[5].get_str().unwrap_or("unknown")),
            confidence: row[6].get_float().unwrap_or(0.5),
            namespace: namespace.to_string(),
        });
    }

    Ok(edges)
}

/// Get causal ancestors of a node (what causes it?).
///
/// Performs iterative BFS over the causal graph, accumulating
/// multiplicative strength along causal chains. Returns
/// (ancestor_id, aggregated_strength) sorted by strength descending.
pub fn get_causal_ancestors(
    store: &CozoStore,
    target_id: Uuid,
    namespace: &str,
    limit: usize,
) -> Result<Vec<(Uuid, f64)>> {
    let mut ancestors: BTreeMap<Uuid, f64> = BTreeMap::new();
    let mut frontier = vec![(target_id, 1.0_f64)];
    let mut visited = std::collections::HashSet::new();
    visited.insert(target_id);

    // BFS up the causal graph
    while let Some((node, path_strength)) = frontier.pop() {
        let mut params = BTreeMap::new();
        params.insert("effect".into(), DataValue::from(node.to_string()));
        params.insert("ns".into(), DataValue::from(namespace));

        let result = store.run_script(
            r#"?[cause_id, causal_strength] :=
                *causal_edges{cause_id, effect_id, causal_strength, namespace: ns, @ 'NOW'},
                effect_id = to_uuid($effect),
                ns = $ns"#,
            params,
        )?;

        for row in &result.rows {
            let cause = crate::store::cozo::parse_uuid_pub(&row[0])?;
            let strength = row[1].get_float().unwrap_or(0.0);
            let accumulated = path_strength * strength;

            if accumulated > 0.1 && !visited.contains(&cause) {
                // Keep the max strength path
                let entry = ancestors.entry(cause).or_insert(0.0);
                if accumulated > *entry {
                    *entry = accumulated;
                }
                visited.insert(cause);
                frontier.push((cause, accumulated));
            }
        }
    }

    let mut result: Vec<(Uuid, f64)> = ancestors.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result.truncate(limit);
    Ok(result)
}

/// Find common confounders of two nodes.
///
/// Returns (confounder_id, strength_to_a, strength_to_b) sorted by combined strength.
pub fn get_common_confounders(
    store: &CozoStore,
    a: Uuid,
    b: Uuid,
    namespace: &str,
) -> Result<Vec<(Uuid, f64, f64)>> {
    // Get direct causes of a
    let mut params_a = BTreeMap::new();
    params_a.insert("effect".into(), DataValue::from(a.to_string()));
    params_a.insert("ns".into(), DataValue::from(namespace));

    let causes_a = store.run_script(
        r#"?[cause_id, causal_strength] :=
            *causal_edges{cause_id, effect_id, causal_strength, namespace: ns, @ 'NOW'},
            effect_id = to_uuid($effect),
            ns = $ns"#,
        params_a,
    )?;

    let mut a_causes: BTreeMap<Uuid, f64> = BTreeMap::new();
    for row in &causes_a.rows {
        let cause = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let strength = row[1].get_float().unwrap_or(0.0);
        a_causes.insert(cause, strength);
    }

    // Get direct causes of b
    let mut params_b = BTreeMap::new();
    params_b.insert("effect".into(), DataValue::from(b.to_string()));
    params_b.insert("ns".into(), DataValue::from(namespace));

    let causes_b = store.run_script(
        r#"?[cause_id, causal_strength] :=
            *causal_edges{cause_id, effect_id, causal_strength, namespace: ns, @ 'NOW'},
            effect_id = to_uuid($effect),
            ns = $ns"#,
        params_b,
    )?;

    // Find common causes
    let mut confounders = Vec::new();
    for row in &causes_b.rows {
        let cause = crate::store::cozo::parse_uuid_pub(&row[0])?;
        let s_b = row[1].get_float().unwrap_or(0.0);

        if let Some(&s_a) = a_causes.get(&cause) {
            confounders.push((cause, s_a, s_b));
        }
    }

    // Sort by combined strength
    confounders.sort_by(|x, y| {
        (y.1 + y.2)
            .partial_cmp(&(x.1 + x.2))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(confounders)
}

/// Boost or reduce confidence of an edge when NOTEARS validates or contradicts it.
pub fn adjust_edge_confidence(
    store: &CozoStore,
    cause_id: Uuid,
    effect_id: Uuid,
    confirmed: bool,
) -> Result<()> {
    let mut params = BTreeMap::new();
    params.insert("cause".into(), DataValue::from(cause_id.to_string()));
    params.insert("effect".into(), DataValue::from(effect_id.to_string()));

    let existing = store.run_script(
        r#"?[strength, obs, last_observed, mechanism, confidence, namespace] :=
            *causal_edges{cause_id, effect_id, causal_strength: strength,
                          observations: obs, last_observed, mechanism,
                          confidence, namespace, @ 'NOW'},
            cause_id = to_uuid($cause),
            effect_id = to_uuid($effect)"#,
        params.clone(),
    )?;

    if let Some(row) = existing.rows.first() {
        let strength = row[0].get_float().unwrap_or(0.5);
        let obs = row[1].get_int().unwrap_or(1);
        let last_observed = row[2].get_int().unwrap_or(0);
        let mechanism = row[3].get_str().unwrap_or("unknown").to_string();
        let old_confidence = row[4].get_float().unwrap_or(0.5);
        let namespace = row[5].get_str().unwrap_or("").to_string();

        let new_confidence = if confirmed {
            (old_confidence + (1.0 - old_confidence) * 0.2).min(0.95)
        } else {
            (old_confidence * 0.7).max(0.05)
        };

        params.insert("strength".into(), DataValue::from(strength));
        params.insert("obs".into(), DataValue::from(obs));
        params.insert("last_observed".into(), DataValue::from(last_observed));
        params.insert("mechanism".into(), DataValue::from(mechanism.as_str()));
        params.insert("confidence".into(), DataValue::from(new_confidence));
        params.insert("namespace".into(), DataValue::from(namespace.as_str()));

        store.run_script(
            r#"?[cause_id, effect_id, valid_at, causal_strength, observations,
               last_observed, mechanism, confidence, namespace] <-
              [[$cause, $effect, 'ASSERT', $strength, $obs,
                $last_observed, $mechanism, $confidence, $namespace]]
            :put causal_edges {cause_id, effect_id, valid_at =>
                 causal_strength, observations, last_observed, mechanism,
                 confidence, namespace}"#,
            params,
        )?;
    }

    Ok(())
}

/// Count total causal edges in the store.
pub fn count_edges(store: &CozoStore, namespace: &str) -> Result<usize> {
    let mut params = BTreeMap::new();
    params.insert("namespace".into(), DataValue::from(namespace));

    let result = store.run_script(
        r#"?[count(cause_id)] :=
            *causal_edges{cause_id, effect_id, namespace, @ 'NOW'},
            namespace = $namespace"#,
        params,
    )?;

    Ok(result
        .rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::now_ms;

    #[test]
    fn test_accumulate_new_edge() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();

        let edge = CausalEdge {
            cause_id: a,
            effect_id: b,
            causal_strength: 0.8,
            observations: 1,
            last_observed: now_ms(),
            mechanism: CausalMechanism::RecallAttribution,
            confidence: 0.5,
            namespace: String::new(),
        };

        accumulate_causal_edge(&store, &edge).unwrap();
        assert_eq!(count_edges(&store, "").unwrap(), 1);
    }

    #[test]
    fn test_bayesian_update() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let ts = now_ms();

        // First observation: strength 0.8
        let edge1 = CausalEdge {
            cause_id: a,
            effect_id: b,
            causal_strength: 0.8,
            observations: 1,
            last_observed: ts,
            mechanism: CausalMechanism::RecallAttribution,
            confidence: 0.5,
            namespace: String::new(),
        };
        accumulate_causal_edge(&store, &edge1).unwrap();

        // Second observation: strength 0.6
        let edge2 = CausalEdge {
            cause_id: a,
            effect_id: b,
            causal_strength: 0.6,
            observations: 1,
            last_observed: ts + 1000,
            mechanism: CausalMechanism::RecallAttribution,
            confidence: 0.5,
            namespace: String::new(),
        };
        accumulate_causal_edge(&store, &edge2).unwrap();

        // Check: posterior = (0.8 * 1 + 0.6) / 2 = 0.7
        let edges = load_all_edges(&store, "").unwrap();
        assert_eq!(edges.len(), 1);
        assert!((edges[0].causal_strength - 0.7).abs() < 0.01);
        assert_eq!(edges[0].observations, 2);
    }

    #[test]
    fn test_load_causal_graph() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        let ts = now_ms();

        // a → b → c
        for (cause, effect) in [(a, b), (b, c)] {
            accumulate_causal_edge(&store, &CausalEdge {
                cause_id: cause,
                effect_id: effect,
                causal_strength: 0.7,
                observations: 1,
                last_observed: ts,
                mechanism: CausalMechanism::RecallAttribution,
                confidence: 0.6,
                namespace: String::new(),
            })
            .unwrap();
        }

        let cg = load_causal_graph(&store, "", 0.0).unwrap();
        assert_eq!(cg.node_count(), 3);
        assert_eq!(cg.edge_count(), 2);
    }

    #[test]
    fn test_causal_ancestors() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        let ts = now_ms();

        // a → b → c
        for (cause, effect) in [(a, b), (b, c)] {
            accumulate_causal_edge(&store, &CausalEdge {
                cause_id: cause,
                effect_id: effect,
                causal_strength: 0.8,
                observations: 1,
                last_observed: ts,
                mechanism: CausalMechanism::RecallAttribution,
                confidence: 0.6,
                namespace: String::new(),
            })
            .unwrap();
        }

        let ancestors = get_causal_ancestors(&store, c, "", 10).unwrap();
        // Both a and b are ancestors of c
        assert!(ancestors.len() >= 1);
        // Direct ancestor (b) should have strength 0.8
        assert!(ancestors.iter().any(|(id, _)| *id == b));
    }

    #[test]
    fn test_common_confounders() {
        let store = CozoStore::open_mem(4).unwrap();
        let confounder = Uuid::now_v7();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let ts = now_ms();

        // confounder → a, confounder → b
        for effect in [a, b] {
            accumulate_causal_edge(&store, &CausalEdge {
                cause_id: confounder,
                effect_id: effect,
                causal_strength: 0.9,
                observations: 1,
                last_observed: ts,
                mechanism: CausalMechanism::RecallAttribution,
                confidence: 0.7,
                namespace: String::new(),
            })
            .unwrap();
        }

        let confounders = get_common_confounders(&store, a, b, "").unwrap();
        assert_eq!(confounders.len(), 1);
        assert_eq!(confounders[0].0, confounder);
    }

    #[test]
    fn test_adjust_edge_confidence() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let ts = now_ms();

        accumulate_causal_edge(&store, &CausalEdge {
            cause_id: a,
            effect_id: b,
            causal_strength: 0.7,
            observations: 1,
            last_observed: ts,
            mechanism: CausalMechanism::LlmProposed,
            confidence: 0.3,
            namespace: String::new(),
        })
        .unwrap();

        // Confirm the edge — confidence should increase
        adjust_edge_confidence(&store, a, b, true).unwrap();
        let edges = load_all_edges(&store, "").unwrap();
        assert!(edges[0].confidence > 0.3);

        // Contradict the edge — confidence should decrease
        adjust_edge_confidence(&store, a, b, false).unwrap();
        let edges2 = load_all_edges(&store, "").unwrap();
        assert!(edges2[0].confidence < edges[0].confidence);
    }

    #[test]
    fn test_mechanism_roundtrip() {
        assert_eq!(
            CausalMechanism::from_str(CausalMechanism::RecallAttribution.as_str()),
            CausalMechanism::RecallAttribution
        );
        assert_eq!(
            CausalMechanism::from_str(CausalMechanism::NotearsDiscovered.as_str()),
            CausalMechanism::NotearsDiscovered
        );
    }
}
