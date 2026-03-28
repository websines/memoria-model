//! Memory compression — episodic → semantic promotion.
//!
//! Compression levels:
//!   Level 0: Raw memories (individual events)
//!       ↓ cluster + summarize
//!   Level 1: Episode summaries
//!       ↓ pattern extraction
//!   Level 2: Abstractions (general patterns)
//!       ↓ meta-abstraction
//!   Level 3: Core beliefs (deeply reinforced)
//!
//! Mechanism:
//! 1. Clustering via CozoDB ConnectedComponents on co-activation graph
//! 2. LLM summarizes cluster into summary memory
//! 3. If abstraction reinforced by N+ episodes → promote confidence
//! 4. Level 0 memories that have been compressed + not accessed → eligible for GC

use std::collections::BTreeMap;
use std::sync::Arc;

use cozo::DataValue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::services::traits::{Embedder, LlmService, Message};
use crate::store::CozoStore;
use crate::types::memory::{Memory, MemoryId};

/// Compression level for memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// Raw individual events.
    Raw = 0,
    /// Episode summaries (cluster of raw events).
    EpisodeSummary = 1,
    /// Abstractions (patterns across episodes).
    Abstraction = 2,
    /// Core beliefs (deeply reinforced patterns).
    CoreBelief = 3,
}

impl CompressionLevel {
    pub fn from_i64(v: i64) -> Self {
        match v {
            0 => Self::Raw,
            1 => Self::EpisodeSummary,
            2 => Self::Abstraction,
            3 => Self::CoreBelief,
            _ => Self::Raw,
        }
    }
}

/// Result of a compression operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    /// The summary memory that was created.
    pub summary_id: MemoryId,
    /// The raw memory IDs that were compressed into this summary.
    pub source_ids: Vec<MemoryId>,
    /// The compression level of the new summary.
    pub level: CompressionLevel,
    /// The summary content.
    pub summary: String,
}

/// Find clusters of co-activated memories and compress them into summaries.
///
/// 1. Find clusters via connected components on the co-activation graph
/// 2. For each cluster above minimum size, LLM summarizes into a single memory
/// 3. New summary memory linked to sources via edges
/// 4. Source memories marked as compressed (eligible for GC)
pub async fn compress_clusters(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    min_cluster_size: usize,
    namespace: &str,
) -> Result<Vec<CompressionResult>> {
    // 1. Find connected components in the co-activation graph
    let clusters = find_coactivation_clusters(store, min_cluster_size)?;

    let mut results = Vec::new();

    for cluster_ids in &clusters {
        // 2. Gather content from cluster members
        let contents = gather_cluster_content(store, cluster_ids)?;
        if contents.is_empty() {
            continue;
        }

        // 3. LLM summarizes the cluster
        let summary = summarize_cluster(llm, &contents).await?;
        if summary.is_empty() {
            continue;
        }

        // 4. Embed the summary
        let embeddings = embedder
            .embed(&[summary.as_str()])
            .await
            .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

        let embedding = embeddings
            .into_iter()
            .next()
            .unwrap_or_else(|| vec![0.0; store.dim()]);

        // 5. Store summary memory
        let mut summary_mem = Memory::new("summary.compressed", &summary, embedding);
        summary_mem.namespace = namespace.to_string();
        summary_mem.provenance = "compressed".to_string();
        summary_mem.source_ids = cluster_ids.clone();
        let summary_id = summary_mem.id;

        store.insert_memory(&summary_mem)?;

        // 6. Link summary to source memories via edges
        for source_id in cluster_ids {
            store.insert_edge(*source_id, summary_id, "compressed_into", 1.0)?;
        }

        // 7. Mark source memories as compressed via fields update
        for source_id in cluster_ids {
            mark_memory_compressed(store, *source_id)?;
        }

        results.push(CompressionResult {
            summary_id,
            source_ids: cluster_ids.clone(),
            level: CompressionLevel::EpisodeSummary,
            summary,
        });
    }

    Ok(results)
}

/// Mark a memory as compressed by setting `compressed: true` in its fields JSON.
///
/// Reads the memory, modifies its fields map in Rust, and writes back.
fn mark_memory_compressed(store: &CozoStore, memory_id: Uuid) -> Result<()> {
    let Some(mut memory) = store.get_memory(memory_id)? else {
        return Ok(()); // already gone — not an error
    };
    memory.fields.insert(
        "compressed".to_string(),
        serde_json::Value::Bool(true),
    );
    store.insert_memory(&memory)?;
    Ok(())
}

/// Find connected components in the co-activation graph.
///
/// Uses CozoDB's graph traversal — memories that are frequently co-accessed
/// form natural clusters.
fn find_coactivation_clusters(
    store: &CozoStore,
    min_size: usize,
) -> Result<Vec<Vec<Uuid>>> {
    // Fetch all co-activation edges, then compute connected components in Rust.
    //
    // CozoDB's `min()` aggregation doesn't work on UUID types, so we can't do
    // connected-component computation in Datalog. Instead, we fetch all edges
    // and use a simple union-find algorithm in Rust.
    let result = store.run_query(
        "?[a, b] := *co_activations{a, b, count, @ 'NOW'}, count >= 1",
        BTreeMap::new(),
    )?;

    if result.rows.is_empty() {
        return Ok(Vec::new());
    }

    // Parse edges
    let mut edges: Vec<(Uuid, Uuid)> = Vec::new();
    for row in &result.rows {
        if let (Ok(a), Ok(b)) = (
            crate::store::cozo::parse_uuid_pub(&row[0]),
            crate::store::cozo::parse_uuid_pub(&row[1]),
        ) {
            edges.push((a, b));
        }
    }

    // Union-Find in Rust
    use std::collections::HashMap;
    let mut parent: HashMap<Uuid, Uuid> = HashMap::new();

    fn find(parent: &mut HashMap<Uuid, Uuid>, x: Uuid) -> Uuid {
        let p = *parent.get(&x).unwrap_or(&x);
        if p == x {
            return x;
        }
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    fn union(parent: &mut HashMap<Uuid, Uuid>, a: Uuid, b: Uuid) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    for &(a, b) in &edges {
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
        union(&mut parent, a, b);
    }

    // Group by root
    let mut groups: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    let all_nodes: Vec<Uuid> = parent.keys().copied().collect();
    for node in all_nodes {
        let root = find(&mut parent, node);
        groups.entry(root).or_default().push(node);
    }

    Ok(groups
        .into_values()
        .filter(|g| g.len() >= min_size)
        .collect())
}

/// Gather text content from a cluster of memory IDs.
fn gather_cluster_content(store: &CozoStore, ids: &[Uuid]) -> Result<Vec<String>> {
    let mut contents = Vec::new();

    for id in ids {
        if let Some(mem) = store.get_memory(*id)? {
            if !mem.content.is_empty() {
                contents.push(mem.content);
            }
        }
    }

    Ok(contents)
}

/// Use LLM to summarize a cluster of memory contents.
async fn summarize_cluster(
    llm: &Arc<dyn LlmService>,
    contents: &[String],
) -> Result<String> {
    let combined = contents.join("\n---\n");

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "Summarize these related memories into a single concise paragraph. \
                              Preserve key facts and relationships. Output only the summary."
                        .into(),
                },
                Message {
                    role: "user".into(),
                    content: combined,
                },
            ],
            1024,
        )
        .await
        .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

    Ok(response.content.trim().to_string())
}

/// Promote an abstraction's confidence when reinforced by additional episodes.
///
/// If an abstraction has been independently confirmed by N+ episodes,
/// increase its confidence toward 1.0.
pub fn promote_abstraction_confidence(
    store: &CozoStore,
    abstraction_id: Uuid,
    promotion_threshold: i64,
) -> Result<bool> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(abstraction_id.to_string()));

    let result = store.run_query(
        r#"?[confidence, evidence_count] :=
            *abstractions{id, confidence, evidence_count},
            id = to_uuid($id)"#,
        params.clone(),
    )?;

    if result.rows.is_empty() {
        return Ok(false);
    }

    let confidence = result.rows[0][0].get_float().unwrap_or(0.5);
    let evidence = result.rows[0][1].get_int().unwrap_or(1);

    if evidence < promotion_threshold {
        return Ok(false);
    }

    // Promote: new_conf = old_conf + (1.0 - old_conf) × 0.1 per excess episode
    let excess = (evidence - promotion_threshold) as f64;
    let new_confidence = (confidence + (1.0 - confidence) * 0.1 * (excess + 1.0)).min(0.99);

    params.insert("new_conf".into(), DataValue::from(new_confidence));
    params.insert(
        "new_count".into(),
        DataValue::from(evidence + 1),
    );

    store.run_script(
        r#"?[id, valid_at, confidence, evidence_count] <-
            [[$id, 'ASSERT', $new_conf, $new_count]]
           :update abstractions {id, valid_at => confidence, evidence_count}"#,
        params,
    )?;

    Ok(true)
}

/// Compress Level 1 summaries into Level 2 abstractions (pattern extraction).
///
/// Groups episode summaries that share similar content, then uses the LLM to
/// extract a cross-episodic pattern/abstraction from them.
pub async fn compress_summaries_to_abstractions(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    min_summaries: usize,
    namespace: &str,
) -> Result<Vec<CompressionResult>> {
    // Find summaries (kind starts with "summary.") that haven't been abstracted yet
    let mut params = BTreeMap::new();
    params.insert("ns".into(), DataValue::from(namespace));

    let result = store.run_query(
        r#"
        summaries[id, content, embedding] :=
            *memories{id, kind, content, embedding, namespace},
            namespace = $ns,
            starts_with(kind, "summary.")
        already_compressed[source] :=
            *edges{source, target, kind},
            kind == "compressed_into"
        uncompressed[id, content, embedding] :=
            summaries[id, content, embedding],
            not already_compressed[id]
        ?[id, content] := uncompressed[id, content, _]
        "#,
        params,
    );

    let summary_mems: Vec<(Uuid, String)> = match result {
        Ok(rows) => rows
            .rows
            .iter()
            .filter_map(|row| {
                let id = crate::store::cozo::parse_uuid_pub(&row[0]).ok()?;
                let content = row[1].get_str()?.to_string();
                Some((id, content))
            })
            .collect(),
        Err(_) => return Ok(Vec::new()),
    };

    if summary_mems.len() < min_summaries {
        return Ok(Vec::new());
    }

    // Group all uncompressed summaries and extract an abstraction
    let contents: Vec<String> = summary_mems.iter().map(|(_, c)| c.clone()).collect();
    let source_ids: Vec<Uuid> = summary_mems.iter().map(|(id, _)| *id).collect();

    let abstraction = extract_abstraction(llm, &contents).await?;
    if abstraction.is_empty() {
        return Ok(Vec::new());
    }

    // Embed the abstraction
    let embeddings = embedder
        .embed(&[abstraction.as_str()])
        .await
        .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

    let embedding = embeddings
        .into_iter()
        .next()
        .unwrap_or_else(|| vec![0.0; store.dim()]);

    // Store abstraction memory
    let mut abs_mem = Memory::new("abstraction", &abstraction, embedding);
    abs_mem.namespace = namespace.to_string();
    abs_mem.provenance = "compressed".to_string();
    abs_mem.source_ids = source_ids.clone();
    let abs_id = abs_mem.id;

    store.insert_memory(&abs_mem)?;

    // Link sources to abstraction
    for source_id in &source_ids {
        store.insert_edge(*source_id, abs_id, "compressed_into", 1.0)?;
    }

    Ok(vec![CompressionResult {
        summary_id: abs_id,
        source_ids,
        level: CompressionLevel::Abstraction,
        summary: abstraction,
    }])
}

/// Use LLM to extract a cross-episodic abstraction/pattern from summaries.
async fn extract_abstraction(
    llm: &Arc<dyn LlmService>,
    summaries: &[String],
) -> Result<String> {
    let combined = summaries.join("\n---\n");

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "These are summaries of different episodes/events. \
                              Extract the recurring PATTERN or general principle that \
                              connects them. Output a single concise statement of the \
                              pattern. If no clear pattern exists, output an empty string."
                        .into(),
                },
                Message {
                    role: "user".into(),
                    content: combined,
                },
            ],
            512,
        )
        .await
        .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

    Ok(response.content.trim().to_string())
}

/// Garbage-collect compressed source memories that are no longer accessed.
///
/// A memory is eligible for GC if:
/// 1. It has been compressed (has a "compressed_into" outgoing edge)
/// 2. It has not been accessed within `max_age_ms`
/// 3. It is not pinned
///
/// Returns the IDs of deleted memories.
pub fn gc_compressed_sources(
    store: &CozoStore,
    max_age_ms: i64,
    namespace: &str,
) -> Result<Vec<Uuid>> {
    let now = crate::types::memory::now_ms();
    let cutoff = now - max_age_ms;

    let mut params = BTreeMap::new();
    params.insert("ns".into(), DataValue::from(namespace));
    params.insert("cutoff".into(), DataValue::from(cutoff));

    // Find memories that are compressed and have no recent access
    let result = store.run_query(
        r#"
        compressed[source] :=
            *edges{source, target: _, kind},
            kind == "compressed_into"
        last_access[mem_id, max(ts)] :=
            *access_log{memory_id: mem_id, ts}
        last_access[mem_id, created] :=
            *memories{id: mem_id, created_at: created},
            not *access_log{memory_id: mem_id}
        eligible[id] :=
            compressed[id],
            last_access[id, last_ts],
            last_ts < $cutoff,
            *memories{id, namespace, pinned},
            namespace = $ns,
            pinned == false
        ?[id] := eligible[id]
        "#,
        params,
    );

    let ids: Vec<Uuid> = match result {
        Ok(rows) => rows
            .rows
            .iter()
            .filter_map(|row| crate::store::cozo::parse_uuid_pub(&row[0]).ok())
            .collect(),
        Err(_) => return Ok(Vec::new()),
    };

    // Delete eligible memories
    for id in &ids {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(id.to_string()));
        let _ = store.run_script(
            r#"?[id] <- [[$id]]
               :rm memories {id}"#,
            params,
        );
    }

    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_level_roundtrip() {
        assert_eq!(CompressionLevel::from_i64(0), CompressionLevel::Raw);
        assert_eq!(CompressionLevel::from_i64(1), CompressionLevel::EpisodeSummary);
        assert_eq!(CompressionLevel::from_i64(2), CompressionLevel::Abstraction);
        assert_eq!(CompressionLevel::from_i64(3), CompressionLevel::CoreBelief);
        assert_eq!(CompressionLevel::from_i64(99), CompressionLevel::Raw);
    }

    #[test]
    fn test_find_clusters_empty_store() {
        let store = CozoStore::open_mem(4).unwrap();
        let clusters = find_coactivation_clusters(&store, 3).unwrap();
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_co_activation_creates_clusters() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = crate::types::memory::now_ms();

        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::now_v7()).collect();
        for &id in &ids {
            let mut m = Memory::new("test", &format!("memory {id}"), vec![0.1, 0.2, 0.3, 0.4]);
            m.id = id;
            store.insert_memory(&m).unwrap();
        }

        // All pairs connected
        store.upsert_co_activation(ids[0], ids[1], now).unwrap();
        store.upsert_co_activation(ids[0], ids[2], now).unwrap();
        store.upsert_co_activation(ids[1], ids[2], now).unwrap();

        let clusters = find_coactivation_clusters(&store, 2).unwrap();
        assert!(
            !clusters.is_empty(),
            "should find clusters after co_activation inserts"
        );
        let biggest = clusters.iter().max_by_key(|c| c.len()).unwrap();
        assert_eq!(biggest.len(), 3, "all 3 nodes should be in one cluster");
    }

    #[test]
    fn test_gather_cluster_content() {
        let store = CozoStore::open_mem(4).unwrap();

        let m1 = Memory::new("test", "Alice works at Acme", vec![0.1, 0.2, 0.3, 0.4]);
        let m2 = Memory::new("test", "Bob works at Acme", vec![0.5, 0.6, 0.7, 0.8]);
        let id1 = m1.id;
        let id2 = m2.id;
        store.insert_memory(&m1).unwrap();
        store.insert_memory(&m2).unwrap();

        let contents = gather_cluster_content(&store, &[id1, id2]).unwrap();
        assert_eq!(contents.len(), 2);
        assert!(contents[0].contains("Alice"));
        assert!(contents[1].contains("Bob"));
    }
}
