//! Reconsolidation — recall-triggered memory rewriting.
//!
//! When a memory is recalled and contradicts newer knowledge, the system
//! rewrites it with a precision-weighted update. CozoDB Validity preserves
//! the original version — nothing is lost.
//!
//! This mimics human memory reconsolidation: each time we recall a memory,
//! it becomes labile and can be updated with current context.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::services::traits::{Embedder, LlmService, Message};
use crate::store::CozoStore;
use crate::types::memory::Memory;

/// Result of a reconsolidation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconsolidationResult {
    pub memory_id: Uuid,
    pub original_content: String,
    pub updated_content: String,
    pub contradictions_resolved: usize,
}

/// Reconsolidate a memory in light of new contradicting knowledge.
///
/// 1. Find contradicting facts for the entities mentioned in this memory
/// 2. If contradictions exist, LLM rewrites the memory content
/// 3. Re-embed and store new version (old preserved via Validity)
/// 4. Record a reconsolidation access event
pub async fn reconsolidate(
    store: &CozoStore,
    memory: &Memory,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
) -> Result<Option<ReconsolidationResult>> {
    // 1. Find entities linked to this memory
    let entity_ids = store.collect_entity_ids_for_memories(&[memory.id])?;
    if entity_ids.is_empty() {
        return Ok(None);
    }

    // 2. Find contradictions for those entities
    let contradictions = store.find_contradictions(&entity_ids)?;
    if contradictions.is_empty() {
        return Ok(None);
    }

    // 3. Build contradiction summary for LLM
    let contradiction_summary: Vec<String> = contradictions
        .iter()
        .map(|c| {
            format!(
                "{}: '{}' vs '{}'",
                c.predicate, c.value_a, c.value_b
            )
        })
        .collect();

    // 4. LLM rewrites the memory
    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "Update this memory to reflect current knowledge. \
                              Keep the same style and length. Output only the updated text."
                        .into(),
                },
                Message {
                    role: "user".into(),
                    content: format!(
                        "Original memory: {}\n\nContradictions found:\n{}\n\n\
                         Rewrite the memory incorporating the newer information.",
                        memory.content,
                        contradiction_summary.join("\n")
                    ),
                },
            ],
            4096,
        )
        .await
        .map_err(|e| crate::error::MemoriaError::Llm(e.to_string()))?;

    let updated_content = response.content.trim().to_string();
    if updated_content.is_empty() || updated_content == memory.content {
        return Ok(None);
    }

    // 5. Re-embed the updated content
    let embeddings = embedder
        .embed(&[updated_content.as_str()])
        .await
        .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

    let new_embedding = embeddings
        .into_iter()
        .next()
        .unwrap_or_else(|| memory.embedding.clone());

    // 6. Store updated version (Validity preserves old version)
    update_memory_content(store, memory.id, &updated_content, &new_embedding)?;

    // 7. Record reconsolidation access
    store.record_accesses(&[memory.id], "", "reconsolidation")?;

    Ok(Some(ReconsolidationResult {
        memory_id: memory.id,
        original_content: memory.content.clone(),
        updated_content,
        contradictions_resolved: contradictions.len(),
    }))
}

/// Update memory content and embedding in CozoDB.
///
/// Reads the full memory, updates content/embedding/version in Rust,
/// and re-inserts using the standard insert_memory path.
/// CozoDB Validity preserves old versions via time-travel.
pub(crate) fn update_memory_content(
    store: &CozoStore,
    id: Uuid,
    new_content: &str,
    new_embedding: &[f32],
) -> Result<()> {
    let mut memory = store
        .get_memory(id)?
        .ok_or(crate::error::MemoriaError::NotFound(id))?;

    memory.content = new_content.to_string();
    memory.embedding = new_embedding.to_vec();
    memory.version += 1;

    store.insert_memory(&memory)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::mock::{MockEmbedder, MockLlm};
    use crate::types::memory::Memory;

    #[tokio::test]
    async fn test_reconsolidate_no_entities_returns_none() {
        let store = CozoStore::open_mem(4).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(4));

        let mem = Memory::new("test", "hello world", vec![0.1, 0.2, 0.3, 0.4]);
        store.insert_memory(&mem).unwrap();

        let result = reconsolidate(&store, &mem, &llm, &embedder).await.unwrap();
        assert!(result.is_none(), "no entities → no reconsolidation");
    }

    #[tokio::test]
    async fn test_reconsolidate_no_contradictions_returns_none() {
        let store = CozoStore::open_mem(4).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(4));

        // Create memory with entity but no contradicting facts
        let mem = Memory::new("test", "Alice is an engineer", vec![0.1, 0.2, 0.3, 0.4]);
        store.insert_memory(&mem).unwrap();

        let entity = crate::types::entity::Entity {
            id: Uuid::now_v7(),
            name: "Alice".to_string(),
            entity_type: "person".to_string(),
            namespace: String::new(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            properties: Default::default(),
            mention_count: 1,
            confidence: 1.0,
            provenance: "extracted".to_string(),
            source_ids: vec![],
        };
        store.insert_entity(&entity).unwrap();
        store
            .link_entity_to_memory(entity.id, mem.id, "mentioned", 1.0)
            .unwrap();

        let result = reconsolidate(&store, &mem, &llm, &embedder).await.unwrap();
        assert!(result.is_none(), "no contradictions → no reconsolidation");
    }

    #[test]
    fn test_update_memory_content() {
        let store = CozoStore::open_mem(4).unwrap();
        let mem = Memory::new("test", "original content", vec![0.1, 0.2, 0.3, 0.4]);
        let id = mem.id;
        store.insert_memory(&mem).unwrap();

        update_memory_content(&store, id, "updated content", &[0.5, 0.6, 0.7, 0.8]).unwrap();

        // CozoDB Validity: with multiple ASSERT versions, the latest by
        // valid_at is returned. Query explicitly to get the latest version.
        let mut params = std::collections::BTreeMap::new();
        params.insert("id".into(), cozo::DataValue::from(id.to_string()));
        let result = store
            .run_query(
                r#"?[content, version] :=
                    *memories{id, content, version},
                    id = to_uuid($id)
                   :order -version
                   :limit 1"#,
                params,
            )
            .unwrap();
        assert!(!result.rows.is_empty());
        let content = result.rows[0][0].get_str().unwrap().to_string();
        let version = result.rows[0][1].get_int().unwrap();
        assert_eq!(content, "updated content");
        assert_eq!(version, 1);
    }

    #[test]
    fn test_update_nonexistent_memory_fails() {
        let store = CozoStore::open_mem(4).unwrap();
        let result = update_memory_content(
            &store,
            Uuid::now_v7(),
            "content",
            &[0.1, 0.2, 0.3, 0.4],
        );
        assert!(result.is_err());
    }
}
