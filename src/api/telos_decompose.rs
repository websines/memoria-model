//! Telos decomposition — LLM-driven goal breakdown into subtelos.
//!
//! When a telos has depth < 3, this module uses the LLM to generate
//! structured subtelos that become children of the parent. Each child
//! inherits the parent's namespace and gets `depth = parent.depth + 1`.

use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::{Embedder, LlmService, Message};
use crate::store::CozoStore;
use crate::types::memory::now_ms;
use crate::types::telos::{Telos, TelosEvent, TelosProvenance, TelosStatus};

/// A subtelos extracted by the LLM during decomposition.
#[derive(Debug, Clone, serde::Deserialize)]
struct SubtelosSpec {
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_priority")]
    priority: f64,
}

fn default_priority() -> f64 {
    0.5
}

/// Decompose a telos into subtelos using the LLM.
///
/// The LLM receives the parent goal's title, description, and depth,
/// then returns a JSON array of subtelos specs. Each subtelos is embedded,
/// stored, and linked as a child of the parent.
///
/// Returns the created child telos list. Records a `decomposed` event on the parent.
pub async fn decompose_telos(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    telos_id: Uuid,
    namespace: &str,
) -> Result<Vec<Telos>> {
    // 1. Load the parent telos
    let parent = store
        .get_telos(telos_id)?
        .ok_or_else(|| MemoriaError::NotFound(telos_id))?;

    // Don't decompose terminal goals or goals at max depth
    if parent.status.is_terminal() || parent.depth >= 3 {
        return Ok(vec![]);
    }

    // Check if already decomposed (has children)
    let existing_children = store.get_children_telos(telos_id)?;
    if !existing_children.is_empty() {
        return Ok(existing_children);
    }

    // 2. Ask LLM for subtelos
    let child_depth = parent.depth + 1;
    let depth_label = match child_depth {
        0 => "north-star",
        1 => "strategic",
        2 => "tactical",
        3 => "operational",
        _ => "task",
    };

    let prompt = format!(
        concat!(
            "Break down this goal into concrete {depth_label}-level subtelos.\n\n",
            "Parent goal (depth {parent_depth}):\n",
            "  Title: {title}\n",
            "  Description: {description}\n\n",
            "Return a JSON array of subtelos. Each subtelos should have:\n",
            "- \"title\": concise action-oriented title\n",
            "- \"description\": what done looks like\n",
            "- \"priority\": 0.0-1.0 relative importance within this parent\n\n",
            "Rules:\n",
            "- 2-5 subtelos (no more)\n",
            "- Each should be independently achievable\n",
            "- Priorities should sum to roughly 1.0\n",
            "- Be specific, not vague\n\n",
            "Return ONLY the JSON array, no markdown fences."
        ),
        depth_label = depth_label,
        parent_depth = parent.depth,
        title = parent.title,
        description = if parent.description.is_empty() {
            &parent.title
        } else {
            &parent.description
        },
    );

    let llm_response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: "You decompose goals into structured subtelos. Return only valid JSON arrays.".into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            1024,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    // 3. Parse LLM response
    let specs: Vec<SubtelosSpec> = parse_subtelos_response(&llm_response.content)?;

    if specs.is_empty() {
        return Ok(vec![]);
    }

    // 4. Embed all subtelos titles+descriptions in one batch
    let texts: Vec<String> = specs
        .iter()
        .map(|s| {
            if s.description.is_empty() {
                s.title.clone()
            } else {
                format!("{}: {}", s.title, s.description)
            }
        })
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let embeddings = embedder
        .embed(&text_refs)
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    // 5. Create and store each child telos
    let now = now_ms();
    let mut children = Vec::new();

    for (i, (spec, embedding)) in specs.iter().zip(embeddings.into_iter()).enumerate() {
        let mut child = Telos::new(
            &spec.title,
            &spec.description,
            embedding,
            &parent.owner,
            "decomposition",
        );
        child.parent = Some(telos_id);
        child.depth = child_depth;
        child.namespace = namespace.to_string();
        child.status = TelosStatus::Active;
        child.provenance = TelosProvenance::Decomposition;
        child.confidence = TelosProvenance::Decomposition.initial_confidence();
        child.priority = spec.priority.clamp(0.0, 1.0);
        child.created_at = now;

        // Inherit deadline from parent if set
        child.deadline = parent.deadline;

        store.insert_telos(&child)?;

        // Record creation event on child
        let mut event = TelosEvent::new(child.id, "created");
        event.agent_id = "decomposition".to_string();
        event.description = format!(
            "Decomposed from parent '{}' (subtelos {}/{})",
            parent.title,
            i + 1,
            specs.len()
        );
        store.insert_telos_event(&event)?;

        children.push(child);
    }

    // 6. Record decomposed event on parent
    let mut parent_event = TelosEvent::new(telos_id, "decomposed");
    parent_event.agent_id = "decomposition".to_string();
    parent_event.description = format!(
        "Decomposed into {} {} subtelos",
        children.len(),
        depth_label
    );
    store.insert_telos_event(&parent_event)?;

    Ok(children)
}

/// Parse the LLM response into subtelos specs.
///
/// Handles both raw JSON arrays and responses with markdown fences.
fn parse_subtelos_response(response: &str) -> Result<Vec<SubtelosSpec>> {
    let trimmed = response.trim();

    // Try direct parse first
    if let Ok(specs) = serde_json::from_str::<Vec<SubtelosSpec>>(trimmed) {
        return Ok(limit_specs(specs));
    }

    // Try stripping markdown code fences
    let stripped = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if let Ok(specs) = serde_json::from_str::<Vec<SubtelosSpec>>(stripped) {
        return Ok(limit_specs(specs));
    }

    // Try finding a JSON array in the response
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            let json_slice = &trimmed[start..=end];
            if let Ok(specs) = serde_json::from_str::<Vec<SubtelosSpec>>(json_slice) {
                return Ok(limit_specs(specs));
            }
        }
    }

    Err(MemoriaError::Llm(format!(
        "Failed to parse subtelos from LLM response: {}",
        &response[..response.len().min(200)]
    )))
}

/// Limit to 5 subtelos maximum.
fn limit_specs(specs: Vec<SubtelosSpec>) -> Vec<SubtelosSpec> {
    specs.into_iter().take(5).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_subtelos_raw_json() {
        let input = r#"[
            {"title": "Research competitors", "description": "Survey market landscape", "priority": 0.3},
            {"title": "Draft slides", "description": "Create initial deck", "priority": 0.5}
        ]"#;
        let specs = parse_subtelos_response(input).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].title, "Research competitors");
        assert!((specs[1].priority - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_subtelos_with_fences() {
        let input = "```json\n[{\"title\": \"Step 1\", \"description\": \"Do thing\", \"priority\": 1.0}]\n```";
        let specs = parse_subtelos_response(input).unwrap();
        assert_eq!(specs.len(), 1);
    }

    #[test]
    fn test_parse_subtelos_embedded_array() {
        let input = "Here are the subtelos:\n[{\"title\": \"A\"}]\nDone.";
        let specs = parse_subtelos_response(input).unwrap();
        assert_eq!(specs.len(), 1);
        assert!((specs[0].priority - 0.5).abs() < 0.01); // default
    }

    #[test]
    fn test_parse_subtelos_limit() {
        let input = r#"[
            {"title":"A"},{"title":"B"},{"title":"C"},
            {"title":"D"},{"title":"E"},{"title":"F"},{"title":"G"}
        ]"#;
        let specs = parse_subtelos_response(input).unwrap();
        assert_eq!(specs.len(), 5); // capped at 5
    }

    #[test]
    fn test_parse_subtelos_invalid() {
        let result = parse_subtelos_response("not json at all");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_decompose_telos_integration() {
        use crate::services::mock::{MockEmbedder, MockLlm};

        let store = CozoStore::open_mem(4).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(4));

        // Create a parent telos
        let mut parent = Telos::new("Ship Q3 deck", "Prepare investor presentation", vec![0.1; 4], "agent-1", "user");
        parent.depth = 1;
        parent.namespace = "test".to_string();
        store.insert_telos(&parent).unwrap();

        // MockLlm returns "mock response" which won't parse as JSON,
        // so decomposition should return an error or empty
        let result = decompose_telos(&store, &llm, &embedder, parent.id, "test").await;
        // MockLlm doesn't return valid JSON, so this will error
        assert!(result.is_err() || result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_decompose_skips_terminal() {
        use crate::services::mock::{MockEmbedder, MockLlm};

        let store = CozoStore::open_mem(4).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(4));

        let mut parent = Telos::new("Done goal", "", vec![0.1; 4], "a", "u");
        parent.status = TelosStatus::Completed;
        store.insert_telos(&parent).unwrap();

        let result = decompose_telos(&store, &llm, &embedder, parent.id, "test")
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_decompose_skips_max_depth() {
        use crate::services::mock::{MockEmbedder, MockLlm};

        let store = CozoStore::open_mem(4).unwrap();
        let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(4));

        let mut parent = Telos::new("Leaf goal", "", vec![0.1; 4], "a", "u");
        parent.depth = 3;
        store.insert_telos(&parent).unwrap();

        let result = decompose_telos(&store, &llm, &embedder, parent.id, "test")
            .await
            .unwrap();
        assert!(result.is_empty());
    }
}
