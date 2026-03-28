use async_trait::async_trait;
use serde::Deserialize;

use super::traits::*;

/// GLiNER2 NER sidecar HTTP client.
///
/// Connects to the FastAPI NER microservice at the configured URL.
pub struct GlinerSidecar {
    base_url: String,
    client: ureq::Agent,
}

#[derive(Deserialize)]
struct GlinerResponse {
    results: Vec<GlinerTextResult>,
}

#[derive(Deserialize)]
struct GlinerTextResult {
    entities: Vec<GlinerEntity>,
    #[serde(default)]
    relations: Vec<GlinerRelation>,
}

#[derive(Deserialize)]
struct GlinerEntity {
    text: String,
    label: String,
    score: f64,
    start: usize,
    end: usize,
}

#[derive(Deserialize)]
struct GlinerRelation {
    head: String,
    tail: String,
    label: String,
    score: f64,
}

/// Map a relation head/tail string to its canonical entity name.
///
/// GLiNER relation extraction often returns full mention text (e.g. "The company")
/// while entity extraction returns the minimal noun form ("company"). This function
/// resolves the mismatch by finding the best-matching extracted entity:
///
/// 1. Exact match — relation text == entity text
/// 2. Containment — entity text is a substring of relation text or vice versa
///    (e.g. "The company" contains "company", or "CEO" is contained in "CEO of TechCorp")
/// 3. No match — return the original text unchanged
///
/// When multiple entities match, the longest entity name wins (most specific match).
fn canonicalize_name(rel_text: &str, entities: &[ExtractedEntity]) -> String {
    // Exact match — fast path
    if entities.iter().any(|e| e.text == rel_text) {
        return rel_text.to_string();
    }

    let rel_lower = rel_text.to_lowercase();
    let mut best: Option<&str> = None;
    let mut best_len = 0;

    for e in entities {
        let e_lower = e.text.to_lowercase();
        if (rel_lower.contains(&e_lower) || e_lower.contains(&rel_lower))
            && e.text.len() > best_len
        {
            best = Some(&e.text);
            best_len = e.text.len();
        }
    }

    best.unwrap_or(rel_text).to_string()
}

impl GlinerSidecar {
    pub fn new(base_url: &str) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(5))
            .timeout_read(std::time::Duration::from_secs(30))
            .build();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }
}

#[async_trait]
impl NerService for GlinerSidecar {
    async fn extract(
        &self,
        texts: &[TextInput],
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> anyhow::Result<Vec<ExtractionResult>> {
        let url = format!("{}/extract", self.base_url);

        let text_inputs: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                ureq::json!({
                    "text": t.text,
                    "id": t.id.clone().unwrap_or_else(|| i.to_string()),
                })
            })
            .collect();

        let payload = ureq::json!({
            "texts": text_inputs,
            "entity_labels": entity_labels,
            "relation_labels": relation_labels,
        });

        // Retry up to 3 times on transient connection errors
        let mut last_err = None;
        for attempt in 0..3 {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_millis(500 * attempt as u64));
            }
            match self.client.post(&url).send_json(payload.clone()) {
                Ok(response) => {
                    let resp: GlinerResponse = response.into_json()?;
                    return Ok(resp
                        .results
                        .into_iter()
                        .map(|r| {
                            let entities: Vec<ExtractedEntity> = r
                                .entities
                                .iter()
                                .map(|e| ExtractedEntity {
                                    text: e.text.clone(),
                                    label: e.label.clone(),
                                    confidence: e.score,
                                    start: e.start,
                                    end: e.end,
                                })
                                .collect();

                            let relations = r
                                .relations
                                .into_iter()
                                .map(|rel| ExtractedRelation {
                                    head: canonicalize_name(&rel.head, &entities),
                                    tail: canonicalize_name(&rel.tail, &entities),
                                    label: rel.label,
                                    confidence: rel.score,
                                })
                                .collect();

                            ExtractionResult { entities, relations }
                        })
                        .collect());
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("Connection reset")
                        || err_str.contains("connection reset")
                        || err_str.contains("Connection refused")
                    {
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        Err(last_err.unwrap().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(text: &str) -> ExtractedEntity {
        ExtractedEntity {
            text: text.to_string(),
            label: "test".to_string(),
            confidence: 0.9,
            start: 0,
            end: text.len(),
        }
    }

    #[test]
    fn exact_match_returns_as_is() {
        let entities = vec![entity("TechCorp"), entity("Alice")];
        assert_eq!(canonicalize_name("TechCorp", &entities), "TechCorp");
    }

    #[test]
    fn determiner_prefix_stripped() {
        let entities = vec![entity("company"), entity("Cupertino")];
        // "The company" contains "company" → resolves to "company"
        assert_eq!(canonicalize_name("The company", &entities), "company");
    }

    #[test]
    fn case_insensitive_containment() {
        let entities = vec![entity("TechCorp")];
        assert_eq!(canonicalize_name("the techcorp", &entities), "TechCorp");
    }

    #[test]
    fn prefers_longest_match() {
        let entities = vec![entity("Corp"), entity("TechCorp")];
        // Both match "The TechCorp company" but "TechCorp" is longer
        assert_eq!(
            canonicalize_name("The TechCorp company", &entities),
            "TechCorp"
        );
    }

    #[test]
    fn no_match_returns_original() {
        let entities = vec![entity("Alice")];
        assert_eq!(
            canonicalize_name("completely unrelated", &entities),
            "completely unrelated"
        );
    }

    #[test]
    fn entity_contained_in_rel_text() {
        let entities = vec![entity("Alice")];
        // "Alice Smith, CEO" contains "Alice"
        assert_eq!(
            canonicalize_name("Alice Smith, CEO", &entities),
            "Alice"
        );
    }
}
