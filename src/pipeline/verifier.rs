use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{MemoriaError, Result};
use crate::services::traits::{ExtractedEntity, ExtractedRelation, LlmService, Message};

/// Temporal status of a verified relation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TemporalStatus {
    Current,
    Past,
    Future,
    Hypothetical,
}

impl TemporalStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Past => "past",
            Self::Future => "future",
            Self::Hypothetical => "hypothetical",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "current" => Self::Current,
            "past" => Self::Past,
            "future" => Self::Future,
            "hypothetical" => Self::Hypothetical,
            _ => Self::Current,
        }
    }
}

impl fmt::Display for TemporalStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A relation that has been refined and verified by an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedRelation {
    pub head: String,
    pub tail: String,
    pub label: String,
    pub original_label: String,
    pub confidence: f64,
    pub temporal_status: TemporalStatus,
    pub verified: bool,
    pub reason: String,
}

/// LLM response for a single refined relation.
#[derive(Debug, Deserialize)]
struct RefinedRelation {
    head: Option<String>,
    tail: Option<String>,
    label: Option<String>,
    temporal_status: Option<String>,
    confidence: Option<f64>,
    negated: Option<bool>,
}

/// Post-NER LLM refinement step.
///
/// Takes raw NER output (entities + relations) and uses an LLM to:
/// 1. Correct relation labels to be semantically accurate
/// 2. Fix head/tail entity names to match extracted entities
/// 3. Detect negation and temporal status
/// 4. Add relations the NER missed
///
/// This replaces the old "verify" approach — instead of rejecting NER output,
/// we refine it. The LLM improves quality rather than gatekeeping.
pub struct RelationVerifier {
    llm: Arc<dyn LlmService>,
}

impl RelationVerifier {
    pub fn new(llm: Arc<dyn LlmService>) -> Self {
        Self { llm }
    }

    /// Refine extracted relations using an LLM.
    ///
    /// The LLM sees the original text, the NER-extracted entities and relations,
    /// and returns refined relations with corrected labels and temporal status.
    /// On LLM/parse failure, falls back to accepting NER output directly.
    pub async fn verify(
        &self,
        text: &str,
        relations: &[ExtractedRelation],
    ) -> Result<Vec<VerifiedRelation>> {
        self.refine(text, relations, &[]).await
    }

    /// Refine with entity context for better accuracy.
    pub async fn refine(
        &self,
        text: &str,
        relations: &[ExtractedRelation],
        entities: &[ExtractedEntity],
    ) -> Result<Vec<VerifiedRelation>> {
        if relations.is_empty() {
            return Ok(Vec::new());
        }

        let relations_json = serde_json::to_string_pretty(relations)
            .map_err(|e| MemoriaError::Llm(e.to_string()))?;

        let entity_names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        let entity_ctx = if entity_names.is_empty() {
            String::new()
        } else {
            format!("\nExtracted entities: {}\n", entity_names.join(", "))
        };

        let prompt = format!(
            "Text: \"{text}\"\n{entity_ctx}\
             NER extracted these relations:\n{relations_json}\n\n\
             Refine each relation:\n\
             - \"label\": What is the actual relationship? Use a short snake_case label \
               that accurately describes what the text states \
               (e.g. \"founded_in\", \"works_at\", \"has_employee_count\", \"headquartered_in\").\n\
             - \"head\": The subject entity name (must match an extracted entity if possible).\n\
             - \"tail\": The object/value.\n\
             - \"temporal_status\": \"current\"|\"past\"|\"future\"|\"hypothetical\"\n\
             - \"confidence\": 0.0-1.0 (how clearly the text states this).\n\
             - \"negated\": true if the text negates or denies this relation.\n\n\
             Return a JSON array with one object per relation. \
             You may also add new relations the NER missed.\n\
             Respond ONLY with a JSON array."
        );

        let response = self
            .llm
            .complete(
                &[
                    Message {
                        role: "system".into(),
                        content: "You refine NER-extracted relations against source text. \
                                  Correct labels to be semantically accurate. \
                                  Keep entity names matching extracted entities. \
                                  Respond ONLY with a JSON array."
                            .into(),
                    },
                    Message {
                        role: "user".into(),
                        content: prompt,
                    },
                ],
                4096,
            )
            .await
            .map_err(|e| MemoriaError::Llm(e.to_string()))?;

        let refined = parse_llm_response(&response.content);

        // Build verified relations from refined output
        let mut result = Vec::new();

        // First, refine existing NER relations
        for (i, rel) in relations.iter().enumerate() {
            if let Some(r) = refined.get(i) {
                let negated = r.negated.unwrap_or(false);
                result.push(VerifiedRelation {
                    head: r.head.clone().unwrap_or_else(|| rel.head.clone()),
                    tail: r.tail.clone().unwrap_or_else(|| rel.tail.clone()),
                    label: r.label.clone().unwrap_or_else(|| rel.label.clone()),
                    original_label: rel.label.clone(),
                    confidence: r.confidence.unwrap_or(rel.confidence),
                    temporal_status: TemporalStatus::from_str_lossy(
                        r.temporal_status.as_deref().unwrap_or("current"),
                    ),
                    verified: !negated,
                    reason: if negated {
                        "negated in text".into()
                    } else {
                        "refined by LLM".into()
                    },
                });
            } else {
                // LLM didn't return enough entries — accept NER output directly
                result.push(VerifiedRelation {
                    head: rel.head.clone(),
                    tail: rel.tail.clone(),
                    label: rel.label.clone(),
                    original_label: rel.label.clone(),
                    confidence: rel.confidence * 0.8,
                    temporal_status: TemporalStatus::Current,
                    verified: true,
                    reason: "accepted from NER (LLM refinement incomplete)".into(),
                });
            }
        }

        // Then, add any extra relations the LLM found beyond the NER set
        for r in refined.iter().skip(relations.len()) {
            if let (Some(head), Some(tail), Some(label)) =
                (&r.head, &r.tail, &r.label)
            {
                let negated = r.negated.unwrap_or(false);
                if !negated {
                    result.push(VerifiedRelation {
                        head: head.clone(),
                        tail: tail.clone(),
                        label: label.clone(),
                        original_label: String::new(),
                        confidence: r.confidence.unwrap_or(0.6),
                        temporal_status: TemporalStatus::from_str_lossy(
                            r.temporal_status.as_deref().unwrap_or("current"),
                        ),
                        verified: true,
                        reason: "added by LLM refinement".into(),
                    });
                }
            }
        }

        Ok(result)
    }
}

/// Parse the LLM response, extracting a JSON array from potentially messy output.
fn parse_llm_response(content: &str) -> Vec<RefinedRelation> {
    // Try direct parse first
    if let Ok(v) = serde_json::from_str::<Vec<RefinedRelation>>(content) {
        return v;
    }

    // Try extracting JSON array from markdown code block or surrounding text
    let trimmed = content.trim();
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            let json_slice = &trimmed[start..=end];
            if let Ok(v) = serde_json::from_str::<Vec<RefinedRelation>>(json_slice) {
                return v;
            }
        }
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::mock::MockLlm;

    fn make_relation(head: &str, tail: &str, label: &str) -> ExtractedRelation {
        ExtractedRelation {
            head: head.into(),
            tail: tail.into(),
            label: label.into(),
            confidence: 0.85,
        }
    }

    #[tokio::test]
    async fn test_verify_empty_relations() {
        let verifier = RelationVerifier::new(Arc::new(MockLlm));
        let result = verifier.verify("some text", &[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_verify_positive_relation() {
        let verifier = RelationVerifier::new(Arc::new(MockLlm));
        let rels = vec![make_relation("Alice", "Acme", "works_at")];
        let result = verifier
            .verify("Alice works at Acme Corp", &rels)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].verified);
        assert_eq!(result[0].temporal_status, TemporalStatus::Current);
    }

    #[test]
    fn test_parse_llm_response_clean_json() {
        let json = r#"[{"head": "Alice", "tail": "Acme", "label": "works_at", "temporal_status": "current", "confidence": 0.9, "negated": false}]"#;
        let result = parse_llm_response(json);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label.as_deref(), Some("works_at"));
        assert_eq!(result[0].negated, Some(false));
    }

    #[test]
    fn test_parse_llm_response_with_markdown() {
        let json = "```json\n[{\"head\": \"X\", \"tail\": \"Y\", \"label\": \"related\", \"negated\": true}]\n```";
        let result = parse_llm_response(json);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].negated, Some(true));
    }

    #[test]
    fn test_parse_llm_response_garbage() {
        let result = parse_llm_response("I cannot process this request.");
        assert!(result.is_empty());
    }

    #[test]
    fn test_temporal_status_roundtrip() {
        assert_eq!(TemporalStatus::from_str_lossy("current"), TemporalStatus::Current);
        assert_eq!(TemporalStatus::from_str_lossy("past"), TemporalStatus::Past);
        assert_eq!(TemporalStatus::from_str_lossy("FUTURE"), TemporalStatus::Future);
        assert_eq!(TemporalStatus::from_str_lossy("hypothetical"), TemporalStatus::Hypothetical);
        assert_eq!(TemporalStatus::from_str_lossy("unknown"), TemporalStatus::Current);
    }
}
