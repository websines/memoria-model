use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use cozo::DataValue;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::rules::types::*;
use crate::services::traits::{LlmService, Message};
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// Rule enforcer that evaluates governance rules before operations.
///
/// Supports two evaluation modes:
/// - **Datalog**: Compiled CozoScript predicates (fast, deterministic)
/// - **LLM**: Natural language rule evaluation (flexible, slower)
pub struct RuleEnforcer {
    store: CozoStore,
    llm: Arc<dyn LlmService>,
}

impl RuleEnforcer {
    pub fn new(store: CozoStore, llm: Arc<dyn LlmService>) -> Self {
        Self { store, llm }
    }

    /// Check all relevant rules for an operation.
    ///
    /// Returns evaluations. Caller should check for Block results.
    /// When `content_embedding` is provided, also finds semantically relevant rules
    /// via HNSW search and merges them with scope-based matches (deduped by rule ID).
    pub async fn check_rules(
        &self,
        operation: &str,
        agent_id: &str,
        scope: &str,
        content: &str,
        target_id: Option<Uuid>,
    ) -> Result<Vec<RuleEvaluation>> {
        self.check_rules_with_embedding(operation, agent_id, scope, content, target_id, None)
            .await
    }

    /// Check rules with optional semantic matching via content embedding.
    pub async fn check_rules_with_embedding(
        &self,
        operation: &str,
        agent_id: &str,
        scope: &str,
        content: &str,
        target_id: Option<Uuid>,
        content_embedding: Option<&[f32]>,
    ) -> Result<Vec<RuleEvaluation>> {
        let mut rules = self.find_active_rules(scope)?;

        // Merge in semantically matched rules if embedding provided
        if let Some(embedding) = content_embedding {
            let semantic_rules = self.find_rules_by_semantic_match(embedding, 10, 0.8)?;
            let existing_ids: HashSet<Uuid> = rules.iter().map(|r| r.id).collect();
            for rule in semantic_rules {
                if !existing_ids.contains(&rule.id) {
                    rules.push(rule);
                }
            }
        }

        if rules.is_empty() {
            return Ok(Vec::new());
        }

        let mut evaluations = Vec::new();

        // Evaluate Datalog rules first (fast)
        for rule in rules.iter().filter(|r| r.compiled_type == CompiledType::Datalog) {
            let eval = self
                .evaluate_datalog_rule(rule, operation, agent_id, target_id)
                .await?;
            self.record_evaluation(&eval)?;
            evaluations.push(eval);
        }

        // Then LLM rules (slower)
        for rule in rules.iter().filter(|r| r.compiled_type == CompiledType::Llm) {
            let eval = self
                .evaluate_llm_rule(rule, operation, agent_id, content, target_id)
                .await?;
            self.record_evaluation(&eval)?;
            evaluations.push(eval);
        }

        Ok(evaluations)
    }

    /// Find rules by semantic similarity to content embedding using HNSW index.
    fn find_rules_by_semantic_match(
        &self,
        embedding: &[f32],
        k: usize,
        max_dist: f64,
    ) -> Result<Vec<Rule>> {
        use ndarray::Array1;

        let arr = Array1::from_vec(embedding.to_vec());
        let query_vec = DataValue::Vec(cozo::Vector::F32(arr));

        let mut params = BTreeMap::new();
        params.insert("q_vec".into(), query_vec);
        params.insert("k".into(), DataValue::from(k as i64));
        params.insert("max_dist".into(), DataValue::from(max_dist));

        let result = self.store.run_query(
            r#"?[id, text, embedding, category, severity, scope, compiled_type,
                compiled_predicate, active, created_by, violation_count] :=
                ~rules:rule_vec_idx{id | query: $q_vec, k: $k, ef: 50, bind_distance: dist},
                *rules{id, text, embedding, category, severity, scope,
                       compiled_type, compiled_predicate, active, created_by,
                       violation_count},
                active == true,
                dist < $max_dist
            :limit $k"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_rule_row(row)).collect()
    }

    /// Find active rules matching the given scope.
    fn find_active_rules(&self, scope: &str) -> Result<Vec<Rule>> {
        let mut params = BTreeMap::new();
        params.insert("scope".into(), DataValue::from(scope));

        let result = self.store.run_query(
            r#"?[id, text, embedding, category, severity, scope, compiled_type,
                compiled_predicate, active, created_by, violation_count] :=
                *rules{id, text, embedding, category, severity, scope,
                       compiled_type, compiled_predicate, active, created_by,
                       violation_count},
                active == true,
                (scope == $scope || scope == "*")"#,
            params,
        )?;

        result.rows.iter().map(|row| parse_rule_row(row)).collect()
    }

    /// Evaluate a Datalog-compiled rule.
    async fn evaluate_datalog_rule(
        &self,
        rule: &Rule,
        operation: &str,
        agent_id: &str,
        target_id: Option<Uuid>,
    ) -> Result<RuleEvaluation> {
        let start = Instant::now();

        let result = if !rule.compiled_predicate.is_empty() {
            // Bind operation context so compiled predicates can reference
            // $operation, $agent_id, $scope, $target_id
            let mut params = BTreeMap::new();
            params.insert("operation".into(), DataValue::from(operation));
            params.insert("agent_id".into(), DataValue::from(agent_id));
            params.insert("scope".into(), DataValue::from(rule.scope.as_str()));
            if let Some(tid) = target_id {
                params.insert("target_id".into(), DataValue::from(tid.to_string()));
            } else {
                params.insert("target_id".into(), DataValue::Null);
            }

            // Run the Datalog predicate with operation context
            match self.store.run_query(&rule.compiled_predicate, params) {
                Ok(r) => {
                    if r.rows.is_empty() {
                        EvalResult::Allow
                    } else {
                        // Non-empty result means the violation condition matched
                        match rule.severity {
                            Severity::Block => EvalResult::Block,
                            Severity::Warn => EvalResult::Warn,
                            Severity::Log => EvalResult::Allow,
                        }
                    }
                }
                // Fail-closed: a broken rule blocks rather than silently passing
                Err(_) => match rule.severity {
                    Severity::Block => EvalResult::Block,
                    _ => EvalResult::Warn,
                },
            }
        } else {
            EvalResult::Allow
        };

        let elapsed = start.elapsed().as_millis() as i64;

        Ok(RuleEvaluation {
            id: Uuid::now_v7(),
            ts: now_ms(),
            rule_id: rule.id,
            operation: operation.to_string(),
            agent_id: agent_id.to_string(),
            target_id,
            result,
            reason: format!("Datalog evaluation of rule '{}'", rule.text),
            evaluation_ms: elapsed,
        })
    }

    /// Evaluate an LLM-based rule.
    async fn evaluate_llm_rule(
        &self,
        rule: &Rule,
        operation: &str,
        agent_id: &str,
        content: &str,
        target_id: Option<Uuid>,
    ) -> Result<RuleEvaluation> {
        let start = Instant::now();

        let prompt = format!(
            "Rule: \"{}\"\n\n\
             Operation: {operation}\n\
             Agent: {agent_id}\n\
             Content: \"{content}\"\n\n\
             Does this operation violate the rule?\n\
             Respond with a JSON object: {{\"result\": \"allow\"|\"block\"|\"warn\", \"reason\": \"...\"}}",
            rule.text
        );

        let response = self
            .llm
            .complete(
                &[
                    Message {
                        role: "system".into(),
                        content: "You evaluate governance rules. Respond ONLY with JSON."
                            .into(),
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

        let (result, reason) = parse_rule_llm_response(&response.content, &rule.severity);
        let elapsed = start.elapsed().as_millis() as i64;

        Ok(RuleEvaluation {
            id: Uuid::now_v7(),
            ts: now_ms(),
            rule_id: rule.id,
            operation: operation.to_string(),
            agent_id: agent_id.to_string(),
            target_id,
            result,
            reason,
            evaluation_ms: elapsed,
        })
    }

    /// Record an evaluation in the rule_evaluations relation.
    fn record_evaluation(&self, eval: &RuleEvaluation) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(eval.id.to_string()));
        params.insert("ts".into(), DataValue::from(eval.ts));
        params.insert("rule_id".into(), DataValue::from(eval.rule_id.to_string()));
        params.insert("operation".into(), DataValue::from(eval.operation.as_str()));
        params.insert("agent_id".into(), DataValue::from(eval.agent_id.as_str()));

        if let Some(tid) = eval.target_id {
            params.insert("target_id".into(), DataValue::from(tid.to_string()));
        } else {
            params.insert("target_id".into(), DataValue::Null);
        }

        params.insert("result".into(), DataValue::from(eval.result.as_str()));
        params.insert("reason".into(), DataValue::from(eval.reason.as_str()));
        params.insert("evaluation_ms".into(), DataValue::from(eval.evaluation_ms));

        self.store.run_script(
            concat!(
                "?[id, ts, rule_id, operation, agent_id, target_id, result, reason, evaluation_ms] <- ",
                "[[$id, $ts, $rule_id, $operation, $agent_id, $target_id, $result, $reason, $evaluation_ms]] ",
                ":put rule_evaluations {id, ts => rule_id, operation, agent_id, target_id, result, reason, evaluation_ms}",
            ),
            params,
        )?;

        // Increment violation count if blocked or warned
        if matches!(eval.result, EvalResult::Block | EvalResult::Warn) {
            self.increment_violation_count(eval.rule_id)?;
        }

        Ok(())
    }

    /// Increment the violation count for a rule (two-step read-modify-write).
    fn increment_violation_count(&self, rule_id: Uuid) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("rule_id".into(), DataValue::from(rule_id.to_string()));

        let existing = self.store.run_query(
            r#"?[violation_count] :=
                *rules{id, violation_count},
                id = to_uuid($rule_id)"#,
            params.clone(),
        )?;

        let old_count = if existing.rows.is_empty() {
            0i64
        } else {
            existing.rows[0][0].get_int().unwrap_or(0)
        };

        params.insert("violation_count".into(), DataValue::from(old_count + 1));

        self.store.run_script(
            r#"?[id, valid_at, violation_count] <-
                [[$rule_id, 'ASSERT', $violation_count]]
            :update rules {id, valid_at => violation_count}"#,
            params,
        )?;

        Ok(())
    }
}

/// Parse the LLM response for a rule evaluation.
fn parse_rule_llm_response(content: &str, _severity: &Severity) -> (EvalResult, String) {
    #[derive(serde::Deserialize)]
    struct Response {
        result: Option<String>,
        reason: Option<String>,
    }

    // Try direct parse
    let parsed = serde_json::from_str::<Response>(content).ok().or_else(|| {
        // Try extracting JSON from markdown
        let trimmed = content.trim();
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                return serde_json::from_str::<Response>(&trimmed[start..=end]).ok();
            }
        }
        None
    });

    match parsed {
        Some(r) => {
            let result = r
                .result
                .map(|s| EvalResult::from_str_lossy(&s))
                .unwrap_or(EvalResult::Allow);
            let reason = r.reason.unwrap_or_default();
            (result, reason)
        }
        None => (
            EvalResult::Warn,
            "LLM response unparseable, defaulting to warn (fail-closed)".to_string(),
        ),
    }
}

/// Parse a CozoDB row into a Rule struct.
pub fn parse_rule_row(row: &[DataValue]) -> Result<Rule> {
    let id = parse_uuid(&row[0])?;
    let text = row[1].get_str().unwrap_or("").to_string();
    let embedding = parse_f32_vec(&row[2]);
    let category = row[3].get_str().unwrap_or("").to_string();
    let severity = Severity::from_str_lossy(row[4].get_str().unwrap_or("warn"));
    let scope = row[5].get_str().unwrap_or("*").to_string();
    let compiled_type = CompiledType::from_str_lossy(row[6].get_str().unwrap_or("llm"));
    let compiled_predicate = row[7].get_str().unwrap_or("").to_string();
    let active = row[8].get_bool().unwrap_or(true);
    let created_by = row[9].get_str().unwrap_or("").to_string();
    let violation_count = row[10].get_int().unwrap_or(0);

    Ok(Rule {
        id,
        text,
        embedding,
        category,
        severity,
        scope,
        compiled_type,
        compiled_predicate,
        active,
        created_by,
        violation_count,
    })
}

fn parse_uuid(val: &DataValue) -> Result<uuid::Uuid> {
    match val {
        DataValue::Str(s) => uuid::Uuid::parse_str(s.as_ref())
            .map_err(|e| MemoriaError::RuleViolation(format!("parsing uuid: {e}"))),
        DataValue::Uuid(u) => Ok(uuid::Uuid::from_bytes(*u.0.as_bytes())),
        _ => Err(MemoriaError::RuleViolation(format!(
            "expected uuid, got {val:?}"
        ))),
    }
}

fn parse_f32_vec(val: &DataValue) -> Vec<f32> {
    match val {
        DataValue::List(list) => list
            .iter()
            .map(|v| v.get_float().unwrap_or(0.0) as f32)
            .collect(),
        DataValue::Vec(v) => match v {
            cozo::Vector::F32(arr) => arr.to_vec(),
            cozo::Vector::F64(arr) => arr.iter().map(|&x| x as f32).collect(),
        },
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rule_llm_response_clean_json() {
        let json = r#"{"result": "block", "reason": "Contains PII"}"#;
        let (result, reason) = parse_rule_llm_response(json, &Severity::Block);
        assert_eq!(result, EvalResult::Block);
        assert_eq!(reason, "Contains PII");
    }

    #[test]
    fn test_parse_rule_llm_response_allow() {
        let json = r#"{"result": "allow", "reason": "No violation found"}"#;
        let (result, _) = parse_rule_llm_response(json, &Severity::Warn);
        assert_eq!(result, EvalResult::Allow);
    }

    #[test]
    fn test_parse_rule_llm_response_garbage() {
        let (result, _) = parse_rule_llm_response("I can't evaluate this", &Severity::Block);
        assert_eq!(result, EvalResult::Warn); // fail-closed: unparseable → warn
    }

    #[test]
    fn test_eval_result_roundtrip() {
        assert_eq!(EvalResult::from_str_lossy("allow"), EvalResult::Allow);
        assert_eq!(EvalResult::from_str_lossy("block"), EvalResult::Block);
        assert_eq!(EvalResult::from_str_lossy("deny"), EvalResult::Block);
        assert_eq!(EvalResult::from_str_lossy("warn"), EvalResult::Warn);
    }
}
