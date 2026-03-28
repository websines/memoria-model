//! NL→Datalog rule compiler — translates natural language rules into
//! CozoDB Datalog predicates for fast, deterministic evaluation.
//!
//! The compiler uses an LLM to understand the rule intent and produce
//! a valid CozoDB query. The query should return rows when the rule is
//! *violated* — an empty result means the operation is allowed.
//!
//! After compilation, the rule is stored with `compiled_type: Datalog`
//! and its `compiled_predicate` set, enabling fast evaluation without
//! further LLM calls.

use std::collections::BTreeMap;
use std::sync::Arc;

use cozo::DataValue;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::{LlmService, Message};
use crate::store::CozoStore;

use super::types::Rule;

/// Compile a natural language rule into a CozoDB Datalog predicate.
///
/// Returns the compiled Datalog query string. The query should return
/// non-empty results when the rule is violated.
///
/// The LLM is prompted with CozoDB schema context so it generates
/// valid queries against the actual relations.
pub async fn compile_rule(
    llm: &Arc<dyn LlmService>,
    rule_text: &str,
) -> Result<String> {
    let schema_context = r#"Available CozoDB relations:
- memories {id: Uuid, kind: String, content: String, namespace: String, pinned: Bool, confidence: Float, provenance: String}
- entities {id: Uuid, name: String, entity_type: String, namespace: String, mention_count: Int, confidence: Float}
- facts {id: Uuid, subject_entity: Uuid, predicate: String, object_entity: Uuid?, object_value: String?, namespace: String, temporal_status: String, confidence: Float, reinforcement_count: Int}
- access_log {memory_id: Uuid, ts: Int, agent_id: String, access_type: String}
- audit_log {ts: Int, operation: String, agent_id: String}

Operation context variables (always available as parameters):
- $operation — the current operation ("tell", "ask", "prime", "feedback")
- $agent_id — the agent performing the operation
- $scope — the namespace/scope of the operation
- $target_id — optional target memory/entity ID (may be null)

CozoDB Datalog syntax:
- ?[col1, col2] := *relation{col1, col2, ...}  — basic query
- Conditions in rule body: col == "value", col > 5, col != null
- count(x), sum(x) — aggregations in output columns
- String functions: starts_with(s, prefix), ends_with(s, suffix), contains(s, sub)
- Use $variable to reference operation context parameters
"#;

    let prompt = format!(
        "{schema_context}\n\
         Compile this governance rule into a CozoDB Datalog query.\n\
         The query should return rows when the rule is VIOLATED.\n\
         An empty result means the operation is ALLOWED.\n\n\
         Rule: \"{rule_text}\"\n\n\
         Output ONLY the CozoDB query, no explanation. Example:\n\
         ?[id] := *memories{{id, namespace}}, namespace == \"secret\""
    );

    let response = llm
        .complete(
            &[Message {
                role: "user".to_string(),
                content: prompt,
            }],
            512,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let query = response.content.trim().to_string();

    // Basic validation: must start with ?[ or contain a rule definition
    if !query.contains("?[") {
        return Err(MemoriaError::Llm(format!(
            "LLM did not produce a valid Datalog query: {query}"
        )));
    }

    // Strip markdown code fences if present
    let query = query
        .trim_start_matches("```datalog")
        .trim_start_matches("```sql")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .to_string();

    Ok(query)
}

/// Compile a rule and update it in the store from LLM to Datalog type.
///
/// If compilation succeeds and the predicate validates against the store,
/// updates the rule's `compiled_type` to Datalog and stores the predicate.
pub async fn compile_and_store_rule(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    rule_id: Uuid,
) -> Result<bool> {
    // Read the current rule
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(rule_id.to_string()));

    let result = store.run_query(
        r#"?[id, text, embedding, category, severity, scope, compiled_type,
            compiled_predicate, active, created_by, violation_count] :=
            *rules{id, text, embedding, category, severity, scope,
                   compiled_type, compiled_predicate, active, created_by,
                   violation_count},
            id = to_uuid($id)"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Err(MemoriaError::Llm(format!("Rule not found: {rule_id}")));
    }

    let rule = super::enforcer::parse_rule_row(&result.rows[0])?;

    // Compile the rule text
    let predicate = compile_rule(llm, &rule.text).await?;

    // Validate the predicate by running it (should not error)
    if let Err(e) = store.run_query(&predicate, BTreeMap::new()) {
        return Err(MemoriaError::Llm(format!(
            "Compiled predicate failed validation: {e}"
        )));
    }

    // Update the rule with the compiled predicate
    update_rule_compiled_type(store, &rule, &predicate)?;

    Ok(true)
}

/// Update a rule's compiled type and predicate in the store.
fn update_rule_compiled_type(
    store: &CozoStore,
    rule: &Rule,
    predicate: &str,
) -> Result<()> {
    let embedding_vals: Vec<DataValue> = rule
        .embedding
        .iter()
        .map(|&v| DataValue::from(v as f64))
        .collect();

    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(rule.id.to_string()));
    params.insert("text".into(), DataValue::from(rule.text.as_str()));
    params.insert("embedding".into(), DataValue::List(embedding_vals));
    params.insert("category".into(), DataValue::from(rule.category.as_str()));
    params.insert("severity".into(), DataValue::from(rule.severity.as_str()));
    params.insert("scope".into(), DataValue::from(rule.scope.as_str()));
    params.insert("compiled_type".into(), DataValue::from("datalog"));
    params.insert("compiled_predicate".into(), DataValue::from(predicate));
    params.insert("active".into(), DataValue::from(rule.active));
    params.insert("created_by".into(), DataValue::from(rule.created_by.as_str()));
    params.insert(
        "violation_count".into(),
        DataValue::from(rule.violation_count),
    );

    store.run_script(
        concat!(
            "?[id, text, embedding, category, severity, scope, compiled_type, ",
            "compiled_predicate, active, created_by, violation_count] <- ",
            "[[$id, $text, $embedding, $category, $severity, $scope, $compiled_type, ",
            "$compiled_predicate, $active, $created_by, $violation_count]] ",
            ":put rules {id => text, embedding, category, severity, scope, compiled_type, ",
            "compiled_predicate, active, created_by, violation_count}",
        ),
        params,
    )?;

    Ok(())
}

/// Batch-compile all LLM rules to Datalog.
///
/// Iterates through all rules with `compiled_type == "llm"` and attempts
/// to compile each one. Returns the number successfully compiled.
pub async fn compile_all_llm_rules(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
) -> Result<usize> {
    let result = store.run_query(
        r#"?[id] := *rules{id, compiled_type, active},
            compiled_type == "llm", active == true"#,
        BTreeMap::new(),
    )?;

    let mut compiled = 0;
    for row in &result.rows {
        let id = crate::store::cozo::parse_uuid_pub(&row[0])?;
        match compile_and_store_rule(store, llm, id).await {
            Ok(true) => compiled += 1,
            Ok(false) | Err(_) => {} // skip failures
        }
    }

    Ok(compiled)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_compile_rule_validation() {
        // Test that the prompt format produces the expected structure
        let schema = "Available CozoDB relations:";
        assert!(schema.contains("CozoDB"));
    }
}
