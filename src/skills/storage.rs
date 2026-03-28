//! CozoDB CRUD operations for skills, usages, niches, lineage, recall contexts,
//! and task outcomes.

use cozo::DataValue;
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::skills::{
    Skill, SkillPerformance, SkillProvenance, SkillStep, SkillUsage,
};
use crate::store::CozoStore;

/// A record of what memories/facts were used for a task.
#[derive(Debug, Clone)]
pub struct RecallContext {
    pub task_id: Uuid,
    pub recall_id: Uuid,
    pub memory_ids: Vec<Uuid>,
    pub fact_ids: Vec<Uuid>,
    pub query_text: String,
    pub ts: i64,
}

/// The outcome of a task execution, with PES fields.
#[derive(Debug, Clone)]
pub struct TaskOutcome {
    pub task_id: Uuid,
    pub outcome: String,
    pub task_type: String,
    pub agent_id: String,
    pub failure_reason: Option<String>,
    pub duration_ms: Option<i64>,
    pub plan: String,
    pub summary: String,
    pub skills_used: Vec<Uuid>,
    pub adaptations: Vec<serde_json::Value>,
}

impl TaskOutcome {
    pub fn new(task_id: Uuid, outcome: &str, agent_id: &str) -> Self {
        Self {
            task_id,
            outcome: outcome.to_string(),
            task_type: String::new(),
            agent_id: agent_id.to_string(),
            failure_reason: None,
            duration_ms: None,
            plan: String::new(),
            summary: String::new(),
            skills_used: vec![],
            adaptations: vec![],
        }
    }
}

// ── Skill Storage ──

/// Store a new skill or update an existing one (via Validity versioning).
///
/// Validates all UUID fields before writing to prevent corrupt data in CozoDB
/// that would cause parse errors on subsequent reads.
pub fn store_skill(store: &CozoStore, skill: &Skill, embedding: &[f32]) -> Result<()> {
    // Validate skill ID is a proper UUID (should always be from Uuid::now_v7())
    if skill.id.is_nil() {
        return Err(MemoriaError::Skill("skill ID is nil UUID".to_string()));
    }

    // Filter source_episodes to only valid UUIDs (LLM-generated content may produce garbage)
    let valid_source_episodes: Vec<Uuid> = skill
        .source_episodes
        .iter()
        .filter(|id| !id.is_nil())
        .copied()
        .collect();

    // Validate parent_skill if present
    let valid_parent = skill.parent_skill.filter(|id| !id.is_nil());

    let embedding_vals: Vec<DataValue> = embedding.iter().map(|&v| DataValue::from(v as f64)).collect();
    let source_ep_vals: Vec<DataValue> = valid_source_episodes.iter().map(|id| DataValue::from(id.to_string())).collect();
    let tag_vals: Vec<DataValue> = skill.tags.iter().map(|t| DataValue::from(t.as_str())).collect();
    let steps_json = serde_json::to_string(&skill.steps).unwrap_or_else(|_| "[]".to_string());
    let preconditions_json = serde_json::to_string(&skill.preconditions).unwrap_or_else(|_| "[]".to_string());
    let postconditions_json = serde_json::to_string(&skill.postconditions).unwrap_or_else(|_| "[]".to_string());
    let performance_json = serde_json::to_string(&skill.performance).unwrap_or_else(|_| "{}".to_string());

    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(skill.id.to_string()));
    params.insert("name".into(), DataValue::from(skill.name.as_str()));
    params.insert("description".into(), DataValue::from(skill.description.as_str()));
    params.insert("embedding".into(), DataValue::List(embedding_vals));
    params.insert("steps".into(), DataValue::from(steps_json.as_str()));
    params.insert("preconditions".into(), DataValue::from(preconditions_json.as_str()));
    params.insert("postconditions".into(), DataValue::from(postconditions_json.as_str()));
    params.insert("confidence".into(), DataValue::from(skill.confidence));
    params.insert("provenance".into(), DataValue::from(skill.provenance.to_string().as_str()));
    params.insert("source_episodes".into(), DataValue::List(source_ep_vals));
    params.insert("domain".into(), DataValue::from(skill.domain.as_str()));
    params.insert("version".into(), DataValue::from(skill.version));
    params.insert("performance".into(), DataValue::from(performance_json.as_str()));
    if let Some(parent) = valid_parent {
        params.insert("parent_skill".into(), DataValue::from(parent.to_string()));
    } else {
        params.insert("parent_skill".into(), DataValue::Null);
    }
    params.insert("tags".into(), DataValue::List(tag_vals));

    store.run_script(
        concat!(
            "?[id, valid_at, name, description, embedding, steps, preconditions, postconditions, ",
            "confidence, provenance, source_episodes, domain, version, performance, parent_skill, tags] <- ",
            "[[$id, 'ASSERT', $name, $description, $embedding, $steps, $preconditions, $postconditions, ",
            "$confidence, $provenance, $source_episodes, $domain, $version, $performance, $parent_skill, $tags]] ",
            ":put skills {id, valid_at => name, description, embedding, steps, preconditions, postconditions, ",
            "confidence, provenance, source_episodes, domain, version, performance, parent_skill, tags}",
        ),
        params,
    )?;

    Ok(())
}

/// Retrieve a skill by ID (latest version via Validity).
pub fn get_skill(store: &CozoStore, id: Uuid) -> Result<Option<Skill>> {
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(id.to_string()));

    let result = store.run_query(
        r#"?[id, name, description, embedding, steps, preconditions, postconditions,
            confidence, provenance, source_episodes, domain, version, performance,
            parent_skill, tags] :=
            *skills{id, name, description, embedding, steps, preconditions, postconditions,
                    confidence, provenance, source_episodes, domain, version, performance,
                    parent_skill, tags, @ 'NOW'},
            id = to_uuid($id)"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    Ok(Some(parse_skill_row(&result.rows[0])?))
}

/// List all current skills (latest versions, confidence > 0).
pub fn list_all_skills(store: &CozoStore) -> Result<Vec<Skill>> {
    let result = store.run_query(
        r#"?[id, name, description, embedding, steps, preconditions, postconditions,
            confidence, provenance, source_episodes, domain, version, performance,
            parent_skill, tags] :=
            *skills{id, name, description, embedding, steps, preconditions, postconditions,
                    confidence, provenance, source_episodes, domain, version, performance,
                    parent_skill, tags, @ 'NOW'},
            confidence > 0.0
        :order -confidence
        :limit 200"#,
        BTreeMap::new(),
    )?;

    result
        .rows
        .iter()
        .map(|row| parse_skill_row(row))
        .collect()
}

/// Find skills by embedding similarity (HNSW search).
pub fn find_skills_by_embedding(
    store: &CozoStore,
    embedding: &[f32],
    k: usize,
    max_dist: f64,
) -> Result<Vec<(Skill, f64)>> {
    use ndarray::Array1;

    let arr = Array1::from_vec(embedding.to_vec());
    let query_vec = DataValue::Vec(cozo::Vector::F32(arr));

    let mut params = BTreeMap::new();
    params.insert("q_vec".into(), query_vec);
    params.insert("k".into(), DataValue::from(k as i64));
    params.insert("max_dist".into(), DataValue::from(max_dist));

    // Query layout: [dist, id, name, description, embedding, steps, ...]
    // Put dist first so that skill fields (id, name, ...) start at offset 1,
    // matching what parse_skill_row_with_offset expects (id at offset+0).
    let result = store.run_query(
        r#"?[dist, id, name, description, embedding, steps, preconditions, postconditions,
            confidence, provenance, source_episodes, domain, version, performance,
            parent_skill, tags] :=
            ~skills:skill_vec_idx{id | query: $q_vec, k: $k, ef: 50, bind_distance: dist},
            *skills{id, name, description, embedding, steps, preconditions, postconditions,
                    confidence, provenance, source_episodes, domain, version, performance,
                    parent_skill, tags, @ 'NOW'},
            dist < $max_dist
        :sort dist
        :limit $k"#,
        params,
    )?;

    result
        .rows
        .iter()
        .map(|row| {
            let dist = row[0].get_float().unwrap_or(1.0);
            let skill = parse_skill_row_with_offset(row, 1)?;
            Ok((skill, dist))
        })
        .collect()
}

/// Store a skill usage record.
pub fn store_skill_usage(store: &CozoStore, usage: &SkillUsage) -> Result<()> {
    let adaptations_json = serde_json::to_string(&usage.adaptations).unwrap_or_else(|_| "[]".to_string());

    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(usage.id.to_string()));
    params.insert("ts".into(), DataValue::from(usage.ts));
    params.insert("skill_id".into(), DataValue::from(usage.skill_id.to_string()));
    params.insert("skill_version".into(), DataValue::from(usage.skill_version));
    if let Some(ep_id) = usage.episode_id {
        params.insert("episode_id".into(), DataValue::from(ep_id.to_string()));
    } else {
        params.insert("episode_id".into(), DataValue::Null);
    }
    params.insert("agent_id".into(), DataValue::from(usage.agent_id.as_str()));
    params.insert("outcome".into(), DataValue::from(usage.outcome.to_string().as_str()));
    if let Some(dur) = usage.duration_ms {
        params.insert("duration_ms".into(), DataValue::from(dur));
    } else {
        params.insert("duration_ms".into(), DataValue::Null);
    }
    params.insert("context_summary".into(), DataValue::from(usage.context_summary.as_str()));
    params.insert("adaptations".into(), DataValue::from(adaptations_json.as_str()));

    store.run_script(
        concat!(
            "?[id, ts, skill_id, skill_version, episode_id, agent_id, outcome, ",
            "duration_ms, context_summary, adaptations] <- ",
            "[[$id, $ts, $skill_id, $skill_version, $episode_id, $agent_id, $outcome, ",
            "$duration_ms, $context_summary, $adaptations]] ",
            ":put skill_usages {id, ts => skill_id, skill_version, episode_id, agent_id, ",
            "outcome, duration_ms, context_summary, adaptations}",
        ),
        params,
    )?;

    Ok(())
}

/// Get skill usage statistics from the skill_usages relation.
pub fn get_skill_usage_stats(store: &CozoStore, skill_id: Uuid) -> Result<SkillPerformance> {
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));

    let result = store.run_query(
        r#"?[success_rate, avg_dur, usage_count, last_ts] :=
            *skill_usages{skill_id, outcome, duration_ms, ts},
            skill_id = to_uuid($skill_id),
            successes = count(if(outcome = "success", 1, null)),
            total = count(outcome),
            success_rate = if(total > 0, to_float(successes) / to_float(total), 0.0),
            avg_dur = mean(duration_ms),
            usage_count = total,
            last_ts = max(ts)"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Ok(SkillPerformance::default());
    }

    let row = &result.rows[0];
    Ok(SkillPerformance {
        success_rate: row[0].get_float().unwrap_or(0.0),
        avg_duration_ms: row[1].get_float().unwrap_or(0.0),
        usage_count: row[2].get_int().unwrap_or(0) as u64,
        last_used: row[3].get_int().unwrap_or(0),
    })
}

/// Performance stats for a specific skill version.
#[derive(Debug, Clone)]
pub struct VersionStats {
    pub version: i64,
    pub total: usize,
    pub successes: usize,
    pub success_rate: f64,
    pub avg_duration_ms: f64,
}

/// Comparison result between two skill versions.
#[derive(Debug, Clone)]
pub struct VersionComparison {
    pub version_a: i64,
    pub version_b: i64,
    pub stats_a: VersionStats,
    pub stats_b: VersionStats,
    pub success_rate_diff: f64,
    /// Whether the difference is statistically significant (chi-squared p < 0.05).
    pub significant: bool,
}

/// Get performance stats grouped by skill version.
pub fn get_skill_version_stats(
    store: &CozoStore,
    skill_id: Uuid,
) -> Result<Vec<VersionStats>> {
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));

    // CozoDB gotcha: aggregation functions must be in output columns
    let result = store.run_query(
        r#"?[version, total, successes, avg_dur] :=
            *skill_usages{skill_id, skill_version: version, outcome, duration_ms},
            skill_id = to_uuid($skill_id),
            total = count(outcome),
            successes = count(if(outcome == "success", 1, null)),
            avg_dur = mean(duration_ms)"#,
        params,
    )?;

    let mut stats = Vec::new();
    for row in &result.rows {
        let version = row[0].get_int().unwrap_or(1);
        let total = row[1].get_int().unwrap_or(0) as usize;
        let successes = row[2].get_int().unwrap_or(0) as usize;
        let success_rate = if total > 0 {
            successes as f64 / total as f64
        } else {
            0.0
        };
        let avg_duration_ms = row[3].get_float().unwrap_or(0.0);

        stats.push(VersionStats {
            version,
            total,
            successes,
            success_rate,
            avg_duration_ms,
        });
    }

    stats.sort_by_key(|s| s.version);
    Ok(stats)
}

/// Compare two skill versions using a chi-squared test for statistical significance.
pub fn compare_skill_versions(
    store: &CozoStore,
    skill_id: Uuid,
    version_a: i64,
    version_b: i64,
) -> Result<VersionComparison> {
    let all_stats = get_skill_version_stats(store, skill_id)?;

    let stats_a = all_stats
        .iter()
        .find(|s| s.version == version_a)
        .cloned()
        .unwrap_or(VersionStats {
            version: version_a,
            total: 0,
            successes: 0,
            success_rate: 0.0,
            avg_duration_ms: 0.0,
        });

    let stats_b = all_stats
        .iter()
        .find(|s| s.version == version_b)
        .cloned()
        .unwrap_or(VersionStats {
            version: version_b,
            total: 0,
            successes: 0,
            success_rate: 0.0,
            avg_duration_ms: 0.0,
        });

    let success_rate_diff = stats_b.success_rate - stats_a.success_rate;

    // Chi-squared test for independence on 2x2 contingency table:
    //              Success  Failure
    // Version A:    a        b
    // Version B:    c        d
    let significant = if stats_a.total >= 5 && stats_b.total >= 5 {
        let a = stats_a.successes as f64;
        let b = (stats_a.total - stats_a.successes) as f64;
        let c = stats_b.successes as f64;
        let d = (stats_b.total - stats_b.successes) as f64;
        let n = a + b + c + d;

        if n > 0.0 {
            // Chi-squared = n * (ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d))
            let numerator = n * (a * d - b * c).powi(2);
            let denominator = (a + b) * (c + d) * (a + c) * (b + d);
            if denominator > 0.0 {
                let chi2 = numerator / denominator;
                chi2 > 3.841 // p < 0.05 with 1 degree of freedom
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false // Not enough data
    };

    Ok(VersionComparison {
        version_a,
        version_b,
        stats_a,
        stats_b,
        success_rate_diff,
        significant,
    })
}

// ── Task Outcome Storage ──

/// Store a task outcome (with optional embeddings for HNSW search).
pub fn store_task_outcome(
    store: &CozoStore,
    outcome: &TaskOutcome,
    task_embedding: Option<&[f32]>,
    summary_embedding: Option<&[f32]>,
) -> Result<()> {
    let skills_vals: Vec<DataValue> = outcome.skills_used.iter().map(|id| DataValue::from(id.to_string())).collect();
    let adaptations_json = serde_json::to_string(&outcome.adaptations).unwrap_or_else(|_| "[]".to_string());

    let mut params = BTreeMap::new();
    params.insert("task_id".into(), DataValue::from(outcome.task_id.to_string()));
    params.insert("outcome".into(), DataValue::from(outcome.outcome.as_str()));
    params.insert("task_type".into(), DataValue::from(outcome.task_type.as_str()));

    if let Some(emb) = task_embedding {
        let vals: Vec<DataValue> = emb.iter().map(|&v| DataValue::from(v as f64)).collect();
        params.insert("task_embedding".into(), DataValue::List(vals));
    } else {
        params.insert("task_embedding".into(), DataValue::Null);
    }

    params.insert("agent_id".into(), DataValue::from(outcome.agent_id.as_str()));

    if let Some(ref reason) = outcome.failure_reason {
        params.insert("failure_reason".into(), DataValue::from(reason.as_str()));
    } else {
        params.insert("failure_reason".into(), DataValue::Null);
    }
    if let Some(dur) = outcome.duration_ms {
        params.insert("duration_ms".into(), DataValue::from(dur));
    } else {
        params.insert("duration_ms".into(), DataValue::Null);
    }

    params.insert("plan".into(), DataValue::from(outcome.plan.as_str()));
    params.insert("summary".into(), DataValue::from(outcome.summary.as_str()));

    if let Some(emb) = summary_embedding {
        let vals: Vec<DataValue> = emb.iter().map(|&v| DataValue::from(v as f64)).collect();
        params.insert("summary_embedding".into(), DataValue::List(vals));
    } else {
        params.insert("summary_embedding".into(), DataValue::Null);
    }

    params.insert("skills_used".into(), DataValue::List(skills_vals));
    params.insert("adaptations".into(), DataValue::from(adaptations_json.as_str()));

    store.run_script(
        concat!(
            "?[task_id, valid_at, outcome, task_type, task_embedding, agent_id, ",
            "failure_reason, duration_ms, plan, summary, summary_embedding, skills_used, adaptations] <- ",
            "[[$task_id, 'ASSERT', $outcome, $task_type, $task_embedding, $agent_id, ",
            "$failure_reason, $duration_ms, $plan, $summary, $summary_embedding, $skills_used, $adaptations]] ",
            ":put task_outcomes {task_id, valid_at => outcome, task_type, task_embedding, agent_id, ",
            "failure_reason, duration_ms, plan, summary, summary_embedding, skills_used, adaptations}",
        ),
        params,
    )?;

    Ok(())
}

/// Get the latest task outcome by task_id.
pub fn get_task_outcome(store: &CozoStore, task_id: Uuid) -> Result<Option<TaskOutcome>> {
    let mut params = BTreeMap::new();
    params.insert("task_id".into(), DataValue::from(task_id.to_string()));

    let result = store.run_query(
        r#"?[task_id, outcome, task_type, agent_id, failure_reason, duration_ms,
            plan, summary, skills_used, adaptations] :=
            *task_outcomes{task_id, outcome, task_type, agent_id, failure_reason,
                          duration_ms, plan, summary, skills_used, adaptations, @ 'NOW'},
            task_id = to_uuid($task_id)"#,
        params,
    )?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    Ok(Some(parse_task_outcome_row(&result.rows[0])?))
}

/// Find similar tasks by embedding (HNSW on task_outcomes:task_vec_idx).
pub fn find_similar_tasks(
    store: &CozoStore,
    task_embedding: &[f32],
    k: usize,
    max_dist: f64,
) -> Result<Vec<(TaskOutcome, f64)>> {
    use ndarray::Array1;

    let arr = Array1::from_vec(task_embedding.to_vec());
    let query_vec = DataValue::Vec(cozo::Vector::F32(arr));

    let mut params = BTreeMap::new();
    params.insert("q_vec".into(), query_vec);
    params.insert("k".into(), DataValue::from(k as i64));
    params.insert("max_dist".into(), DataValue::from(max_dist));

    // Put dist first so task_outcome fields (task_id, outcome, ...) start at offset 1
    let result = store.run_query(
        r#"?[dist, task_id, outcome, task_type, agent_id, failure_reason, duration_ms,
            plan, summary, skills_used, adaptations] :=
            ~task_outcomes:task_vec_idx{task_id | query: $q_vec, k: $k, ef: 50, bind_distance: dist},
            *task_outcomes{task_id, outcome, task_type, agent_id, failure_reason,
                          duration_ms, plan, summary, skills_used, adaptations, @ 'NOW'},
            dist < $max_dist
        :sort dist
        :limit $k"#,
        params,
    )?;

    result
        .rows
        .iter()
        .map(|row| {
            let dist = row[0].get_float().unwrap_or(1.0);
            let to = parse_task_outcome_row_with_offset(row, 1)?;
            Ok((to, dist))
        })
        .collect()
}

// ── Recall Context Storage ──

/// Store a recall context (what memories/facts were used for a task).
pub fn store_recall_context(store: &CozoStore, ctx: &RecallContext) -> Result<()> {
    let mem_vals: Vec<DataValue> = ctx.memory_ids.iter().map(|id| DataValue::from(id.to_string())).collect();
    let fact_vals: Vec<DataValue> = ctx.fact_ids.iter().map(|id| DataValue::from(id.to_string())).collect();

    let mut params = BTreeMap::new();
    params.insert("task_id".into(), DataValue::from(ctx.task_id.to_string()));
    params.insert("recall_id".into(), DataValue::from(ctx.recall_id.to_string()));
    params.insert("memory_ids".into(), DataValue::List(mem_vals));
    params.insert("fact_ids".into(), DataValue::List(fact_vals));
    params.insert("query_text".into(), DataValue::from(ctx.query_text.as_str()));
    params.insert("ts".into(), DataValue::from(ctx.ts));

    store.run_script(
        concat!(
            "?[task_id, recall_id, memory_ids, fact_ids, query_text, ts] <- ",
            "[[$task_id, $recall_id, $memory_ids, $fact_ids, $query_text, $ts]] ",
            ":put recall_contexts {task_id, recall_id => memory_ids, fact_ids, query_text, ts}",
        ),
        params,
    )?;

    Ok(())
}

/// Get recall contexts for a task.
pub fn get_recall_contexts(store: &CozoStore, task_id: Uuid) -> Result<Vec<RecallContext>> {
    let mut params = BTreeMap::new();
    params.insert("task_id".into(), DataValue::from(task_id.to_string()));

    let result = store.run_query(
        r#"?[task_id, recall_id, memory_ids, fact_ids, query_text, ts] :=
            *recall_contexts{task_id, recall_id, memory_ids, fact_ids, query_text, ts},
            task_id = to_uuid($task_id)"#,
        params,
    )?;

    result.rows.iter().map(|row| parse_recall_context_row(row)).collect()
}

/// Find all memory IDs that were used in recall contexts for a given task.
pub fn find_recall_contexts_for_task(store: &CozoStore, task_id: Uuid) -> Result<Vec<Uuid>> {
    let contexts = get_recall_contexts(store, task_id)?;
    let mut memory_ids: Vec<Uuid> = contexts.iter()
        .flat_map(|ctx| ctx.memory_ids.iter().copied())
        .collect();
    memory_ids.sort();
    memory_ids.dedup();
    Ok(memory_ids)
}

/// Update skill confidence (using Validity versioning).
pub fn update_skill_confidence(store: &CozoStore, skill_id: Uuid, new_confidence: f64) -> Result<()> {
    // Read current skill, update confidence, write back
    let skill = get_skill(store, skill_id)?
        .ok_or_else(|| MemoriaError::Skill(format!("skill not found: {skill_id}")))?;

    let mut updated = skill;
    updated.confidence = new_confidence;

    // Re-embed is not needed since we're just updating confidence.
    // We need the current embedding to write back.
    let mut params = BTreeMap::new();
    params.insert("id".into(), DataValue::from(skill_id.to_string()));

    let emb_result = store.run_query(
        r#"?[embedding] :=
            *skills{id, embedding, @ 'NOW'},
            id = to_uuid($id)"#,
        params,
    )?;

    if emb_result.rows.is_empty() {
        return Err(MemoriaError::Skill(format!("skill embedding not found: {skill_id}")));
    }

    let embedding = parse_f32_vec(&emb_result.rows[0][0]);
    store_skill(store, &updated, &embedding)?;

    Ok(())
}

/// Count all skills.
pub fn count_skills(store: &CozoStore) -> Result<usize> {
    let result = store.run_query(
        "?[count(id)] := *skills{id, @ 'NOW'}",
        BTreeMap::new(),
    )?;

    Ok(result.rows.first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0) as usize)
}

// ── Row Parsers ──

fn parse_skill_row(row: &[DataValue]) -> Result<Skill> {
    parse_skill_row_with_offset(row, 0)
}

fn parse_skill_row_with_offset(row: &[DataValue], offset: usize) -> Result<Skill> {
    let id = crate::store::parse_uuid_pub(&row[offset])
        .map_err(|e| MemoriaError::Skill(format!("invalid skill UUID: {e}")))?;
    let name = parse_string(&row[offset + 1]);
    let description = parse_string(&row[offset + 2]);
    let _embedding = &row[offset + 3]; // skip embedding in parsed struct
    let steps: Vec<SkillStep> = serde_json::from_str(&datavalue_to_json_string(&row[offset + 4])).unwrap_or_default();
    let preconditions: Vec<serde_json::Value> = serde_json::from_str(&datavalue_to_json_string(&row[offset + 5])).unwrap_or_default();
    let postconditions: Vec<serde_json::Value> = serde_json::from_str(&datavalue_to_json_string(&row[offset + 6])).unwrap_or_default();
    let confidence = row[offset + 7].get_float().unwrap_or(0.5);
    let provenance = SkillProvenance::from_str(&parse_string(&row[offset + 8]));
    let source_episodes = parse_uuid_list(&row[offset + 9]);
    let domain = parse_string(&row[offset + 10]);
    let version = row[offset + 11].get_int().unwrap_or(1);
    let performance: SkillPerformance = serde_json::from_str(&datavalue_to_json_string(&row[offset + 12])).unwrap_or_default();
    let parent_skill = if row[offset + 13] == DataValue::Null {
        None
    } else {
        crate::store::parse_uuid_pub(&row[offset + 13]).ok()
    };
    let tags = parse_string_list(&row[offset + 14]);

    Ok(Skill {
        id,
        name,
        description,
        steps,
        preconditions,
        postconditions,
        confidence,
        provenance,
        source_episodes,
        domain,
        version,
        performance,
        parent_skill,
        tags,
    })
}

fn parse_task_outcome_row(row: &[DataValue]) -> Result<TaskOutcome> {
    parse_task_outcome_row_with_offset(row, 0)
}

fn parse_task_outcome_row_with_offset(row: &[DataValue], offset: usize) -> Result<TaskOutcome> {
    let task_id = crate::store::parse_uuid_pub(&row[offset])?;
    let outcome = parse_string(&row[offset + 1]);
    let task_type = parse_string(&row[offset + 2]);
    let agent_id = parse_string(&row[offset + 3]);
    let failure_reason = if row[offset + 4] == DataValue::Null {
        None
    } else {
        Some(parse_string(&row[offset + 4]))
    };
    let duration_ms = if row[offset + 5] == DataValue::Null {
        None
    } else {
        row[offset + 5].get_int()
    };
    let plan = parse_string(&row[offset + 6]);
    let summary = parse_string(&row[offset + 7]);
    let skills_used = parse_uuid_list(&row[offset + 8]);
    let adaptations: Vec<serde_json::Value> = serde_json::from_str(&parse_string(&row[offset + 9])).unwrap_or_default();

    Ok(TaskOutcome {
        task_id,
        outcome,
        task_type,
        agent_id,
        failure_reason,
        duration_ms,
        plan,
        summary,
        skills_used,
        adaptations,
    })
}

fn parse_recall_context_row(row: &[DataValue]) -> Result<RecallContext> {
    let task_id = crate::store::parse_uuid_pub(&row[0])?;
    let recall_id = crate::store::parse_uuid_pub(&row[1])?;
    let memory_ids = parse_uuid_list(&row[2]);
    let fact_ids = parse_uuid_list(&row[3]);
    let query_text = parse_string(&row[4]);
    let ts = row[5].get_int().unwrap_or(0);

    Ok(RecallContext {
        task_id,
        recall_id,
        memory_ids,
        fact_ids,
        query_text,
        ts,
    })
}

// ── Local Helpers ──

fn parse_string(val: &DataValue) -> String {
    val.get_str().unwrap_or("").to_string()
}

fn parse_f32_vec(val: &DataValue) -> Vec<f32> {
    match val {
        DataValue::List(list) => list.iter().map(|v| v.get_float().unwrap_or(0.0) as f32).collect(),
        DataValue::Vec(v) => match v {
            cozo::Vector::F32(arr) => arr.to_vec(),
            cozo::Vector::F64(arr) => arr.iter().map(|&x| x as f32).collect(),
        },
        _ => Vec::new(),
    }
}

fn parse_uuid_list(val: &DataValue) -> Vec<Uuid> {
    match val {
        DataValue::List(list) => list
            .iter()
            .filter_map(|v| match v {
                DataValue::Str(s) => Uuid::parse_str(s.as_ref()).ok(),
                DataValue::Uuid(u) => Some(Uuid::from_bytes(*u.0.as_bytes())),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_string_list(val: &DataValue) -> Vec<String> {
    match val {
        DataValue::List(list) => list.iter().filter_map(|v| v.get_str().map(|s| s.to_string())).collect(),
        _ => Vec::new(),
    }
}

/// Convert a CozoDB DataValue (which may be a Json-wrapped value) to a JSON string.
///
/// CozoDB's `Json` column type returns `DataValue::Json(JsonData)` which has
/// a `Debug` format like `json("...")`. We extract the inner JSON string.
fn datavalue_to_json_string(val: &DataValue) -> String {
    // First try get_str for plain strings
    if let Some(s) = val.get_str() {
        return s.to_string();
    }

    // For CozoDB Json type, the Debug format is json("..."), but we need
    // to convert it to a proper JSON string. Use the Display format or
    // extract from the debug representation.
    let debug = format!("{val:?}");

    // CozoDB Json values have format: json("escaped_json_string")
    if debug.starts_with("json(\"") && debug.ends_with("\")") {
        let inner = &debug[6..debug.len() - 2];
        // Unescape the inner string
        inner.replace("\\\"", "\"").replace("\\\\", "\\")
    } else if debug.starts_with("json(") && debug.ends_with(")") {
        // json(null) or json(123) etc
        debug[5..debug.len() - 1].to_string()
    } else {
        // Fallback: try to use it as-is
        debug
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{SkillOutcome, SkillStep};

    #[test]
    fn test_store_and_get_skill() {
        let store = CozoStore::open_mem(4).unwrap();
        let mut skill = Skill::new(
            "debug_auth",
            "Debug authentication issues",
            vec![
                SkillStep { step: 1, action: "Check logs".into() },
                SkillStep { step: 2, action: "Verify tokens".into() },
            ],
        );
        skill.domain = "security".into();
        skill.tags = vec!["auth".into(), "debug".into()];

        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        store_skill(&store, &skill, &embedding).unwrap();

        let retrieved = get_skill(&store, skill.id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "debug_auth");
        assert_eq!(retrieved.description, "Debug authentication issues");
        assert_eq!(retrieved.steps.len(), 2);
        assert_eq!(retrieved.domain, "security");
        assert_eq!(retrieved.confidence, 0.5);
    }

    #[test]
    fn test_store_skill_usage() {
        let store = CozoStore::open_mem(4).unwrap();
        let skill_id = Uuid::now_v7();
        let usage = SkillUsage::new(skill_id, 1, "agent-1", SkillOutcome::Success);
        store_skill_usage(&store, &usage).unwrap();
    }

    #[test]
    fn test_store_and_get_task_outcome() {
        let store = CozoStore::open_mem(4).unwrap();
        let task_id = Uuid::now_v7();
        let mut outcome = TaskOutcome::new(task_id, "success", "agent-1");
        outcome.plan = "debug the auth module".into();
        outcome.summary = "found null pointer in validator".into();

        let task_emb = vec![0.1, 0.2, 0.3, 0.4];
        store_task_outcome(&store, &outcome, Some(&task_emb), None).unwrap();

        let retrieved = get_task_outcome(&store, task_id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.outcome, "success");
        assert_eq!(retrieved.plan, "debug the auth module");
    }

    #[test]
    fn test_store_recall_context() {
        let store = CozoStore::open_mem(4).unwrap();
        let ctx = RecallContext {
            task_id: Uuid::now_v7(),
            recall_id: Uuid::now_v7(),
            memory_ids: vec![Uuid::now_v7()],
            fact_ids: vec![],
            query_text: "test query".into(),
            ts: crate::types::memory::now_ms(),
        };
        store_recall_context(&store, &ctx).unwrap();

        let retrieved = get_recall_contexts(&store, ctx.task_id).unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].query_text, "test query");
    }

    #[test]
    fn test_count_skills() {
        let store = CozoStore::open_mem(4).unwrap();
        assert_eq!(count_skills(&store).unwrap(), 0);

        let skill = Skill::new("test", "test skill", vec![]);
        store_skill(&store, &skill, &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(count_skills(&store).unwrap(), 1);
    }

    #[test]
    fn test_update_skill_confidence() {
        let store = CozoStore::open_mem(4).unwrap();
        let skill = Skill::new("test", "test skill", vec![]);
        let id = skill.id;
        store_skill(&store, &skill, &[0.1, 0.2, 0.3, 0.4]).unwrap();

        update_skill_confidence(&store, id, 0.9).unwrap();
        let updated = get_skill(&store, id).unwrap().unwrap();
        assert!((updated.confidence - 0.9).abs() < 0.01);
    }
}
