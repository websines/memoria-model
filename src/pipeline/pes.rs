//! Plan-Execute-Summarize (PES) loop.
//!
//! Every task execution follows a structured cognitive loop:
//! 1. PLAN: Retrieve lineage context + relevant skills → build plan
//! 2. EXECUTE: Record what memories/skills/facts were provided → return for external execution
//! 3. SUMMARIZE: After outcome received, LLM structured reflection → store as task_outcome

use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::{Embedder, LlmService, Message};
use crate::skills::selection::{self, ScoredSkill};
use crate::skills::storage::{self, RecallContext, TaskOutcome};
use crate::store::CozoStore;
use crate::types::memory::now_ms;
use crate::types::query::AgentContext;

/// Result of the Plan phase of PES.
#[derive(Debug)]
pub struct PesResult {
    pub plan: String,
    pub selected_skills: Vec<ScoredSkill>,
    pub recall_context_id: Uuid,
    pub task_id: Uuid,
}

/// Phase 1: Plan — select skills and build a plan for the task.
pub async fn plan(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    task_id: Uuid,
    task_description: &str,
    _ctx: &AgentContext,
    beta: f64,
) -> Result<PesResult> {
    // Select relevant skills via EFE
    let selected_skills = selection::select_skills(store, embedder, task_description, beta, 5).await?;

    // Build plan from selected skills and their lineage
    let mut plan_parts = Vec::new();
    for scored in &selected_skills {
        let skill = &scored.skill;
        plan_parts.push(format!(
            "Skill: {} (confidence: {:.2}, EFE: {:.2})",
            skill.name, skill.confidence, scored.efe_score
        ));

        // Include lineage context if available
        let ancestry = crate::skills::lineage::get_full_ancestry(store, skill.id, 5)?;
        for entry in &ancestry {
            if !entry.mutation_summary.is_empty() {
                plan_parts.push(format!(
                    "  Previous version note: {}",
                    entry.mutation_summary
                ));
            }
        }

        for step in &skill.steps {
            plan_parts.push(format!("  Step {}: {}", step.step, step.action));
        }
    }

    let skill_context = plan_parts.join("\n");

    // LLM builds the plan
    let plan = if !skill_context.is_empty() {
        let response = llm
            .complete(
                &[
                    Message {
                        role: "system".into(),
                        content: concat!(
                            "Given relevant skills and a task, create a concrete plan. ",
                            "Output a numbered plan with specific actions. ",
                            "Note any past failures from skill lineage to avoid."
                        )
                        .into(),
                    },
                    Message {
                        role: "user".into(),
                        content: format!(
                            "Task: {task_description}\n\nAvailable skills:\n{skill_context}"
                        ),
                    },
                ],
                2048,
            )
            .await
            .map_err(|e| MemoriaError::Llm(e.to_string()))?;
        response.content
    } else {
        format!("No matching skills found. Execute task directly: {task_description}")
    };

    // Record recall context
    let recall_id = Uuid::now_v7();
    let recall_context = RecallContext {
        task_id,
        recall_id,
        memory_ids: vec![], // filled during execution
        fact_ids: vec![],
        query_text: task_description.to_string(),
        ts: now_ms(),
    };
    storage::store_recall_context(store, &recall_context)?;

    Ok(PesResult {
        plan,
        selected_skills,
        recall_context_id: recall_id,
        task_id,
    })
}

/// Phase 3: Summarize — structured reflection after task outcome.
pub async fn summarize(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    task_id: Uuid,
    plan: &str,
    outcome: &str,
    failure_reason: Option<&str>,
    ctx: &AgentContext,
    duration_ms: Option<i64>,
    skills_used: &[Uuid],
) -> Result<TaskOutcome> {
    // LLM structured abductive reflection
    // Enhanced prompt with full plan context, divergence analysis, causal hypotheses,
    // and counterfactual reasoning for deeper learning from outcomes.
    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: concat!(
                        "Perform abductive reflection on a task execution. Analyze:\n\n",
                        "1. **Plan-Outcome Alignment**: Did the outcome match the plan? ",
                        "Identify each step that succeeded, failed, or was skipped.\n\n",
                        "2. **Divergence Analysis**: What diverged from prediction? ",
                        "Was the divergence due to missing knowledge, incorrect assumptions, ",
                        "or environmental changes?\n\n",
                        "3. **Causal Hypotheses**: Generate 2-3 causal hypotheses for why ",
                        "the outcome occurred. Rank by plausibility.\n\n",
                        "4. **Counterfactual Analysis**: What would have happened if a key ",
                        "decision had been made differently? Identify the highest-leverage ",
                        "intervention point.\n\n",
                        "5. **Actionable Advice**: Provide 1-2 sentences of specific advice ",
                        "for the next similar task, incorporating the causal analysis.\n\n",
                        "Output a concise structured summary covering all 5 points."
                    )
                    .into(),
                },
                Message {
                    role: "user".into(),
                    content: format!(
                        "Task ID: {task_id}\n\nPlan:\n{plan}\n\nOutcome: {outcome}\n{}\n\n\
                         Skills used: {} skill(s)\nDuration: {}",
                        failure_reason
                            .map(|r| format!("Failure reason: {r}"))
                            .unwrap_or_default(),
                        skills_used.len(),
                        duration_ms
                            .map(|d| format!("{}ms", d))
                            .unwrap_or_else(|| "unknown".to_string()),
                    ),
                },
            ],
            2048,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let summary = response.content;

    // Embed the task description and summary for future retrieval
    let texts: Vec<&str> = vec![plan, summary.as_str()];
    let embeddings = embedder
        .embed(&texts)
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    let task_emb = embeddings.first().map(|e| e.as_slice());
    let summary_emb = embeddings.get(1).map(|e| e.as_slice());

    // Build and store the outcome
    let mut task_outcome = TaskOutcome::new(task_id, outcome, &ctx.agent_id);
    task_outcome.plan = plan.to_string();
    task_outcome.summary = summary;
    task_outcome.failure_reason = failure_reason.map(String::from);
    task_outcome.duration_ms = duration_ms;
    task_outcome.skills_used = skills_used.to_vec();

    storage::store_task_outcome(store, &task_outcome, task_emb, summary_emb)?;

    Ok(task_outcome)
}
