//! EFE-based skill selection: Expected Free Energy = pragmatic + β × epistemic.
//!
//! Pragmatic value measures expected success (confidence × success_rate).
//! Epistemic value measures information gain from trying the skill (outcome
//! entropy + novelty from low usage count).

use std::sync::Arc;

use crate::error::Result;
use crate::services::traits::Embedder;
use crate::skills::storage;
use crate::skills::Skill;
use crate::store::CozoStore;

/// A skill scored by Expected Free Energy.
#[derive(Debug, Clone)]
pub struct ScoredSkill {
    pub skill: Skill,
    pub efe_score: f64,
    pub pragmatic: f64,
    pub epistemic: f64,
    pub distance: f64,
}

/// Select skills for a task using EFE scoring.
///
/// 1. Embed the task description
/// 2. HNSW search for semantically similar skills
/// 3. Score each candidate with EFE = pragmatic + β × epistemic + γ × telos_alignment
/// 4. Return top k sorted by EFE descending
pub async fn select_skills(
    store: &CozoStore,
    embedder: &Arc<dyn Embedder>,
    task_description: &str,
    beta: f64,
    k: usize,
) -> Result<Vec<ScoredSkill>> {
    select_skills_with_gamma(store, embedder, task_description, beta, 0.0, k).await
}

/// Select skills with explicit telos alignment weight (γ).
///
/// EFE = pragmatic + β × epistemic + γ × telos_alignment
/// where telos_alignment is the max cosine similarity between the skill's
/// embedding and active telos embeddings, weighted by telos priority.
pub async fn select_skills_with_gamma(
    store: &CozoStore,
    embedder: &Arc<dyn Embedder>,
    task_description: &str,
    beta: f64,
    gamma: f64,
    k: usize,
) -> Result<Vec<ScoredSkill>> {
    // Step 1: Embed the task
    let embeddings = embedder
        .embed(&[task_description])
        .await
        .map_err(|e| crate::error::MemoriaError::Embedding(e.to_string()))?;

    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Find candidate skills by semantic similarity
    let candidates = storage::find_skills_by_embedding(store, &embeddings[0], k * 3, 0.8)?;

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2b: Load active telos for alignment computation (only if γ > 0)
    let active_telos = if gamma > 0.0 {
        store.list_active_telos("default", 20).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Step 3: Score each candidate with EFE + diversity bonus + telos alignment
    let mut scored: Vec<ScoredSkill> = candidates
        .into_iter()
        .map(|(skill, dist)| {
            let (pragmatic, epistemic) = compute_efe_components(store, &skill);
            // Diversity bonus: underexplored niches get epistemic boost
            let div_bonus = crate::skills::niche::diversity_bonus(store, skill.id)
                .unwrap_or(1.0);

            // Telos alignment: max similarity between task and active goals, weighted by priority
            // We use the task embedding (which the skill was matched to) as a proxy for
            // skill-goal alignment — a skill matched to a goal-aligned task is goal-aligned.
            let telos_alignment = if !active_telos.is_empty() {
                compute_telos_alignment(&embeddings[0], &active_telos)
            } else {
                0.0
            };

            let efe_score = pragmatic + beta * epistemic * div_bonus + gamma * telos_alignment;

            ScoredSkill {
                skill,
                efe_score,
                pragmatic,
                epistemic,
                distance: dist,
            }
        })
        .collect();

    // Step 4: Sort by EFE descending, take top k
    scored.sort_by(|a, b| b.efe_score.partial_cmp(&a.efe_score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    Ok(scored)
}

/// Compute pragmatic and epistemic components of EFE for a skill.
///
/// Pragmatic = precision-weighted expected success (precision × success_rate).
/// Epistemic = outcome_entropy + novelty (information gain).
///
/// Uses `aif::belief_update::precision()` to weight pragmatic value by how
/// well-established the skill is (confidence × ln(version + 1)), not just
/// raw confidence.
fn compute_efe_components(store: &CozoStore, skill: &Skill) -> (f64, f64) {
    let stats = storage::get_skill_usage_stats(store, skill.id).unwrap_or_default();

    // Pragmatic: precision-weighted expected success
    let success_rate = if stats.usage_count > 0 {
        stats.success_rate
    } else {
        0.5 // prior for untested skills
    };
    // Use canonical precision formula: well-reinforced skills weigh more
    let skill_precision = crate::aif::belief_update::precision(
        skill.confidence,
        skill.version as i64,
    );
    let pragmatic = skill_precision * success_rate;

    // Epistemic: information gain
    let usage_count = stats.usage_count as f64;
    let p = if usage_count > 0.0 { success_rate } else { 0.5 };

    // Outcome entropy: H(p) = -p·ln(p) - (1-p)·ln(1-p)
    let outcome_entropy = if p > 0.0 && p < 1.0 {
        -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
    } else {
        1.0 // maximum uncertainty for degenerate cases
    };

    // Novelty: inversely proportional to usage count
    let novelty = 1.0 / (usage_count + 1.0);

    let epistemic = outcome_entropy + novelty;

    (pragmatic, epistemic)
}

/// Compute telos alignment for a task embedding.
///
/// Returns the maximum cosine similarity between the task embedding
/// and any active telos embedding, weighted by the telos priority.
///
/// Uses the task embedding as a proxy for the skill — if a skill was
/// semantically matched to a goal-aligned task, the skill is goal-aligned.
fn compute_telos_alignment(
    task_embedding: &[f32],
    active_telos: &[crate::types::telos::Telos],
) -> f64 {
    if task_embedding.is_empty() || active_telos.is_empty() {
        return 0.0;
    }

    active_telos
        .iter()
        .filter(|t| !t.embedding.is_empty())
        .map(|t| {
            let sim = crate::store::cozo::cosine_similarity(task_embedding, &t.embedding);
            // Weight by priority × (1 - progress): achieved goals don't pull
            let goal_distance = 1.0 - t.progress;
            sim * t.priority * goal_distance
        })
        .fold(0.0_f64, f64::max)
}

/// Select skills for a specific niche context.
pub async fn select_skills_for_niche(
    store: &CozoStore,
    embedder: &Arc<dyn Embedder>,
    task_description: &str,
    desired_niche: &str,
    beta: f64,
    k: usize,
) -> Result<Vec<ScoredSkill>> {
    // Get base EFE scores
    let mut scored = select_skills(store, embedder, task_description, beta, k * 2).await?;

    // Boost skills that match the desired niche
    for s in &mut scored {
        if let Ok(niches) = crate::skills::niche::get_niches_for_skill(store, s.skill.id) {
            for n in &niches {
                if n.niche_key == desired_niche {
                    s.efe_score *= 1.0 + n.fitness; // niche fitness boost
                }
            }
        }
    }

    scored.sort_by(|a, b| b.efe_score.partial_cmp(&a.efe_score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_select_skills_empty_store() {
        let store = CozoStore::open_mem(4).unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(crate::services::mock::MockEmbedder::new(4));

        let result = select_skills(&store, &embedder, "test task", 1.0, 5).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_efe_components_no_usages() {
        let store = CozoStore::open_mem(4).unwrap();
        let skill = Skill::new("test", "test", vec![]);

        let (prag, epist) = compute_efe_components(&store, &skill);
        // No usages: pragmatic = precision(0.5, 1) * prior_success(0.5)
        // precision(0.5, 1) = 0.5 * ln(1 + 1) = 0.5 * ln(2) ≈ 0.347
        // pragmatic ≈ 0.347 * 0.5 ≈ 0.173
        let expected_precision = crate::aif::belief_update::precision(0.5, 1);
        assert!((prag - expected_precision * 0.5).abs() < 0.01);
        // Epistemic = entropy(0.5) + novelty(1.0) ≈ 0.693 + 1.0 = 1.693
        assert!(epist > 1.5);
    }
}
