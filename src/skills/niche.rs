//! Behavioral niche preservation (MAP-Elites).
//!
//! Don't just keep the "best" skill — preserve diverse approaches that solve
//! the same problem in different ways. Different contexts call for different
//! behavioral niches (quick vs thorough, autonomous vs interactive, etc.).

use cozo::DataValue;
use std::collections::BTreeMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::{LlmService, Message};
use crate::store::CozoStore;

/// A skill's placement in a behavioral niche.
#[derive(Debug, Clone)]
pub struct SkillNiche {
    pub skill_id: Uuid,
    pub niche_key: String,
    pub feature_vector: serde_json::Value,
    pub fitness: f64,
    pub usage_count: i64,
}

/// Classify a skill into a behavioral niche via LLM analysis.
pub async fn classify_niche(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    skill_id: Uuid,
) -> Result<SkillNiche> {
    let skill = crate::skills::storage::get_skill(store, skill_id)?
        .ok_or_else(|| MemoriaError::Skill(format!("skill not found: {skill_id}")))?;

    let steps_json = serde_json::to_string(&skill.steps).unwrap_or_default();

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: concat!(
                        "Classify this skill into a behavioral niche. ",
                        "Consider these dimensions: speed (quick/thorough), ",
                        "autonomy (autonomous/interactive), risk (safe/aggressive), ",
                        "scope (targeted/comprehensive). ",
                        "Output a JSON object:\n",
                        r#"{"niche_key": "domain:style", "features": {"speed": "quick", "autonomy": "autonomous", "risk": "safe", "scope": "targeted"}}"#
                    )
                    .into(),
                },
                Message {
                    role: "user".into(),
                    content: format!(
                        "Skill: {} — {}\nSteps: {}\nDomain: {}",
                        skill.name, skill.description, steps_json, skill.domain
                    ),
                },
            ],
            512,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let parsed: serde_json::Value = serde_json::from_str(&response.content)
        .unwrap_or_else(|_| serde_json::json!({"niche_key": "general:default", "features": {}}));

    let niche_key = parsed["niche_key"]
        .as_str()
        .unwrap_or("general:default")
        .to_string();
    let features = parsed["features"].clone();

    let niche = SkillNiche {
        skill_id,
        niche_key: niche_key.clone(),
        feature_vector: features,
        fitness: skill.confidence * skill.performance.success_rate.max(0.5),
        usage_count: 0,
    };

    store_niche(store, &niche)?;

    Ok(niche)
}

/// Store or update a skill niche entry.
pub fn store_niche(store: &CozoStore, niche: &SkillNiche) -> Result<()> {
    let features_json = serde_json::to_string(&niche.feature_vector).unwrap_or_else(|_| "{}".to_string());

    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(niche.skill_id.to_string()));
    params.insert("niche_key".into(), DataValue::from(niche.niche_key.as_str()));
    params.insert("feature_vector".into(), DataValue::from(features_json.as_str()));
    params.insert("fitness".into(), DataValue::from(niche.fitness));
    params.insert("usage_count".into(), DataValue::from(niche.usage_count));

    store.run_script(
        concat!(
            "?[skill_id, niche_key, feature_vector, fitness, usage_count] <- ",
            "[[$skill_id, $niche_key, $feature_vector, $fitness, $usage_count]] ",
            ":put skill_niches {skill_id, niche_key => feature_vector, fitness, usage_count}",
        ),
        params,
    )?;

    Ok(())
}

/// Update niche fitness (only if the new fitness is higher).
pub fn update_niche_fitness(
    store: &CozoStore,
    skill_id: Uuid,
    niche_key: &str,
    new_fitness: f64,
) -> Result<bool> {
    // Read current
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));
    params.insert("niche_key".into(), DataValue::from(niche_key));

    let result = store.run_query(
        r#"?[fitness, usage_count, feature_vector] :=
            *skill_niches{skill_id, niche_key, fitness, usage_count, feature_vector},
            skill_id = to_uuid($skill_id),
            niche_key = $niche_key"#,
        params.clone(),
    )?;

    if result.rows.is_empty() {
        return Ok(false); // niche doesn't exist
    }

    let current_fitness = result.rows[0][0].get_float().unwrap_or(0.0);
    let usage_count = result.rows[0][1].get_int().unwrap_or(0);

    if new_fitness <= current_fitness {
        // Just increment usage count
        params.insert("fitness".into(), DataValue::from(current_fitness));
        params.insert("usage_count".into(), DataValue::from(usage_count + 1));
    } else {
        params.insert("fitness".into(), DataValue::from(new_fitness));
        params.insert("usage_count".into(), DataValue::from(usage_count + 1));
    }

    let feature_vector = result.rows[0][2].get_str().unwrap_or("{}").to_string();
    params.insert("feature_vector".into(), DataValue::from(feature_vector.as_str()));

    store.run_script(
        concat!(
            "?[skill_id, niche_key, feature_vector, fitness, usage_count] <- ",
            "[[$skill_id, $niche_key, $feature_vector, $fitness, $usage_count]] ",
            ":put skill_niches {skill_id, niche_key => feature_vector, fitness, usage_count}",
        ),
        params,
    )?;

    Ok(new_fitness > current_fitness)
}

/// Get all niches for a specific skill.
pub fn get_niches_for_skill(store: &CozoStore, skill_id: Uuid) -> Result<Vec<SkillNiche>> {
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));

    let result = store.run_query(
        r#"?[skill_id, niche_key, feature_vector, fitness, usage_count] :=
            *skill_niches{skill_id, niche_key, feature_vector, fitness, usage_count},
            skill_id = to_uuid($skill_id)"#,
        params,
    )?;

    result.rows.iter().map(|row| parse_niche_row(row)).collect()
}

/// Get the best skill for each niche in a domain.
pub fn get_best_per_niche(store: &CozoStore) -> Result<Vec<SkillNiche>> {
    let result = store.run_query(
        r#"best[niche_key, max(fitness)] :=
            *skill_niches{niche_key, fitness}
        ?[skill_id, niche_key, feature_vector, fitness, usage_count] :=
            best[niche_key, fitness],
            *skill_niches{skill_id, niche_key, feature_vector, fitness, usage_count}"#,
        BTreeMap::new(),
    )?;

    result.rows.iter().map(|row| parse_niche_row(row)).collect()
}

fn parse_niche_row(row: &[DataValue]) -> Result<SkillNiche> {
    let skill_id = crate::store::parse_uuid_pub(&row[0])?;
    let niche_key = row[1].get_str().unwrap_or("").to_string();
    let feature_vector: serde_json::Value = serde_json::from_str(
        row[2].get_str().unwrap_or("{}"),
    )
    .unwrap_or_default();
    let fitness = row[3].get_float().unwrap_or(0.0);
    let usage_count = row[4].get_int().unwrap_or(0);

    Ok(SkillNiche {
        skill_id,
        niche_key,
        feature_vector,
        fitness,
        usage_count,
    })
}

/// Cross-pollinate skills across niches — recombine steps from the best skill
/// in each niche to create hybrid candidates.
///
/// For each pair of niches, takes the first half of one skill's steps and
/// the second half of the other's, creating a "cross-pollinated" hybrid.
/// Returns the recombined step lists (caller decides whether to create new skills).
pub fn cross_pollinate(store: &CozoStore) -> Result<Vec<CrossPollinationCandidate>> {
    let best = get_best_per_niche(store)?;
    if best.len() < 2 {
        return Ok(Vec::new());
    }

    let mut candidates = Vec::new();

    // Load skills for each niche champion
    let mut niche_skills: Vec<(SkillNiche, crate::skills::Skill)> = Vec::new();
    for niche in &best {
        if let Ok(Some(skill)) = crate::skills::storage::get_skill(store, niche.skill_id) {
            niche_skills.push((niche.clone(), skill));
        }
    }

    // Pairwise recombination
    for i in 0..niche_skills.len() {
        for j in (i + 1)..niche_skills.len() {
            let (niche_a, skill_a) = &niche_skills[i];
            let (niche_b, skill_b) = &niche_skills[j];

            if skill_a.steps.is_empty() || skill_b.steps.is_empty() {
                continue;
            }

            // Crossover: first half of A + second half of B
            let mid_a = skill_a.steps.len() / 2;
            let mid_b = skill_b.steps.len() / 2;

            let mut hybrid_steps = skill_a.steps[..mid_a.max(1)].to_vec();
            hybrid_steps.extend_from_slice(&skill_b.steps[mid_b..]);

            candidates.push(CrossPollinationCandidate {
                parent_a: niche_a.skill_id,
                parent_b: niche_b.skill_id,
                niche_a: niche_a.niche_key.clone(),
                niche_b: niche_b.niche_key.clone(),
                hybrid_steps,
                hybrid_name: format!("{}×{}", skill_a.name, skill_b.name),
            });
        }
    }

    Ok(candidates)
}

/// A candidate for cross-pollinated skill creation.
#[derive(Debug, Clone)]
pub struct CrossPollinationCandidate {
    pub parent_a: Uuid,
    pub parent_b: Uuid,
    pub niche_a: String,
    pub niche_b: String,
    pub hybrid_steps: Vec<crate::skills::SkillStep>,
    pub hybrid_name: String,
}

/// Compute a diversity bonus for a skill based on its niche's underrepresentation.
///
/// Skills in underexplored niches (low usage_count relative to others) get a bonus
/// to encourage exploration of diverse behavioral strategies.
///
/// Returns a multiplier >= 1.0 to apply to the skill's EFE epistemic component.
pub fn diversity_bonus(store: &CozoStore, skill_id: Uuid) -> Result<f64> {
    let all_niches = get_best_per_niche(store)?;
    if all_niches.is_empty() {
        return Ok(1.0);
    }

    let total_usage: i64 = all_niches.iter().map(|n| n.usage_count).sum();
    let avg_usage = if all_niches.is_empty() {
        1.0
    } else {
        (total_usage as f64) / (all_niches.len() as f64)
    };

    // Find this skill's niche usage
    let skill_niches = get_niches_for_skill(store, skill_id)?;
    let min_usage = skill_niches
        .iter()
        .map(|n| n.usage_count)
        .min()
        .unwrap_or(0);

    // Bonus = avg / (this_usage + 1) — underused niches get higher bonus
    // Clamped to [1.0, 3.0] to prevent extreme dominance
    let bonus = (avg_usage + 1.0) / (min_usage as f64 + 1.0);
    Ok(bonus.clamp(1.0, 3.0))
}

/// Compute aggregate behavioral diversity metrics across all niches.
pub fn behavioral_diversity_metrics(store: &CozoStore) -> Result<DiversityMetrics> {
    let all_niches = get_best_per_niche(store)?;
    let n = all_niches.len();

    if n == 0 {
        return Ok(DiversityMetrics {
            niche_count: 0,
            coverage: 0.0,
            fitness_spread: 0.0,
            usage_uniformity: 0.0,
        });
    }

    // Coverage: number of distinct niches populated
    let niche_count = n;

    // Fitness spread: std deviation of fitness values
    let fitnesses: Vec<f64> = all_niches.iter().map(|n| n.fitness).collect();
    let mean_fitness: f64 = fitnesses.iter().sum::<f64>() / n as f64;
    let variance: f64 = fitnesses.iter().map(|f| (f - mean_fitness).powi(2)).sum::<f64>() / n as f64;
    let fitness_spread = variance.sqrt();

    // Usage uniformity: 1 - normalized std dev of usage counts
    // Uniformity of 1.0 means all niches used equally
    let usages: Vec<f64> = all_niches.iter().map(|n| n.usage_count as f64).collect();
    let mean_usage: f64 = usages.iter().sum::<f64>() / n as f64;
    let usage_uniformity = if mean_usage > 0.0 {
        let usage_var: f64 = usages.iter().map(|u| (u - mean_usage).powi(2)).sum::<f64>() / n as f64;
        let cv = usage_var.sqrt() / mean_usage; // coefficient of variation
        (1.0 - cv).max(0.0) // 1.0 = perfectly uniform
    } else {
        1.0 // no usage yet → perfectly uniform
    };

    Ok(DiversityMetrics {
        niche_count,
        coverage: niche_count as f64, // absolute count for now
        fitness_spread,
        usage_uniformity,
    })
}

/// Aggregate behavioral diversity metrics.
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Number of distinct behavioral niches populated.
    pub niche_count: usize,
    /// Coverage score (number of niches).
    pub coverage: f64,
    /// Standard deviation of fitness across niches (higher = more variation).
    pub fitness_spread: f64,
    /// Usage uniformity: 1.0 = all niches equally used, 0.0 = highly skewed.
    pub usage_uniformity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_get_niche() {
        let store = CozoStore::open_mem(4).unwrap();
        let skill_id = Uuid::now_v7();

        let niche = SkillNiche {
            skill_id,
            niche_key: "debug:quick".into(),
            feature_vector: serde_json::json!({"speed": "quick"}),
            fitness: 0.8,
            usage_count: 0,
        };
        store_niche(&store, &niche).unwrap();

        let niches = get_niches_for_skill(&store, skill_id).unwrap();
        assert_eq!(niches.len(), 1);
        assert_eq!(niches[0].niche_key, "debug:quick");
        assert!((niches[0].fitness - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_update_niche_fitness() {
        let store = CozoStore::open_mem(4).unwrap();
        let skill_id = Uuid::now_v7();

        let niche = SkillNiche {
            skill_id,
            niche_key: "test:niche".into(),
            feature_vector: serde_json::json!({}),
            fitness: 0.5,
            usage_count: 0,
        };
        store_niche(&store, &niche).unwrap();

        // Higher fitness → updates
        let updated = update_niche_fitness(&store, skill_id, "test:niche", 0.9).unwrap();
        assert!(updated);

        // Lower fitness → doesn't update fitness but increments count
        let not_updated = update_niche_fitness(&store, skill_id, "test:niche", 0.3).unwrap();
        assert!(!not_updated);

        let niches = get_niches_for_skill(&store, skill_id).unwrap();
        assert!((niches[0].fitness - 0.9).abs() < 0.01);
        assert_eq!(niches[0].usage_count, 2);
    }

    #[test]
    fn test_diversity_metrics_empty() {
        let store = CozoStore::open_mem(4).unwrap();
        let metrics = behavioral_diversity_metrics(&store).unwrap();
        assert_eq!(metrics.niche_count, 0);
        assert_eq!(metrics.coverage, 0.0);
    }

    #[test]
    fn test_diversity_metrics_with_niches() {
        let store = CozoStore::open_mem(4).unwrap();

        // Create 3 niches with different fitness values
        for (i, key) in ["debug:quick", "deploy:safe", "refactor:thorough"].iter().enumerate() {
            let niche = SkillNiche {
                skill_id: Uuid::now_v7(),
                niche_key: key.to_string(),
                feature_vector: serde_json::json!({}),
                fitness: 0.5 + (i as f64 * 0.2),
                usage_count: (i + 1) as i64,
            };
            store_niche(&store, &niche).unwrap();
        }

        let metrics = behavioral_diversity_metrics(&store).unwrap();
        assert_eq!(metrics.niche_count, 3);
        assert!(metrics.fitness_spread > 0.0, "fitness should vary");
        assert!(metrics.usage_uniformity > 0.0, "usage should have some uniformity");
        assert!(metrics.usage_uniformity < 1.0, "usage is not perfectly uniform");
    }

    #[test]
    fn test_diversity_bonus_no_niches() {
        let store = CozoStore::open_mem(4).unwrap();
        let bonus = diversity_bonus(&store, Uuid::now_v7()).unwrap();
        assert_eq!(bonus, 1.0, "no niches → neutral bonus");
    }
}
