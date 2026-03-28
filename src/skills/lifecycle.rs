//! Skill lifecycle: bootstrap, crystallize, discover, improve, generalize.
//!
//! Skills evolve through five stages:
//! 1. Bootstrap: SKILL.md → LLM parse → structured skills
//! 2. Crystallize: 3+ similar successful episodes → new skill
//! 3. Discover: Novel success without skill → extract procedure
//! 4. Improve: Recurring adaptations → version the skill
//! 5. Generalize: Similar skills across domains → lift to general

use std::sync::Arc;
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::services::traits::{Embedder, LlmService, Message};
use crate::skills::{Skill, SkillProvenance, SkillStep};
use crate::skills::storage;
use crate::skills::lineage;
use crate::store::CozoStore;

/// Result of bootstrapping skills from a markdown document.
#[derive(Debug)]
pub struct BootstrapResult {
    pub skill_ids: Vec<Uuid>,
    pub skills_created: usize,
}

/// Result of crystallizing skills from episodes.
#[derive(Debug)]
pub struct CrystallizeResult {
    pub new_skill_ids: Vec<Uuid>,
    pub reinforced_skill_ids: Vec<Uuid>,
}

// ── 1. Bootstrap ──

/// Parse a SKILL.md document into structured skills via LLM.
pub async fn bootstrap_from_markdown(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    markdown: &str,
    _agent_id: &str,
) -> Result<BootstrapResult> {
    if markdown.trim().is_empty() {
        return Ok(BootstrapResult {
            skill_ids: vec![],
            skills_created: 0,
        });
    }

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: concat!(
                        "You parse skill documentation into structured JSON. ",
                        "For each skill section, output a JSON object on a single line:\n",
                        r#"{"name": "skill_name", "description": "what it does", "#,
                        r#""steps": [{"step": 1, "action": "step description"}], "#,
                        r#""preconditions": ["when to use"], "#,
                        r#""postconditions": ["expected outcome"], "#,
                        r#""domain": "general", "tags": ["tag1"]}"#,
                        "\nOutput one JSON object per line, nothing else."
                    )
                    .into(),
                },
                Message {
                    role: "user".into(),
                    content: format!("Parse these skills:\n\n{markdown}"),
                },
            ],
            4096,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let mut skill_ids = Vec::new();

    // Strip thinking tags from response
    let content = if let Some(pos) = response.content.find("</think>") {
        &response.content[pos + 8..]
    } else {
        &response.content
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }

        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
            let name = parsed["name"].as_str().unwrap_or("unnamed");
            let description = parsed["description"].as_str().unwrap_or("");
            let steps: Vec<SkillStep> = parsed["steps"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .enumerate()
                        .map(|(i, v)| SkillStep {
                            step: v["step"].as_u64().unwrap_or(i as u64 + 1) as usize,
                            action: v["action"].as_str().unwrap_or("").to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();

            let preconditions: Vec<serde_json::Value> = parsed["preconditions"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let postconditions: Vec<serde_json::Value> = parsed["postconditions"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let domain = parsed["domain"].as_str().unwrap_or("general");
            let tags: Vec<String> = parsed["tags"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let mut skill = Skill::new(name, description, steps);
            skill.preconditions = preconditions;
            skill.postconditions = postconditions;
            skill.provenance = SkillProvenance::Bootstrapped;
            skill.confidence = 0.5;
            skill.domain = domain.to_string();
            skill.tags = tags;

            // Embed the skill description
            let embed_text = format!("{name}: {description}");
            let embeddings = embedder
                .embed(&[embed_text.as_str()])
                .await
                .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

            if let Some(emb) = embeddings.first() {
                storage::store_skill(store, &skill, emb)?;
                skill_ids.push(skill.id);
            }
        }
    }

    let count = skill_ids.len();
    Ok(BootstrapResult {
        skill_ids,
        skills_created: count,
    })
}

// ── 2. Crystallize ──

/// Find repeated successful patterns across episodes and crystallize into skills.
///
/// Looks for 3+ similar successful episodes that don't match any existing skill,
/// then extracts the common procedure via LLM.
pub async fn crystallize_skills(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
) -> Result<CrystallizeResult> {
    // Find successful episodes with summaries
    let episodes = store.run_query(
        r#"?[id, summary] :=
            *episodes{id, outcome, summary, @ 'NOW'},
            outcome = "success",
            summary != """#,
        std::collections::BTreeMap::new(),
    )?;

    if episodes.rows.len() < 3 {
        return Ok(CrystallizeResult {
            new_skill_ids: vec![],
            reinforced_skill_ids: vec![],
        });
    }

    // Embed episode summaries and look for clusters
    let summaries: Vec<String> = episodes
        .rows
        .iter()
        .map(|r| r[1].get_str().unwrap_or("").to_string())
        .collect();

    let summary_refs: Vec<&str> = summaries.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder
        .embed(&summary_refs)
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    // Adaptive clustering: compute similarity matrix, derive threshold from data
    let n = embeddings.len();
    let mut sim_matrix = vec![vec![0.0f64; n]; n];
    let mut all_sims = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
            all_sims.push(sim);
        }
    }

    // Adaptive threshold: mean similarity + 0.5 * std_dev above the minimum.
    // Episodes that are more similar than the population mean form clusters.
    let threshold = if all_sims.is_empty() {
        0.5
    } else {
        let mean = all_sims.iter().sum::<f64>() / all_sims.len() as f64;
        let variance = all_sims.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / all_sims.len() as f64;
        let std_dev = variance.sqrt();
        // Use mean as threshold: episodes above average similarity cluster together
        // Floor at 0.3 to avoid clustering unrelated episodes
        (mean - 0.5 * std_dev).max(0.3)
    };

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut used = vec![false; n];

    for i in 0..n {
        if used[i] {
            continue;
        }
        let mut cluster = vec![i];
        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            if sim_matrix[i][j] > threshold {
                cluster.push(j);
                used[j] = true;
            }
        }
        if cluster.len() >= 3 {
            used[i] = true;
            clusters.push(cluster);
        }
    }

    let mut new_skill_ids = Vec::new();
    let mut reinforced_skill_ids = Vec::new();

    for cluster in &clusters {
        let cluster_summaries: Vec<&str> = cluster.iter().map(|&i| summaries[i].as_str()).collect();
        let combined = cluster_summaries.join("\n---\n");

        // Check if existing skill covers this pattern
        let cluster_embedding = &embeddings[cluster[0]];
        let existing = storage::find_skills_by_embedding(store, cluster_embedding, 3, 0.3)?;

        if !existing.is_empty() {
            // Reinforce existing skill's confidence
            for (skill, _) in &existing {
                let new_conf = (skill.confidence + 0.1).min(0.95);
                storage::update_skill_confidence(store, skill.id, new_conf)?;
                reinforced_skill_ids.push(skill.id);
            }
            continue;
        }

        // No existing skill matches — crystallize a new one
        let response = llm
            .complete(
                &[
                    Message {
                        role: "system".into(),
                        content: concat!(
                            "Extract the common procedure from these successful episodes. ",
                            "Output a single JSON object:\n",
                            r#"{"name": "skill_name", "description": "what it does", "#,
                            r#""steps": [{"step": 1, "action": "..."}], "#,
                            r#""domain": "general", "tags": ["tag"]}"#
                        )
                        .into(),
                    },
                    Message {
                        role: "user".into(),
                        content: format!("Episodes:\n{combined}"),
                    },
                ],
                2048,
            )
            .await
            .map_err(|e| MemoriaError::Llm(e.to_string()))?;

        if let Some(parsed) = extract_json_from_llm(&response.content) {
            let name = parsed["name"].as_str().unwrap_or("crystallized_skill");
            let description = parsed["description"].as_str().unwrap_or("");
            let steps: Vec<SkillStep> = parsed["steps"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .enumerate()
                        .map(|(i, v)| SkillStep {
                            step: v["step"].as_u64().unwrap_or(i as u64 + 1) as usize,
                            action: v["action"].as_str().unwrap_or("").to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();

            let episode_ids: Vec<Uuid> = cluster
                .iter()
                .filter_map(|&i| crate::store::parse_uuid_pub(&episodes.rows[i][0]).ok())
                .collect();

            let mut skill = Skill::new(name, description, steps);
            skill.provenance = SkillProvenance::Crystallized;
            skill.confidence = (0.3 * episode_ids.len() as f64).min(0.9);
            skill.source_episodes = episode_ids;
            if let Some(domain) = parsed["domain"].as_str() {
                skill.domain = domain.to_string();
            }

            storage::store_skill(store, &skill, cluster_embedding)?;
            new_skill_ids.push(skill.id);
        }
    }

    Ok(CrystallizeResult {
        new_skill_ids,
        reinforced_skill_ids,
    })
}

// ── 3. Discover ──

/// Discover a new skill from a novel successful episode.
///
/// Called when an agent completes a task successfully without using any existing skill.
pub async fn discover_skill(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    episode_id: Uuid,
    episode_summary: &str,
) -> Result<Option<Uuid>> {
    if episode_summary.trim().is_empty() {
        return Ok(None);
    }

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: concat!(
                        "Extract a reusable procedure from this successful episode. ",
                        "Output a single JSON object:\n",
                        r#"{"name": "skill_name", "description": "what it does", "#,
                        r#""steps": [{"step": 1, "action": "..."}], "#,
                        r#""domain": "general", "tags": ["tag"]}"#
                    )
                    .into(),
                },
                Message {
                    role: "user".into(),
                    content: format!("Successful episode summary:\n{episode_summary}"),
                },
            ],
            2048,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let parsed = extract_json_from_llm(&response.content)
        .ok_or_else(|| MemoriaError::Skill("failed to parse LLM response as JSON".into()))?;

    let name = parsed["name"].as_str().unwrap_or("discovered_skill");
    let description = parsed["description"].as_str().unwrap_or("");
    let steps: Vec<SkillStep> = parsed["steps"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, v)| SkillStep {
                    step: v["step"].as_u64().unwrap_or(i as u64 + 1) as usize,
                    action: v["action"].as_str().unwrap_or("").to_string(),
                })
                .collect()
        })
        .unwrap_or_default();

    let mut skill = Skill::new(name, description, steps);
    skill.provenance = SkillProvenance::Discovered;
    skill.confidence = 0.3; // single observation, needs validation
    skill.source_episodes = vec![episode_id];

    let embed_text = format!("{name}: {description}");
    let embeddings = embedder
        .embed(&[embed_text.as_str()])
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    if let Some(emb) = embeddings.first() {
        storage::store_skill(store, &skill, emb)?;
        Ok(Some(skill.id))
    } else {
        Ok(None)
    }
}

// ── 4. Improve ──

/// Improve a skill by incorporating recurring adaptations.
///
/// When the same adaptation appears in 3+ successful usages, the skill is
/// versioned with the adaptation incorporated.
pub async fn improve_skill(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    skill_id: Uuid,
) -> Result<Option<Uuid>> {
    let skill = storage::get_skill(store, skill_id)?
        .ok_or_else(|| MemoriaError::Skill(format!("skill not found: {skill_id}")))?;

    // Find successful usages with adaptations
    let mut params = std::collections::BTreeMap::new();
    params.insert("skill_id".into(), cozo::DataValue::from(skill_id.to_string()));

    let result = store.run_query(
        r#"?[adaptations] :=
            *skill_usages{skill_id, outcome, adaptations},
            skill_id = to_uuid($skill_id),
            outcome = "success",
            adaptations != "[]""#,
        params,
    )?;

    if result.rows.len() < 3 {
        return Ok(None); // not enough data to improve
    }

    // Collect all adaptations
    let all_adaptations: Vec<String> = result
        .rows
        .iter()
        .map(|r| r[0].get_str().unwrap_or("[]").to_string())
        .collect();

    let combined = all_adaptations.join("\n");
    let steps_json = serde_json::to_string(&skill.steps).unwrap_or_default();

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: concat!(
                        "A skill has been used multiple times with adaptations. ",
                        "Incorporate the most common adaptations into an improved version. ",
                        "Output a single JSON object with the updated steps:\n",
                        r#"{"steps": [{"step": 1, "action": "..."}], "mutation_summary": "what changed"}"#
                    )
                    .into(),
                },
                Message {
                    role: "user".into(),
                    content: format!(
                        "Original steps:\n{steps_json}\n\nAdaptations from successful usages:\n{combined}"
                    ),
                },
            ],
            2048,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let parsed = extract_json_from_llm(&response.content)
        .ok_or_else(|| MemoriaError::Skill("failed to parse improved steps as JSON".into()))?;

    let new_steps: Vec<SkillStep> = parsed["steps"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, v)| SkillStep {
                    step: v["step"].as_u64().unwrap_or(i as u64 + 1) as usize,
                    action: v["action"].as_str().unwrap_or("").to_string(),
                })
                .collect()
        })
        .unwrap_or_default();

    let mutation_summary = parsed["mutation_summary"]
        .as_str()
        .unwrap_or("incorporated recurring adaptations")
        .to_string();

    // Create a new version
    let mut improved = skill.clone();
    improved.id = Uuid::now_v7();
    improved.steps = new_steps;
    improved.version = skill.version + 1;
    improved.parent_skill = Some(skill.id);

    let embed_text = format!("{}: {}", improved.name, improved.description);
    let embeddings = embedder
        .embed(&[embed_text.as_str()])
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    if let Some(emb) = embeddings.first() {
        storage::store_skill(store, &improved, emb)?;
        lineage::record_lineage(store, improved.id, skill.id, "evolved_from", &mutation_summary)?;
        Ok(Some(improved.id))
    } else {
        Ok(None)
    }
}

// ── 5. Generalize ──

/// Find similar skills across different domains and lift to a general skill.
pub async fn generalize_skills(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
) -> Result<Vec<Uuid>> {
    // Find high-confidence skills across different domains
    let result = store.run_query(
        r#"?[id, name, description, domain, steps, embedding] :=
            *skills{id, name, description, domain, steps, embedding, confidence, @ 'NOW'},
            confidence > 0.6,
            domain != "general""#,
        std::collections::BTreeMap::new(),
    )?;

    if result.rows.len() < 2 {
        return Ok(Vec::new());
    }

    // Parse skills and their embeddings
    let mut domain_skills: Vec<(Uuid, String, String, String, String, Vec<f32>)> = Vec::new();
    for row in &result.rows {
        let id = crate::store::parse_uuid_pub(&row[0])?;
        let name = row[1].get_str().unwrap_or("").to_string();
        let description = row[2].get_str().unwrap_or("").to_string();
        let domain = row[3].get_str().unwrap_or("").to_string();
        let steps = row[4].get_str().unwrap_or("[]").to_string();
        let emb = parse_f32_vec_local(&row[5]);
        domain_skills.push((id, name, description, domain, steps, emb));
    }

    // Find cross-domain pairs with high similarity
    let mut generalized_ids = Vec::new();

    for i in 0..domain_skills.len() {
        for j in (i + 1)..domain_skills.len() {
            if domain_skills[i].3 == domain_skills[j].3 {
                continue; // same domain, skip
            }

            let sim = cosine_similarity(&domain_skills[i].5, &domain_skills[j].5);
            if sim <= 0.6 {
                continue; // not similar enough
            }

            // Found cross-domain similar pair — generalize
            let response = llm
                .complete(
                    &[
                        Message {
                            role: "system".into(),
                            content: concat!(
                                "Two skills from different domains are similar. ",
                                "Extract a general skill that covers both. ",
                                "Output a single JSON object:\n",
                                r#"{"name": "general_skill_name", "description": "...", "#,
                                r#""steps": [{"step": 1, "action": "..."}], "tags": ["tag"]}"#
                            )
                            .into(),
                        },
                        Message {
                            role: "user".into(),
                            content: format!(
                                "Skill 1 (domain: {}):\n{}: {}\nSteps: {}\n\nSkill 2 (domain: {}):\n{}: {}\nSteps: {}",
                                domain_skills[i].3, domain_skills[i].1, domain_skills[i].2, domain_skills[i].4,
                                domain_skills[j].3, domain_skills[j].1, domain_skills[j].2, domain_skills[j].4,
                            ),
                        },
                    ],
                    2048,
                )
                .await
                .map_err(|e| MemoriaError::Llm(e.to_string()))?;

            if let Some(parsed) = extract_json_from_llm(&response.content) {
                let name = parsed["name"].as_str().unwrap_or("generalized_skill");
                let description = parsed["description"].as_str().unwrap_or("");
                let steps: Vec<SkillStep> = parsed["steps"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .enumerate()
                            .map(|(idx, v)| SkillStep {
                                step: v["step"].as_u64().unwrap_or(idx as u64 + 1) as usize,
                                action: v["action"].as_str().unwrap_or("").to_string(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let mut general = Skill::new(name, description, steps);
                general.provenance = SkillProvenance::Generalized;
                general.confidence = 0.7; // proven in at least 2 domains
                general.domain = "general".to_string();

                let embed_text = format!("{name}: {description}");
                let embeddings = embedder
                    .embed(&[embed_text.as_str()])
                    .await
                    .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

                if let Some(emb) = embeddings.first() {
                    storage::store_skill(store, &general, emb)?;

                    // Record lineage from both parent skills
                    lineage::record_lineage(
                        store,
                        general.id,
                        domain_skills[i].0,
                        "generalized_from",
                        &format!("generalized from {} domain", domain_skills[i].3),
                    )?;
                    lineage::record_lineage(
                        store,
                        general.id,
                        domain_skills[j].0,
                        "generalized_from",
                        &format!("generalized from {} domain", domain_skills[j].3),
                    )?;

                    generalized_ids.push(general.id);
                }
            }
        }
    }

    Ok(generalized_ids)
}

// ── 6. Specialize ──

/// Result of specializing a skill to a target domain.
#[derive(Debug)]
pub struct SpecializeResult {
    pub specialized_skill_id: Uuid,
    pub parent_skill_id: Uuid,
    pub target_domain: String,
}

/// Specialize a general skill to a specific domain using context examples.
///
/// Takes a general-purpose skill and adapts its steps, preconditions, and
/// postconditions to be domain-specific. Records `specialized_from` lineage.
pub async fn specialize_skill(
    store: &CozoStore,
    llm: &Arc<dyn LlmService>,
    embedder: &Arc<dyn Embedder>,
    general_skill_id: Uuid,
    target_domain: &str,
    context_examples: &[&str],
) -> Result<SpecializeResult> {
    let general = storage::get_skill(store, general_skill_id)?
        .ok_or_else(|| MemoriaError::Skill(format!("skill not found: {general_skill_id}")))?;

    let steps_json = serde_json::to_string(&general.steps).unwrap_or_default();
    let examples = context_examples.join("\n---\n");

    let response = llm
        .complete(
            &[
                Message {
                    role: "system".into(),
                    content: format!(
                        concat!(
                            "Adapt a general skill to the '{}' domain. ",
                            "Use the provided examples to make steps concrete and domain-specific. ",
                            "Output a single JSON object:\n",
                            r#"{{"name": "specialized_name", "description": "...", "#,
                            r#""steps": [{{"step": 1, "action": "..."}}], "#,
                            r#""preconditions": ["..."], "postconditions": ["..."], "tags": ["tag"]}}"#
                        ),
                        target_domain
                    ),
                },
                Message {
                    role: "user".into(),
                    content: format!(
                        "General skill: {} — {}\nSteps: {}\n\nDomain examples:\n{}",
                        general.name, general.description, steps_json, examples
                    ),
                },
            ],
            2048,
        )
        .await
        .map_err(|e| MemoriaError::Llm(e.to_string()))?;

    let parsed = extract_json_from_llm(&response.content)
        .ok_or_else(|| MemoriaError::Skill("failed to parse specialized skill as JSON".into()))?;

    let default_name = format!("{}_{}", general.name, target_domain);
    let name = parsed["name"].as_str().unwrap_or(&default_name);
    let description = parsed["description"].as_str().unwrap_or(&general.description);
    let steps: Vec<SkillStep> = parsed["steps"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, v)| SkillStep {
                    step: v["step"].as_u64().unwrap_or(i as u64 + 1) as usize,
                    action: v["action"].as_str().unwrap_or("").to_string(),
                })
                .collect()
        })
        .unwrap_or_else(|| general.steps.clone());

    let preconditions: Vec<serde_json::Value> = parsed["preconditions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let postconditions: Vec<serde_json::Value> = parsed["postconditions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let tags: Vec<String> = parsed["tags"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let mut specialized = Skill::new(name, description, steps);
    specialized.preconditions = preconditions;
    specialized.postconditions = postconditions;
    specialized.provenance = SkillProvenance::Specialized;
    specialized.confidence = general.confidence * 0.8; // slightly lower until validated
    specialized.domain = target_domain.to_string();
    specialized.parent_skill = Some(general_skill_id);
    specialized.tags = tags;
    specialized.version = 1;

    let embed_text = format!("{name}: {description}");
    let embeddings = embedder
        .embed(&[embed_text.as_str()])
        .await
        .map_err(|e| MemoriaError::Embedding(e.to_string()))?;

    if let Some(emb) = embeddings.first() {
        storage::store_skill(store, &specialized, emb)?;
        lineage::record_lineage(
            store,
            specialized.id,
            general_skill_id,
            "specialized_from",
            &format!("specialized to {} domain", target_domain),
        )?;
    }

    Ok(SpecializeResult {
        specialized_skill_id: specialized.id,
        parent_skill_id: general_skill_id,
        target_domain: target_domain.to_string(),
    })
}

// ── Helpers ──

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Extract JSON from LLM response that may contain thinking tags, markdown fences, etc.
fn extract_json_from_llm(response: &str) -> Option<serde_json::Value> {
    let content = response.trim();

    // Direct parse first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(content) {
        return Some(v);
    }

    // Strip <think>...</think> tags (local LLMs with reasoning)
    let stripped = if let Some(pos) = content.find("</think>") {
        content[pos + 8..].trim()
    } else {
        content
    };

    // Try direct parse after stripping think tags
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(stripped) {
        return Some(v);
    }

    // Strip markdown code fences
    let stripped = if stripped.starts_with("```") {
        let inner = stripped.trim_start_matches("```json").trim_start_matches("```");
        inner.trim_end_matches("```").trim()
    } else {
        stripped
    };

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(stripped) {
        return Some(v);
    }

    // Last resort: find first { ... } block
    if let Some(start) = stripped.find('{') {
        if let Some(end) = stripped.rfind('}') {
            if end > start {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&stripped[start..=end]) {
                    return Some(v);
                }
            }
        }
    }

    None
}

fn parse_f32_vec_local(val: &cozo::DataValue) -> Vec<f32> {
    match val {
        cozo::DataValue::List(list) => list.iter().map(|v| v.get_float().unwrap_or(0.0) as f32).collect(),
        cozo::DataValue::Vec(v) => match v {
            cozo::Vector::F32(arr) => arr.to_vec(),
            cozo::Vector::F64(arr) => arr.iter().map(|&x| x as f32).collect(),
        },
        _ => Vec::new(),
    }
}
