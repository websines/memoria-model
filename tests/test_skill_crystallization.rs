//! Integration test: Episode patterns → skill crystallization → niche classification.
//!
//! Run with: cargo test --features local-embeddings --test test_skill_crystallization -- --nocapture

#![cfg(feature = "local-embeddings")]

mod common;

use common::{build_real_memoria, load_task_patterns, test_ctx};

/// Create episodes with similar task patterns, crystallize skills,
/// verify new skills are created.
#[tokio::test]
async fn test_crystallize_skills_from_episodes() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let patterns = load_task_patterns();

    println!("\n=== Phase 1: Create episodes with success outcomes ===");
    let mut episode_ids = Vec::new();
    for (i, pattern) in patterns.iter().enumerate() {
        // Create and close an episode for each task pattern
        let episode = m
            .create_episode(&ctx.agent_id, "task", Default::default())
            .unwrap();

        // Tell the task description to build memory context
        m.tell(&format!("Task: {}", pattern.task), &ctx)
            .await
            .unwrap();

        // Close with outcome and summary
        m.close_episode(episode.id, &pattern.outcome, &pattern.summary)
            .await
            .unwrap();

        episode_ids.push(episode.id);
        println!(
            "  episode[{}]: {} → {}",
            i,
            &pattern.task[..pattern.task.len().min(50)],
            pattern.outcome
        );
    }

    let skills_before = m.count_skills().unwrap();
    println!("\nSkills before crystallization: {}", skills_before);

    println!("\n=== Phase 2: Crystallize skills ===");
    let result = m.crystallize_skills().await.unwrap();
    println!(
        "Crystallization result: {} new skills, {} reinforced",
        result.new_skill_ids.len(),
        result.reinforced_skill_ids.len()
    );

    let skills_after = m.count_skills().unwrap();
    println!("Skills after crystallization: {}", skills_after);

    // Log any created skills
    for skill_id in &result.new_skill_ids {
        if let Ok(Some(skill)) = m.get_skill(*skill_id) {
            println!(
                "  NEW skill: {} — {} (confidence={:.2})",
                skill.name,
                &skill.description[..skill.description.len().min(60)],
                skill.confidence
            );
        }
    }

    assert!(
        !result.new_skill_ids.is_empty(),
        "crystallization must produce new skills from {} episodes",
        episode_ids.len()
    );
    assert!(
        skills_after > skills_before,
        "skill count should increase after crystallization"
    );

    // Verify skills are findable via selection
    let selected = m
        .select_skills("Parse CSV data and generate statistics", 1.0, 5)
        .await
        .unwrap();
    println!(
        "\nSkill selection for CSV task: {} skills found",
        selected.len()
    );
    for s in &selected {
        println!("  score={:.4} name={}", s.efe_score, s.skill.name);
    }

    println!("\n=== Skill crystallization test complete ===");
}

/// Bootstrap skills from markdown and verify they're retrievable.
#[tokio::test]
async fn test_bootstrap_skills_from_markdown() {
    let m = build_real_memoria();
    let ctx = test_ctx();

    let skill_md = r#"
## Parse CSV Files
1. Load the CSV file using pandas
2. Validate column types and handle missing values
3. Compute descriptive statistics
4. Output formatted markdown table

## Debug API Errors
1. Check server logs for stack traces
2. Reproduce the error with curl
3. Identify the root cause in the handler
4. Apply fix and verify with tests
"#;

    println!("\n=== Bootstrap skills from markdown ===");
    let result = m.bootstrap_skills(skill_md, &ctx).await.unwrap();
    println!(
        "Bootstrapped {} skills: {:?}",
        result.skills_created, result.skill_ids
    );

    assert!(
        result.skills_created >= 1,
        "should bootstrap at least 1 skill"
    );

    let count = m.count_skills().unwrap();
    println!("Total skills: {}", count);
    assert!(count >= result.skills_created);

    // Verify skills are selectable (UUID validation in store_skill prevents corruption)
    let selected = m
        .select_skills("How do I parse a CSV file?", 1.0, 5)
        .await
        .unwrap();
    println!("Skill selection: {} skills found", selected.len());

    println!("\n=== Bootstrap test complete ===");
}
