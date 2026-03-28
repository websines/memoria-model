use cozo::{DbInstance, ScriptMutability};
use std::collections::BTreeMap;

use crate::error::{MemoriaError, Result};

/// Bootstrap all CozoDB relations and indexes.
///
/// Each relation is created as a separate script call because CozoDB
/// doesn't support multiple `:create` statements in one script.
///
/// `dim` is the embedding dimension, resolved from `embedder.dim()` at runtime.
pub fn bootstrap_schema(db: &DbInstance, dim: usize) -> Result<()> {
    let statements = build_schema_statements(dim);

    for (name, script) in &statements {
        db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
            .map_err(|e| MemoriaError::SchemaBootstrap(format!("{name}: {e}")))?;
    }

    Ok(())
}

/// Build all schema creation statements as (name, script) pairs.
fn build_schema_statements(dim: usize) -> Vec<(&'static str, String)> {
    vec![
        // ── Core Memory Store ──
        (
            "memories",
            format!(
                r#":create memories {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    kind: String,
    content: String default "",
    embedding: <F32; {dim}>,
    fields: Json default {{}},
    namespace: String default "",
    pinned: Bool default false,
    expires_at: Int? default null,
    version: Int default 0,
    created_at: Int default now(),
    confidence: Float default 1.0,
    provenance: String default "direct",
    source_ids: [Uuid] default []
}}"#
            ),
        ),
        (
            "memories:vec_idx",
            format!(
                r#"::hnsw create memories:vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        // ── Knowledge Graph ──
        (
            "edges",
            r#":create edges {
    source: Uuid,
    target: Uuid,
    kind: String,
    valid_at: Validity
    =>
    weight: Float default 1.0,
    fields: Json default {}
}"#
            .to_string(),
        ),
        (
            "entities",
            format!(
                r#":create entities {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    name: String,
    entity_type: String,
    namespace: String default "",
    embedding: <F32; {dim}>,
    properties: Json default {{}},
    mention_count: Int default 1,
    confidence: Float default 1.0,
    provenance: String default "extracted",
    source_ids: [Uuid] default []
}}"#
            ),
        ),
        (
            "entities:entity_vec_idx",
            format!(
                r#"::hnsw create entities:entity_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        (
            "entity_mentions",
            r#":create entity_mentions {
    entity_id: Uuid,
    memory_id: Uuid,
    valid_at: Validity
    =>
    role: String default "mentioned",
    confidence: Float default 1.0
}"#
            .to_string(),
        ),
        (
            "facts",
            r#":create facts {
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    subject_entity: Uuid,
    predicate: String,
    object_entity: Uuid? default null,
    object_value: String? default null,
    namespace: String default "",
    temporal_status: String default "current",
    confidence: Float default 1.0,
    provenance: String default "extracted",
    source_ids: [Uuid] default [],
    reinforcement_count: Int default 1
}"#
            .to_string(),
        ),
        // ── Episodic Memory ──
        (
            "episodes",
            format!(
                r#":create episodes {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    agent_id: String,
    started_at: Int,
    ended_at: Int? default null,
    summary: String default "",
    summary_embedding: <F32; {dim}>? default null,
    episode_type: String default "session",
    outcome: String? default null,
    properties: Json default {{}}
}}"#
            ),
        ),
        (
            "episodes:episode_vec_idx",
            format!(
                r#"::hnsw create episodes:episode_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [summary_embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        (
            "episode_memories",
            r#":create episode_memories {
    episode_id: Uuid,
    memory_id: Uuid
    =>
    position: Int default 0,
    role: String default "member"
}"#
            .to_string(),
        ),
        // ── Learning / Temporal ──
        (
            "accesses",
            r#":create accesses {
    memory_id: Uuid,
    ts: Int
    =>
    event_type: String,
    agent_id: String default ""
}"#
            .to_string(),
        ),
        (
            "associations",
            r#":create associations {
    a: Uuid,
    b: Uuid,
    valid_at: Validity
    =>
    weight: Float default 0.1,
    co_access_count: Int default 1,
    last_access: Int
}"#
            .to_string(),
        ),
        (
            "co_activations",
            r#":create co_activations {
    a: Uuid,
    b: Uuid,
    valid_at: Validity
    =>
    count: Int default 1,
    last_seen: Int
}"#
            .to_string(),
        ),
        (
            "scorer_weights",
            r#":create scorer_weights {
    scorer: String,
    valid_at: Validity
    =>
    weight: Float,
    baseline: Float default 0.0
}"#
            .to_string(),
        ),
        // ── Audit ──
        (
            "audit_log",
            r#":create audit_log {
    id: Uuid default rand_uuid_v7(),
    ts: Int
    =>
    operation: String,
    agent_id: String,
    target_id: Uuid? default null,
    details: Json default {}
}"#
            .to_string(),
        ),
        // ── Routing ──
        (
            "route_decisions",
            r#":create route_decisions {
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    state_vec: <F32; 128>,
    chosen_agent: Int,
    confidence: Float,
    success: Bool? default null
}"#
            .to_string(),
        ),
        (
            "router_checkpoints",
            r#":create router_checkpoints {
    checkpoint_id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    component: String,
    weights_blob: Bytes,
    metadata: Json default {}
}"#
            .to_string(),
        ),
        (
            "agent_profiles",
            r#":create agent_profiles {
    agent_idx: Int,
    valid_at: Validity
    =>
    centroid: <F32; 128>,
    success_rate: Float,
    total_decisions: Int
}"#
            .to_string(),
        ),
        // ── Task Queue ──
        (
            "task_queue",
            r#":create task_queue {
    id: Uuid default rand_uuid_v7(),
    enqueued_at: Int default now()
    =>
    task_type: String,
    priority: Int default 0,
    payload: String default "{}",
    status: String default "pending",
    attempts: Int default 0,
    max_attempts: Int default 3,
    locked_until: Int? default null,
    last_error: String? default null,
    completed_at: Int? default null,
    result: String? default null
}"#
            .to_string(),
        ),
        // ── Rules Engine ──
        (
            "rules",
            format!(
                r#":create rules {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    text: String,
    embedding: <F32; {dim}>,
    category: String,
    severity: String default "warn",
    scope: String default "*",
    compiled_type: String default "datalog",
    compiled_predicate: String default "",
    active: Bool default true,
    created_by: String default "",
    violation_count: Int default 0
}}"#
            ),
        ),
        (
            "rules:rule_vec_idx",
            format!(
                r#"::hnsw create rules:rule_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        (
            "rule_evaluations",
            r#":create rule_evaluations {
    id: Uuid default rand_uuid_v7(),
    ts: Int default now()
    =>
    rule_id: Uuid,
    operation: String,
    agent_id: String,
    target_id: Uuid? default null,
    result: String,
    reason: String default "",
    evaluation_ms: Int default 0
}"#
            .to_string(),
        ),
        (
            "agent_profiles:profile_idx",
            r#"::hnsw create agent_profiles:profile_idx {
    dim: 128,
    dtype: F32,
    fields: [centroid],
    distance: Cosine,
    m: 8,
    ef_construction: 100
}"#
            .to_string(),
        ),
        // ── Phase 4: Memory Dynamics ──
        (
            "surprise_log",
            r#":create surprise_log {
    id: Uuid default rand_uuid_v7(),
    ts: Int default now()
    =>
    surprise: Float,
    source: String,
    variable_id: Uuid? default null,
    factor_id: Uuid? default null,
    observation_summary: String default "",
    resolved: Bool default false
}"#
            .to_string(),
        ),
        (
            "abstractions",
            format!(
                r#":create abstractions {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    content: String,
    embedding: <F32; {dim}>,
    confidence: Float default 0.5,
    evidence_count: Int default 1,
    source_episodes: [Uuid],
    category: String default "pattern"
}}"#
            ),
        ),
        (
            "abstractions:abs_vec_idx",
            format!(
                r#"::hnsw create abstractions:abs_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        (
            "reflections",
            r#":create reflections {
    id: Uuid default rand_uuid_v7(),
    ts: Int default now()
    =>
    episodes_reviewed: [Uuid],
    facts_created: Int default 0,
    facts_updated: Int default 0,
    entities_created: Int default 0,
    abstractions_created: Int default 0,
    duration_ms: Int default 0
}"#
            .to_string(),
        ),
        (
            "memory_importance",
            r#":create memory_importance {
    memory_id: Uuid
    =>
    pagerank: Float default 0.0,
    community_id: Int default 0,
    in_degree: Int default 0,
    out_degree: Int default 0,
    computed_at: Int default now()
}"#
            .to_string(),
        ),
        // ── Phase 5: Procedural Memory ──
        (
            "skills",
            format!(
                r#":create skills {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    name: String,
    description: String,
    embedding: <F32; {dim}>,
    steps: Json,
    preconditions: Json default [],
    postconditions: Json default [],
    confidence: Float default 0.5,
    provenance: String default "bootstrapped",
    source_episodes: [Uuid] default [],
    domain: String default "general",
    version: Int default 1,
    performance: Json default {{}},
    parent_skill: Uuid? default null,
    tags: [String] default []
}}"#
            ),
        ),
        (
            "skills:skill_vec_idx",
            format!(
                r#"::hnsw create skills:skill_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        (
            "skill_usages",
            r#":create skill_usages {
    id: Uuid default rand_uuid_v7(),
    ts: Int
    =>
    skill_id: Uuid,
    skill_version: Int default 1,
    episode_id: Uuid? default null,
    agent_id: String,
    outcome: String,
    duration_ms: Int? default null,
    context_summary: String default "",
    adaptations: Json default []
}"#
            .to_string(),
        ),
        (
            "skill_niches",
            r#":create skill_niches {
    skill_id: Uuid,
    niche_key: String
    =>
    feature_vector: Json default {},
    fitness: Float default 0.0,
    usage_count: Int default 0
}"#
            .to_string(),
        ),
        (
            "lineage",
            r#":create lineage {
    child_id: Uuid,
    parent_id: Uuid
    =>
    relation: String,
    generation: Int,
    mutation_summary: String default "",
    ts: Int
}"#
            .to_string(),
        ),
        (
            "recall_contexts",
            r#":create recall_contexts {
    task_id: Uuid,
    recall_id: Uuid
    =>
    memory_ids: [Uuid] default [],
    fact_ids: [Uuid] default [],
    query_text: String default "",
    ts: Int
}"#
            .to_string(),
        ),
        (
            "task_outcomes",
            format!(
                r#":create task_outcomes {{
    task_id: Uuid,
    valid_at: Validity
    =>
    outcome: String,
    task_type: String default "",
    task_embedding: <F32; {dim}>? default null,
    agent_id: String,
    failure_reason: String? default null,
    duration_ms: Int? default null,
    plan: String default "",
    summary: String default "",
    summary_embedding: <F32; {dim}>? default null,
    skills_used: [Uuid] default [],
    adaptations: Json default []
}}"#
            ),
        ),
        (
            "task_outcomes:task_vec_idx",
            format!(
                r#"::hnsw create task_outcomes:task_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [task_embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        // ── Phase 6: Active Inference ──
        (
            "model_state",
            r#":create model_state {
    valid_at: Validity
    =>
    free_energy: Float,
    accuracy: Float,
    complexity: Float,
    beta: Float,
    unresolved_surprise: Float,
    agent_id: String default ""
}"#
            .to_string(),
        ),
        // ── Meta-Learning ──
        (
            "meta_params",
            r#":create meta_params {
    name: String
    =>
    value: Float,
    min_bound: Float,
    max_bound: Float,
    step: Float,
    generation: Int,
    updated_at: Int,
}"#
            .to_string(),
        ),
        (
            "meta_snapshots",
            r#":create meta_snapshots {
    id: Uuid default rand_uuid_v7()
    =>
    free_energy: Float,
    beta: Float,
    unresolved_surprise: Float,
    params_json: String,
    generation: Int,
    phase: String,
    ts: Int,
}"#
            .to_string(),
        ),
        (
            "meta_optimizer",
            r#":create meta_optimizer {
    id: Int default 0
    =>
    phase: String,
    state_json: String,
    generation: Int,
    updated_at: Int,
}"#
            .to_string(),
        ),
        // ── Multi-Agent: Agent Registry ──
        (
            "agent_registry",
            r#":create agent_registry {
    agent_id: String
    =>
    display_name: String,
    status: String default "active",
    team_id: String default "",
    org_id: String default "",
    role: String default "",
    capabilities: String default "[]",
    metadata: String default "{}",
    registered_at: Int default now(),
    last_seen_at: Int default now()
}"#
            .to_string(),
        ),
        (
            "agent_team_membership",
            r#":create agent_team_membership {
    agent_id: String,
    team_id: String
    =>
    joined_at: Int default now()
}"#
            .to_string(),
        ),
        // ── Multi-Agent: Scope Grants ──
        (
            "scope_grants",
            r#":create scope_grants {
    id: Uuid
    =>
    agent_pattern: String,
    namespace_pattern: String,
    permissions: String default "[]",
    granted_by: String,
    granted_at: Int,
    expires_at: Int? default null,
    revoked: Bool default false
}"#
            .to_string(),
        ),
        // ── Multi-Agent: Scratchpad ──
        (
            "scratchpad",
            r#":create scratchpad {
    namespace: String,
    key: String
    =>
    value: String,
    visibility: String default "private",
    owner_agent: String,
    updated_at: Int,
    expires_at: Int? default null
}"#
            .to_string(),
        ),
        // ── Multi-Agent: Agent Trust Profiles ──
        (
            "agent_trust_profiles",
            r#":create agent_trust_profiles {
    agent_id: String
    =>
    trust_score: Float default 1.0,
    success_count: Int default 0,
    failure_count: Int default 0,
    attribution_penalty_sum: Float default 0.0
}"#
            .to_string(),
        ),
        // ── Telos: Goal System ──
        (
            "telos",
            format!(
                r#":create telos {{
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    title: String,
    description: String default "",
    embedding: <F32; {dim}>,
    parent: Uuid? default null,
    depth: Int default 0,
    owner: String,
    set_by: String,
    namespace: String default "",
    status: String default "active",
    priority: Float default 0.5,
    urgency: Float default 0.0,
    confidence: Float default 1.0,
    provenance: String default "user_stated",
    deadline: Int? default null,
    created_at: Int default now(),
    started_at: Int? default null,
    completed_at: Int? default null,
    progress: Float default 0.0,
    stalled_since: Int? default null,
    success_criteria: Json default [],
    related_entities: [Uuid] default [],
    required_skills: [Uuid] default [],
    depends_on: [Uuid] default [],
    last_attended: Int default 0,
    attention_count: Int default 0
}}"#
            ),
        ),
        (
            "telos:telos_vec_idx",
            format!(
                r#"::hnsw create telos:telos_vec_idx {{
    dim: {dim},
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 16,
    ef_construction: 200
}}"#
            ),
        ),
        (
            "telos_events",
            r#":create telos_events {
    id: Uuid default rand_uuid_v7(),
    ts: Int
    =>
    telos_id: Uuid,
    event_type: String,
    agent_id: String default "",
    description: String default "",
    impact: Float default 0.0,
    source_memory: Uuid? default null,
    source_episode: Uuid? default null,
    metadata: Json default {}
}"#
            .to_string(),
        ),
        (
            "telos_attention",
            r#":create telos_attention {
    telos_id: Uuid,
    started_at: Int
    =>
    ended_at: Int? default null,
    agent_id: String,
    episode_id: Uuid? default null,
    outcome: String default "ongoing"
}"#
            .to_string(),
        ),
        // ── Predictive Generation ──
        (
            "predictions",
            r#":create predictions {
    id: Uuid default rand_uuid_v7(),
    valid_at: Validity
    =>
    kind: String default "",
    content: String default "",
    predicted_at: Int default 0,
    expected_by: Int default 0,
    confidence: Float default 0.5,
    ci_lower: Float default 0.0,
    ci_upper: Float default 1.0,
    source: String default "",
    context_ids: String default "[]",
    namespace: String default "",
    resolved: Bool default false,
    matched: Bool default false,
    prediction_error: Float default 1.0,
    actual_memory_id: String default "",
    resolved_at: Int default 0
}"#
            .to_string(),
        ),
        (
            "ppm_tree",
            r#":create ppm_tree {
    agent_id: String,
    valid_at: Validity
    =>
    tree_blob: Bytes,
    max_depth: Int,
    alphabet_size: Int,
    updated_at: Int
}"#
            .to_string(),
        ),
        (
            "changepoint_state",
            r#":create changepoint_state {
    stream_name: String,
    valid_at: Validity
    =>
    state_blob: Bytes,
    last_changepoint_at: Int,
    run_length: Int
}"#
            .to_string(),
        ),
        // ── Multi-Agent: Hash-Chained Audit Log ──
        (
            "audit_chain",
            r#":create audit_chain {
    seq: Int
    =>
    ts: Int,
    operation: String,
    agent_id: String,
    namespace: String default "",
    details: String default "{}",
    prev_hash: String default "",
    hash: String
}"#
            .to_string(),
        ),
        // ── Structural Causal Models (Initiative 5) ──
        (
            "causal_edges",
            r#":create causal_edges {
    cause_id: Uuid,
    effect_id: Uuid,
    valid_at: Validity
    =>
    causal_strength: Float default 0.5,
    observations: Int default 1,
    last_observed: Int default 0,
    mechanism: String default "unknown",
    confidence: Float default 0.5,
    namespace: String default ""
}"#
            .to_string(),
        ),
        // ── Embedding Projection ──
        (
            "embedding_projection",
            r#":create embedding_projection {
    id: Int default 0
    =>
    weights_blob: Bytes,
    dim: Int,
    last_loss: Float default 1.0,
    train_count: Int default 0,
    trained_at: Int default 0
}"#
            .to_string(),
        ),
        (
            "projection_stats",
            r#":create projection_stats {
    id: Uuid default rand_uuid_v7(),
    ts: Int default now()
    =>
    loss_before: Float,
    loss_after: Float,
    triplet_count: Int,
    epochs: Int,
    duration_ms: Int
}"#
            .to_string(),
        ),
    ]
}

/// Check if the schema is already bootstrapped by testing for the `memories` relation.
pub fn schema_exists(db: &DbInstance) -> bool {
    // Try querying the memories relation; if it doesn't exist we get an error
    db.run_script(
        "?[count(id)] := *memories{id}",
        BTreeMap::new(),
        ScriptMutability::Immutable,
    )
    .is_ok()
}

/// Count the number of relations created (for verification).
pub fn count_relations(db: &DbInstance) -> Result<usize> {
    let result = db
        .run_script(
            "::relations",
            BTreeMap::new(),
            ScriptMutability::Immutable,
        )
        .map_err(|e| MemoriaError::SchemaBootstrap(format!("listing relations: {e}")))?;

    Ok(result.rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db() -> DbInstance {
        DbInstance::new("mem", "", Default::default()).unwrap()
    }

    #[test]
    fn test_schema_bootstrap_succeeds() {
        let db = create_test_db();
        bootstrap_schema(&db, 128).unwrap();
    }

    #[test]
    fn test_schema_creates_all_relations() {
        let db = create_test_db();
        bootstrap_schema(&db, 128).unwrap();

        // Phase 1-3: 18, Phase 4: 5, Phase 5: 7+1, Phase 6: 1,
        // Meta-learning: meta_params, meta_snapshots, meta_optimizer = 3
        // Predictive generation: predictions, ppm_tree, changepoint_state = 3
        // Multi-agent: agent_registry, agent_team_membership, scope_grants, scratchpad, audit_chain, agent_trust_profiles = 6
        let count = count_relations(&db).unwrap();
        assert!(count >= 50, "expected at least 50 relations, got {count}");
    }

    #[test]
    fn test_schema_exists_check() {
        let db = create_test_db();
        assert!(!schema_exists(&db));
        bootstrap_schema(&db, 128).unwrap();
        assert!(schema_exists(&db));
    }

    #[test]
    fn test_schema_idempotent_check() {
        let db = create_test_db();
        bootstrap_schema(&db, 128).unwrap();
        // Second bootstrap should fail because relations already exist
        assert!(bootstrap_schema(&db, 128).is_err());
    }
}
