use uuid::Uuid;

/// Events emitted by the Memoria runtime.
///
/// Subscribe via `Memoria::subscribe_all()` to receive all events,
/// or filter on specific variants.
#[derive(Debug, Clone)]
pub enum MemoryEvent {
    /// A new memory was created.
    Created {
        memory_ids: Vec<Uuid>,
        namespace: String,
        agent_id: String,
    },
    /// A memory was updated (reconsolidation, confidence change, etc.).
    Updated {
        memory_id: Uuid,
        namespace: String,
    },
    /// A memory was deleted.
    Deleted {
        memory_id: Uuid,
        namespace: String,
    },
    /// A version conflict was detected during reconsolidation.
    VersionConflict {
        memory_id: Uuid,
        expected_version: i32,
        actual_version: i32,
    },
    /// A scratchpad write occurred.
    ScratchpadWrite {
        namespace: String,
        key: String,
        agent_id: String,
    },
    /// A governance rule was violated.
    RuleViolation {
        rule_id: Uuid,
        agent_id: String,
        operation: String,
        reason: String,
    },
    /// A scope grant was changed (created or revoked).
    GrantChanged {
        grant_id: Uuid,
        action: String,
    },
    /// Surprise threshold was exceeded — consolidation needed.
    SurpriseThresholdExceeded {
        surprise: f64,
        threshold: f64,
    },
    /// Consolidation (reflection) was triggered.
    ConsolidationTriggered {
        episodes_reviewed: usize,
        abstractions_created: usize,
    },
    /// A memory needs reconsolidation due to contradictions.
    ReconsolidationNeeded {
        memory_id: Uuid,
        contradiction_count: usize,
    },
    /// Compression was triggered for a namespace.
    CompressionTriggered {
        namespace: String,
        memory_count: usize,
    },
    /// A new skill was discovered from an episode.
    SkillDiscovered {
        skill_id: Uuid,
    },
    /// A skill was improved (new version created).
    SkillImproved {
        skill_id: Uuid,
        new_version_id: Uuid,
    },
    /// Skills were generalized into new cross-domain skills.
    SkillGeneralized {
        skill_ids: Vec<Uuid>,
    },
    /// A new telos (goal) was created.
    TelosCreated {
        telos_id: Uuid,
        title: String,
        namespace: String,
        agent_id: String,
    },
    /// A telos status changed.
    TelosStatusChanged {
        telos_id: Uuid,
        new_status: String,
        agent_id: String,
    },
    /// Progress was made on a telos.
    TelosProgress {
        telos_id: Uuid,
        progress: f64,
        agent_id: String,
    },
    /// A meta-learning step completed — hyperparameters were adjusted.
    MetaLearningStep {
        adjustments: Vec<(String, f64, f64)>,
        generation: u64,
        phase: String,
    },
    /// Predictions were generated during a prediction cycle.
    PredictionGenerated {
        count: usize,
        namespace: String,
        sources: Vec<String>,
    },
    /// A prediction was resolved (matched or expired).
    PredictionResolved {
        prediction_id: uuid::Uuid,
        matched: bool,
        prediction_error: f64,
    },
    /// A regime change was detected by BOCPD.
    RegimeChange {
        namespace: String,
        confidence: f64,
    },
    /// Embedding projection was trained (loss improved).
    ProjectionTrained {
        loss_before: f64,
        loss_after: f64,
        triplet_count: usize,
    },
    /// Projected index rebuild was triggered after training.
    ProjectionRebuildTriggered {
        memory_count: usize,
    },
    /// A causal edge was strengthened by a new observation.
    CausalEdgeStrengthened {
        cause_id: Uuid,
        effect_id: Uuid,
        strength: f64,
        observations: u64,
    },
    /// NOTEARS structure learning completed.
    CausalStructureLearned {
        edges_discovered: usize,
        edges_confirmed: usize,
    },
    /// Intrinsic goals were generated from surprise hotspots.
    IntrinsicGoalGenerated {
        telos_ids: Vec<Uuid>,
        hotspots: usize,
        beta: f64,
    },
}
