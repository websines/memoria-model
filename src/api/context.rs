use uuid::Uuid;

use crate::types::query::AgentContext;

impl AgentContext {
    /// Create a new agent context with the given agent ID and namespace.
    pub fn new(agent_id: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            namespace: namespace.into(),
            current_episode: None,
            task_id: None,
            limit: None,
            context_memory_ids: Vec::new(),
            team_id: None,
            org_id: None,
            role: None,
            session_id: None,
            parent_agent_id: None,
            niche_hint: None,
        }
    }

    /// Set the current episode for this context.
    pub fn with_episode(mut self, episode_id: Uuid) -> Self {
        self.current_episode = Some(episode_id);
        self
    }

    /// Set the task ID for attribution and audit trail.
    pub fn with_task(mut self, task_id: Uuid) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Set the result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the working context memory IDs (for Hebbian scoring).
    pub fn with_context_memories(mut self, ids: Vec<Uuid>) -> Self {
        self.context_memory_ids = ids;
        self
    }

    /// Set the team ID for multi-agent coordination.
    pub fn with_team(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = Some(team_id.into());
        self
    }

    /// Set the organization ID for multi-tenant isolation.
    pub fn with_org(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    /// Set the agent's role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Set the session ID for grouping operations.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the parent agent ID for hierarchical topologies.
    pub fn with_parent_agent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_agent_id = Some(parent_id.into());
        self
    }

    /// Set the niche hint for skill selection in prime().
    pub fn with_niche_hint(mut self, niche: impl Into<String>) -> Self {
        self.niche_hint = Some(niche.into());
        self
    }
}
