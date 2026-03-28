use crate::error::Result;
use crate::runtime::Memoria;
use crate::types::agent::{AgentFilter, AgentRecord, AgentRegistration, AgentStatus};

impl Memoria {
    /// Register a new agent in the registry.
    pub fn register_agent(&self, reg: &AgentRegistration) -> Result<()> {
        self.store.register_agent(reg)
    }

    /// Get an agent record by ID.
    pub fn get_agent(&self, agent_id: &str) -> Result<Option<AgentRecord>> {
        self.store.get_agent(agent_id)
    }

    /// Update an agent's status.
    pub fn update_agent_status(&self, agent_id: &str, status: &AgentStatus) -> Result<()> {
        self.store.update_agent_status(agent_id, status)
    }

    /// Query agents with optional filters.
    pub fn query_agents(&self, filter: &AgentFilter) -> Result<Vec<AgentRecord>> {
        self.store.query_agents(filter)
    }

    /// Deregister an agent.
    pub fn deregister_agent(&self, agent_id: &str) -> Result<()> {
        self.store.deregister_agent(agent_id)
    }

    /// Add an agent to a team.
    pub fn add_team_membership(&self, agent_id: &str, team_id: &str) -> Result<()> {
        self.store.add_team_membership(agent_id, team_id)
    }

    /// Remove an agent from a team.
    pub fn remove_team_membership(&self, agent_id: &str, team_id: &str) -> Result<()> {
        self.store.remove_team_membership(agent_id, team_id)
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::Memoria;
    use crate::types::agent::{AgentFilter, AgentRegistration, AgentStatus};

    #[test]
    fn test_register_and_get_agent() {
        let m = Memoria::with_mocks(128).unwrap();
        let reg = AgentRegistration::new("agent-1", "Agent One")
            .with_team("eng")
            .with_role("executor");

        m.register_agent(&reg).unwrap();

        let record = m.get_agent("agent-1").unwrap().unwrap();
        assert_eq!(record.agent_id, "agent-1");
        assert_eq!(record.display_name, "Agent One");
        assert_eq!(record.status, AgentStatus::Active);
        assert_eq!(record.team_id.as_deref(), Some("eng"));
        assert_eq!(record.role.as_deref(), Some("executor"));
    }

    #[test]
    fn test_get_nonexistent_agent() {
        let m = Memoria::with_mocks(128).unwrap();
        let result = m.get_agent("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_update_agent_status() {
        let m = Memoria::with_mocks(128).unwrap();
        let reg = AgentRegistration::new("agent-1", "Agent One");
        m.register_agent(&reg).unwrap();

        m.update_agent_status("agent-1", &AgentStatus::Suspended).unwrap();

        let record = m.get_agent("agent-1").unwrap().unwrap();
        assert_eq!(record.status, AgentStatus::Suspended);
    }

    #[test]
    fn test_deregister_agent() {
        let m = Memoria::with_mocks(128).unwrap();
        let reg = AgentRegistration::new("agent-1", "Agent One");
        m.register_agent(&reg).unwrap();

        m.deregister_agent("agent-1").unwrap();

        let record = m.get_agent("agent-1").unwrap().unwrap();
        assert_eq!(record.status, AgentStatus::Deregistered);
    }

    #[test]
    fn test_query_agents_by_team() {
        let m = Memoria::with_mocks(128).unwrap();
        m.register_agent(&AgentRegistration::new("a1", "A1").with_team("eng")).unwrap();
        m.register_agent(&AgentRegistration::new("a2", "A2").with_team("eng")).unwrap();
        m.register_agent(&AgentRegistration::new("a3", "A3").with_team("design")).unwrap();

        let filter = AgentFilter {
            team_id: Some("eng".to_string()),
            ..Default::default()
        };
        let agents = m.query_agents(&filter).unwrap();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_team_membership() {
        let m = Memoria::with_mocks(128).unwrap();
        let reg = AgentRegistration::new("agent-1", "Agent One");
        m.register_agent(&reg).unwrap();

        m.add_team_membership("agent-1", "new-team").unwrap();
        m.remove_team_membership("agent-1", "new-team").unwrap();
    }
}
