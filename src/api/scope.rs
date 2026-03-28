use crate::error::Result;
use crate::runtime::Memoria;
use crate::types::scope::{GrantFilter, Permission, ScopeGrant};

impl Memoria {
    /// Grant scope (permissions) to agents matching a pattern.
    pub fn grant_scope(&self, grant: &ScopeGrant) -> Result<()> {
        self.store.insert_grant(grant)?;

        // Emit event if channel is available
        self.emit(crate::types::event::MemoryEvent::GrantChanged {
            grant_id: grant.id,
            action: "granted".to_string(),
        });

        Ok(())
    }

    /// Revoke a scope grant.
    pub fn revoke_scope(&self, grant_id: uuid::Uuid) -> Result<()> {
        self.store.revoke_grant(grant_id)?;

        self.emit(crate::types::event::MemoryEvent::GrantChanged {
            grant_id,
            action: "revoked".to_string(),
        });

        Ok(())
    }

    /// Query active scope grants.
    pub fn query_grants(&self, filter: &GrantFilter) -> Result<Vec<ScopeGrant>> {
        self.store.query_grants(filter)
    }

    /// Check if an agent has a specific permission on a namespace.
    pub fn check_permission(
        &self,
        agent_id: &str,
        agent_team: Option<&str>,
        agent_org: Option<&str>,
        agent_role: Option<&str>,
        namespace: &str,
        permission: &Permission,
    ) -> Result<bool> {
        self.store
            .check_permission(agent_id, agent_team, agent_org, agent_role, namespace, permission)
    }

    /// Internal scope check — called before tell/ask when enforcement is enabled.
    pub(crate) fn check_scope(
        &self,
        ctx: &crate::types::query::AgentContext,
        permission: &Permission,
    ) -> Result<()> {
        if !self.config.load().scope_enforcement_enabled {
            return Ok(());
        }

        let allowed = self.store.check_permission(
            &ctx.agent_id,
            ctx.team_id.as_deref(),
            ctx.org_id.as_deref(),
            ctx.role.as_deref(),
            &ctx.namespace,
            permission,
        )?;

        if !allowed {
            return Err(crate::error::MemoriaError::PermissionDenied {
                agent: ctx.agent_id.clone(),
                permission: permission.to_string(),
                namespace: ctx.namespace.clone(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::Memoria;
    use crate::types::query::AgentContext;
    use crate::types::scope::{AgentPattern, GrantFilter, Permission, ScopeGrant};

    #[test]
    fn test_grant_and_check_permission() {
        let m = Memoria::with_mocks(128).unwrap();

        let grant = ScopeGrant::new(
            AgentPattern::Team("eng".to_string()),
            "team:eng:*",
            vec![Permission::Read, Permission::Write],
            "admin",
        );
        m.grant_scope(&grant).unwrap();

        // Agent in the team should have access
        let has = m
            .check_permission("a1", Some("eng"), None, None, "team:eng:docs", &Permission::Read)
            .unwrap();
        assert!(has, "team member should have read access");

        // Agent NOT in the team should not
        let has = m
            .check_permission("a2", Some("design"), None, None, "team:eng:docs", &Permission::Read)
            .unwrap();
        assert!(!has, "non-team member should not have access");
    }

    #[test]
    fn test_revoke_grant() {
        let m = Memoria::with_mocks(128).unwrap();

        let grant = ScopeGrant::new(
            AgentPattern::Exact("a1".to_string()),
            "private:a1",
            vec![Permission::Read],
            "admin",
        );
        let gid = grant.id;
        m.grant_scope(&grant).unwrap();

        assert!(
            m.check_permission("a1", None, None, None, "private:a1", &Permission::Read)
                .unwrap()
        );

        m.revoke_scope(gid).unwrap();

        assert!(
            !m.check_permission("a1", None, None, None, "private:a1", &Permission::Read)
                .unwrap()
        );
    }

    #[test]
    fn test_wildcard_grant() {
        let m = Memoria::with_mocks(128).unwrap();

        let grant = ScopeGrant::new(
            AgentPattern::Any,
            "*",
            vec![Permission::Read],
            "admin",
        );
        m.grant_scope(&grant).unwrap();

        let has = m
            .check_permission("anyone", None, None, None, "any-namespace", &Permission::Read)
            .unwrap();
        assert!(has, "wildcard grant should match any agent and namespace");
    }

    #[test]
    fn test_query_grants() {
        let m = Memoria::with_mocks(128).unwrap();

        m.grant_scope(&ScopeGrant::new(
            AgentPattern::Team("eng".to_string()),
            "team:eng:*",
            vec![Permission::Read],
            "admin",
        )).unwrap();

        m.grant_scope(&ScopeGrant::new(
            AgentPattern::Team("design".to_string()),
            "team:design:*",
            vec![Permission::Read],
            "admin",
        )).unwrap();

        let all = m.query_grants(&GrantFilter::default()).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_scope_enforcement() {
        let m = Memoria::with_mocks(128).unwrap();
        {
            let mut cfg: crate::config::MemoriaConfig = (**m.config.load()).clone();
            cfg.scope_enforcement_enabled = true;
            m.config.store(std::sync::Arc::new(cfg));
        }

        let ctx = AgentContext::new("a1", "restricted").with_team("eng");

        // No grants → should fail
        let result = m.check_scope(&ctx, &Permission::Write);
        assert!(result.is_err());

        // Grant access
        m.grant_scope(&ScopeGrant::new(
            AgentPattern::Team("eng".to_string()),
            "restricted",
            vec![Permission::Write],
            "admin",
        )).unwrap();

        // Now should succeed
        m.check_scope(&ctx, &Permission::Write).unwrap();
    }

    #[test]
    fn test_scope_enforcement_disabled() {
        let m = Memoria::with_mocks(128).unwrap();
        // Default: enforcement disabled
        let ctx = AgentContext::new("a1", "anything");
        m.check_scope(&ctx, &Permission::Write).unwrap();
    }

    #[test]
    fn test_admin_implies_all() {
        let m = Memoria::with_mocks(128).unwrap();

        let grant = ScopeGrant::new(
            AgentPattern::Exact("admin-agent".to_string()),
            "*",
            vec![Permission::Admin],
            "system",
        );
        m.grant_scope(&grant).unwrap();

        // Admin should imply read, write, delete
        for perm in &[Permission::Read, Permission::Write, Permission::Delete] {
            assert!(
                m.check_permission("admin-agent", None, None, None, "any-ns", perm)
                    .unwrap(),
                "admin should imply {perm}"
            );
        }
    }
}
