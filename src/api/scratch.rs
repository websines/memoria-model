use crate::error::Result;
use crate::runtime::Memoria;
use crate::types::query::AgentContext;
use crate::types::scratch::{ScratchEntry, ScratchValue, Visibility};

impl Memoria {
    /// Put a value into the scratchpad (upsert).
    pub fn scratch_put(
        &self,
        namespace: &str,
        key: &str,
        value: ScratchValue,
        ctx: &AgentContext,
        expires_at: Option<i64>,
    ) -> Result<()> {
        let visibility = if ctx.team_id.is_some() {
            Visibility::Team
        } else {
            Visibility::Private
        };

        self.store.scratch_put(
            namespace,
            key,
            &value,
            &ctx.agent_id,
            &visibility,
            expires_at,
        )?;

        // Emit event
        self.emit(crate::types::event::MemoryEvent::ScratchpadWrite {
            namespace: namespace.to_string(),
            key: key.to_string(),
            agent_id: ctx.agent_id.clone(),
        });

        Ok(())
    }

    /// Put a value with explicit visibility.
    pub fn scratch_put_with_visibility(
        &self,
        namespace: &str,
        key: &str,
        value: ScratchValue,
        ctx: &AgentContext,
        visibility: Visibility,
        expires_at: Option<i64>,
    ) -> Result<()> {
        self.store.scratch_put(
            namespace,
            key,
            &value,
            &ctx.agent_id,
            &visibility,
            expires_at,
        )?;

        self.emit(crate::types::event::MemoryEvent::ScratchpadWrite {
            namespace: namespace.to_string(),
            key: key.to_string(),
            agent_id: ctx.agent_id.clone(),
        });

        Ok(())
    }

    /// Get a scratchpad entry.
    pub fn scratch_get(&self, namespace: &str, key: &str) -> Result<Option<ScratchEntry>> {
        self.store.scratch_get(namespace, key)
    }

    /// List all scratchpad entries in a namespace.
    pub fn scratch_list(&self, namespace: &str) -> Result<Vec<ScratchEntry>> {
        self.store.scratch_list(namespace)
    }

    /// Delete a scratchpad entry.
    pub fn scratch_delete(&self, namespace: &str, key: &str) -> Result<()> {
        self.store.scratch_delete(namespace, key)
    }

    /// Clear all scratchpad entries in a namespace.
    pub fn scratch_clear(&self, namespace: &str) -> Result<()> {
        self.store.scratch_clear(namespace)
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::Memoria;
    use crate::types::query::AgentContext;
    use crate::types::scratch::ScratchValue;

    #[test]
    fn test_scratch_put_and_get() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("a1", "default");

        m.scratch_put("scratch:default", "status", ScratchValue::Text("working".into()), &ctx, None)
            .unwrap();

        let entry = m.scratch_get("scratch:default", "status").unwrap().unwrap();
        assert!(matches!(entry.value, ScratchValue::Text(ref s) if s == "working"));
        assert_eq!(entry.owner_agent, "a1");
    }

    #[test]
    fn test_scratch_get_nonexistent() {
        let m = Memoria::with_mocks(128).unwrap();
        let result = m.scratch_get("ns", "missing").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_scratch_list() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("a1", "default");

        m.scratch_put("ns", "k1", ScratchValue::Number(42.0), &ctx, None).unwrap();
        m.scratch_put("ns", "k2", ScratchValue::Bool(true), &ctx, None).unwrap();
        m.scratch_put("other", "k3", ScratchValue::Text("x".into()), &ctx, None).unwrap();

        let entries = m.scratch_list("ns").unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_scratch_delete() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("a1", "default");

        m.scratch_put("ns", "k", ScratchValue::Text("v".into()), &ctx, None).unwrap();
        m.scratch_delete("ns", "k").unwrap();

        assert!(m.scratch_get("ns", "k").unwrap().is_none());
    }

    #[test]
    fn test_scratch_clear() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("a1", "default");

        m.scratch_put("ns", "k1", ScratchValue::Text("a".into()), &ctx, None).unwrap();
        m.scratch_put("ns", "k2", ScratchValue::Text("b".into()), &ctx, None).unwrap();
        m.scratch_clear("ns").unwrap();

        let entries = m.scratch_list("ns").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_scratch_upsert() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("a1", "default");

        m.scratch_put("ns", "k", ScratchValue::Text("v1".into()), &ctx, None).unwrap();
        m.scratch_put("ns", "k", ScratchValue::Text("v2".into()), &ctx, None).unwrap();

        let entry = m.scratch_get("ns", "k").unwrap().unwrap();
        assert!(matches!(entry.value, ScratchValue::Text(ref s) if s == "v2"));
    }

    #[test]
    fn test_scratch_json_value() {
        let m = Memoria::with_mocks(128).unwrap();
        let ctx = AgentContext::new("a1", "default");

        let val = ScratchValue::Json(serde_json::json!({"progress": 0.5, "step": 3}));
        m.scratch_put("ns", "state", val, &ctx, None).unwrap();

        let entry = m.scratch_get("ns", "state").unwrap().unwrap();
        if let ScratchValue::Json(v) = &entry.value {
            assert_eq!(v["step"], 3);
        } else {
            panic!("expected Json variant");
        }
    }
}
