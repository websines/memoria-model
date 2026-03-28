//! Telos deadline enforcement.
//!
//! Periodically checks for telos that have exceeded their deadline
//! and transitions them to "stalled" status with an event record.

use crate::store::CozoStore;
use crate::types::memory::now_ms;
use crate::types::telos::TelosEvent;

/// Enforce deadlines on active telos.
///
/// Finds all active telos with a deadline in the past and transitions
/// them to "stalled" status. Records a `deadline_exceeded` event for each.
///
/// This operates directly on the store (not on `Memoria`) so it can be
/// called from the `QueueWorker` without requiring a full `Memoria` instance.
///
/// Returns the number of telos that were stalled.
pub fn enforce_deadlines(store: &CozoStore) -> crate::error::Result<u64> {
    let now = now_ms();

    // Query all active telos with a deadline in the past
    let overdue = store.list_overdue_telos(now)?;

    let mut stalled_count = 0u64;
    for telos in overdue {
        eprintln!(
            "[deadline_enforcer] telos {} '{}' — deadline exceeded ({}), marking stalled",
            telos.id,
            telos.title,
            deadline_display(telos.deadline, now),
        );

        store.update_telos_status(telos.id, "stalled")?;

        // Record a deadline_exceeded event
        let mut event = TelosEvent::new(telos.id, "deadline_exceeded");
        event.agent_id = "system:deadline_enforcer".to_string();
        event.description = format!(
            "Deadline exceeded (was {})",
            deadline_display(telos.deadline, now),
        );
        store.insert_telos_event(&event)?;

        // Set stalled_since if not already set
        store.update_telos_stalled_since(telos.id, now)?;

        stalled_count += 1;
    }

    if stalled_count > 0 {
        eprintln!("[deadline_enforcer] enforced {} telos deadlines", stalled_count);
    }

    Ok(stalled_count)
}

/// Format a deadline timestamp into a human-readable relative string.
///
/// Returns e.g. "overdue by 2h", "overdue by 3d", or the raw ms value
/// if the deadline cannot be interpreted.
fn deadline_display(deadline: Option<i64>, now: i64) -> String {
    match deadline {
        None => "no deadline".to_string(),
        Some(d) => {
            let overdue_ms = now - d;
            if overdue_ms <= 0 {
                // Should not happen (we only call this for past deadlines), but be safe
                return format!("{d}ms");
            }
            let hours = overdue_ms / 3_600_000;
            if hours < 24 {
                format!("overdue by {hours}h")
            } else {
                let days = hours / 24;
                format!("overdue by {days}d")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::CozoStore;
    use crate::types::telos::{Telos, TelosStatus};

    fn make_telos_with_deadline(store: &CozoStore, deadline: Option<i64>) -> Telos {
        let mut t = Telos::new("Test goal", "desc", vec![0.1; 4], "agent", "user");
        t.namespace = "test".to_string();
        t.deadline = deadline;
        store.insert_telos(&t).unwrap();
        t
    }

    #[test]
    fn test_enforce_deadlines_marks_overdue() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = now_ms();

        // Overdue telos (deadline in the past)
        let t_overdue = make_telos_with_deadline(&store, Some(now - 3_600_000));
        // Future telos (deadline 1h from now — should NOT be touched)
        let t_future = make_telos_with_deadline(&store, Some(now + 3_600_000));
        // No-deadline telos — should NOT be touched
        let t_nodl = make_telos_with_deadline(&store, None);

        let stalled = enforce_deadlines(&store).unwrap();
        assert_eq!(stalled, 1, "only one telos should be stalled");

        // Overdue telos should now be stalled
        let updated_overdue = store.get_telos(t_overdue.id).unwrap().unwrap();
        assert_eq!(updated_overdue.status, TelosStatus::Stalled);
        assert!(updated_overdue.stalled_since.is_some());

        // Future telos should still be active
        let updated_future = store.get_telos(t_future.id).unwrap().unwrap();
        assert_eq!(updated_future.status, TelosStatus::Active);

        // No-deadline telos should still be active
        let updated_nodl = store.get_telos(t_nodl.id).unwrap().unwrap();
        assert_eq!(updated_nodl.status, TelosStatus::Active);
    }

    #[test]
    fn test_enforce_deadlines_no_overdue() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = now_ms();

        // Only future deadlines
        make_telos_with_deadline(&store, Some(now + 86_400_000));
        make_telos_with_deadline(&store, None);

        let stalled = enforce_deadlines(&store).unwrap();
        assert_eq!(stalled, 0);
    }

    #[test]
    fn test_enforce_deadlines_idempotent() {
        let store = CozoStore::open_mem(4).unwrap();
        let now = now_ms();

        let t = make_telos_with_deadline(&store, Some(now - 1_000));

        // Run twice
        let first = enforce_deadlines(&store).unwrap();
        let second = enforce_deadlines(&store).unwrap();

        assert_eq!(first, 1);
        // Second run finds 0 because telos is now "stalled", not "active"
        assert_eq!(second, 0);

        let updated = store.get_telos(t.id).unwrap().unwrap();
        assert_eq!(updated.status, TelosStatus::Stalled);
    }

    #[test]
    fn test_deadline_display() {
        let now = now_ms();

        assert_eq!(
            deadline_display(Some(now - 3_600_000), now),
            "overdue by 1h"
        );
        assert_eq!(
            deadline_display(Some(now - 2 * 86_400_000), now),
            "overdue by 2d"
        );
        assert_eq!(deadline_display(None, now), "no deadline");
    }
}
