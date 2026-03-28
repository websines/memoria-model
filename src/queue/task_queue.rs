use std::collections::BTreeMap;

use cozo::DataValue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{MemoriaError, Result};
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// Status of a queued task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    DeadLetter,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::DeadLetter => "dead_letter",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "pending" => Self::Pending,
            "running" => Self::Running,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "dead_letter" => Self::DeadLetter,
            _ => Self::Pending,
        }
    }
}

/// A task in the queue.
#[derive(Debug, Clone)]
pub struct QueueTask {
    pub id: Uuid,
    pub enqueued_at: i64,
    pub task_type: String,
    pub priority: i64,
    pub payload: String,
    pub status: TaskStatus,
    pub attempts: i64,
    pub max_attempts: i64,
    pub locked_until: Option<i64>,
    pub last_error: Option<String>,
    pub completed_at: Option<i64>,
    pub result: Option<String>,
}

/// CozoDB-backed persistent task queue.
///
/// Uses optimistic locking: `dequeue` sets `locked_until` to prevent
/// concurrent workers from claiming the same task. Tasks are retried
/// up to `max_attempts` before being dead-lettered.
pub struct TaskQueue {
    store: CozoStore,
}

impl TaskQueue {
    pub fn new(store: CozoStore) -> Self {
        Self { store }
    }

    /// Enqueue a new task. Returns the task ID.
    pub fn enqueue(
        &self,
        task_type: &str,
        priority: i64,
        payload: &str,
        max_attempts: i64,
    ) -> Result<Uuid> {
        let id = Uuid::now_v7();
        let now = now_ms();

        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(id.to_string()));
        params.insert("enqueued_at".into(), DataValue::from(now));
        params.insert("task_type".into(), DataValue::from(task_type));
        params.insert("priority".into(), DataValue::from(priority));
        params.insert("payload".into(), DataValue::from(payload));
        params.insert("status".into(), DataValue::from("pending"));
        params.insert("attempts".into(), DataValue::from(0i64));
        params.insert("max_attempts".into(), DataValue::from(max_attempts));
        params.insert("locked_until".into(), DataValue::Null);
        params.insert("last_error".into(), DataValue::Null);
        params.insert("completed_at".into(), DataValue::Null);
        params.insert("result".into(), DataValue::Null);

        self.store.run_script(
            "?[id, enqueued_at, task_type, priority, payload, status, attempts, max_attempts, locked_until, last_error, completed_at, result] <- [[$id, $enqueued_at, $task_type, $priority, $payload, $status, $attempts, $max_attempts, $locked_until, $last_error, $completed_at, $result]]\n:put task_queue {id, enqueued_at => task_type, priority, payload, status, attempts, max_attempts, locked_until, last_error, completed_at, result}",
            params,
        )?;

        Ok(id)
    }

    /// Dequeue the highest-priority pending task. Returns None if queue is empty.
    ///
    /// Sets `locked_until` to `now + lock_duration_ms` and status to "running".
    pub fn dequeue(&self, lock_duration_ms: i64) -> Result<Option<QueueTask>> {
        let now = now_ms();

        let mut params = BTreeMap::new();
        params.insert("now".into(), DataValue::from(now));

        // Find the highest-priority pending task.
        let result = self.store.run_query(
            r#"?[id, enqueued_at, task_type, priority, payload, status, attempts,
                max_attempts, locked_until, last_error, completed_at, result] :=
                *task_queue{id, enqueued_at, task_type, priority, payload, status,
                            attempts, max_attempts, locked_until, last_error,
                            completed_at, result},
                status == 'pending'
            :sort -priority, enqueued_at
            :limit 1"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        let mut task = parse_queue_task_row(row)?;

        // Lock the task
        let lock_until = now + lock_duration_ms;
        task.status = TaskStatus::Running;
        task.locked_until = Some(lock_until);
        task.attempts += 1;

        self.put_full_row(&task)?;

        Ok(Some(task))
    }

    /// Mark a task as completed with a result.
    pub fn complete(&self, id: Uuid, enqueued_at: i64, result_str: &str) -> Result<()> {
        // Read current state, modify, write back
        let mut task = self.get_task(id, enqueued_at)?
            .ok_or_else(|| MemoriaError::TaskQueue("task not found".into()))?;

        task.status = TaskStatus::Completed;
        task.completed_at = Some(now_ms());
        task.result = Some(result_str.to_string());
        task.locked_until = None;

        self.put_full_row(&task)
    }

    /// Mark a task as failed. If attempts >= max_attempts, dead-letter it.
    pub fn fail(&self, id: Uuid, enqueued_at: i64, error: &str, max_attempts: i64, current_attempts: i64) -> Result<()> {
        let mut task = self.get_task(id, enqueued_at)?
            .ok_or_else(|| MemoriaError::TaskQueue("task not found".into()))?;

        task.status = if current_attempts >= max_attempts {
            TaskStatus::DeadLetter
        } else {
            TaskStatus::Pending
        };
        task.last_error = Some(error.to_string());
        task.locked_until = None;

        self.put_full_row(&task)
    }

    /// Get a task by id and enqueued_at.
    fn get_task(&self, id: Uuid, enqueued_at: i64) -> Result<Option<QueueTask>> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(id.to_string()));
        params.insert("enqueued_at".into(), DataValue::from(enqueued_at));

        let result = self.store.run_query(
            r#"?[id, enqueued_at, task_type, priority, payload, status, attempts,
                max_attempts, locked_until, last_error, completed_at, result] :=
                *task_queue{id, enqueued_at, task_type, priority, payload, status,
                            attempts, max_attempts, locked_until, last_error,
                            completed_at, result},
                id == to_uuid($id), enqueued_at == $enqueued_at"#,
            params,
        )?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        Ok(Some(parse_queue_task_row(&result.rows[0])?))
    }

    /// Write a complete QueueTask row back to CozoDB.
    fn put_full_row(&self, task: &QueueTask) -> Result<()> {
        let mut params = BTreeMap::new();
        params.insert("id".into(), DataValue::from(task.id.to_string()));
        params.insert("enqueued_at".into(), DataValue::from(task.enqueued_at));
        params.insert("task_type".into(), DataValue::from(task.task_type.as_str()));
        params.insert("priority".into(), DataValue::from(task.priority));
        params.insert("payload".into(), DataValue::from(task.payload.as_str()));
        params.insert("status".into(), DataValue::from(task.status.as_str()));
        params.insert("attempts".into(), DataValue::from(task.attempts));
        params.insert("max_attempts".into(), DataValue::from(task.max_attempts));

        if let Some(lu) = task.locked_until {
            params.insert("locked_until".into(), DataValue::from(lu));
        } else {
            params.insert("locked_until".into(), DataValue::Null);
        }
        if let Some(ref e) = task.last_error {
            params.insert("last_error".into(), DataValue::from(e.as_str()));
        } else {
            params.insert("last_error".into(), DataValue::Null);
        }
        if let Some(ca) = task.completed_at {
            params.insert("completed_at".into(), DataValue::from(ca));
        } else {
            params.insert("completed_at".into(), DataValue::Null);
        }
        if let Some(ref r) = task.result {
            params.insert("result".into(), DataValue::from(r.as_str()));
        } else {
            params.insert("result".into(), DataValue::Null);
        }

        self.store.run_script(
            "?[id, enqueued_at, task_type, priority, payload, status, attempts, max_attempts, locked_until, last_error, completed_at, result] <- [[$id, $enqueued_at, $task_type, $priority, $payload, $status, $attempts, $max_attempts, $locked_until, $last_error, $completed_at, $result]]\n:put task_queue {id, enqueued_at => task_type, priority, payload, status, attempts, max_attempts, locked_until, last_error, completed_at, result}",
            params,
        )?;

        Ok(())
    }

    /// Count pending tasks.
    pub fn count_pending(&self) -> Result<usize> {
        let result = self.store.run_query(
            r#"?[count(id)] := *task_queue{id, status}, status == "pending""#,
            BTreeMap::new(),
        )?;

        if result.rows.is_empty() {
            return Ok(0);
        }

        Ok(result.rows[0][0].get_int().unwrap_or(0) as usize)
    }
}

/// Parse a CozoDB row into a QueueTask.
fn parse_queue_task_row(row: &[DataValue]) -> Result<QueueTask> {
    let id = parse_uuid(&row[0])?;
    let enqueued_at = row[1].get_int().unwrap_or(0);
    let task_type = row[2].get_str().unwrap_or("").to_string();
    let priority = row[3].get_int().unwrap_or(0);
    let payload = row[4].get_str().unwrap_or("{}").to_string();
    let status = TaskStatus::from_str_lossy(row[5].get_str().unwrap_or("pending"));
    let attempts = row[6].get_int().unwrap_or(0);
    let max_attempts = row[7].get_int().unwrap_or(3);
    let locked_until = row[8].get_int();
    let last_error = row[9].get_str().map(|s| s.to_string());
    let completed_at = row[10].get_int();
    let result = row[11].get_str().map(|s| s.to_string());

    Ok(QueueTask {
        id,
        enqueued_at,
        task_type,
        priority,
        payload,
        status,
        attempts,
        max_attempts,
        locked_until,
        last_error,
        completed_at,
        result,
    })
}

fn parse_uuid(val: &DataValue) -> Result<uuid::Uuid> {
    match val {
        DataValue::Str(s) => uuid::Uuid::parse_str(s.as_ref())
            .map_err(|e| MemoriaError::TaskQueue(format!("parsing uuid: {e}"))),
        DataValue::Uuid(u) => Ok(uuid::Uuid::from_bytes(*u.0.as_bytes())),
        _ => Err(MemoriaError::TaskQueue(format!(
            "expected uuid, got {val:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_queue() -> TaskQueue {
        let store = CozoStore::open_mem(4).unwrap();
        TaskQueue::new(store)
    }

    #[test]
    fn test_enqueue_and_dequeue() {
        let q = make_queue();
        let id = q.enqueue("verify_relations", 0, r#"{"memory_id": "abc"}"#, 3)
            .expect("enqueue should succeed");
        assert_ne!(id, Uuid::nil());

        let task = q.dequeue(60_000).expect("dequeue should succeed");
        assert!(task.is_some());
        let task = task.unwrap();
        assert_eq!(task.task_type, "verify_relations");
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.attempts, 1);
    }

    #[test]
    fn test_dequeue_empty_queue() {
        let q = make_queue();
        let task = q.dequeue(60_000).unwrap();
        assert!(task.is_none());
    }

    #[test]
    fn test_dequeue_respects_priority() {
        let q = make_queue();
        q.enqueue("low_priority", 0, "{}", 3).unwrap();
        q.enqueue("high_priority", 10, "{}", 3).unwrap();

        let task = q.dequeue(60_000).unwrap().unwrap();
        assert_eq!(task.task_type, "high_priority");
    }

    #[test]
    fn test_complete_task() {
        let q = make_queue();
        q.enqueue("test_task", 0, "{}", 3).unwrap();
        let task = q.dequeue(60_000).unwrap().unwrap();

        q.complete(task.id, task.enqueued_at, r#"{"status": "ok"}"#).unwrap();

        // Should not dequeue completed tasks
        let next = q.dequeue(60_000).unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_fail_and_retry() {
        let q = make_queue();
        q.enqueue("retry_task", 0, "{}", 3).unwrap();

        // First attempt
        let task = q.dequeue(60_000).unwrap().unwrap();
        q.fail(task.id, task.enqueued_at, "timeout", task.max_attempts, task.attempts).unwrap();

        // Should be available for retry (attempts=1 < max_attempts=3)
        let retry = q.dequeue(60_000).unwrap();
        assert!(retry.is_some());
    }

    #[test]
    fn test_dead_letter_after_max_attempts() {
        let q = make_queue();
        q.enqueue("failing_task", 0, "{}", 2).unwrap();

        // Attempt 1
        let t1 = q.dequeue(60_000).unwrap().unwrap();
        q.fail(t1.id, t1.enqueued_at, "error 1", t1.max_attempts, t1.attempts).unwrap();

        // Attempt 2
        let t2 = q.dequeue(60_000).unwrap().unwrap();
        q.fail(t2.id, t2.enqueued_at, "error 2", t2.max_attempts, t2.attempts).unwrap();

        // Should be dead-lettered now, not available
        let t3 = q.dequeue(60_000).unwrap();
        assert!(t3.is_none(), "should be dead-lettered after max attempts");
    }

    #[test]
    fn test_count_pending() {
        let q = make_queue();
        assert_eq!(q.count_pending().unwrap(), 0);

        q.enqueue("task_a", 0, "{}", 3).unwrap();
        q.enqueue("task_b", 0, "{}", 3).unwrap();
        assert_eq!(q.count_pending().unwrap(), 2);

        q.dequeue(60_000).unwrap();
        assert_eq!(q.count_pending().unwrap(), 1);
    }
}
