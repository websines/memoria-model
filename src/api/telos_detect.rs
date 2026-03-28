/// Heuristic goal detection from natural language.
///
/// Runs on the hot path (no LLM call) inside `tell()`.
/// Detects goal-like intent from common patterns:
/// - "I need to X by Y" → telos + deadline
/// - "Let's work on X" → telos
/// - "Goal: X" / "Objective: X" → explicit telos
/// - "Don't forget to X" → depth 3-4 telos
/// - "We should X" → proposed telos (lower confidence)

/// A goal detected from natural language text.
#[derive(Debug, Clone)]
pub struct DetectedGoal {
    pub title: String,
    pub deadline: Option<i64>,
    pub depth: i32,
    pub confidence: f64,
}

/// Try to detect a goal from natural language text.
///
/// Returns `None` if no goal-like pattern is found.
/// Uses simple regex-free pattern matching for speed.
pub fn detect_goal(text: &str) -> Option<DetectedGoal> {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    // Explicit goal declarations (highest confidence)
    if let Some(goal) = try_explicit_goal(trimmed, &lower) {
        return Some(goal);
    }

    // "I need to X by Y" pattern
    if let Some(goal) = try_need_to_pattern(trimmed, &lower) {
        return Some(goal);
    }

    // "Let's work on X" pattern
    if let Some(goal) = try_lets_work_pattern(trimmed, &lower) {
        return Some(goal);
    }

    // "Don't forget to X" pattern
    if let Some(goal) = try_reminder_pattern(trimmed, &lower) {
        return Some(goal);
    }

    // "We should X" pattern (lowest confidence — inferred)
    if let Some(goal) = try_should_pattern(trimmed, &lower) {
        return Some(goal);
    }

    None
}

/// "Goal: X", "Objective: X", "Mission: X"
fn try_explicit_goal(text: &str, lower: &str) -> Option<DetectedGoal> {
    for prefix in ["goal:", "objective:", "mission:", "target:"] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let title = text[prefix.len()..].trim().to_string();
            if title.is_empty() || rest.trim().is_empty() {
                continue;
            }
            return Some(DetectedGoal {
                title,
                deadline: None,
                depth: 1, // Explicit goals are strategic
                confidence: 1.0,
            });
        }
    }
    None
}

/// "I need to X by Y", "I have to X by Y", "I must X by Y"
fn try_need_to_pattern(text: &str, lower: &str) -> Option<DetectedGoal> {
    let prefixes = [
        "i need to ",
        "i have to ",
        "i must ",
        "we need to ",
        "we have to ",
        "we must ",
    ];

    for prefix in prefixes {
        if !lower.starts_with(prefix) {
            continue;
        }

        // Use byte length of lowercase prefix to slice the original text
        // (safe because the prefix is ASCII-only)
        let rest = &text[prefix.len()..].trim();
        if rest.is_empty() {
            continue;
        }
        let (title, deadline) = extract_deadline(rest);

        if title.is_empty() {
            continue;
        }

        return Some(DetectedGoal {
            title,
            deadline,
            depth: 2, // Tactical
            confidence: 0.9,
        });
    }
    None
}

/// "Let's work on X", "Let's focus on X"
fn try_lets_work_pattern(text: &str, lower: &str) -> Option<DetectedGoal> {
    let prefixes = [
        "let's work on ",
        "let's focus on ",
        "lets work on ",
        "lets focus on ",
        "let us work on ",
    ];

    for prefix in prefixes {
        if !lower.starts_with(prefix) {
            continue;
        }

        // The apostrophe in "let's" may differ in byte length from the lowercase version.
        // Find the prefix end position by matching the lowercase prefix length.
        let title = text[prefix.len()..].trim().to_string();
        if title.is_empty() {
            continue;
        }

        return Some(DetectedGoal {
            title,
            deadline: None,
            depth: 2,
            confidence: 0.8,
        });
    }
    None
}

/// "Don't forget to X", "Remember to X", "Make sure to X"
fn try_reminder_pattern(text: &str, lower: &str) -> Option<DetectedGoal> {
    let prefixes = [
        "don't forget to ",
        "dont forget to ",
        "remember to ",
        "make sure to ",
        "make sure we ",
    ];

    for prefix in prefixes {
        if !lower.starts_with(prefix) {
            continue;
        }

        let rest = &text[prefix.len()..].trim();
        if rest.is_empty() {
            continue;
        }
        let (title, deadline) = extract_deadline(rest);

        if title.is_empty() {
            continue;
        }

        return Some(DetectedGoal {
            title,
            deadline,
            depth: 3, // Operational — reminder-like
            confidence: 0.85,
        });
    }
    None
}

/// "We should X", "I should X", "It would be good to X"
fn try_should_pattern(text: &str, lower: &str) -> Option<DetectedGoal> {
    let prefixes = [
        "we should ",
        "i should ",
        "it would be good to ",
        "we ought to ",
    ];

    for prefix in prefixes {
        if !lower.starts_with(prefix) {
            continue;
        }

        let title = text[prefix.len()..].trim().to_string();
        if title.is_empty() {
            continue;
        }

        return Some(DetectedGoal {
            title,
            deadline: None,
            depth: 2,
            confidence: 0.5, // Low confidence — speculative
        });
    }
    None
}

/// Extract a deadline phrase from text like "X by Friday" or "X by March 30".
///
/// Returns (title_without_deadline, parsed_deadline_ms).
/// Deadline parsing is best-effort — returns None if unrecognized.
fn extract_deadline(text: &str) -> (String, Option<i64>) {
    let lower = text.to_lowercase();

    // Look for "by <date>" at the end
    if let Some(by_pos) = lower.rfind(" by ") {
        let title = text[..by_pos].trim().to_string();
        let date_part = text[by_pos + 4..].trim();

        if let Some(deadline) = parse_relative_deadline(date_part) {
            return (title, Some(deadline));
        }

        // Even if we can't parse the deadline, keep the title clean
        return (title, None);
    }

    (text.trim().to_string(), None)
}

/// Parse relative deadline strings into unix ms.
/// Returns None for unrecognized formats.
fn parse_relative_deadline(text: &str) -> Option<i64> {
    let lower = text.to_lowercase();
    let lower = lower.trim_end_matches('.');
    let now = crate::types::memory::now_ms();
    let day_ms: i64 = 86_400_000;

    match lower {
        "today" | "tonight" | "end of day" | "eod" => Some(now + day_ms),
        "tomorrow" => Some(now + day_ms),
        "this week" | "end of week" | "eow" | "friday" => {
            // Approximate: up to 7 days
            Some(now + 7 * day_ms)
        }
        "next week" => Some(now + 7 * day_ms),
        "this month" | "end of month" | "eom" => Some(now + 30 * day_ms),
        "next month" => Some(now + 30 * day_ms),
        _ => {
            // Try "N days/weeks/months"
            let parts: Vec<&str> = lower.split_whitespace().collect();
            if parts.len() == 2 {
                if let Ok(n) = parts[0].parse::<i64>() {
                    return match parts[1].trim_end_matches('s') {
                        "day" => Some(now + n * day_ms),
                        "week" => Some(now + n * 7 * day_ms),
                        "month" => Some(now + n * 30 * day_ms),
                        _ => None,
                    };
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_goal() {
        let g = detect_goal("Goal: reduce customer churn by 20%").unwrap();
        assert_eq!(g.title, "reduce customer churn by 20%");
        assert_eq!(g.depth, 1);
        assert_eq!(g.confidence, 1.0);
    }

    #[test]
    fn test_need_to_with_deadline() {
        let g = detect_goal("I need to ship the Q3 deck by Friday").unwrap();
        assert_eq!(g.title, "ship the Q3 deck");
        assert!(g.deadline.is_some());
        assert_eq!(g.depth, 2);
    }

    #[test]
    fn test_lets_work_on() {
        let g = detect_goal("Let's work on the migration").unwrap();
        assert_eq!(g.title, "the migration");
        assert!(g.deadline.is_none());
        assert_eq!(g.depth, 2);
    }

    #[test]
    fn test_reminder_pattern() {
        let g = detect_goal("Don't forget to update the docs").unwrap();
        assert_eq!(g.title, "update the docs");
        assert_eq!(g.depth, 3);
    }

    #[test]
    fn test_should_pattern() {
        let g = detect_goal("We should clean up the API").unwrap();
        assert_eq!(g.title, "clean up the API");
        assert_eq!(g.confidence, 0.5);
    }

    #[test]
    fn test_no_goal_detected() {
        assert!(detect_goal("Hello, how are you?").is_none());
        assert!(detect_goal("The weather is nice today").is_none());
        assert!(detect_goal("").is_none());
    }

    #[test]
    fn test_deadline_parsing() {
        let (title, deadline) = extract_deadline("ship the deck by tomorrow");
        assert_eq!(title, "ship the deck");
        assert!(deadline.is_some());

        let (title, deadline) = extract_deadline("finish the report by 3 weeks");
        assert_eq!(title, "finish the report");
        assert!(deadline.is_some());
    }

    #[test]
    fn test_empty_title_rejected() {
        assert!(detect_goal("Goal:").is_none());
        assert!(detect_goal("Goal:  ").is_none());
        assert!(detect_goal("I need to ").is_none());
    }
}
