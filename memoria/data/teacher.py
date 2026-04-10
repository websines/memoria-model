"""Teacher LLM for synthetic data generation across all training phases.

Configurable frontier model that generates:
- Phase 1: Hard reasoning problems, scientific QA with traces
- Phase 2: Agentic trajectories, code problems, tool-call scenarios
- Phase 3a SFT: Multi-session conversations, fact revision, goal tracking
- Phase 3b DPO: Preference pairs (retrieve-over-regenerate, revise-over-accumulate)
- Online: Evaluation grading, curriculum steering, weakness-targeted examples

Uses Anthropic-compatible API (works with Claude, MiniMax, or any compatible provider).
Rate-limited with error recovery. 500 errors are silently dropped — never teach bad outputs.

Configuration via environment variables:
  TEACHER_API_KEY     — API key (required)
  TEACHER_BASE_URL    — Base URL (optional, defaults to Anthropic)
  TEACHER_MODEL       — Model name (optional, defaults to config)

Usage:
  teacher = TeacherLLM()  # reads from env + defaults
  teacher = TeacherLLM(TeacherConfig(model="claude-sonnet-4-20250514"))  # override
  response = teacher.generate("You are helpful.", [{"role": "user", "content": "..."}])
"""

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeacherConfig:
    """Configuration for teacher LLM. All fields overridable."""
    model: str = "MiniMax-M2.7"
    base_url: str = ""           # empty = use env TEACHER_BASE_URL, then Anthropic default
    api_key: str = ""            # empty = use env TEACHER_API_KEY
    max_tokens: int = 8192
    max_context: int = 131072
    temperature: float = 0.7     # some creativity for synthetic data
    # Rate limiting
    rate_limit_calls: int = 4000
    rate_limit_window: int = 5 * 3600  # 5 hours in seconds
    # Error handling
    max_retries: int = 3
    retry_base_delay: float = 2.0  # exponential backoff base
    # Thinking support (MiniMax-M2.7 has <thinking> blocks)
    include_thinking: bool = True


class _RateLimiter:
    """Sliding window rate limiter. Thread-safe."""

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window = window_seconds
        self.timestamps: deque[float] = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Block until a call is allowed under the rate limit."""
        with self.lock:
            now = time.time()
            # Purge expired timestamps
            cutoff = now - self.window
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.max_calls:
                # Must wait until the oldest call expires
                wait_until = self.timestamps[0] + self.window
                wait_time = wait_until - now
                if wait_time > 0:
                    print(f"  [teacher] Rate limit reached ({self.max_calls}/{self.window}s). "
                          f"Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)

            self.timestamps.append(time.time())

    @property
    def calls_remaining(self) -> int:
        with self.lock:
            cutoff = time.time() - self.window
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
            return self.max_calls - len(self.timestamps)


@dataclass
class TeacherResponse:
    """Parsed response from teacher LLM."""
    text: str                          # final text output
    thinking: str = ""                 # thinking trace (if model supports it)
    raw_content: list = field(default_factory=list)  # full response.content for multi-turn
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True
    error: str = ""


class TeacherLLM:
    """Configurable frontier LLM teacher.

    Handles rate limiting, 500 error recovery, and multi-turn state.
    Configurable to swap models — change TeacherConfig or env vars.
    """

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()
        self._client = None
        self._rate_limiter = _RateLimiter(
            self.config.rate_limit_calls,
            self.config.rate_limit_window,
        )
        self._total_calls = 0
        self._total_errors = 0

    def _get_client(self):
        """Lazy-init Anthropic client from config + env vars."""
        if self._client is not None:
            return self._client

        import anthropic

        api_key = (
            self.config.api_key
            or os.environ.get("TEACHER_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Teacher API key not set. Set TEACHER_API_KEY or ANTHROPIC_API_KEY env var, "
                "or pass api_key in TeacherConfig."
            )

        base_url = (
            self.config.base_url
            or os.environ.get("TEACHER_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or None  # use Anthropic default
        )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = anthropic.Anthropic(**kwargs)
        model_name = os.environ.get("TEACHER_MODEL") or self.config.model
        self.config.model = model_name
        print(f"  [teacher] Initialized: model={model_name}, "
              f"base_url={base_url or 'default'}, "
              f"rate_limit={self.config.rate_limit_calls}/{self.config.rate_limit_window}s")
        return self._client

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> TeacherResponse:
        """Single generation with rate limiting and error recovery.

        Args:
            system: system prompt
            messages: list of {"role": "user"|"assistant", "content": ...}
            max_tokens: override config max_tokens
            temperature: override config temperature

        Returns:
            TeacherResponse with text, thinking, and raw_content.
            On error: ok=False, text="", error=<message>.
        """
        client = self._get_client()
        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        for attempt in range(self.config.max_retries):
            self._rate_limiter.wait_if_needed()
            self._total_calls += 1

            try:
                response = client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tok,
                    system=system,
                    messages=messages,
                    temperature=temp,
                )

                # Parse response content blocks
                text_parts = []
                thinking_parts = []
                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "thinking":
                            thinking_parts.append(block.thinking)

                return TeacherResponse(
                    text="\n".join(text_parts),
                    thinking="\n".join(thinking_parts),
                    raw_content=response.content,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    ok=True,
                )

            except Exception as e:
                self._total_errors += 1
                error_str = str(e)

                # Server errors: retry with backoff (500, 502, 503, 529 overloaded)
                is_server_error = any(code in error_str for code in ("500", "502", "503", "529"))
                is_rate_error = "429" in error_str or "rate" in error_str.lower()

                if attempt < self.config.max_retries - 1 and (is_server_error or is_rate_error):
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"  [teacher] Attempt {attempt + 1} failed ({error_str[:80]}), "
                          f"retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                # Final failure — return error, DON'T propagate
                return TeacherResponse(
                    text="",
                    ok=False,
                    error=error_str[:200],
                )

        return TeacherResponse(text="", ok=False, error="max retries exceeded")

    def multi_turn(
        self,
        system: str,
        messages: list[dict],
        continuations: int = 1,
        max_tokens: int | None = None,
    ) -> list[dict]:
        """Multi-turn conversation preserving full response.content.

        Important: appends the complete response.content list (including thinking
        and tool_use blocks) to maintain reasoning chain continuity.

        Args:
            system: system prompt
            messages: initial messages
            continuations: number of assistant turns to generate
            max_tokens: per-turn max tokens

        Returns:
            Full message history with assistant responses appended.
        """
        history = list(messages)

        for _ in range(continuations):
            response = self.generate(system, history, max_tokens=max_tokens)
            if not response.ok:
                break

            # Append full content blocks to maintain reasoning chain
            # (thinking + text + tool_use all preserved)
            content_blocks = []
            for block in response.raw_content:
                if hasattr(block, 'type'):
                    if block.type == "thinking":
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                        })
                    elif block.type == "text":
                        content_blocks.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif block.type == "tool_use":
                        content_blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

            history.append({
                "role": "assistant",
                "content": content_blocks if len(content_blocks) > 1 else response.text,
            })

        return history

    def generate_batch(
        self,
        system: str,
        prompts: list[str],
        max_tokens: int | None = None,
        show_progress: bool = True,
    ) -> list[TeacherResponse]:
        """Generate responses for a batch of prompts.

        Rate-limited. Failed generations (500 errors) are excluded from results
        with ok=False — caller should filter these out before using as training data.

        Args:
            system: shared system prompt
            prompts: list of user messages
            max_tokens: per-generation max tokens
            show_progress: print progress

        Returns:
            List of TeacherResponse (check .ok before using)
        """
        results = []
        n_ok = 0
        n_fail = 0

        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            response = self.generate(system, messages, max_tokens=max_tokens)
            results.append(response)

            if response.ok:
                n_ok += 1
            else:
                n_fail += 1

            if show_progress and (i + 1) % 10 == 0:
                remaining = self._rate_limiter.calls_remaining
                print(f"  [teacher] {i + 1}/{len(prompts)} done "
                      f"(ok={n_ok}, fail={n_fail}, remaining={remaining})")

        if show_progress:
            print(f"  [teacher] Batch complete: {n_ok}/{len(prompts)} ok, {n_fail} failed")

        return results

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "calls_remaining": self._rate_limiter.calls_remaining,
            "model": self.config.model,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Preset generators for each training phase
# ═══════════════════════════════════════════════════════════════════════════

# ── Phase 3a: SFT Data Generation ──

SYSTEM_MULTI_SESSION = """You generate realistic multi-session conversation pairs for training an AI with persistent memory.

Requirements:
- Generate TWO sessions between a user and an assistant
- Session 1: user shares facts, preferences, or context naturally
- Session 2: user asks something that REQUIRES remembering session 1
- The assistant in session 2 should demonstrate recall of session 1 facts
- Make conversations natural, not contrived
- Vary topics: personal preferences, project details, health info, travel plans, work context
- Format as JSON with keys: session_1 (list of turns), session_2 (list of turns), required_memory (what must be remembered)"""

SYSTEM_FACT_REVISION = """You generate fact revision dialogues for training an AI that updates its beliefs when corrected.

Requirements:
- Generate a conversation where:
  1. User states a fact
  2. Assistant acknowledges it
  3. User CORRECTS the fact (it was wrong or changed)
  4. Assistant updates its understanding
  5. User queries the fact — assistant gives the CORRECTED version
- The correction should be natural (not "you're wrong" — more like "actually, it changed" or "I misspoke")
- Vary domains: people, companies, science, geography, current events, project details
- Format as JSON with keys: turns (list of {role, content}), original_fact, corrected_fact, query, expected_answer"""

SYSTEM_DPO_COGNITIVE = """You generate preference pairs for training an AI with persistent memory.

Given a scenario, generate:
- chosen: response that uses retrieved memories/beliefs (demonstrates recall)
- rejected: plausible response that ignores available memories (generic/asks again)

Both responses should be fluent and helpful-sounding. The ONLY difference is memory usage.
Format as JSON with keys: prompt, chosen, rejected, memory_type (one of: recall, revision, goal_tracking)"""

SYSTEM_PROGRESSIVE_TASK = """You generate multi-session progressive task conversations for training an AI with goal tracking.

Requirements:
- Generate 3-5 sessions of an ongoing project/task
- Each session builds on previous progress
- The assistant should track what's done, what's next, blockers
- Tasks: code refactoring, research projects, trip planning, learning a skill, debugging
- Format as JSON with keys: sessions (list of {session_id, turns}), goal, progress_checkpoints"""


def generate_multi_session(teacher: TeacherLLM, count: int = 500) -> list[str]:
    """Generate multi-session conversation pairs for SFT Phase 3a."""
    prompts = [
        f"Generate a multi-session conversation pair #{i+1}. "
        f"Topic hint: {topic}"
        for i, topic in enumerate(_cycle_topics(count))
    ]
    responses = teacher.generate_batch(SYSTEM_MULTI_SESSION, prompts, max_tokens=4096)
    return [r.text for r in responses if r.ok and r.text.strip()]


def generate_fact_revision(teacher: TeacherLLM, count: int = 500) -> list[str]:
    """Generate fact revision dialogues for SFT Phase 3a."""
    prompts = [
        f"Generate a fact revision dialogue #{i+1}. "
        f"Domain hint: {domain}"
        for i, domain in enumerate(_cycle_domains(count))
    ]
    responses = teacher.generate_batch(SYSTEM_FACT_REVISION, prompts, max_tokens=2048)
    return [r.text for r in responses if r.ok and r.text.strip()]


def generate_dpo_pairs(teacher: TeacherLLM, count: int = 500) -> list[str]:
    """Generate cognitive DPO preference pairs for Phase 3b."""
    memory_types = ["recall", "revision", "goal_tracking"]
    prompts = [
        f"Generate a preference pair #{i+1}. "
        f"Memory type: {memory_types[i % 3]}. "
        f"Topic hint: {topic}"
        for i, topic in enumerate(_cycle_topics(count))
    ]
    responses = teacher.generate_batch(SYSTEM_DPO_COGNITIVE, prompts, max_tokens=2048)
    return [r.text for r in responses if r.ok and r.text.strip()]


def generate_progressive_tasks(teacher: TeacherLLM, count: int = 200) -> list[str]:
    """Generate multi-session progressive task conversations for SFT Phase 3a."""
    prompts = [
        f"Generate a progressive task conversation #{i+1}. "
        f"Task type: {task}"
        for i, task in enumerate(_cycle_tasks(count))
    ]
    responses = teacher.generate_batch(SYSTEM_PROGRESSIVE_TASK, prompts, max_tokens=6000)
    return [r.text for r in responses if r.ok and r.text.strip()]


# ── Online Evaluation ──

SYSTEM_EVAL_BELIEF = """You are evaluating whether an AI model correctly used its persistent memory/beliefs.

Given the model's input and output, score 1-5:
1 = completely ignored relevant memory
2 = partially used memory but with errors
3 = used memory but could be better
4 = good memory usage
5 = excellent — recalled, updated, or tracked goals perfectly

Respond with JSON: {"score": N, "reason": "brief explanation", "suggestion": "what to improve"}"""


def evaluate_belief_usage(
    teacher: TeacherLLM,
    model_input: str,
    model_output: str,
    available_beliefs: str = "",
) -> dict | None:
    """Use teacher to evaluate model's belief usage quality."""
    prompt = (
        f"Model input:\n{model_input}\n\n"
        f"Model output:\n{model_output}\n\n"
        f"Available beliefs in state:\n{available_beliefs or 'unknown'}\n\n"
        f"Score the model's memory usage."
    )
    response = teacher.generate(
        SYSTEM_EVAL_BELIEF,
        [{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,  # deterministic for evaluation
    )
    if not response.ok:
        return None
    try:
        import json
        return json.loads(response.text)
    except (json.JSONDecodeError, ValueError):
        return None


# ── Helpers ──

_TOPICS = [
    "cooking preferences", "work project", "health condition", "travel plans",
    "programming language choice", "book recommendations", "pet care",
    "home renovation", "fitness routine", "financial planning",
    "child's school schedule", "car maintenance", "garden planning",
    "recipe modifications", "team management", "research paper",
    "conference presentation", "job interview prep", "moving to new city",
    "learning a musical instrument", "dietary restrictions", "timezone",
]

_DOMAINS = [
    "company leadership", "scientific discovery", "geography", "sports records",
    "technology specs", "medical guidelines", "legal regulations",
    "historical dates", "product pricing", "team membership",
    "project deadlines", "software versions", "restaurant hours",
    "event schedules", "contact information", "policy changes",
]

_TASKS = [
    "refactoring a codebase", "planning a trip", "learning Rust",
    "debugging a production issue", "writing a research paper",
    "preparing a presentation", "migrating a database",
    "building a side project", "studying for an exam",
    "organizing a team offsite", "redesigning an API",
]


def _cycle_topics(n: int):
    for i in range(n):
        yield _TOPICS[i % len(_TOPICS)]

def _cycle_domains(n: int):
    for i in range(n):
        yield _DOMAINS[i % len(_DOMAINS)]

def _cycle_tasks(n: int):
    for i in range(n):
        yield _TASKS[i % len(_TASKS)]
