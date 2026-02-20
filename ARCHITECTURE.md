## ARCHITECTURE.md — Living Document

This file lives at the project root and must be kept current. Every Claude Code
session and every dev_agent change should update it if the change affects
architecture. It contains:

- What each module does and why it exists
- Key design decisions and the reasoning behind them (see Decision Log below)
- Current known limitations and planned improvements
- The current phase of the roadmap and what has been completed
- A list of all files carrying `[SAFETY-CRITICAL]` annotations and what they protect

This is the first file any agent reads before modifying the codebase.

---

## Current Status

**Phase 2 — COMPLETE**

Three-tier context management is live. Every event is now assigned to a project
(defaulting to the `"default"` project from `config.yaml`). After each invocation
`invoke()` appends the user task and agent response to `history`, then triggers
`maybe_summarise()` if the buffer is over budget. The next invocation receives
`[BRIEF] / [SUMMARY] / [RECENT HISTORY]` injected as a user message before the
task description. RAG / vector retrieval is deferred to a later phase.
All 47 unit and integration tests pass.

---

## Module Map

### `main.py`
Phase 1 event loop entry point. Loads `config.yaml`, initialises the DB,
registers all Phase 1 tools, starts the timer adapter, resets any stale events
from a previous crash, and loops: dequeue → route → invoke → complete/fail.
SIGINT and SIGTERM trigger a graceful shutdown after the current event finishes.

### `config.yaml`
System configuration: model routing, allowed filesystem paths, tool shell
allowlist, per-agent tool permissions, approval mode, context token budgets.

### `core/context_manager.py`
Three-tier project memory. Provides:
- `ensure_project()` — idempotent project row creation (called before every invoke).
- `assemble_context()` — reads brief, summary, and recent history from the DB and
  formats them into a context block injected before the task description.
- `append_to_history()` — writes one history row (called twice per invocation).
- `maybe_summarise()` — compresses old history into `projects.summary` via a model
  call when the accumulated token count exceeds the configured budget.

### `core/agent_invoker.py`
The execution engine. Given a task description, it:
1. Builds the system prompt via `agents/base_prompt.py`
2. If `project_id + db_path + config` are all set, injects the assembled context
   (brief / summary / recent history) as a user message before the task
3. Calls the model through LiteLLM
4. Parses `<tool_call>...</tool_call>` blocks from the response
5. For high-risk tools, calls the injected `approval_gate` callable before execution
6. Executes approved tool calls through the registry
7. Feeds results back into the conversation and loops
8. Appends task + response to `history` and triggers summarisation if over budget
9. Logs the complete invocation record (prompt, response, all tool calls) to SQLite

### `core/approval_gate.py` [SAFETY-CRITICAL]
Terminal-based human approval for high-risk tool calls. Writes the approval
request to the DB before prompting (audit trail). Auto-denies when stdin is
not a TTY (safe default for unattended runs). Transport is terminal in Phase 1;
later phases replace the transport without changing the interface.

### `core/event_queue.py`
Persistent SQLite-backed queue. `push_event()` uses `INSERT OR IGNORE` with a
hash-based ID for deduplication. `dequeue_next_event()` uses a two-step
read-then-write so the approval gate has a cancellation window between
status transitions. `reset_stale_events()` recovers from crashes at startup.

### `core/router.py`
Rules-based dispatcher. Given an event, returns agent type, model key, and
task description. Template expansion supports `{payload[key]}` references only
(no eval, no Jinja). Unknown events raise `UnroutableEvent`.

### `agents/base_prompt.py`
Constructs the system prompt: base agent instructions + tool catalogue as JSON.
Tool schemas are injected as JSON (not prose) so smaller local models have
exact parameter names and types to work with.

### `agents/task_agent.py`
Defines `AGENT_TYPE = "task_agent"` and `make_registry()`, which filters the
full tool registry down to the tools the task_agent is permitted to call
(as configured in `config.yaml`).

### `adapters/timer.py`
Background daemon thread that pushes a `tick` event every `interval_seconds`.
Self-healing: push failures are logged to stderr; the thread keeps running.
Each tick includes the current minute in the payload for distinct event IDs.

### `tools/registry.py` [SAFETY-CRITICAL]
Defines the `Tool` dataclass and `ToolRegistry`. Every tool call goes through
`Tool.execute()`, which validates parameters against the JSON Schema before
dispatching to the implementation. This is the sole validation checkpoint.

### `tools/filesystem.py` [SAFETY-CRITICAL]
Implements `read_file`, `write_file`, and `list_directory`. Path allowlist
enforcement via `_is_allowed()` uses `.resolve()` and `.relative_to()` to
defeat directory traversal attacks.

### `tools/shell.py` [SAFETY-CRITICAL]
Implements `shell_run` (allowlisted commands, `shell=False`) and `run_tests`
(fixed pytest command). The allowlist check and `shell=False` are the two
enforcement points for preventing arbitrary command injection.

### `tools/git.py`
Implements `git_status`, `git_commit`, `git_branch`, `git_rollback`. All use
`subprocess.run(..., shell=False)` with fixed argv arrays.

### `tools/tasks.py`
Implements `read_task_list` and `write_task` over the `tasks` DB table.

### `tools/notify.py`
Implements `notify_user`. Phase 1 transport is stderr. Interface is stable.

### `storage/schema.sql`
All table definitions. Safe to re-run (uses `IF NOT EXISTS`). Tables:
`events`, `invocations`, `tool_calls`, `approvals`, `projects`, `history`, `tasks`.

### `storage/db.py`
`init_db()` loads and executes `schema.sql`. `get_conn()` is a context manager
that yields a WAL-mode SQLite connection, commits on clean exit, rolls back on
exception. WAL mode is set here and nowhere else.

### `tests/test_agent_invoker.py`
Unit tests: invocation written to DB, tool call executed and logged, unknown
tool handled gracefully, iteration cap fires cleanly.

### `tests/test_event_queue.py`
Unit tests: push dedup, dequeue ordering, lifecycle transitions,
reset_stale_events recovery.

### `tests/test_router.py`
Unit tests: timer tick routing, user message routing, template expansion,
UnroutableEvent for unmatched events.

### `tests/test_tool_registry.py`
Unit tests: parameter validation, unknown tool error, schema_for_agent format.

### `tests/test_context_manager.py`
Unit tests: ensure_project idempotency, assemble_context empty/brief/summary/history
paths, token-budget truncation, append_to_history role preservation, maybe_summarise
skip-when-within-budget and trigger-with-update paths (LiteLLM mocked).

### `tests/integration/test_end_to_end.py`
Integration test: real tool execution (read_file + notify_user), LiteLLM
mocked, full pipeline verified against a real temp DB.

### `scripts/setup.sh`
Installs Python dependencies and initialises the DB. Run once before `main.py`.

### `scripts/run_tests.sh`
Runs `pytest tests/ -v` from the project root.

---

## Safety-Critical Files

| File | What it protects |
|---|---|
| `tools/registry.py` | Parameter validation before every tool execution |
| `tools/filesystem.py` | Path allowlist enforcement for all file access |
| `core/approval_gate.py` | Human-in-the-loop gate for high-risk tool calls |
| `tools/shell.py` | Shell command allowlist and shell=False enforcement |

Do not modify safety-critical files as part of autonomous improvement tasks.
Any change requires human review.

---

## Known Limitations (Phase 2)

- No RAG / vector retrieval: context is limited to brief + summary + recent buffer.
  Phase 3 will add sqlite-vec or ChromaDB for semantic retrieval.
- Approval gate is terminal-only: unattended runs auto-deny all high-risk calls.
- Single-writer SQLite: fine for current event rates; revisit if multi-host needed.
- LiteLLM Ollama calls require a running local Ollama server.
- Summarisation uses the same model as the main invocation; a cheaper local model
  would be more efficient for this task. Revisit when model routing is expanded.
- No reflection or task backlog automation (Phase 4).

---

## Phased Roadmap

### Phase 0 — Bare Scaffold ✓ COMPLETE

**Done when:** `python main.py` runs, calls Ollama, executes at least one tool
call, and the invocation appears in the DB.

### Phase 1 — Event Queue & Router ✓ COMPLETE

Deliverables: `event_queue.py`, `router.py`, `timer.py` adapter, rewritten
event-loop `main.py`, `approval_gate.py`, full MVP tool set, 31 passing tests.

### Phase 2 — Project Context & Multi-Project Support ✓ COMPLETE

Deliverables: `context_manager.py` (three-tier brief/summary/recent history),
per-project history accumulated across invocations, summarisation trigger,
`config.yaml` default project, 47 passing tests. Vector retrieval deferred.

### Phase 3 — Email & External Event Adapters

Deliverables: `email.py` IMAP IDLE adapter, backpressure, real notification channel.

### Phase 4 — Reflection & Task Backlog

Deliverables: nightly reflection workflow, task backlog, global memory.

### Phase 5 — Dev Agent & Self-Improvement Loop

Deliverables: `dev_agent.py`, branch-per-change workflow, pre-merge test gate.

---

## Decision Log

---

### DECISION — SQLite as sole persistence layer (no separate message broker)
Date: 2026-02-20
Status: active

**Decision:** Use SQLite for the event queue, invocation log, tool call log,
projects, history, and task backlog. No external broker (Redis, RabbitMQ, etc.).

**Reasoning:** The system runs on a single host. SQLite with WAL mode provides
sufficient concurrent read throughput for the expected event rates. A separate
broker would add an operational dependency with no concrete benefit at this
scale.

**Alternatives considered:**
- Redis Streams: operationally heavier, requires a running Redis process, adds
  a new failure mode with no benefit for single-host throughput.
- SQLite + PostgreSQL NOTIFY: NOTIFY is not available in SQLite; the hybrid
  would require two DBs.
- In-memory queue (Python queue.Queue): loses state on crash; makes replay and
  audit impossible.

**Consequences:**
- All persistence is in one file, easy to inspect and back up.
- Concurrent write throughput is limited by SQLite's single-writer model; if
  the system ever moves to multi-host, this decision must be revisited.
- WAL mode is required and must remain enabled (enforced in `db.py`).

---

### DECISION — LiteLLM as the model abstraction layer
Date: 2026-02-20
Status: active

**Decision:** All model calls go through LiteLLM. No direct Ollama or Anthropic
SDK calls anywhere in the codebase.

**Reasoning:** LiteLLM presents a unified `completion()` interface across
Ollama, Anthropic, OpenAI, and other providers. This lets the orchestrator
switch between local (Ollama) and remote (Anthropic) models purely through
config without touching invocation code.

**Alternatives considered:**
- Direct Ollama HTTP calls: ties the invocation code to one provider; switching
  requires code changes.
- Anthropic SDK directly: same problem in the other direction; local models
  become awkward.
- LangChain: significantly heavier dependency with its own abstractions that
  conflict with the orchestrator's explicit tool-call protocol.

**Consequences:**
- Model routing is purely configuration (`config.yaml` models section).
- LiteLLM's provider support determines which models are available; if a model
  is needed that LiteLLM does not support, a thin adapter in `tools/model.py`
  can wrap it.
- LiteLLM version pinning matters; check the changelog on upgrades.

---

### DECISION — Deferred RAG / vector retrieval to Phase 3
Date: 2026-02-20
Status: active

**Decision:** Phase 2 ships three-tier context (brief, summary, recent buffer)
without any vector embedding or semantic retrieval step.

**Reasoning:** The three-tier model delivers meaningful memory without adding
a new dependency (sqlite-vec or ChromaDB). For the expected project sizes in
Phase 2, summary + recent buffer covers all reasonable context needs. Adding
a vector store prematurely would complicate the setup and testing with no
concrete benefit yet.

**Alternatives considered:**
- sqlite-vec: promising but still early-stage; API may change.
- ChromaDB: adds a server process or persistent client; operational overhead.
- No summarisation (raw truncation only): loses older context completely;
  summarisation is cheap enough to be worth having from day one.

**Consequences:**
- Context is limited to what fits in brief + summary + recent buffer.
- If a project's history grows faster than summarisation can compress it,
  oldest events are permanently lost (not retrievable by RAG). This is
  acceptable for Phase 2 scope.
- Phase 3 can add vector retrieval as an additive change without touching
  the existing three-tier logic.

---

### DECISION — Fixed tool registry over dynamic tool loading
Date: 2026-02-20
Status: active

**Decision:** All tools are registered at startup in `main.py` (and later in
the event loop). There is no mechanism for agents to load new tools at runtime,
no plugin directory scanning, no `importlib` dynamic loading.

**Reasoning:** Dynamic tool loading would allow an agent to introduce arbitrary
code execution by convincing the system to load a new tool. The fixed registry
is a hard boundary: the set of things the system can do is known at startup and
cannot be expanded without a human editing the code and restarting.

**Alternatives considered:**
- Plugin directory (drop a .py file, it auto-loads): too easy to exploit; an
  agent that can write files could add a tool.
- Agent-proposed tools added to the registry at runtime: requires trusting
  agent-generated code, which violates the "no eval on agent strings" rule.
- Config-driven tool registration (tools listed in config.yaml, loaded by name):
  still requires the implementation to exist in the codebase, so provides no
  real flexibility gain over the current approach.

**Consequences:**
- Adding a new tool requires a human to write the implementation and register it.
- The dev_agent (Phase 5) can propose new tools but cannot add them itself;
  implementation is always a human step.
- The complete capability surface of the running system is auditable by reading
  the startup code.
