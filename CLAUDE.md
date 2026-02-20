# Agentic Orchestration System — MVP Design & Roadmap

## Purpose

This document defines the architecture and phased build roadmap for a home-hosted agentic AI orchestration system. It is intended to be used directly by Claude Code as a specification for bootstrapping the system — the system that will eventually run agents capable of continuing to build itself.

---

## Guiding Principles

- **Simplicity first.** Every component should be the simplest thing that works. Complexity is added only when a concrete need justifies it.
- **Legibility over cleverness.** Code must be understandable by an LLM agent. Explicit over implicit, verbose over terse. Good naming and small functions are the foundation; *why* comments and constraint annotations are required on top of that.
- **Security by default.** Agents never execute anything directly. All side effects flow through the orchestrator's tool registry.
- **Human in the loop.** Any irreversible or high-risk action requires explicit human approval before execution.
- **Everything is logged.** Every agent invocation, tool call, and event is written to a durable log. Nothing is fire-and-forget.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                        │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────────┐ │
│  │  Event   │   │  Router  │   │   Context Manager   │ │
│  │  Queue   │──▶│ (Rules)  │──▶│  (Project / RAG)    │ │
│  └──────────┘   └──────────┘   └─────────────────────┘ │
│                                         │               │
│                               ┌─────────▼─────────┐    │
│                               │   Agent Invoker   │    │
│                               │ (builds prompt,   │    │
│                               │  calls model)     │    │
│                               └─────────┬─────────┘    │
│                                         │               │
│                               ┌─────────▼─────────┐    │
│                               │   Tool Registry   │    │
│                               │ (all side effects)│    │
│                               └───────────────────┘    │
└─────────────────────────────────────────────────────────┘
         ▲                                   │
         │ Events                            │ Results / Notifications
┌────────┴────────┐                 ┌────────▼────────┐
│ Source Adapters │                 │   Human I/O     │
│ (email, timers, │                 │ (approval gate, │
│  messages, etc) │                 │  notifications) │
└─────────────────┘                 └─────────────────┘
```

---

## Directory Structure

```
orchestrator/
├── main.py                  # Entry point, starts all services
├── config.yaml              # System configuration (models, paths, rules)
├── ARCHITECTURE.md          # Living document: describes every module (kept up to date)
│
├── core/
│   ├── event_queue.py       # Persistent SQLite-backed event queue
│   ├── router.py            # Rules-based event router
│   ├── agent_invoker.py     # Assembles prompt, calls model, parses response
│   ├── context_manager.py   # Project context, summarisation, retrieval
│   └── approval_gate.py     # Human-in-the-loop approval mechanism
│
├── tools/
│   ├── registry.py          # Tool registry: schema + dispatch
│   ├── filesystem.py        # Read/write files within allowed paths
│   ├── shell.py             # Allowlisted shell command execution
│   ├── git.py               # git status, diff, commit, branch, rollback
│   ├── model.py             # Calls Ollama or hosted API via LiteLLM
│   └── notify.py            # Sends notifications to the user
│
├── adapters/
│   ├── timer.py             # Cron-like timer events
│   ├── email.py             # IMAP IDLE adapter (Phase 3)
│   └── messaging.py        # Messaging platform adapter (Phase 4)
│
├── storage/
│   ├── schema.sql           # All table definitions
│   ├── db.py                # SQLite connection pool and helpers
│   └── vector.py            # Embedding storage and retrieval (Phase 2)
│
├── agents/
│   ├── base_prompt.py       # Shared system prompt construction
│   ├── task_agent.py        # General task execution agent
│   └── dev_agent.py         # Self-improvement / development agent (Phase 5)
│
├── tests/
│   ├── test_event_queue.py
│   ├── test_router.py
│   ├── test_tool_registry.py
│   ├── test_agent_invoker.py
│   └── integration/
│       └── test_end_to_end.py
│
└── scripts/
    ├── setup.sh             # Install dependencies, initialise DB
    └── run_tests.sh         # Run full test suite
```

---

## Core Data Model (SQLite)

```sql
-- Events arriving from any source
CREATE TABLE events (
    id          TEXT PRIMARY KEY,  -- deterministic hash of source+content
    source      TEXT NOT NULL,     -- 'timer', 'email', 'user', etc.
    type        TEXT NOT NULL,     -- 'message', 'tick', 'approval_response', etc.
    project_id  TEXT,              -- NULL means unclassified / inbox
    priority    INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    payload     TEXT NOT NULL,     -- JSON
    status      TEXT DEFAULT 'pending', -- pending | processing | done | failed
    created_at  TEXT NOT NULL,
    processed_at TEXT
);

-- Agent invocations: one row per model call
CREATE TABLE invocations (
    id           TEXT PRIMARY KEY,
    event_id     TEXT REFERENCES events(id),
    agent_type   TEXT NOT NULL,
    project_id   TEXT,
    model        TEXT NOT NULL,
    context      TEXT NOT NULL,    -- full assembled prompt (JSON)
    response     TEXT,             -- raw model response
    tool_calls   TEXT,             -- JSON array of tool calls made
    status       TEXT DEFAULT 'running',
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);

-- All tool calls: every side effect the system takes
CREATE TABLE tool_calls (
    id             TEXT PRIMARY KEY,
    invocation_id  TEXT REFERENCES invocations(id),
    tool_name      TEXT NOT NULL,
    parameters     TEXT NOT NULL,  -- JSON
    risk_level     TEXT NOT NULL,  -- low | medium | high
    approval_id    TEXT,           -- NULL if auto-approved
    result         TEXT,           -- JSON
    status         TEXT DEFAULT 'pending', -- pending | approved | denied | executed | failed
    created_at     TEXT NOT NULL,
    executed_at    TEXT
);

-- Human approval requests
CREATE TABLE approvals (
    id           TEXT PRIMARY KEY,
    tool_call_id TEXT REFERENCES tool_calls(id),
    prompt       TEXT NOT NULL,    -- description shown to human
    response     TEXT,             -- 'approved' | 'denied'
    responded_at TEXT,
    created_at   TEXT NOT NULL
);

-- Projects
CREATE TABLE projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    brief       TEXT,              -- stable facts, goals, constraints
    summary     TEXT,              -- medium-term compressed history
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Conversation and action history per project
CREATE TABLE history (
    id          TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(id),
    role        TEXT NOT NULL,     -- 'user' | 'agent' | 'tool'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Task backlog (improvement candidates, user tasks, etc.)
CREATE TABLE tasks (
    id          TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(id),
    title       TEXT NOT NULL,
    description TEXT,
    source      TEXT,              -- 'user' | 'reflection' | 'agent'
    priority    INTEGER DEFAULT 5,
    status      TEXT DEFAULT 'open', -- open | in_progress | done | cancelled
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
```

---

## Tool Registry Design

Every tool is a Python class registered at startup. The orchestrator exposes the schema to agents; agents emit structured tool call requests; the orchestrator validates, authorizes, and executes.

```python
# tools/registry.py (illustrative)

class Tool:
    name: str
    description: str          # shown to agent in prompt
    parameters_schema: dict   # JSON Schema for validation
    risk_level: str           # 'low' | 'medium' | 'high'
    allowed_agents: list[str] # which agent types may call this

    def execute(self, parameters: dict) -> dict:
        raise NotImplementedError
```

**MVP Tool Set:**

| Tool | Risk | Description |
|---|---|---|
| `read_file` | low | Read a file within allowed paths |
| `write_file` | medium | Write a file within allowed paths |
| `list_directory` | low | List directory contents |
| `run_tests` | low | Execute the test suite, return output |
| `git_status` | low | Return current git status and diff |
| `git_commit` | medium | Commit staged changes with a message |
| `git_branch` | medium | Create or switch branches |
| `git_rollback` | high | Revert to a previous commit |
| `shell_run` | high | Run an allowlisted shell command |
| `notify_user` | low | Send a notification to the user |
| `read_task_list` | low | Read the current task backlog |
| `write_task` | low | Add or update a task |

Risk levels map to authorization behaviour: `low` auto-approves, `medium` logs but auto-approves in dev mode, `high` always requires human approval.

---

## Agent Invocation Flow

For every agent call the sequence is:

1. **Context assembly** — context manager pulls project brief, recent history (last N tokens), and (in Phase 2) RAG-retrieved relevant chunks. Global user preferences appended.
2. **Prompt construction** — base system prompt + tool schema + assembled context + task description.
3. **Model call** — via LiteLLM, routing to Ollama or hosted API based on config.
4. **Response parsing** — extract reasoning text and any tool call requests from the response.
5. **Tool execution loop** — for each tool call: validate schema, check agent permissions, check risk level / approval, execute, append result to context, continue loop.
6. **Result logging** — full invocation record written to DB including assembled context, response, and all tool calls.
7. **History update** — agent response and tool results appended to project history.

The agent communicates tool calls in a simple structured format within its response:

```
<tool_call>
{
  "tool": "read_file",
  "parameters": { "path": "core/event_queue.py" }
}
</tool_call>
```

The invoker parses these tags after each model response and runs the execution loop until no more tool calls are emitted or the iteration cap is reached.

---

## Context Management

Each project maintains three tiers:

- **Brief** — stable facts, goals, constraints. Rarely changes. Always included. Target: ~500 tokens.
- **Summary** — compressed medium-term history, updated after every N invocations. Target: ~1000 tokens.
- **Recent buffer** — raw last M invocations (tool calls + results + agent responses). Flushed into summary when it exceeds a token threshold.

At prompt assembly time the context packet is: `[global_prefs] + [brief] + [summary] + [recent_buffer]`.

In Phase 2, a RAG retrieval step runs against the project's vector store and injects the top-K relevant chunks between summary and recent buffer.

---

## Human Approval Gate

The approval gate is the primary safety mechanism. In the MVP it operates via the terminal (stdout prompt + stdin response). Later phases replace the transport but the interface contract stays the same.

```
[APPROVAL REQUIRED]
Agent: dev_agent | Project: orchestrator | Risk: high
Tool: git_rollback
Parameters: { "to_commit": "a3f9c12", "reason": "tests failing after last change" }

Approve? [y/N]:
```

The orchestrator blocks on this prompt. The agent waits. On approval the tool executes and the result is returned to the agent. On denial the agent receives a denial message and may attempt an alternative approach or halt.

---

## Configuration (config.yaml)

```yaml
models:
  default_local: "ollama/llama3.2"
  default_remote: "anthropic/claude-sonnet-4-6"
  routing:
    # use local for low-complexity tasks, remote for high
    local_for: ["summarize", "classify", "reflect"]
    remote_for: ["dev", "research", "plan"]

paths:
  allowed_read:  ["./", "~/projects/"]
  allowed_write: ["./", "~/projects/"]
  db: "./storage/orchestrator.db"
  logs: "./logs/"

tools:
  shell_allowlist:
    - "pytest"
    - "python"
    - "git"
    - "ls"
    - "cat"
    - "grep"

agents:
  dev_agent:
    allowed_tools: ["read_file", "write_file", "git_status", "git_commit",
                    "git_branch", "git_rollback", "run_tests", "notify_user",
                    "read_task_list", "write_task"]
    max_iterations: 10
    approval_required_for: ["high"]

approval:
  mode: "terminal"   # terminal | file | (later: web | notification)

context:
  recent_buffer_tokens: 4000
  summary_tokens: 1000
  brief_tokens: 500
```

---

## Comment Policy

Code comments in this codebase follow a specific philosophy shaped by the fact that LLM agents are primary readers alongside humans.

**What comments are not for**

Comments that restate what code obviously does are noise and will drift. They are prohibited. If a function named `dequeue_next_event` needs a comment explaining that it dequeues the next event, the comment is the problem. Fix the name or the structure instead.

**What comments are required for**

*Why* comments are mandatory wherever a non-obvious decision was made. These capture reasoning that cannot be inferred from the code itself — constraints being respected, alternatives that were tried and rejected, assumptions the implementation depends on, and gotchas that would trap a future reader trying to "improve" the code.

Examples of required why comments:

```python
# We poll at 500ms rather than using NOTIFY because sqlite-vec does not
# support PostgreSQL-style listeners. If the DB is ever replaced, revisit this.

# Do NOT simplify this to a single UPDATE. The two-step read-then-write is
# intentional: it gives the approval gate a window to cancel between status
# transitions. A single atomic update removes that window.

# The event ID is a hash of source + content + timestamp truncated to the hour.
# Truncating to the hour (not the minute) is deliberate — it tolerates clock
# skew between adapters without producing duplicate events.
```

**Agent-oriented annotations**

A third category exists specifically to communicate intent and constraints to future agents. These are not explaining code — they are governance signals.

Use the following conventions consistently:

```python
# [SAFETY-CRITICAL] This module implements the approval gate. Any change here
# requires human review regardless of risk classification. Do not modify as
# part of autonomous improvement tasks.

# [DO NOT SIMPLIFY] This check appears redundant but guards against a subtle
# race condition described in ARCHITECTURE.md § Known Limitations.

# [INVARIANT] approved must be checked before executed. Reversing this order
# breaks the audit log integrity guarantee.
```

Any file containing `[SAFETY-CRITICAL]` in its header is treated as off-limits for autonomous dev_agent modification. The dev agent must flag these files for human attention rather than modifying them directly.

**Security-critical files**

The following files must carry a `[SAFETY-CRITICAL]` header from Phase 0 onward:

- `core/approval_gate.py`
- `tools/registry.py`
- `tools/shell.py`
- `tools/filesystem.py`

---

## Decision Log Convention

`ARCHITECTURE.md` maintains a running decision log in addition to its structural description. Every significant design decision is recorded using this format:

```
### DECISION — <short title>
Date: YYYY-MM-DD
Status: active | superseded | reversed

**Decision:** What was decided.
**Reasoning:** Why this option was chosen over alternatives.
**Alternatives considered:** What else was evaluated and why it was rejected.
**Consequences:** What this decision makes easier, harder, or impossible.
```

Decisions are never deleted — only marked superseded or reversed, with a reference to the decision that replaced them. This gives agents reading the log a complete picture of why the system is shaped the way it is, not just what it currently looks like.

Claude Code should create the first decision log entries as part of Phase 0, recording at minimum: the choice of SQLite over a separate message broker, the choice of LiteLLM as the model abstraction layer, and the choice of a fixed tool registry over dynamic tool loading.

---
