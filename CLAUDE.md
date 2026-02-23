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
│   └── <other tools>
│
├── adapters/
│   ├── timer.py             # Cron-like timer events
│   ├── email.py             # IMAP IDLE adapter (Phase 3)
│   └── messaging.py        # Messaging platform adapter (Phase 4)
│
├── storage/
│   ├── schema.sql           # All table definitions
│   └── db.py                # SQLite connection pool and helpers
│
├── agents/
│   ├── base_prompt.py       # Shared system prompt construction
│   ├── task_agent.py        # General task execution agent
│   └── dev_agent.py         # Self-improvement / development agent (Phase 5)
│
├── tests/
│   ├── test_event_queue.py
│   ├── <other unit tests>
│   └── integration/
│       └── <end to end tests>
│
├── cli/
│   └── <cli tools>
│
└── scripts/
    ├── setup.sh             # Install dependencies, initialise DB
    └── run_tests.sh         # Run full test suite
```

---

## Core Data Model (SQLite)

```sql
-- Events arriving from any source
CREATE TABLE events ...
-- Agent invocations: one row per model call
CREATE TABLE invocations ...

-- All tool calls: every side effect the system takes
CREATE TABLE tool_calls ...

-- Human approval requests
CREATE TABLE approvals ...
-- Projects
CREATE TABLE projects ...
-- Conversation and action history per project
CREATE TABLE history ...

```

---

## Tool Registry Design

We're using the Pydantic library to programmatically define the tool parameter schemas, and to serialise/deserialise tool invocations and results.

## Agent Invocation Flow

The prompt loop will has two levels:

Outer Loop: keep prompting until the model's last response did not include a tool call or until a network error or max iterations exceeded.
Note: when a parsing or validation error occurs on the model's response, feed the error back into the next prompt.
Inner Loop: keep prompting until model response is not 'length'.

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

Tunable config is kept in `config.yaml` at the root of the orchestrator.

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
