-- Orchestrator SQLite schema.
-- All tables are created with IF NOT EXISTS so this file is safe to re-run.

-- Mental model:
--- projects is the long-term memory, history is the working memory, and everything else is the audit trail.

CREATE TABLE IF NOT EXISTS events (
    id           TEXT PRIMARY KEY,  -- deterministic hash of source+content+hour
    source       TEXT NOT NULL,     -- 'timer', 'email', 'user', etc.
    type         TEXT NOT NULL,     -- 'message', 'tick', 'approval_response', etc.
    project_id   TEXT,              -- NULL means unclassified / inbox
    priority     INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    payload      TEXT NOT NULL,     -- JSON
    status       TEXT DEFAULT 'pending', -- pending | processing | done | failed
    created_at   TEXT NOT NULL,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS invocations (
    id           TEXT PRIMARY KEY,
    event_id     TEXT REFERENCES events(id),  -- NULL in Phase 0 (no event queue yet)
    agent_type   TEXT NOT NULL,
    project_id   TEXT,
    model        TEXT NOT NULL,
    context      TEXT NOT NULL,    -- full assembled prompt as JSON
    response     TEXT,             -- raw final model response
    tool_calls   TEXT,             -- JSON array of all tool calls made
    status       TEXT DEFAULT 'running', -- running | done | failed
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
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

CREATE TABLE IF NOT EXISTS approvals (
    id           TEXT PRIMARY KEY,
    tool_call_id TEXT REFERENCES tool_calls(id),
    prompt       TEXT NOT NULL,    -- description shown to human
    response     TEXT,             -- 'approved' | 'denied'
    responded_at TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    brief       TEXT,              -- stable facts, goals, constraints (~500 tokens)
    summary     TEXT,              -- compressed medium-term history (~1000 tokens)
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS history (
    id          TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(id),
    role        TEXT NOT NULL,     -- 'user' | 'agent' | 'tool'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
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
