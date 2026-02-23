# Purpose: Persistent SQLite-backed event queue.
# Relationships: Used by adapters (producers) and main.py (consumer).
#               Reads/writes the `events` table in storage/db.py.

# Events flow through statuses: pending → processing → done | failed.
# The queue is the sole coordination point between adapters (producers)
# and the main event loop (consumer). It survives process restarts; any
# 'processing' events from a crashed run are reset to 'pending' by
# reset_stale_events(), which main.py calls at startup.

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..storage.db import get_conn


class EventEnvelope(BaseModel):
    """
    Validated envelope for an event entering the queue.

    Adapters construct this model before calling push_event so that
    malformed events — wrong types, missing required fields, out-of-range
    priority — are caught at the adapter boundary rather than reaching the
    queue or the router with bad data.
    """
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Event source (e.g. 'timer', 'user', 'email').")
    type: str = Field(description="Event type (e.g. 'tick', 'message').")
    payload: dict[str, Any] = Field(description="JSON-serialisable event payload.")
    project_id: Optional[str] = Field(
        default=None,
        description="Project ID to assign the event to, or None for the inbox.",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority 1 (highest urgency) to 10 (lowest).",
    )


# [INVARIANT] Only one consumer (the main event loop) calls dequeue_next_event().
# This is not enforced by the DB but is a system design constraint.
# SQLite's single-writer guarantee makes the read-then-write safe here,
# but adding a second concurrent consumer would break that assumption.


def push_event(
    db_path: str,
    source: str,
    type: str,
    payload: dict,
    project_id: str | None = None,
    priority: int = 5,
) -> str:
    """
    Insert a new event into the queue. Returns the event id.

    Uses INSERT OR IGNORE: if an identical event (same id) was already
    enqueued this hour, the insert is silently skipped and the existing
    id is returned. This provides natural deduplication for adapters
    that may report the same logical event more than once.
    """
    event_id = _event_id(source, type, payload)
    created_at = _now()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO events
                (id, source, type, project_id, priority, payload, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (event_id, source, type, project_id, priority, json.dumps(payload), created_at),
        )
    return event_id


def dequeue_next_event(db_path: str) -> dict | None:
    """
    Fetch and mark the highest-priority pending event as 'processing'.
    Returns None if the queue is empty.

    Priority is 1 (highest urgency) to 10 (lowest). Events at equal
    priority are ordered by created_at (FIFO).

    # [DO NOT SIMPLIFY] The two-step read-then-write is intentional:
    # it gives the approval gate a window to cancel between status
    # transitions. A single atomic UPDATE removes that window.
    # SQLite's WAL mode and single-writer guarantee make this safe
    # without explicit locking as long as there is one consumer.
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM events
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE events SET status = 'processing' WHERE id = ?",
            (row["id"],),
        )
        return dict(row)


def complete_event(db_path: str, event_id: str) -> None:
    """Mark a processed event as done."""
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE events SET status = 'done', processed_at = ? WHERE id = ?",
            (_now(), event_id),
        )


def fail_event(db_path: str, event_id: str) -> None:
    """Mark a processed event as failed."""
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE events SET status = 'failed', processed_at = ? WHERE id = ?",
            (_now(), event_id),
        )


def reset_stale_events(db_path: str) -> int:
    """
    Reset any 'processing' events back to 'pending'.

    Called at startup to recover from crashes. A process killed mid-run
    may leave events in 'processing' indefinitely; this function re-queues
    them so they are retried.

    Returns the number of events reset.
    """
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE events SET status = 'pending' WHERE status = 'processing'"
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _event_id(source: str, type: str, payload: dict) -> str:
    """
    Deterministic event ID: hash of source + type + payload + hour-truncated timestamp.

    # Truncating to the hour (not the minute) is deliberate — it tolerates
    # clock skew between adapters without producing duplicate events.
    # Two adapters reporting the same logical event within the same hour
    # will generate the same ID and only one will be inserted.
    # For adapters that emit distinct events per minute (like timer ticks),
    # the payload must include a per-tick unique field (e.g. the minute string)
    # so that each tick produces a distinct ID.
    """
    hour_key = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    content = json.dumps(
        {"source": source, "type": type, "payload": payload}, sort_keys=True
    )
    raw = f"{source}:{content}:{hour_key}"
    return "evt_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
