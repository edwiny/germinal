# Purpose: Tests for core/event_queue.py.
# Covers: push_event dedup, dequeue ordering, lifecycle transitions,
#         reset_stale_events recovery.

import pytest

from core.event_queue import (
    complete_event,
    dequeue_next_event,
    fail_event,
    push_event,
    reset_stale_events,
)
from storage.db import get_conn, init_db


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


def test_push_event_returns_id(tmp_db):
    """push_event must return a string event ID."""
    event_id = push_event(tmp_db, source="test", type="ping", payload={"x": 1})
    assert isinstance(event_id, str)
    assert event_id.startswith("evt_")


def test_push_event_dedup(tmp_db):
    """Pushing the same logical event twice in the same hour inserts only one row."""
    id1 = push_event(tmp_db, source="timer", type="tick", payload={"minute": "00"})
    id2 = push_event(tmp_db, source="timer", type="tick", payload={"minute": "00"})
    assert id1 == id2
    with get_conn(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events WHERE id = ?", (id1,)).fetchone()[0]
    assert count == 1


def test_push_distinct_payloads_create_distinct_events(tmp_db):
    """Different payloads produce distinct IDs and two rows."""
    id1 = push_event(tmp_db, source="timer", type="tick", payload={"minute": "01"})
    id2 = push_event(tmp_db, source="timer", type="tick", payload={"minute": "02"})
    assert id1 != id2
    with get_conn(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 2


def test_dequeue_returns_none_when_empty(tmp_db):
    """dequeue_next_event must return None when there are no pending events."""
    assert dequeue_next_event(tmp_db) is None


def test_dequeue_marks_event_processing(tmp_db):
    """dequeue_next_event must set the DB status to 'processing'."""
    push_event(tmp_db, source="user", type="message", payload={"msg": "hi"})
    event = dequeue_next_event(tmp_db)
    assert event is not None
    # The returned dict is a pre-UPDATE snapshot; check the DB directly.
    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row["status"] == "processing"


def test_dequeue_returns_none_after_dequeue(tmp_db):
    """A dequeued event is no longer visible to a second dequeue call."""
    push_event(tmp_db, source="user", type="message", payload={"msg": "hi"})
    dequeue_next_event(tmp_db)
    assert dequeue_next_event(tmp_db) is None


def test_dequeue_priority_ordering(tmp_db):
    """Lower priority number (higher urgency) is dequeued first."""
    push_event(tmp_db, source="s", type="t", payload={"n": "low"}, priority=8)
    push_event(tmp_db, source="s", type="t", payload={"n": "high"}, priority=2)
    event = dequeue_next_event(tmp_db)
    import json
    payload = json.loads(event["payload"])
    assert payload["n"] == "high"


def test_complete_event(tmp_db):
    """complete_event sets status to 'done' and records processed_at."""
    push_event(tmp_db, source="u", type="m", payload={"v": 1})
    event = dequeue_next_event(tmp_db)
    complete_event(tmp_db, event["id"])
    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status, processed_at FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row["status"] == "done"
    assert row["processed_at"] is not None


def test_fail_event(tmp_db):
    """fail_event sets status to 'failed' and records processed_at."""
    push_event(tmp_db, source="u", type="m", payload={"v": 1})
    event = dequeue_next_event(tmp_db)
    fail_event(tmp_db, event["id"])
    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status, processed_at FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row["status"] == "failed"
    assert row["processed_at"] is not None


def test_reset_stale_events(tmp_db):
    """reset_stale_events returns 'processing' events to 'pending'."""
    push_event(tmp_db, source="u", type="m", payload={"v": 1})
    event = dequeue_next_event(tmp_db)

    # Confirm the event is now 'processing' in the DB (simulates a crash mid-run).
    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row["status"] == "processing"

    recovered = reset_stale_events(tmp_db)
    assert recovered == 1

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row["status"] == "pending"


def test_reset_stale_events_returns_zero_when_nothing_stale(tmp_db):
    """reset_stale_events must return 0 when there are no stale events."""
    assert reset_stale_events(tmp_db) == 0
