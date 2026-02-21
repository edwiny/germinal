# Purpose: Tests for core/context_manager.py.
# Covers: ensure_project idempotency, assemble_context all tiers,
#         history token budget truncation, append_to_history, maybe_summarise.

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context_manager import (
    append_to_history,
    assemble_context,
    ensure_project,
    maybe_summarise,
)
from storage.db import get_conn, init_db


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


# Small token budget so tests don't need hundreds of messages to exercise
# the truncation / summarisation paths.
_CONFIG = {
    "context": {
        "recent_buffer_tokens": 100,
        "summary_tokens": 1000,
        "brief_tokens": 500,
    }
}


# ---------------------------------------------------------------------------
# ensure_project
# ---------------------------------------------------------------------------


def test_ensure_project_creates_row(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", ("proj-1",)
        ).fetchone()

    assert row is not None
    assert row["name"] == "Project One"


def test_ensure_project_is_idempotent(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    ensure_project("proj-1", "Project One", tmp_db)  # second call must not raise

    with get_conn(tmp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM projects WHERE id = ?", ("proj-1",)
        ).fetchone()[0]

    assert count == 1


# ---------------------------------------------------------------------------
# assemble_context
# ---------------------------------------------------------------------------


def test_assemble_context_returns_empty_for_unknown_project(tmp_db):
    result = assemble_context("nonexistent", tmp_db, _CONFIG)
    assert result == ""


def test_assemble_context_returns_empty_when_all_tiers_empty(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    result = assemble_context("proj-1", tmp_db, _CONFIG)
    assert result == ""


def test_assemble_context_includes_brief_and_summary(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    with get_conn(tmp_db) as conn:
        conn.execute(
            "UPDATE projects SET brief = ?, summary = ? WHERE id = ?",
            ("This is the brief.", "This is the summary.", "proj-1"),
        )

    result = assemble_context("proj-1", tmp_db, _CONFIG)

    assert "[BRIEF]" in result
    assert "This is the brief." in result
    assert "[SUMMARY]" in result
    assert "This is the summary." in result


def test_assemble_context_includes_recent_history(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    with get_conn(tmp_db) as conn:
        conn.execute(
            "UPDATE projects SET brief = ? WHERE id = ?", ("brief text", "proj-1")
        )
    append_to_history("proj-1", "user", "Hello agent", tmp_db)
    append_to_history("proj-1", "agent", "Hello user", tmp_db)

    result = assemble_context("proj-1", tmp_db, _CONFIG)

    assert "[RECENT HISTORY]" in result
    assert "[USER] Hello agent" in result
    assert "[AGENT] Hello user" in result


def test_assemble_context_truncates_oldest_when_budget_exceeded(tmp_db):
    """Only the newest rows should appear when history exceeds the token budget."""
    ensure_project("proj-1", "Project One", tmp_db)
    with get_conn(tmp_db) as conn:
        conn.execute(
            "UPDATE projects SET brief = ? WHERE id = ?", ("brief", "proj-1")
        )

    # Budget is 100 tokens (~400 chars). Write 30 messages so early ones are cut.
    for i in range(30):
        append_to_history("proj-1", "user", f"message number {i}", tmp_db)
        # Tiny sleep to ensure distinct created_at ordering in SQLite.
        time.sleep(0.002)

    result = assemble_context("proj-1", tmp_db, _CONFIG)

    # The oldest message must not appear (budget pushed it out).
    assert "message number 0" not in result
    # The newest message must appear.
    assert "message number 29" in result


# ---------------------------------------------------------------------------
# append_to_history
# ---------------------------------------------------------------------------


def test_append_to_history_writes_row(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    append_to_history("proj-1", "user", "test content", tmp_db)

    with get_conn(tmp_db) as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE project_id = ?", ("proj-1",)
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "test content"


def test_append_to_history_preserves_role(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    append_to_history("proj-1", "agent", "agent reply", tmp_db)

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT role FROM history WHERE project_id = ?", ("proj-1",)
        ).fetchone()

    assert row["role"] == "agent"


# ---------------------------------------------------------------------------
# maybe_summarise
# ---------------------------------------------------------------------------


async def test_maybe_summarise_skips_when_within_budget(tmp_db):
    ensure_project("proj-1", "Project One", tmp_db)
    # "short" is well under 100 tokens so no summarisation should happen.
    append_to_history("proj-1", "user", "short", tmp_db)

    with patch("core.context_manager.litellm.acompletion") as mock_completion:
        await maybe_summarise("proj-1", tmp_db, "ollama/llama3.2", None, _CONFIG)
        mock_completion.assert_not_called()


async def test_maybe_summarise_triggers_updates_summary_and_deletes_old_rows(tmp_db):
    """Over-budget history must be summarised, projects.summary updated, old rows deleted."""
    ensure_project("proj-1", "Project One", tmp_db)

    # Write 20 messages with substantial content — total will exceed 100 tokens.
    for i in range(20):
        append_to_history(
            "proj-1",
            "user",
            f"This is message {i} with extra padding text to ensure we hit the token budget.",
            tmp_db,
        )

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Compressed summary text."

    with patch(
        "core.context_manager.litellm.acompletion",
        new=AsyncMock(return_value=mock_resp),
    ):
        await maybe_summarise("proj-1", tmp_db, "ollama/llama3.2", None, _CONFIG)

    with get_conn(tmp_db) as conn:
        project = conn.execute(
            "SELECT summary FROM projects WHERE id = ?", ("proj-1",)
        ).fetchone()
        remaining_rows = conn.execute(
            "SELECT COUNT(*) FROM history WHERE project_id = ?", ("proj-1",)
        ).fetchone()[0]

    assert project["summary"] == "Compressed summary text."
    # Some rows were deleted; only the recent-buffer portion remains.
    assert remaining_rows < 20
