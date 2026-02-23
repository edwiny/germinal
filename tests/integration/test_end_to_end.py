# Purpose: End-to-end integration test for Phase 0.
# Verifies that invoker + tool registry + DB form a working pipeline
# without mocking the tool layer (only LiteLLM is mocked).

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.core.agent_invoker import invoke
from orchestrator.storage.db import get_conn, init_db
from orchestrator.tools.filesystem import make_read_file_tool
from orchestrator.tools.notify import make_notify_user_tool
from orchestrator.tools.registry import ToolRegistry


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "e2e.db")
    init_db(db_path)
    return db_path


@pytest.fixture()
def tmp_file(tmp_path):
    """A real file on disk for the read_file tool to read."""
    f = tmp_path / "readme.txt"
    f.write_text("This is the test file content.")
    return str(f), str(tmp_path)


def _mock_response(content: str) -> MagicMock:
    r = MagicMock()
    r.choices[0].message.content = content
    return r


async def test_end_to_end_read_file_and_notify(tmp_db, tmp_file, caplog):
    """
    Full pipeline: agent reads a real file, notifies user, invocation in DB.
    LiteLLM is mocked; all tool execution is real.
    """
    file_path, allowed_dir = tmp_file

    registry = ToolRegistry()
    registry.register(make_read_file_tool([allowed_dir]))
    registry.register(make_notify_user_tool())

    read_response = _mock_response(
        f'Reading the file now.\n'
        f'<tool_call>\n{{"tool": "read_file", "parameters": {{"path": "{file_path}"}}}}\n</tool_call>'
    )
    notify_response = _mock_response(
        '<tool_call>\n'
        '{"tool": "notify_user", "parameters": {"message": "File read successfully.", "level": "info"}}\n'
        '</tool_call>'
    )
    done_response = _mock_response("Task complete.")

    with caplog.at_level(logging.INFO, logger="notify"), patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[read_response, notify_response, done_response]),
    ):
        result = await invoke(
            task_description="Read the readme and notify the user.",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 2

    # File content was actually read.
    read_tc = result["tool_calls"][0]
    assert read_tc["tool"] == "read_file"
    assert read_tc["result"]["content"] == "This is the test file content."

    # Notification was delivered via the logger.
    assert "File read successfully" in caplog.text

    # Invocation row is in the DB with status done.
    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM invocations WHERE id = ?", (result["invocation_id"],)
        ).fetchone()
    assert row["status"] == "done"
