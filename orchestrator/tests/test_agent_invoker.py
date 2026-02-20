# Purpose: Tests for core/agent_invoker.py.
# Covers: invocation written to DB, tool call executed and logged,
#         unknown tool handled gracefully, iteration cap fires cleanly.

from unittest.mock import MagicMock, patch

import pytest

from core.agent_invoker import invoke
from storage.db import get_conn, init_db
from tools.registry import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


@pytest.fixture()
def registry():
    """Minimal registry with a single low-risk echo tool."""
    reg = ToolRegistry()

    def echo_execute(params: dict) -> dict:
        return {"echo": params.get("message", "")}

    reg.register(
        Tool(
            name="echo",
            description="Echo a message back.",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
            risk_level="low",
            allowed_agents=["task_agent"],
            _execute=echo_execute,
        )
    )
    return reg


def _mock_response(content: str) -> MagicMock:
    """Build a minimal litellm-shaped response mock."""
    r = MagicMock()
    r.choices[0].message.content = content
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_invocation_row_written_on_completion(tmp_db, registry):
    """A completed invocation must appear in the DB with status 'done'."""
    with patch(
        "core.agent_invoker.litellm.completion",
        return_value=_mock_response("All done, no tools needed."),
    ):
        result = invoke(
            task_description="Say hello",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"

    with get_conn(tmp_db) as conn:
        rows = conn.execute(
            "SELECT * FROM invocations WHERE id = ?", (result["invocation_id"],)
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["status"] == "done"
    assert rows[0]["agent_type"] == "task_agent"


def test_tool_call_executed_and_logged(tmp_db, registry):
    """Invoker must execute a tool call and record it in tool_calls table."""
    tool_response = _mock_response(
        "I will echo the message.\n"
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "hello"}}\n</tool_call>'
    )
    done_response = _mock_response("Done. The echo returned 'hello'.")

    with patch(
        "core.agent_invoker.litellm.completion",
        side_effect=[tool_response, done_response],
    ):
        result = invoke(
            task_description="Echo hello",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "echo"
    assert result["tool_calls"][0]["result"] == {"echo": "hello"}

    with get_conn(tmp_db) as conn:
        tc_rows = conn.execute(
            "SELECT * FROM tool_calls WHERE invocation_id = ?",
            (result["invocation_id"],),
        ).fetchall()

    assert len(tc_rows) == 1
    assert tc_rows[0]["status"] == "executed"
    assert tc_rows[0]["tool_name"] == "echo"


def test_unknown_tool_returns_error_and_continues(tmp_db, registry):
    """Calling an unregistered tool must return an error dict, not raise."""
    bad_tool_response = _mock_response(
        '<tool_call>\n{"tool": "nonexistent", "parameters": {}}\n</tool_call>'
    )
    done_response = _mock_response("Acknowledged the error. Task complete.")

    with patch(
        "core.agent_invoker.litellm.completion",
        side_effect=[bad_tool_response, done_response],
    ):
        result = invoke(
            task_description="Call a nonexistent tool",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert "error" in result["tool_calls"][0]["result"]


def test_iteration_cap_sets_failed_status(tmp_db, registry):
    """An agent that keeps emitting tool calls must fail after max_iterations."""
    # Every response includes a tool call so the loop never terminates.
    looping_response = _mock_response(
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "loop"}}\n</tool_call>'
    )

    with patch(
        "core.agent_invoker.litellm.completion",
        return_value=looping_response,
    ):
        result = invoke(
            task_description="Loop forever",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
            max_iterations=3,
        )

    assert result["status"] == "failed"
    assert len(result["tool_calls"]) == 3

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM invocations WHERE id = ?", (result["invocation_id"],)
        ).fetchone()
    assert row["status"] == "failed"
