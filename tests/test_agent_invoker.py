# Purpose: Tests for core/agent_invoker.py.
# Covers: invocation written to DB, tool call executed and logged,
#         unknown tool handled gracefully, iteration cap fires cleanly,
#         truncated responses detected and surfaced as failures.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.core.agent_invoker import _MAX_CONTINUATIONS, _parse_tool_call, invoke
from orchestrator.storage.db import get_conn, init_db
from orchestrator.tools.registry import Tool, ToolRegistry, model_to_json_schema


class _EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(description="Message to echo.")


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
            parameters_schema=model_to_json_schema(_EchoParams),
            risk_level="low",
            allowed_agents=["task_agent"],
            _execute=echo_execute,
            params_model=_EchoParams,
        )
    )
    return reg


def _mock_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Build a minimal litellm-shaped response mock."""
    r = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].finish_reason = finish_reason
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_invocation_row_written_on_completion(tmp_db, registry):
    """A completed invocation must appear in the DB with status 'done'."""
    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_mock_response("All done, no tools needed.")),
    ):
        result = await invoke(
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


async def test_tool_call_executed_and_logged(tmp_db, registry):
    """Invoker must execute a tool call and record it in tool_calls table."""
    tool_response = _mock_response(
        "I will echo the message.\n"
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "hello"}}\n</tool_call>'
    )
    done_response = _mock_response("Done. The echo returned 'hello'.")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[tool_response, done_response]),
    ):
        result = await invoke(
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

    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "echo"
    assert result["steps"][0]["parameters"] == {"message": "hello"}
    assert "I will echo the message" in result["steps"][0]["reasoning"]

    with get_conn(tmp_db) as conn:
        tc_rows = conn.execute(
            "SELECT * FROM tool_calls WHERE invocation_id = ?",
            (result["invocation_id"],),
        ).fetchall()

    assert len(tc_rows) == 1
    assert tc_rows[0]["status"] == "executed"
    assert tc_rows[0]["tool_name"] == "echo"


async def test_unknown_tool_returns_error_and_continues(tmp_db, registry):
    """Calling an unregistered tool must return an error dict, not raise."""
    bad_tool_response = _mock_response(
        '<tool_call>\n{"tool": "nonexistent", "parameters": {}}\n</tool_call>'
    )
    done_response = _mock_response("Acknowledged the error. Task complete.")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[bad_tool_response, done_response]),
    ):
        result = await invoke(
            task_description="Call a nonexistent tool",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert "error" in result["tool_calls"][0]["result"]


async def test_iteration_cap_sets_failed_status(tmp_db, registry):
    """An agent that keeps emitting tool calls must fail after max_iterations."""
    # Every response includes a tool call so the loop never terminates.
    looping_response = _mock_response(
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "loop"}}\n</tool_call>'
    )

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=looping_response),
    ):
        result = await invoke(
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


# ---------------------------------------------------------------------------
# _parse_tool_call — robustness
# ---------------------------------------------------------------------------


def test_parse_tool_call_valid():
    """A well-formed tool call is parsed correctly."""
    text = '<tool_call>\n{"tool": "git_status", "parameters": {}}\n</tool_call>'
    result = _parse_tool_call(text)
    assert result == {"tool": "git_status", "parameters": {}}


def test_parse_tool_call_trailing_brace_tolerated():
    """An extra closing brace after the JSON object is ignored (not a parse failure)."""
    text = '<tool_call>\n{"tool": "git_status", "parameters": {}}\n}\n</tool_call>'
    result = _parse_tool_call(text)
    assert result is not None
    assert result["tool"] == "git_status"


def test_parse_tool_call_trailing_brace_exact_reproduction():
    """Reproduce the exact malformed output from the bug report."""
    text = '<tool_call>\n{"tool": "git_status", "parameters": {}}}' + "\n</tool_call>"
    result = _parse_tool_call(text)
    assert result is not None
    assert result == {"tool": "git_status", "parameters": {}}


def test_parse_tool_call_nested_params():
    """Tool call with nested parameters is parsed in full."""
    text = (
        '<tool_call>\n'
        '{"tool": "write_file", "parameters": {"path": "f.py", "content": "x=1"}}\n'
        '</tool_call>'
    )
    result = _parse_tool_call(text)
    assert result == {"tool": "write_file", "parameters": {"path": "f.py", "content": "x=1"}}


def test_parse_tool_call_no_tags_returns_none():
    """Text with no tool_call tags returns None."""
    assert _parse_tool_call("Task complete.") is None


def test_parse_tool_call_truncated_json_returns_none():
    """Genuinely broken JSON (not just trailing garbage) returns None."""
    text = '<tool_call>\n{"tool": "write_file", "parameters": {"path":\n</tool_call>'
    assert _parse_tool_call(text) is None


async def test_trailing_brace_tool_call_executes(tmp_db, registry):
    """The invoker must execute a tool call even when the model appends an extra brace."""
    # Reproduce the exact response from the bug report.
    malformed = _mock_response(
        'Let me check the status.\n'
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "hi"}}}\n</tool_call>'
    )
    done = _mock_response("Done.")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[malformed, done]),
    ):
        result = await invoke(
            task_description="Echo hi",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "echo"


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


def test_parse_tool_call_lenient_no_closing_tag():
    """Valid JSON after an unclosed <tool_call> tag is parsed successfully."""
    text = '<tool_call>\n{"tool": "git_diff", "parameters": {}}'
    result = _parse_tool_call(text)
    assert result == {"tool": "git_diff", "parameters": {}}


def test_parse_tool_call_lenient_incomplete_json_returns_none():
    """Incomplete JSON after an unclosed <tool_call> tag returns None."""
    text = '<tool_call>\n{"tool": "write_file", "parameters": {"content": "blah'
    assert _parse_tool_call(text) is None


def test_parse_tool_call_lenient_exact_bug_reproduction():
    """Reproduce the exact log line from the reported bug."""
    text = '<tool_call>\n{"tool": "git_diff", "parameters": {}}\n'
    result = _parse_tool_call(text)
    assert result is not None
    assert result["tool"] == "git_diff"


async def test_finish_reason_length_sets_failed_status(tmp_db, registry):
    """finish_reason='length' must set status=failed, not status=done."""
    truncated = _mock_response(
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "hi"',
        finish_reason="length",
    )

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=truncated),
    ):
        result = await invoke(
            task_description="Echo hi",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "failed"
    assert "truncated" in result["response"].lower()
    # The incomplete tool call must not have been executed.
    assert len(result["tool_calls"]) == 0


async def test_incomplete_json_in_tool_call_sets_failed_status(tmp_db, registry):
    """An unclosed <tool_call> tag with incomplete JSON must set status=failed.

    Distinguishes the failure case (JSON cut off) from the benign case where
    the model omits the closing tag but the JSON is complete and valid.
    """
    # JSON is cut off mid-string — raw_decode cannot recover this.
    truncated = _mock_response(
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "hi"',
        finish_reason="stop",
    )

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=truncated),
    ):
        result = await invoke(
            task_description="Echo hi",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "failed"
    assert len(result["tool_calls"]) == 0


async def test_unclosed_tag_with_valid_json_executes_tool(tmp_db, registry):
    """An unclosed <tool_call> with complete JSON must execute — not fail.

    This is the exact bug: the model emits valid JSON but omits </tool_call>
    on the last line, and the invoker was incorrectly treating it as truncation.
    """
    no_closing_tag = _mock_response(
        'Let me check the diff.\n'
        '<tool_call>\n{"tool": "echo", "parameters": {"message": "hello"}}',
        finish_reason="stop",
    )
    done = _mock_response("Done.")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[no_closing_tag, done]),
    ):
        result = await invoke(
            task_description="Run echo",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "echo"


async def test_truncation_written_to_db(tmp_db, registry):
    """A truncated invocation must be recorded in the DB with status=failed."""
    truncated = _mock_response(
        '<tool_call>\n{"tool": "echo"', finish_reason="length"
    )

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=truncated),
    ):
        result = await invoke(
            task_description="Truncation test",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM invocations WHERE id = ?", (result["invocation_id"],)
        ).fetchone()
    assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# Continuation (multi-chunk response assembly)
# ---------------------------------------------------------------------------


async def test_continuation_assembles_split_tool_call(tmp_db, registry):
    """A tool call split across two chunks must be assembled and executed.

    First response: truncated mid-JSON (finish_reason='length').
    Continuation response: completes the JSON (finish_reason='stop').
    The invoker must concatenate the two chunks, parse the tool call, and
    execute it — producing status=done as if the JSON had arrived whole.
    """
    chunk1 = _mock_response(
        'I will echo the message.\n'
        '<tool_call>\n{"tool": "echo", "parameters": {"message":',
        finish_reason="length",
    )
    chunk2 = _mock_response(' "hello"}}\n</tool_call>', finish_reason="stop")
    done = _mock_response("Done. The echo returned 'hello'.")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[chunk1, chunk2, done]),
    ):
        result = await invoke(
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


async def test_continuation_assembles_split_final_response(tmp_db, registry):
    """A plain text final response split across two chunks must be joined."""
    chunk1 = _mock_response("This is the first part of ", finish_reason="length")
    chunk2 = _mock_response("the final answer.", finish_reason="stop")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[chunk1, chunk2]),
    ):
        result = await invoke(
            task_description="Describe something long",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert result["response"] == "This is the first part of the final answer."


async def test_continuation_cap_exhausted_sets_failed(tmp_db, registry):
    """If every chunk is truncated the invoker must fail after the cap is reached."""
    truncated = _mock_response('<tool_call>\n{"tool": "echo"', finish_reason="length")

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=truncated),
    ) as mock_llm:
        result = await invoke(
            task_description="Truncation cap test",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "failed"
    assert len(result["tool_calls"]) == 0
    # initial attempt + _MAX_CONTINUATIONS continuation attempts
    assert mock_llm.call_count == 1 + _MAX_CONTINUATIONS
