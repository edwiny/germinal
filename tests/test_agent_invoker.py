# Purpose: Tests for core/agent_invoker.py.
# Covers: invocation written to DB, tool call executed and logged,
#         unknown tool handled gracefully, iteration cap fires cleanly,
#         truncated responses detected and surfaced as failures,
#         continuation loop reassembles split responses.
#
# Mocking strategy: tests patch _instructor_client.chat.completions.create
# (the module-level instructor client) to return AgentResponse objects or
# raise IncompleteOutputException without making real LLM calls.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from instructor.core import IncompleteOutputException
from pydantic import BaseModel, ConfigDict, Field

import orchestrator.core.agent_invoker as _invoker_mod
from orchestrator.core.agent_invoker import (
    AgentResponse,
    ToolCallRequest,
    _MAX_CONTINUATIONS,
    invoke,
)
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
            _execute=echo_execute,
            params_model=_EchoParams,
        )
    )
    return reg


def _ok(reasoning: str, tool: str | None = None, parameters: dict | None = None) -> AgentResponse:
    """Build a successful AgentResponse, optionally with a tool call."""
    tc = ToolCallRequest(tool=tool, parameters=parameters or {}) if tool else None
    return AgentResponse(reasoning=reasoning, tool_call=tc)


def _truncated() -> IncompleteOutputException:
    """Build an IncompleteOutputException simulating a token-limit cutoff."""
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = '{"reasoning": "partial'
    return IncompleteOutputException(last_completion=mock_completion)


def _patch_create(*return_values):
    """
    Context manager that replaces _instructor_client.chat.completions.create
    with an AsyncMock returning the given values in sequence.

    Values may be AgentResponse instances (returned normally) or exceptions
    (raised). IncompleteOutputException instances are raised; all others are
    returned.
    """
    async def _side_effect(**kwargs):
        val = next(iter_vals)
        if isinstance(val, BaseException):
            raise val
        return val

    iter_vals = iter(return_values)
    return patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_side_effect,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_invocation_row_written_on_completion(tmp_db, registry):
    """A completed invocation must appear in the DB with status 'done'."""
    with _patch_create(_ok("All done, no tools needed.")):
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
    with _patch_create(
        _ok("I will echo the message.", tool="echo", parameters={"message": "hello"}),
        _ok("Done. The echo returned 'hello'."),
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
    with _patch_create(
        _ok("Calling nonexistent.", tool="nonexistent", parameters={}),
        _ok("Acknowledged the error. Task complete."),
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
    looping = _ok("Looping.", tool="echo", parameters={"message": "loop"})

    # Every call returns the same looping response; we need multiple copies.
    async def _always_loop(**kwargs):
        return looping

    with patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_always_loop,
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
# Truncation detection
# ---------------------------------------------------------------------------


async def test_finish_reason_length_sets_failed_status(tmp_db, registry):
    """IncompleteOutputException on every attempt must set status=failed."""
    # The continuation cap will be exhausted; all attempts truncate.
    exc = _truncated()

    async def _always_truncate(**kwargs):
        raise exc

    with patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_always_truncate,
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
    # No tool call was ever executed because we never got a complete response.
    assert len(result["tool_calls"]) == 0


async def test_truncation_written_to_db(tmp_db, registry):
    """A truncated invocation must be recorded in the DB with status=failed."""
    async def _always_truncate(**kwargs):
        raise _truncated()

    with patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_always_truncate,
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
# Continuation (multi-attempt response assembly)
# ---------------------------------------------------------------------------


async def test_continuation_recovers_tool_call(tmp_db, registry):
    """A tool call that requires continuation must be assembled and executed.

    First attempt: instructor raises IncompleteOutputException (token limit).
    Continuation attempt: returns a valid AgentResponse with a tool call.
    Final iteration: agent reports done.
    """
    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _truncated()
        if call_count == 2:
            return _ok("I will echo.", tool="echo", parameters={"message": "hello"})
        return _ok("Done. Echo returned 'hello'.")

    with patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_side_effect,
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


async def test_continuation_recovers_plain_response(tmp_db, registry):
    """A plain text final response that requires continuation must be returned."""
    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _truncated()
        return _ok("This is the complete final answer.")

    with patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_side_effect,
    ):
        result = await invoke(
            task_description="Describe something long",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
        )

    assert result["status"] == "done"
    assert result["response"] == "This is the complete final answer."


async def test_continuation_cap_exhausted_sets_failed(tmp_db, registry):
    """If every attempt is truncated the invoker must fail after the cap."""
    call_count = 0

    async def _always_truncate(**kwargs):
        nonlocal call_count
        call_count += 1
        raise _truncated()

    with patch.object(
        _invoker_mod._instructor_client.chat.completions,
        "create",
        side_effect=_always_truncate,
    ):
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
    assert call_count == 1 + _MAX_CONTINUATIONS
