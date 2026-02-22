# Purpose: End-to-end tests for the agent tool-call loop.
# Covers scenarios not addressed by test_end_to_end.py:
#   - Immediate completion (no tool calls at all)
#   - Multi-step chained tool calls with results fed back into context
#   - Pydantic validation error returned to agent as a structured tool result
#   - Unknown tool name handled gracefully
#   - Approval gate: approved path (high-risk tool executes)
#   - Approval gate: denied path (high-risk tool blocked)
#   - DB tool_calls table populated correctly across multiple calls
#
# LiteLLM is always mocked; all tool execution (filesystem, notify) is real.

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from core.agent_invoker import invoke
from storage.db import get_conn, init_db
from tools.filesystem import make_read_file_tool, make_write_file_tool
from tools.notify import make_notify_user_tool
from tools.registry import Tool, ToolRegistry, model_to_json_schema


class _DangerousOpParams(BaseModel):
    """Params model for the test-only high-risk tool used in approval gate tests."""
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(default="", description="Reason for the operation.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "e2e_tool.db")
    init_db(db_path)
    return db_path


@pytest.fixture()
def tmp_dir(tmp_path):
    """Return a temp directory path and pre-create a seed file inside it."""
    seed = tmp_path / "seed.txt"
    seed.write_text("seed content")
    return str(tmp_path)


@pytest.fixture()
def registry(tmp_dir, tmp_db):
    """Registry with the full tool set used across most tests."""
    reg = ToolRegistry()
    reg.register(make_read_file_tool([tmp_dir]))
    reg.register(make_write_file_tool([tmp_dir]))
    reg.register(make_notify_user_tool())
    return reg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resp(content: str) -> MagicMock:
    """Build a mock LiteLLM completion response."""
    r = MagicMock()
    r.choices[0].message.content = content
    return r


def _tool_call_block(tool: str, parameters: dict) -> str:
    """Format a <tool_call> block as the agent would emit it."""
    return f'<tool_call>\n{json.dumps({"tool": tool, "parameters": parameters})}\n</tool_call>'


async def _run(task: str, responses: list, reg: ToolRegistry, db: str, approval_gate=None) -> dict:
    """Thin wrapper around invoke() with LiteLLM mocked."""
    with patch(
        "core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=responses),
    ):
        return await invoke(
            task_description=task,
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=reg,
            db_path=db,
            approval_gate=approval_gate,
        )


def _tool_call_rows(db: str, invocation_id: str) -> list[dict]:
    """Fetch all tool_calls rows for an invocation, ordered by created_at."""
    with get_conn(db) as conn:
        rows = conn.execute(
            """
            SELECT tool_name, status, result, risk_level
            FROM tool_calls
            WHERE invocation_id = ?
            ORDER BY created_at ASC
            """,
            (invocation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Test: immediate completion — no tool calls emitted
# ---------------------------------------------------------------------------

async def test_agent_completes_without_tool_calls(registry, tmp_db):
    """Agent that never emits a tool call reaches done status in one iteration."""
    result = await _run(
        "Summarise the system.",
        [_resp("The system is healthy. No action needed.")],
        registry,
        tmp_db,
    )

    assert result["status"] == "done"
    assert result["tool_calls"] == []
    assert "healthy" in result["response"]


# ---------------------------------------------------------------------------
# Test: single tool call — result appears in return value and DB
# ---------------------------------------------------------------------------

async def test_single_tool_call_result_in_return_value(registry, tmp_db, tmp_dir):
    """Tool result is returned in result['tool_calls'] and in the DB row."""
    import os
    seed_path = os.path.join(tmp_dir, "seed.txt")

    responses = [
        _resp(_tool_call_block("read_file", {"path": seed_path})),
        _resp("Done reading."),
    ]
    result = await _run("Read the seed file.", responses, registry, tmp_db)

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["tool"] == "read_file"
    assert tc["result"]["content"] == "seed content"

    # Verify the DB row.
    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "read_file"
    assert rows[0]["status"] == "executed"
    stored_result = json.loads(rows[0]["result"])
    assert stored_result["content"] == "seed content"


# ---------------------------------------------------------------------------
# Test: multi-step chain — write then read back
# ---------------------------------------------------------------------------

async def test_multi_step_write_then_read(registry, tmp_db, tmp_dir):
    """
    Agent writes a file, then reads it back, then reports done.
    Verifies that tool results are injected into the conversation so the
    second tool call can reference data from the first.
    """
    import os
    output_path = os.path.join(tmp_dir, "output.txt")

    responses = [
        _resp(_tool_call_block("write_file", {"path": output_path, "content": "hello world"})),
        _resp(_tool_call_block("read_file", {"path": output_path})),
        _resp("Written and verified."),
    ]
    result = await _run("Write a file and read it back.", responses, registry, tmp_db)

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 2

    write_tc = result["tool_calls"][0]
    assert write_tc["tool"] == "write_file"
    assert write_tc["result"]["success"] is True

    read_tc = result["tool_calls"][1]
    assert read_tc["tool"] == "read_file"
    assert read_tc["result"]["content"] == "hello world"

    # DB should have two rows in order.
    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert [r["tool_name"] for r in rows] == ["write_file", "read_file"]
    assert all(r["status"] == "executed" for r in rows)


# ---------------------------------------------------------------------------
# Test: Pydantic validation error is returned to agent, not raised
# ---------------------------------------------------------------------------

async def test_validation_error_returned_as_tool_result(registry, tmp_db):
    """
    Agent emits a tool call with missing required params.
    The registry returns {"error": "..."} as the tool result; invoke() feeds
    it back to the agent as a <tool_result> message. The invocation continues.
    """
    responses = [
        # Missing required "path" parameter — Pydantic will reject this.
        _resp(_tool_call_block("read_file", {})),
        _resp("I see the error. Task done."),
    ]
    result = await _run("Read a file (bad params first).", responses, registry, tmp_db)

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert "error" in tc["result"]
    assert "validation" in tc["result"]["error"].lower()

    # The DB row for a validation-failed call is still "executed" (the tool
    # ran through registry.execute() and returned an error dict — it was not
    # an uncaught exception, so the call_status in DB should be "executed").
    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert rows[0]["status"] == "executed"


# ---------------------------------------------------------------------------
# Test: extra property rejected by Pydantic, returned as tool result
# ---------------------------------------------------------------------------

async def test_extra_property_validation_error_returned(registry, tmp_db, tmp_dir):
    """Extra fields forbidden by ConfigDict(extra='forbid') return an error dict."""
    import os
    seed_path = os.path.join(tmp_dir, "seed.txt")

    responses = [
        # "mode" is not a field on ReadFileParams — extra='forbid' rejects it.
        _resp(_tool_call_block("read_file", {"path": seed_path, "mode": "binary"})),
        _resp("Got the error. Done."),
    ]
    result = await _run("Read file with bad extra param.", responses, registry, tmp_db)

    tc = result["tool_calls"][0]
    assert "error" in tc["result"]
    assert "validation" in tc["result"]["error"].lower()


# ---------------------------------------------------------------------------
# Test: unknown tool name is handled gracefully
# ---------------------------------------------------------------------------

async def test_unknown_tool_returns_error_and_invocation_continues(registry, tmp_db, tmp_dir):
    """
    Agent calls a tool that does not exist in the registry. The invoker
    returns {"error": "Unknown tool: ..."} and the loop continues without
    crashing. The invocation finishes done after the agent recovers.
    """
    import os
    seed_path = os.path.join(tmp_dir, "seed.txt")

    responses = [
        _resp(_tool_call_block("does_not_exist", {"x": 1})),
        # Agent sees the error and switches to a valid tool.
        _resp(_tool_call_block("read_file", {"path": seed_path})),
        _resp("Recovered successfully."),
    ]
    result = await _run("Try a bad tool then recover.", responses, registry, tmp_db)

    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 2

    bad_tc = result["tool_calls"][0]
    assert bad_tc["tool"] == "does_not_exist"
    assert "error" in bad_tc["result"]
    assert "Unknown tool" in bad_tc["result"]["error"]

    good_tc = result["tool_calls"][1]
    assert good_tc["tool"] == "read_file"
    assert "content" in good_tc["result"]

    # DB: unknown-tool row has status "failed"; real tool row has "executed".
    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "executed"


# ---------------------------------------------------------------------------
# Test: approval gate — high-risk tool approved
# ---------------------------------------------------------------------------

async def test_high_risk_tool_approved_executes(tmp_db):
    """A high-risk tool is executed when the approval gate returns True."""
    executed_calls = []

    def _high_risk_execute(params: dict) -> dict:
        executed_calls.append(params)
        return {"done": True, "reason": params.get("reason", "")}

    high_risk_tool = Tool(
        name="dangerous_op",
        description="Test-only high-risk operation.",
        parameters_schema=model_to_json_schema(_DangerousOpParams),
        risk_level="high",
        allowed_agents=["task_agent"],
        _execute=_high_risk_execute,
        params_model=_DangerousOpParams,
    )
    reg = ToolRegistry()
    reg.register(high_risk_tool)
    reg.register(make_notify_user_tool())

    def _always_approve(**_kwargs) -> bool:
        return True

    responses = [
        _resp(_tool_call_block("dangerous_op", {"reason": "testing approval"})),
        _resp("High-risk op complete."),
    ]
    result = await _run("Run the dangerous op.", responses, reg, tmp_db, approval_gate=_always_approve)

    assert result["status"] == "done"
    assert len(executed_calls) == 1
    assert executed_calls[0]["reason"] == "testing approval"

    tc = result["tool_calls"][0]
    assert tc["result"]["done"] is True

    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert rows[0]["status"] == "executed"


# ---------------------------------------------------------------------------
# Test: approval gate — high-risk tool denied
# ---------------------------------------------------------------------------

async def test_high_risk_tool_denied_returns_error(tmp_db):
    """When the approval gate denies a high-risk call, the agent gets an error dict."""
    executed_calls = []

    def _high_risk_execute(params: dict) -> dict:
        executed_calls.append(params)
        return {"done": True}

    high_risk_tool = Tool(
        name="dangerous_op",
        description="Test-only high-risk operation.",
        parameters_schema=model_to_json_schema(_DangerousOpParams),
        risk_level="high",
        allowed_agents=["task_agent"],
        _execute=_high_risk_execute,
        params_model=_DangerousOpParams,
    )
    reg = ToolRegistry()
    reg.register(high_risk_tool)

    def _always_deny(**_kwargs) -> bool:
        return False

    responses = [
        _resp(_tool_call_block("dangerous_op", {})),
        _resp("Understood, the op was denied."),
    ]
    result = await _run("Run the dangerous op.", responses, reg, tmp_db, approval_gate=_always_deny)

    # The execute callable must never have been called.
    assert executed_calls == []

    tc = result["tool_calls"][0]
    assert "error" in tc["result"]
    assert "denied" in tc["result"]["error"].lower()

    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert rows[0]["status"] == "denied"


# ---------------------------------------------------------------------------
# Test: multiple tool calls all appear in DB with correct data
# ---------------------------------------------------------------------------

async def test_db_records_all_tool_calls(registry, tmp_db, tmp_dir):
    """Every tool call in a multi-step sequence appears in the tool_calls table."""
    import os
    path_a = os.path.join(tmp_dir, "a.txt")
    path_b = os.path.join(tmp_dir, "b.txt")

    responses = [
        _resp(_tool_call_block("write_file", {"path": path_a, "content": "aaa"})),
        _resp(_tool_call_block("write_file", {"path": path_b, "content": "bbb"})),
        _resp(_tool_call_block("read_file", {"path": path_a})),
        _resp("All done."),
    ]
    result = await _run("Write two files and read one back.", responses, registry, tmp_db)

    assert result["status"] == "done"
    rows = _tool_call_rows(tmp_db, result["invocation_id"])
    assert len(rows) == 3

    names = [r["tool_name"] for r in rows]
    assert names == ["write_file", "write_file", "read_file"]
    assert all(r["status"] == "executed" for r in rows)

    # Risk levels stored correctly.
    assert rows[0]["risk_level"] == "medium"  # write_file
    assert rows[2]["risk_level"] == "low"     # read_file

    # Results are valid JSON stored in the DB.
    for row in rows:
        parsed = json.loads(row["result"])
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Test: notify_user level variants reach the logger at the right level
# ---------------------------------------------------------------------------

async def test_notify_user_levels_routed_to_correct_log_level(registry, tmp_db, caplog):
    """notify_user with level='warning' must appear at WARNING in the log."""
    responses = [
        _resp(_tool_call_block("notify_user", {"message": "something bad", "level": "warning"})),
        _resp("Notified."),
    ]
    with caplog.at_level(logging.WARNING, logger="notify"):
        result = await _run("Notify the user.", responses, registry, tmp_db)

    assert result["status"] == "done"
    assert "something bad" in caplog.text
    # caplog records include the level name.
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("something bad" in r.getMessage() for r in warning_records)


# ---------------------------------------------------------------------------
# Test: iteration cap fires cleanly
# ---------------------------------------------------------------------------

async def test_iteration_cap_sets_failed_status(registry, tmp_db, tmp_dir):
    """If the agent never stops emitting tool calls, the cap fires and status=failed."""
    import os
    seed_path = os.path.join(tmp_dir, "seed.txt")

    # Two tool calls — agent always emits more, capped at max_iterations=2.
    responses = [
        _resp(_tool_call_block("read_file", {"path": seed_path})),
        _resp(_tool_call_block("read_file", {"path": seed_path})),
        # This third response would only be reached if the cap didn't fire.
        _resp("Should never get here."),
    ]

    with patch(
        "core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=responses),
    ):
        result = await invoke(
            task_description="Read forever.",
            agent_type="task_agent",
            model="ollama/llama3.2",
            registry=registry,
            db_path=tmp_db,
            max_iterations=2,
        )

    assert result["status"] == "failed"
    assert len(result["tool_calls"]) == 2
    assert "Iteration cap" in result["response"]

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT status FROM invocations WHERE id = ?",
            (result["invocation_id"],),
        ).fetchone()
    assert row["status"] == "failed"
