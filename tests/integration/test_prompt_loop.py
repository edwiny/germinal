# Purpose: End-to-end tests for the agent_invoker prompt loop.
#          The LLM is replaced with a mock so tests run offline and
#          deterministically. Fixtures spin up a real (temporary) SQLite DB
#          so the DB helpers run against actual schema rather than being
#          stubbed out.

import pytest
from unittest.mock import AsyncMock, patch

from instructor.core import IncompleteOutputException

from orchestrator.core.agent_invoker import (
    AgentResponse,
    _MAX_CONTINUATIONS,
    invoke,
)
from orchestrator.storage.db import init_db
from orchestrator.tools.registry import ToolRegistry

# Patch target — module-level instructor client used by _collect_full_response.
# Matches the comment in agent_invoker.py: "Tests replace this by patching
# _instructor_client.chat.completions.create."
_PATCH = "orchestrator.core.agent_invoker._instructor_client.chat.completions.create"


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def registry():
    """Empty tool registry — tests that don't exercise tools use this."""
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Continuation loop
# ---------------------------------------------------------------------------


async def test_continuation_succeeds_within_cap(db_path, registry):
    """
    When the LLM response is truncated fewer times than _MAX_CONTINUATIONS,
    invoke() recovers and returns status='done'.

    max_tokens=50 is passed to document the small budget that would trigger
    real truncation; the mock controls what the LLM "returns" in tests.
    """
    truncations = _MAX_CONTINUATIONS - 1  # stay one short of the cap
    mock_create = AsyncMock(
        side_effect=(
            [IncompleteOutputException()] * truncations
            + [AgentResponse(reasoning="Task complete.", tool_call=None)]
        )
    )

    with patch(_PATCH, mock_create):
        result = await invoke(
            task_description="hello",
            agent_type="task_agent",
            model="test/model",
            registry=registry,
            max_tokens=50,
            db_path=db_path,
        )

    assert result["status"] == "done"
    assert result["response"] == "Task complete."
    assert mock_create.call_count == truncations + 1


async def test_continuation_cap_exhausted_returns_failed(db_path, registry):
    """
    When every attempt is truncated the continuation cap (_MAX_CONTINUATIONS)
    is exhausted and invoke() returns status='failed'.

    The loop runs _MAX_CONTINUATIONS + 1 times: one initial attempt plus one
    per continuation slot before giving up.
    """
    mock_create = AsyncMock(side_effect=IncompleteOutputException())

    with patch(_PATCH, mock_create):
        result = await invoke(
            task_description="hello",
            agent_type="task_agent",
            model="test/model",
            registry=registry,
            max_tokens=50,
            db_path=db_path,
        )

    assert result["status"] == "failed"
    assert mock_create.call_count == _MAX_CONTINUATIONS + 1


async def test_continuation_appends_continue_message(db_path, registry):
    """
    After each truncation the loop must append a [CONTINUE] message so
    the model knows to regenerate its full response from scratch rather
    than picking up mid-JSON (which would produce an invalid schema fragment).
    """
    mock_create = AsyncMock(
        side_effect=[
            IncompleteOutputException(),
            AgentResponse(reasoning="Done.", tool_call=None),
        ]
    )

    with patch(_PATCH, mock_create):
        await invoke(
            task_description="hello",
            agent_type="task_agent",
            model="test/model",
            registry=registry,
            max_tokens=50,
            db_path=db_path,
        )

    assert mock_create.call_count == 2

    # The messages list sent on the second call must include the [CONTINUE] prompt.
    second_call_messages = mock_create.call_args_list[1].kwargs["messages"]
    assert any(
        "[CONTINUE]" in msg.get("content", "")
        for msg in second_call_messages
        if isinstance(msg.get("content"), str)
    )
