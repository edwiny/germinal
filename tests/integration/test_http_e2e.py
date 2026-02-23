# Purpose: End-to-end tests for the HTTP network adapter + event loop pipeline.
# Covers: full request → event queue → agent invocation → response cycle,
#         for both streaming and non-streaming modes, with tool calls, and
#         error cases.
#
# Architecture: each test runs a real asyncio event loop (_event_loop from
# main.py) as a background task alongside a real aiohttp app (NetworkAdapter).
# The aiohttp TestServer picks an ephemeral port so there are no port conflicts.
# LiteLLM calls are mocked via AsyncMock; all other components are real.

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from orchestrator.adapters.network import NetworkAdapter
from orchestrator.main_loop import _event_loop
from orchestrator.storage.db import get_conn, init_db
from orchestrator.tools.notify import make_notify_user_tool
from orchestrator.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm(content: str) -> MagicMock:
    """Build a minimal litellm-shaped response mock."""
    r = MagicMock()
    r.choices[0].message.content = content
    return r


def _parse_sse(raw: str) -> list[dict]:
    """
    Parse an SSE body into a list of JSON objects, skipping the [DONE] sentinel.

    Each 'data: <json>' line becomes one entry in the returned list.
    """
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))
    return chunks


def _test_config(db_path: str) -> dict:
    return {
        "models": {
            "list": [{"name": "test-model", "model": "ollama/test", "api_key_env": ""}],
            "categories": [{"category": "default", "model": "test-model", "used_for": []}],
        },
        "paths": {
            "allowed_read": ["/tmp"],
            "allowed_write": ["/tmp"],
            "db": db_path,
            "logs": "/tmp/",
        },
        "tools": {"shell_allowlist": []},
        "agents": {
            "task_agent": {
                "allowed_tools": ["notify_user"],
                "max_iterations": 5,
                "approval_required_for": ["high"],
            }
        },
        "approval": {"mode": "terminal"},
        "logging": {"level": "WARNING"},
        "context": {
            "recent_buffer_tokens": 4000,
            "summary_tokens": 1000,
            "brief_tokens": 500,
        },
        "projects": {
            "default_project_id": "default",
            "default_project_name": "Default Project",
        },
        "network": {
            "enabled": True,
            "tcp": {"host": "127.0.0.1", "port": 8099},
            "unix_socket": None,
            "request_timeout_s": 10,
            "require_auth": False,
            "api_key": "",
            "model_name": "orchestrator",
            "default_agent_type": "task_agent",
        },
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
async def system(tmp_path):
    """
    Full system: clean SQLite DB, minimal tool registry (notify_user only),
    NetworkAdapter, and _event_loop running as a background asyncio task.

    Yields (TestClient, db_path). The TestServer uses an ephemeral port so
    multiple test runs never conflict.

    Teardown: stops the event loop task and closes the HTTP client.
    """
    db_path = str(tmp_path / "orchestrator.db")
    config = _test_config(db_path)
    init_db(db_path)

    registry = ToolRegistry()
    registry.register(make_notify_user_tool())

    pending_http: dict = {}
    stop_event = asyncio.Event()

    net_adapter = NetworkAdapter(
        config=config,
        db_path=db_path,
        pending=pending_http,
    )

    loop_task = asyncio.create_task(
        _event_loop(
            db_path=db_path,
            config=config,
            full_registry=registry,
            approval_gate=None,
            pending_http=pending_http,
            stop_event=stop_event,
        ),
        name="test-event-loop",
    )

    # TestServer wraps the aiohttp app without binding to a real configured port.
    server = TestServer(net_adapter._app)
    client = TestClient(server)
    await client.start_server()

    yield client, db_path

    await client.close()
    stop_event.set()
    try:
        await asyncio.wait_for(loop_task, timeout=2.0)
    except asyncio.TimeoutError:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Infrastructure endpoints — no LLM involved
# ---------------------------------------------------------------------------


async def test_health_endpoint(system):
    client, _ = system
    resp = await client.get("/health")
    assert resp.status == 200
    assert await resp.json() == {"status": "ok"}


async def test_models_endpoint_returns_single_orchestrator_model(system):
    """GET /v1/models must list exactly one model whose id matches config.model_name."""
    client, _ = system
    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "orchestrator"
    assert data["data"][0]["object"] == "model"


async def test_unknown_path_returns_404_with_error_body(system):
    """Unregistered routes must return 404 with a JSON error body."""
    client, _ = system
    resp = await client.get("/v1/not_a_real_endpoint")
    assert resp.status == 404
    data = await resp.json()
    assert data["error"]["type"] == "not_found"
    assert "/v1/not_a_real_endpoint" in data["error"]["message"]


# ---------------------------------------------------------------------------
# Non-streaming chat completions
# ---------------------------------------------------------------------------


async def test_non_streaming_returns_agent_response(system):
    """Full pipeline: POST → event queue → invoke → JSON response."""
    client, db_path = system

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_llm("The answer is 42.")),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "What is the answer?"}],
                "stream": False,
            },
        )

    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "orchestrator"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "The answer is 42."
    assert data["choices"][0]["finish_reason"] == "stop"

    # Invocation recorded in DB.
    with get_conn(db_path) as conn:
        inv = conn.execute(
            "SELECT status, response FROM invocations ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    assert inv["status"] == "done"
    assert inv["response"] == "The answer is 42."


async def test_model_field_is_ignored_routing_uses_default(system):
    """
    The model field sent by the client is irrelevant.
    The orchestrator routes to the default agent regardless of its value,
    and echoes config.model_name (not the client's string) in the response.
    """
    client, _ = system

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_llm("Still works.")),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-or-anything-really",
                "messages": [{"role": "user", "content": "Ping"}],
                "stream": False,
            },
        )

    assert resp.status == 200
    data = await resp.json()
    assert data["model"] == "orchestrator"
    assert data["choices"][0]["message"]["content"] == "Still works."


async def test_multiple_sequential_requests(system):
    """Each request gets its own independent invocation and response."""
    client, db_path = system

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[_llm("First."), _llm("Second.")]),
    ):
        resp1 = await client.post(
            "/v1/chat/completions",
            json={"model": "orchestrator", "messages": [{"role": "user", "content": "One"}], "stream": False},
        )
        resp2 = await client.post(
            "/v1/chat/completions",
            json={"model": "orchestrator", "messages": [{"role": "user", "content": "Two"}], "stream": False},
        )

    data1 = await resp1.json()
    data2 = await resp2.json()

    assert data1["choices"][0]["message"]["content"] == "First."
    assert data2["choices"][0]["message"]["content"] == "Second."

    # Two separate invocation rows in the DB.
    with get_conn(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM invocations").fetchone()[0]
    assert count == 2


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------


async def test_streaming_returns_sse_content_type(system):
    """stream:true must produce Content-Type: text/event-stream."""
    client, _ = system

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_llm("Streamed.")),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Stream this"}],
                "stream": True,
            },
        )

    assert resp.status == 200
    assert "text/event-stream" in resp.content_type


async def test_streaming_sse_structure(system):
    """
    The SSE stream must contain exactly three data chunks plus [DONE]:
      1. role announcement delta
      2. content delta with the full agent response
      3. empty delta with finish_reason
    """
    client, _ = system

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_llm("Hello from the agent.")),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    raw = await resp.text()
    assert "data: [DONE]" in raw

    chunks = _parse_sse(raw)
    assert len(chunks) == 3

    # Chunk 0: role announcement.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[0]["choices"][0]["delta"].get("content") == ""

    # Chunk 1: full content in one delta.
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hello from the agent."
    assert chunks[1]["choices"][0]["finish_reason"] is None

    # Chunk 2: finish signal.
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[2]["choices"][0]["delta"] == {}

    # All chunks share the same completion id and model name.
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["model"] == "orchestrator" for c in chunks)


# ---------------------------------------------------------------------------
# Tool calls within an invocation
# ---------------------------------------------------------------------------


async def test_agent_tool_call_executed_before_final_response(system, caplog):
    """
    Agent emits a notify_user tool call on the first turn, then produces a
    final response on the second turn. The tool must execute for real and
    the final response must be returned to the HTTP client.
    """
    client, db_path = system

    tool_turn = _llm(
        "I will notify the user now.\n"
        "<tool_call>\n"
        '{"tool": "notify_user", "parameters": {"message": "E2E ping", "level": "info"}}\n'
        "</tool_call>"
    )
    final_turn = _llm("Done. Notification delivered.")

    with caplog.at_level(logging.INFO, logger="notify"), patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(side_effect=[tool_turn, final_turn]),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Notify me"}],
                "stream": False,
            },
        )

    assert resp.status == 200
    data = await resp.json()
    content = data["choices"][0]["message"]["content"]
    # Intermediate reasoning and tool info are prepended before the final response.
    assert "I will notify the user now." in content
    assert "[Tool: notify_user" in content
    assert "Done. Notification delivered." in content

    # notify_user logs at INFO — confirms real execution, not just a recorded call.
    assert "E2E ping" in caplog.text

    # tool_calls row in DB reflects the real execution.
    with get_conn(db_path) as conn:
        tc = conn.execute(
            "SELECT tool_name, status FROM tool_calls ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert tc["tool_name"] == "notify_user"
    assert tc["status"] == "executed"


# ---------------------------------------------------------------------------
# Event queue state
# ---------------------------------------------------------------------------


async def test_http_event_marked_done_in_db_after_response(system):
    """The events row created by the HTTP adapter must reach status='done'."""
    client, db_path = system

    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_llm("OK")),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Check state"}],
                "stream": False,
            },
        )

    assert resp.status == 200

    with get_conn(db_path) as conn:
        event = conn.execute(
            "SELECT status, source FROM events WHERE source = 'http' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert event is not None
    assert event["source"] == "http"
    assert event["status"] == "done"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_missing_user_message_returns_400(system):
    """A messages array with no user-role entry must return 400."""
    client, _ = system

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "orchestrator",
            "messages": [{"role": "system", "content": "You are helpful."}],
            "stream": False,
        },
    )

    assert resp.status == 400
    data = await resp.json()
    assert data["error"]["type"] == "invalid_request_error"


async def test_malformed_json_returns_400(system):
    """A non-JSON body must return 400 without crashing the server."""
    client, _ = system

    resp = await client.post(
        "/v1/chat/completions",
        data=b"this is not json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status == 400


async def test_server_remains_healthy_after_error(system):
    """A bad request must not affect subsequent valid requests."""
    client, _ = system

    # Send a bad request first.
    await client.post(
        "/v1/chat/completions",
        data=b"bad",
        headers={"Content-Type": "application/json"},
    )

    # Server must still respond correctly.
    with patch(
        "orchestrator.core.agent_invoker.litellm.acompletion",
        new=AsyncMock(return_value=_llm("Still alive.")),
    ):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "orchestrator",
                "messages": [{"role": "user", "content": "Are you there?"}],
                "stream": False,
            },
        )

    assert resp.status == 200
    data = await resp.json()
    assert data["choices"][0]["message"]["content"] == "Still alive."
