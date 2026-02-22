# Purpose: OpenAI-compatible HTTP adapter. Listens on TCP/IP and/or a UNIX
#          domain socket, accepts chat completion requests, injects them as
#          events into the orchestrator's event queue, and returns the agent
#          response in OpenAI chat.completion format.
# Relationships: Calls core/event_queue.push_event to enqueue work.
#               Receives results via the _pending dict (asyncio.Future keyed
#               by event_id) that main.py resolves after invoke() completes.
#               Started as an asyncio task by main.py.
#
# Wire protocol: OpenAI Chat Completions v1 (non-streaming).
#   POST /v1/chat/completions  — accept a chat turn, return agent response
#   GET  /v1/models            — list available agent:project combinations
#   GET  /health               — liveness check
#
# Model name convention: "{agent_type}:{project_id}"
#   e.g. "task_agent:default", "dev_agent:orchestrator"
#
# Conversation context: only the last user-role message in the request's
# messages array is used as the task_description. The orchestrator's own
# context management (brief/summary/recent history per project) provides
# continuity. Client-side conversation history is intentionally ignored —
# keeping the invocation interface unchanged and avoiding duplicate context.
#
# Authentication: when config["network"]["require_auth"] is true, every
# request must carry "Authorization: Bearer <api_key>". Requests without
# or with a wrong key receive 401.
#
# [SAFETY-CRITICAL] High-risk tool calls from HTTP-sourced events auto-deny
# because sys.stdin.isatty() is False in a server context. The existing
# approval_gate already handles this. Do not weaken that check here.

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from aiohttp import web

from core.event_queue import push_event

logger = logging.getLogger("network")

# Separate logger for HTTP access lines so they can be filtered independently.
_access_log = logging.getLogger("network.access")


@web.middleware
async def _request_log_middleware(request: web.Request, handler):
    """
    Log every inbound request before routing so unmatched paths are visible.

    This fires before aiohttp route matching, so a 404 from an unknown path
    still produces a log line showing exactly what the client sent.
    """
    logger.debug(
        "→ %s %s  [%s] headers=%s",
        request.method,
        request.path_qs,
        request.remote,
        dict(request.headers),
    )
    response = await handler(request)
    logger.debug(
        "← %s %s  status=%d",
        request.method,
        request.path,
        response.status,
    )
    return response


async def _handle_404(request: web.Request) -> web.Response:
    """
    Catch-all handler for unregistered routes.

    Logs the full request details at WARNING level so the exact path the
    client sent is visible without needing DEBUG logging enabled.
    """
    logger.warning(
        "404 — no route for %s %s  remote=%s  headers=%s",
        request.method,
        request.path_qs,
        request.remote,
        dict(request.headers),
    )
    return web.Response(
        status=404,
        content_type="application/json",
        text=json.dumps({
            "error": {
                "message": (
                    f"No route matched {request.method} {request.path!r}. "
                    "Available routes: GET /health, GET /v1/models, "
                    "POST /v1/chat/completions"
                ),
                "type": "not_found",
            }
        }),
    )


class NetworkAdapter:
    """
    aiohttp-based HTTP server exposing an OpenAI-compatible API.

    pending: shared dict {event_id: asyncio.Future} maintained by main.py.
             The event loop sets future.set_result(invoke_result) after invoke()
             completes; handle_chat_completions awaits the future to get the result.

    available_models: list of {agent_type, project_id} dicts describing which
                      model strings to advertise on GET /v1/models. Built by
                      main.py from config at startup.
    """

    def __init__(
        self,
        config: dict,
        db_path: str,
        pending: dict[str, asyncio.Future],
    ) -> None:
        self._cfg = config["network"]
        self._db_path = db_path
        self._pending = pending
        self._require_auth: bool = self._cfg.get("require_auth", False)
        self._api_key: str = self._cfg.get("api_key", "")
        self._timeout_s: int = int(self._cfg.get("request_timeout_s", 300))

        # The name shown to clients in GET /v1/models and echoed back in responses.
        # Clients must send this exact string in the model field of their requests
        # (the OpenAI protocol requires a model field) but the value is ignored
        # for routing — all HTTP requests go to the default agent and project.
        self._model_name: str = self._cfg.get("model_name", "orchestrator")

        # Routing defaults: all HTTP requests are dispatched to this agent type
        # and project. The orchestrator decides which LLM to use; clients cannot
        # influence that choice through the wire protocol.
        self._default_agent_type: str = self._cfg.get("default_agent_type", "task_agent")
        self._default_project_id: str = (
            config.get("projects", {}).get("default_project_id", "default")
        )

        self._app = web.Application(middlewares=[_request_log_middleware])
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/v1/models", self._handle_models)
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)

        # Catch-all: any path/method not matched above goes here.
        # Must be added last so explicit routes take priority.
        self._app.router.add_route("*", "/{path_info:.*}", _handle_404)

        self._runner: web.AppRunner | None = None
        self._tcp_site: web.TCPSite | None = None
        self._unix_site: web.UnixSite | None = None

    async def start(self) -> None:
        """Start TCP and/or UNIX socket listeners."""
        # Route aiohttp's built-in access log through our logger hierarchy so
        # it respects the configured log level and format. Each request produces
        # one line: method, path, status, response size, and latency.
        self._runner = web.AppRunner(self._app, access_log=_access_log)
        await self._runner.setup()

        tcp_cfg = self._cfg.get("tcp", {})
        host = tcp_cfg.get("host", "127.0.0.1")
        port = int(tcp_cfg.get("port", 8080))
        self._tcp_site = web.TCPSite(self._runner, host, port)
        await self._tcp_site.start()
        logger.info("Network adapter listening on TCP %s:%d", host, port)

        unix_path = self._cfg.get("unix_socket")
        if unix_path:
            # Remove a stale socket file from a previous run so the bind succeeds.
            if os.path.exists(unix_path):
                os.unlink(unix_path)
            self._unix_site = web.UnixSite(self._runner, unix_path)
            await self._unix_site.start()
            logger.info("Network adapter listening on UNIX socket %s", unix_path)

    async def stop(self) -> None:
        """Shut down all listeners and clean up the socket file."""
        if self._runner:
            await self._runner.cleanup()
        unix_path = self._cfg.get("unix_socket")
        if unix_path and os.path.exists(unix_path):
            os.unlink(unix_path)

    # -------------------------------------------------------------------------
    # Request handlers
    # -------------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "ok"}),
        )

    async def _handle_models(self, request: web.Request) -> web.Response:
        if not self._check_auth(request):
            return self._unauthorized()

        # Expose a single model entry. Clients pick this name; the orchestrator
        # decides which underlying LLM to use — that detail is not exposed here.
        models = [
            {
                "id": self._model_name,
                "object": "model",
                "created": 0,
                "owned_by": "orchestrator",
            }
        ]
        return web.Response(
            content_type="application/json",
            text=json.dumps({"object": "list", "data": models}),
        )

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        if not self._check_auth(request):
            return self._unauthorized()

        try:
            body = await request.json()
        except Exception as exc:
            raw = await request.text()
            logger.warning("chat/completions — invalid JSON body: %s | raw=%r", exc, raw[:500])
            return web.Response(status=400, text="Invalid JSON body")

        # The model field is required by the OpenAI protocol but we ignore its
        # value entirely. Routing and LLM selection are the orchestrator's job.
        stream: bool = bool(body.get("stream", False))
        logger.debug(
            "chat/completions — model=%r (ignored) messages_count=%d stream=%s",
            body.get("model"),
            len(body.get("messages", [])),
            stream,
        )

        agent_type = self._default_agent_type
        project_id = self._default_project_id

        # Extract only the last user message as the task description.
        # The orchestrator manages conversation continuity via its own context tier;
        # the client's full history is not injected into the prompt.
        messages: list[dict] = body.get("messages", [])
        task_description = _last_user_message(messages)
        if not task_description:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({
                    "error": {
                        "message": "No user message found in messages array.",
                        "type": "invalid_request_error",
                    }
                }),
            )

        # Push the message as a 'http' source event and register a Future so
        # we can await the result without polling the DB.
        event_id = push_event(
            db_path=self._db_path,
            source="http",
            type="message",
            payload={
                "message": task_description,
                "agent_type": agent_type,
                "project_id": project_id,
                # Unique per-request timestamp prevents deduplication collisions
                # when the same message is sent twice in the same hour.
                "_ts": time.time_ns() // 1_000_000,
            },
            project_id=project_id,
            priority=3,  # HTTP requests are interactive — higher priority than timer ticks.
        )
        logger.info(
            "chat/completions — event queued event_id=%s agent=%s project=%s msg=%r",
            event_id, agent_type, project_id, task_description[:120],
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[event_id] = future

        try:
            result = await asyncio.wait_for(future, timeout=self._timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(event_id, None)
            return web.Response(
                status=504,
                content_type="application/json",
                text=json.dumps({
                    "error": {
                        "message": (
                            f"Agent did not respond within {self._timeout_s}s. "
                            "The event remains in the queue and will still be processed."
                        ),
                        "type": "timeout",
                    }
                }),
            )

        response_text: str = _build_response_text(result)
        invocation_id: str = result.get("invocation_id", f"inv_{uuid.uuid4().hex[:16]}")
        finish_reason = "stop" if result.get("status") == "done" else "length"
        completion_id = f"chatcmpl-{invocation_id}"
        created = int(time.time())

        logger.info(
            "chat/completions — responding event_id=%s invocation=%s status=%s stream=%s",
            event_id, invocation_id, result.get("status"), stream,
        )

        if stream:
            return await self._stream_response(
                request, completion_id, created, response_text, finish_reason
            )

        openai_response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": finish_reason,
                }
            ],
            # Token counts are not tracked by the orchestrator. Report zeros.
            # OpenAI clients treat usage as informational and handle zeros gracefully.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        return web.Response(
            content_type="application/json",
            text=json.dumps(openai_response),
        )

    async def _stream_response(
        self,
        request: web.Request,
        completion_id: str,
        created: int,
        response_text: str,
        finish_reason: str,
    ) -> web.StreamResponse:
        """
        Return the completed response as a single-chunk SSE stream.

        Real token-by-token streaming would require litellm.acompletion(stream=True)
        piped through here, which is a future enhancement. For now we fake it:
        wait for the full response, emit one content chunk, then finish.

        This satisfies clients that send stream:true (e.g. Open WebUI) without
        requiring us to restructure the agent invocation pipeline.
        """
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        # Chunk 1: role announcement (matches OpenAI's first chunk format).
        await resp.write(_sse(completion_id, created, self._model_name, {
            "role": "assistant", "content": "",
        }, finish_reason=None))

        # Chunk 2: the full response content as a single delta.
        await resp.write(_sse(completion_id, created, self._model_name, {
            "content": response_text,
        }, finish_reason=None))

        # Chunk 3: empty delta with finish_reason signals end of stream.
        await resp.write(_sse(completion_id, created, self._model_name, {
        }, finish_reason=finish_reason))

        await resp.write(b"data: [DONE]\n\n")
        return resp

    # -------------------------------------------------------------------------
    # Auth helpers
    # -------------------------------------------------------------------------

    def _check_auth(self, request: web.Request) -> bool:
        if not self._require_auth:
            return True
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        return auth_header[len("Bearer "):] == self._api_key

    def _unauthorized(self) -> web.Response:
        return web.Response(
            status=401,
            content_type="application/json",
            text=json.dumps({
                "error": {
                    "message": "Invalid or missing API key.",
                    "type": "authentication_error",
                }
            }),
            headers={"WWW-Authenticate": 'Bearer realm="orchestrator"'},
        )


# -------------------------------------------------------------------------
# Pure helpers (no I/O)
# -------------------------------------------------------------------------


def _sse(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: str | None,
) -> bytes:
    """Encode one SSE data line in OpenAI chat.completion.chunk format."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _build_response_text(result: dict) -> str:
    """
    Build the text shown to the user from an invoke() result.

    When the agent used tools, the intermediate reasoning steps are prepended
    so the user can see what the agent was thinking while it worked, not just
    the bare final answer. Format:

        <reasoning prose>
        [Tool: <name> | Parameters: <json>]

        <reasoning prose>
        [Tool: <name> | Parameters: <json>]

        <final response>
    """
    steps: list[dict] = result.get("steps", [])
    final: str = result.get("response", "")

    if not steps:
        return final

    parts: list[str] = []
    for step in steps:
        reasoning = step.get("reasoning", "").strip()
        tool = step.get("tool", "")
        params = step.get("parameters", {})
        params_str = json.dumps(params) if params else "{}"
        if reasoning:
            parts.append(reasoning)
        parts.append(f"[Tool: {tool} | Parameters: {params_str}]")

    if final:
        parts.append(final)

    return "\n\n".join(parts)


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last message with role=='user', or '' if none."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


