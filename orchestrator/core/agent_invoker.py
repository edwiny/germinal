# Purpose: Assembles prompts, calls the language model via instructor + LiteLLM,
#          extracts structured tool calls from the response, executes them through
#          the registry, and writes every step to the database.
# Relationships: Uses storage/db.py, tools/registry.py, agents/base_prompt.py.
#               Called by main.py's event loop; approval_gate.py injected as callable.
#
# invoke() is an async coroutine. It must be awaited. The only blocking calls
# inside it are SQLite helpers (sub-ms, acceptable) and tool.execute() calls
# (also sync — tool authors must not perform long I/O in execute()). The LLM
# call uses instructor.from_litellm(acompletion) which is truly async.
#
# Response format: instructor extracts a structured AgentResponse (Pydantic model)
# from every LLM call. The model includes a reasoning field (free text) and an
# optional tool_call field. When tool_call is None the agent is considered done.
# This replaces the previous XML <tool_call> tag parsing approach.

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import litellm
import instructor
from instructor.core import IncompleteOutputException
from litellm import acompletion
from pydantic import BaseModel, Field

logger = logging.getLogger("agent_invoker")

from ..agents.base_prompt import build_system_prompt
from .context_manager import append_to_history, assemble_context, maybe_summarise
from ..storage.db import get_conn
from ..tools.registry import ToolRegistry

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# [INVARIANT] Iteration cap prevents runaway loops. An agent that needs more
# than DEFAULT_MAX_ITERATIONS for a sensible task is almost certainly stuck.
# Raise only after confirming the model is reasoning correctly on the task.
DEFAULT_MAX_ITERATIONS = 100

# Maximum number of continuation requests to send after a truncated response
# before giving up. When instructor raises IncompleteOutputException (finish_reason
# = 'length'), we cannot meaningfully concatenate partial JSON chunks — the schema
# would be invalid. Instead we ask the model to regenerate from scratch.
_MAX_CONTINUATIONS = 5

# How many times instructor re-prompts the model when the response fails Pydantic
# validation before raising. Each retry feeds the validation error back to the model
# so it can correct its output.
_MAX_VALIDATION_RETRIES = 3

# Module-level instructor client wrapping LiteLLM's async completion function.
# Tests replace this by patching _instructor_client.chat.completions.create.
#
# We use JSON mode rather than the default TOOLS mode. In TOOLS mode instructor
# wraps AgentResponse as an OpenAI tool definition, which conflicts with the
# system prompt that already instructs the model to return a plain JSON object.
# That conflict causes models to correctly identify the outer AgentResponse
# structure but leave the inner ToolCallRequest.parameters dict empty. JSON
# mode keeps the extraction mechanism consistent with the system prompt.
_instructor_client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    tool: str = Field(description="Name of the registered tool to invoke.")
    parameters: dict = Field(
        default_factory=dict,
        description=(
            "Parameters for the tool. You MUST populate this with the actual "
            "key/value pairs required by the tool's schema as listed in AVAILABLE "
            "TOOLS. Do not leave this empty if the tool has required parameters."
        ),
    )


class AgentResponse(BaseModel):
    reasoning: str = Field(
        description="The agent's explanation of what it is doing or has concluded."
    )
    tool_call: Optional[ToolCallRequest] = Field(
        None,
        description=(
            "The tool to invoke next. Set to null when the task is complete "
            "or no tool is needed."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def invoke(
    task_description: str,
    agent_type: str,
    model: str,
    registry: ToolRegistry,
    api_key: str | None = None,
    max_tokens: int | None = None,
    project_id: str | None = None,
    event_id: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    db_path: str | None = None,
    approval_gate: Callable | None = None,
    config: dict | None = None,
) -> dict:
    """
    Run a single agent invocation to completion. Must be awaited.

    Assembles the prompt, drives the tool-call loop, logs everything to the
    DB, and returns a summary dict:
        {
            "invocation_id": str,
            "status": "done" | "failed",
            "response": str,          # final agent reasoning text
            "tool_calls": list[dict], # summary of every tool call made
            "steps": list[dict],      # intermediate reasoning + tool requests
        }

    Each entry in "steps" has:
        {"reasoning": str, "tool": str, "parameters": dict}
    "reasoning" is the agent's prose from a response that contained a tool
    call. Callers can present this to the user so they see what the agent
    was thinking while it worked.
    """
    invocation_id = _new_id("inv")
    started_at = _now()
    tool_calls_log: list[dict] = []

    system_prompt = build_system_prompt(registry.schema_for_agent())
    messages = [
        {
            "role": "system",
            "content": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if project_id and db_path and config:
        ctx = assemble_context(project_id, db_path, config)
        if ctx:
            messages.append({"role": "user", "content": ctx})
    messages.append({"role": "user", "content": task_description})

    _db_insert_invocation(
        invocation_id=invocation_id,
        agent_type=agent_type,
        model=model,
        project_id=project_id,
        context=json.dumps(messages),
        started_at=started_at,
        db_path=db_path,
    )

    final_response = ""
    status = "failed"
    # Each entry: {"reasoning": str, "tool": str, "parameters": dict}.
    # Populated whenever the agent emits a tool call so callers can show
    # the agent's intermediate reasoning to the user rather than just the
    # final response.
    steps: list[dict] = []

    for iteration in range(max_iterations):
        _log_outgoing(messages[-1], iteration)

        logger.info(
            "→ LLM  agent=%s  model=%s  iter=%d/%d  msgs=%d",
            agent_type, model, iteration + 1, max_iterations, len(messages),
        )
        _t0 = time.monotonic()
        try:
            response, assistant_text, _elapsed = await _collect_full_response(
                messages=messages,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                iteration=iteration,
            )
        except IncompleteOutputException:
            _elapsed = time.monotonic() - _t0
            logger.warning(
                "LLM response truncated (continuation cap exhausted) "
                "agent=%s  model=%s  iter=%d  elapsed=%.1fs",
                agent_type, model, iteration + 1, _elapsed,
            )
            final_response = (
                "Response truncated by model token limit "
                "(continuation cap exhausted)."
            )
            status = "failed"
            break
        except Exception as exc:
            _elapsed = time.monotonic() - _t0
            status_code = getattr(exc, "status_code", None)
            logger.error(
                "LLM call failed  agent=%s  model=%s  iter=%d  elapsed=%.1fs"
                "  status_code=%s  error=%s",
                agent_type, model, iteration + 1, _elapsed, status_code, exc,
                exc_info=True,
            )
            final_response = f"LLM call failed (status_code={status_code}): {exc}"
            status = "failed"
            break

        logger.info(
            "← LLM  agent=%s  iter=%d  chars=%d  elapsed=%.1fs",
            agent_type, iteration + 1, len(assistant_text), _elapsed,
        )
        _log_incoming(assistant_text, iteration)

        # Append the structured response as JSON text so the next prompt has
        # a coherent conversation history matching what the model produced.
        messages.append({"role": "assistant", "content": assistant_text})

        if response.tool_call is None:
            # No tool call — the agent has finished its work.
            # The orchestrator has no independent view of whether the work is
            # actually complete; it trusts that tool_call=null means done.
            # TODO: improve task completion evaluation criteria
            final_response = response.reasoning or ""
            status = "done"
            break

        tool_name = response.tool_call.tool
        parameters = response.tool_call.parameters
        reasoning = response.reasoning or ""

        if reasoning:
            logger.info("agent reasoning iter=%d:\n%s", iteration + 1, _truncate_log(reasoning))
        logger.info(
            "tool request iter=%d  tool=%r  params=%s",
            iteration + 1, tool_name, json.dumps(parameters, separators=(",", ":")),
        )
        steps.append({"reasoning": reasoning, "tool": tool_name, "parameters": parameters})

        tc_id = _new_id("tc")
        result = _run_tool(
            tool_call_id=tc_id,
            invocation_id=invocation_id,
            tool_name=tool_name,
            parameters=parameters,
            registry=registry,
            db_path=db_path,
            agent_type=agent_type,
            project_id=project_id,
            approval_gate=approval_gate,
        )
        tool_calls_log.append(
            {"id": tc_id, "tool": tool_name, "parameters": parameters, "result": result}
        )

        # Feed the tool result back into the conversation so the agent can
        # reason about it before deciding what to do next.
        result_text = (
            "<tool_result>\n" + json.dumps(result, indent=2) + "\n</tool_result>"
        )
        messages.append({"role": "user", "content": result_text})

    else:
        # We must log and return cleanly even when the
        # iteration cap fires. Raising here would lose the partial DB record.
        final_response = "Iteration cap reached without task completion."
        status = "failed"

    # Persist the user task and agent response to history so the next
    # invocation can see what happened this time. Summarisation is triggered
    # here rather than in main.py so it is always called, even in tests that
    # invoke() directly.
    if project_id and db_path and config:
        append_to_history(project_id, "user", task_description, db_path)
        append_to_history(project_id, "agent", final_response, db_path)
        await maybe_summarise(project_id, db_path, model, api_key, config)

    _db_finish_invocation(
        invocation_id=invocation_id,
        response=final_response,
        tool_calls=json.dumps(tool_calls_log),
        status=status,
        finished_at=_now(),
        db_path=db_path,
    )

    return {
        "invocation_id": invocation_id,
        "status": status,
        "response": final_response,
        "tool_calls": tool_calls_log,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _collect_full_response(
    messages: list[dict],
    model: str,
    api_key: str | None,
    max_tokens: int | None,
    iteration: int,
) -> tuple[AgentResponse, str, float]:
    """
    Call the LLM via instructor and, if the response is truncated, retry
    with a continuation prompt until the model returns a complete structured
    response or the continuation cap is reached.

    Returns (parsed_response, raw_json_text, total_elapsed_seconds).

    Continuation strategy: when instructor raises IncompleteOutputException
    (finish_reason='length'), the partial JSON cannot be meaningfully appended
    to form a valid schema — the chunk boundary may land anywhere inside the
    JSON object. We therefore ask the model to regenerate from the beginning
    rather than attempting to merge partial chunks. The continuation turns are
    kept in a LOCAL copy of the message list; the caller's list is not touched.

    Validation retries (_MAX_VALIDATION_RETRIES) are handled inside instructor
    itself: on each failed parse it appends the Pydantic error as a user
    message and re-calls the model, up to max_retries times.

    If the continuation cap is exhausted, IncompleteOutputException is re-raised
    so invoke() can mark the invocation as failed.
    """
    local_messages = list(messages)  # snapshot; continuation turns stay local
    total_elapsed = 0.0

    kwargs: dict = {
        "model": model,
        "messages": local_messages,
        "response_model": AgentResponse,
        "max_retries": _MAX_VALIDATION_RETRIES,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(_MAX_CONTINUATIONS + 1):
        kwargs["messages"] = local_messages
        t0 = time.monotonic()
        try:
            response: AgentResponse = await _instructor_client.chat.completions.create(
                **kwargs
            )
            total_elapsed += time.monotonic() - t0
            # Serialize back to JSON for the conversation history. We use
            # model_dump_json() rather than the raw LLM output string because
            # instructor may have corrected the model's output during retries.
            assistant_text = response.model_dump_json()
            return response, assistant_text, total_elapsed

        except IncompleteOutputException:
            total_elapsed += time.monotonic() - t0
            if attempt < _MAX_CONTINUATIONS:
                logger.warning(
                    "iter=%d continuation=%d/%d — response truncated "
                    "(IncompleteOutputException), requesting regeneration",
                    iteration + 1, attempt + 1, _MAX_CONTINUATIONS,
                )
                local_messages.append({
                    "role": "user",
                    "content": (
                        "[CONTINUE] Your previous JSON response was cut off by the "
                        "token limit. Please regenerate your complete response from "
                        "the beginning."
                    ),
                })
            else:
                logger.warning(
                    "iter=%d continuation cap (%d) reached — response still "
                    "truncated, giving up",
                    iteration + 1, _MAX_CONTINUATIONS,
                )
                raise


def _run_tool(
    tool_call_id: str,
    invocation_id: str,
    tool_name: str,
    parameters: dict,
    registry: ToolRegistry,
    db_path: str | None,
    agent_type: str = "unknown",
    project_id: str | None = None,
    approval_gate: Callable | None = None,
) -> dict:
    """Resolve, execute, and log a single tool call."""
    created_at = _now()

    try:
        tool = registry.get(tool_name)
    except KeyError:
        result = {"error": f"Unknown tool: {tool_name!r}"}
        _db_insert_tool_call(
            tool_call_id, invocation_id, tool_name, parameters,
            "unknown", result, "failed", created_at, db_path,
        )
        return result

    # Log before execution so there is always a record, even if execution
    # crashes the process. The result column is updated after execution.
    _db_insert_tool_call(
        tool_call_id, invocation_id, tool_name, parameters,
        tool.risk_level, None, "pending", created_at, db_path,
    )

    # High-risk tools require explicit human approval before execution.
    # If no approval_gate is provided (e.g. in unit tests), high-risk tools
    # proceed unguarded — only the production event loop wires a gate in.
    if tool.risk_level == "high" and approval_gate is not None:
        approved = approval_gate(
            tool_name=tool_name,
            parameters=parameters,
            agent_type=agent_type,
            project_id=project_id,
            tool_call_id=tool_call_id,
        )
        if not approved:
            result = {"error": f"Tool call {tool_name!r} denied by approval gate."}
            _db_update_tool_call(tool_call_id, result, "denied", db_path)
            return result

    try:
        result = tool.execute(parameters)
        _db_update_tool_call(tool_call_id, result, "executed", db_path)
    except Exception as exc:
        result = {"error": str(exc)}
        _db_update_tool_call(tool_call_id, result, "failed", db_path)

    return result


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

_MAX_LOG_CHARS = 4_000


def _truncate_log(text: str) -> str:
    """Trim text to _MAX_LOG_CHARS and append a count of the dropped chars."""
    text = text.strip()
    if len(text) > _MAX_LOG_CHARS:
        dropped = len(text) - _MAX_LOG_CHARS
        return text[:_MAX_LOG_CHARS] + f"\n… [{dropped} chars truncated]"
    return text


def _log_outgoing(message: dict, iteration: int) -> None:
    """Log the newest message being sent to the LLM (→ direction)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    role = message.get("role", "?")
    content = _truncate_log(message.get("content") or "")
    logger.debug("→ LLM  iter=%d  role=%s\n%s", iteration + 1, role, content)


def _log_incoming(text: str, iteration: int) -> None:
    """Log the raw response received from the LLM (← direction)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    content = _truncate_log(text)
    logger.debug("← LLM  iter=%d  role=assistant\n%s", iteration + 1, content)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _db_insert_invocation(
    invocation_id, agent_type, model, project_id, context, started_at, db_path
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO invocations
                (id, agent_type, model, project_id, context, status, started_at)
            VALUES (?, ?, ?, ?, ?, 'running', ?)
            """,
            (invocation_id, agent_type, model, project_id, context, started_at),
        )


def _db_finish_invocation(
    invocation_id, response, tool_calls, status, finished_at, db_path
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE invocations
            SET response = ?, tool_calls = ?, status = ?, finished_at = ?
            WHERE id = ?
            """,
            (response, tool_calls, status, finished_at, invocation_id),
        )


def _db_insert_tool_call(
    tc_id, invocation_id, tool_name, parameters,
    risk_level, result, status, created_at, db_path,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tool_calls
                (id, invocation_id, tool_name, parameters, risk_level,
                 result, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tc_id, invocation_id, tool_name,
                json.dumps(parameters), risk_level,
                json.dumps(result) if result is not None else None,
                status, created_at,
            ),
        )


def _db_update_tool_call(tc_id, result, status, db_path) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE tool_calls
            SET result = ?, status = ?, executed_at = ?
            WHERE id = ?
            """,
            (json.dumps(result), status, _now(), tc_id),
        )


def _new_id(prefix: str = "") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
