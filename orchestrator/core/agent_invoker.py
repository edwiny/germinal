# Purpose: Assembles prompts, calls the language model via LiteLLM, parses
#          tool calls from the response, executes them through the registry,
#          and writes every step to the database.
# Relationships: Uses storage/db.py, tools/registry.py, agents/base_prompt.py.
#               Called directly by main.py in Phase 0; by router.py in Phase 1+.

import json
import re
import uuid
from datetime import datetime, timezone

import litellm

from agents.base_prompt import build_system_prompt
from storage.db import get_conn
from tools.registry import ToolRegistry

# [INVARIANT] Iteration cap prevents runaway loops. An agent that needs more
# than DEFAULT_MAX_ITERATIONS for a sensible task is almost certainly stuck.
# Raise only after confirming the model is reasoning correctly on the task.
DEFAULT_MAX_ITERATIONS = 10

# Regex for extracting <tool_call>...</tool_call> blocks.
# re.DOTALL so the JSON body can span multiple lines.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def invoke(
    task_description: str,
    agent_type: str,
    model: str,
    registry: ToolRegistry,
    project_id: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    db_path: str | None = None,
) -> dict:
    """
    Run a single agent invocation to completion.

    Assembles the prompt, drives the tool-call loop, logs everything to the
    DB, and returns a summary dict:
        {
            "invocation_id": str,
            "status": "done" | "failed",
            "response": str,          # final agent text
            "tool_calls": list[dict], # summary of every tool call made
        }
    """
    invocation_id = _new_id("inv")
    started_at = _now()
    tool_calls_log: list[dict] = []

    system_prompt = build_system_prompt(registry.schema_for_agent())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_description},
    ]

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

    for _ in range(max_iterations):
        raw = litellm.completion(model=model, messages=messages)
        assistant_text: str = raw.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        call_request = _parse_tool_call(assistant_text)
        if call_request is None:
            # No tool call emitted — agent has finished.
            final_response = assistant_text
            status = "done"
            break

        tool_name = call_request.get("tool", "")
        parameters = call_request.get("parameters", {})

        tc_id = _new_id("tc")
        result = _run_tool(
            tool_call_id=tc_id,
            invocation_id=invocation_id,
            tool_name=tool_name,
            parameters=parameters,
            registry=registry,
            db_path=db_path,
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
        # [DO NOT SIMPLIFY] We must log and return cleanly even when the
        # iteration cap fires. Raising here would lose the partial DB record.
        final_response = "Iteration cap reached without task completion."
        status = "failed"

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
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_tool_call(text: str) -> dict | None:
    """Return the first <tool_call> JSON object found in text, or None."""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _run_tool(
    tool_call_id: str,
    invocation_id: str,
    tool_name: str,
    parameters: dict,
    registry: ToolRegistry,
    db_path: str | None,
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

    try:
        result = tool.execute(parameters)
        _db_update_tool_call(tool_call_id, result, "executed", db_path)
    except Exception as exc:
        result = {"error": str(exc)}
        _db_update_tool_call(tool_call_id, result, "failed", db_path)

    return result


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
