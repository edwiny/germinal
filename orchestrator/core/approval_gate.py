# [SAFETY-CRITICAL] This module implements the human-in-the-loop approval gate.
# Any change here requires human review regardless of risk classification.
# Do not modify as part of autonomous improvement tasks.
#
# Purpose: Block high-risk tool calls until a human approves or denies them.
# Relationships: Called by core/agent_invoker.py before executing high-risk
#               tools. Writes approval records to the `approvals` table.

# The approval gate is the primary safety mechanism for high-risk tool calls.
# In Phase 1 the transport is terminal (stdout prompt + stdin response).
# Later phases replace the transport (e.g. web push, notification) without
# changing the interface: approve() always returns True (approved) or
# False (denied), and the caller treats the result the same way regardless.

import json
import logging
import sys
import uuid
from datetime import datetime, timezone

from storage.db import get_conn

logger = logging.getLogger("approval_gate")


def request_approval(
    tool_name: str,
    parameters: dict,
    agent_type: str,
    project_id: str | None,
    tool_call_id: str,
    db_path: str | None,
) -> bool:
    """
    Show an approval prompt on stdout and wait for a human response on stdin.

    Records the approval request to the `approvals` table before prompting
    and records the response immediately after. If stdin is not a terminal
    (e.g. in tests or piped runs), auto-denies and logs the denial.

    Returns True if approved, False if denied.

    # [INVARIANT] The approval record must be written before the prompt is
    # shown and updated before this function returns. Reversing that order
    # would create a window where the tool could execute with no DB record.
    """
    approval_id = "appr_" + uuid.uuid4().hex[:16]
    created_at = _now()

    prompt_text = _build_prompt(tool_name, parameters, agent_type, project_id)

    _db_insert_approval(approval_id, tool_call_id, prompt_text, created_at, db_path)

    # Auto-deny when stdin is not interactive (e.g. piped, test harness).
    # This is the safe default: an unattended process cannot approve a
    # high-risk action on behalf of the human.
    if not sys.stdin.isatty():
        _db_record_response(approval_id, "denied", db_path)
        logger.warning("Non-interactive stdin detected — auto-denying %r.", tool_name)
        return False

    # Print the full formatted prompt directly so the human sees it clearly
    # on the terminal without a logger prefix cluttering the approval block.
    print(prompt_text, flush=True)

    try:
        answer = input("Approve? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    approved = answer == "y"
    response = "approved" if approved else "denied"
    _db_record_response(approval_id, response, db_path)
    return approved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(
    tool_name: str,
    parameters: dict,
    agent_type: str,
    project_id: str | None,
) -> str:
    params_pretty = json.dumps(parameters, indent=2)
    project_str = project_id or "(none)"
    return (
        f"\n{'=' * 60}\n"
        f"[APPROVAL REQUIRED]\n"
        f"Agent: {agent_type}  |  Project: {project_str}  |  Risk: high\n"
        f"Tool: {tool_name}\n"
        f"Parameters:\n{params_pretty}\n"
        f"{'=' * 60}"
    )


def _db_insert_approval(
    approval_id: str,
    tool_call_id: str,
    prompt: str,
    created_at: str,
    db_path: str | None,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO approvals (id, tool_call_id, prompt, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (approval_id, tool_call_id, prompt, created_at),
        )


def _db_record_response(approval_id: str, response: str, db_path: str | None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE approvals
            SET response = ?, responded_at = ?
            WHERE id = ?
            """,
            (response, _now(), approval_id),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
