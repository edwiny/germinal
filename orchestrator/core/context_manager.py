# Purpose: Three-tier project context management (brief, summary, recent history).
#          Assembles context for prompt injection; triggers summarisation when
#          the recent history buffer overflows.
# Relationships: Used by core/agent_invoker.py. Reads/writes projects and history
#               tables via storage/db.py. Calls litellm for summarisation.

import uuid
from datetime import datetime, timezone

import litellm

from ..storage.db import get_conn


def _count_tokens(text: str) -> int:
    # Cheap approximation: one token ≈ four characters. This avoids a
    # tokeniser dependency while remaining accurate enough for budget decisions.
    # Do not replace with a real tokeniser unless off-by-30% errors cause
    # visible problems in context assembly.
    return len(text) // 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def ensure_project(project_id: str, name: str, db_path: str) -> None:
    """
    Guarantee a row exists in the projects table for project_id.

    INSERT OR IGNORE is idempotent: safe to call before every invocation
    without checking first. Does not overwrite existing name or brief/summary.
    """
    now = _now()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO projects (id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (project_id, name, now, now),
        )


def assemble_context(project_id: str, db_path: str, config: dict) -> str:
    """
    Build the context string to inject between the system prompt and the task.

    Returns "" if the project does not exist or all three tiers are empty,
    so the caller can skip injection entirely in that case.

    Recent history is collected newest-first until the token budget is consumed,
    then reversed to chronological order so the prompt reads naturally.
    """
    with get_conn(db_path) as conn:
        project = conn.execute(
            "SELECT brief, summary FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

    if project is None:
        return ""

    brief = project["brief"] or ""
    summary = project["summary"] or ""

    recent_buffer_tokens: int = config["context"]["recent_buffer_tokens"]

    with get_conn(db_path) as conn:
        # Newest first so we fill the budget with the most recent entries.
        rows = conn.execute(
            """
            SELECT role, content FROM history
            WHERE project_id = ?
            ORDER BY created_at DESC
            """,
            (project_id,),
        ).fetchall()

    # Walk newest-first, accumulate rows until the token budget is consumed.
    # We reverse the collected slice afterward so the prompt reads oldest-first.
    recent_rows: list[tuple[str, str]] = []
    budget = recent_buffer_tokens
    for row in rows:
        if budget <= 0:
            break
        entry = f"[{row['role'].upper()}] {row['content']}"
        tokens = _count_tokens(entry)
        recent_rows.append((row["role"], row["content"]))
        budget -= tokens

    recent_rows.reverse()

    # Return "" when all tiers are empty so the caller injects nothing.
    if not brief and not summary and not recent_rows:
        return ""

    lines = ["=== PROJECT CONTEXT ===", ""]
    lines.append("[BRIEF]")
    lines.append(brief if brief else "(none)")
    lines.append("")
    lines.append("[SUMMARY]")
    lines.append(summary if summary else "(none)")
    lines.append("")
    lines.append("[RECENT HISTORY]")
    for role, content in recent_rows:
        lines.append(f"[{role.upper()}] {content}")
    lines.append("=== END CONTEXT ===")

    return "\n".join(lines)


def append_to_history(project_id: str, role: str, content: str, db_path: str) -> None:
    """Insert one history row. Called twice after each invocation: user task + agent response."""
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO history (id, project_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_new_id(), project_id, role, content, _now()),
        )


async def maybe_summarise(
    project_id: str,
    db_path: str,
    model: str,
    api_key: str | None,
    config: dict,
) -> None:
    """
    Compress old history into projects.summary if the buffer is over budget.

    Split logic: walk rows oldest-first, accumulate tokens, stop when
    accumulated >= total - recent_buffer_tokens. Everything up to that split
    point is summarised and deleted; the rest stays as the recent buffer.

    Skips without action when total history is within budget. This keeps the
    common case (short-lived projects) cheap — no model call, no DB writes.
    """
    recent_buffer_tokens: int = config["context"]["recent_buffer_tokens"]

    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, role, content FROM history
            WHERE project_id = ?
            ORDER BY created_at ASC
            """,
            (project_id,),
        ).fetchall()

    if not rows:
        return

    total_tokens = sum(_count_tokens(row["content"]) for row in rows)
    if total_tokens <= recent_buffer_tokens:
        return

    # Determine the split point: summarise oldest rows until we have
    # compressed enough that the remainder fits within the budget.
    # [INVARIANT] target_summarise_tokens must be > 0 here because
    # total_tokens > recent_buffer_tokens (checked above).
    target_summarise_tokens = total_tokens - recent_buffer_tokens
    accumulated = 0
    split_index = 0
    for i, row in enumerate(rows):
        accumulated += _count_tokens(row["content"])
        if accumulated >= target_summarise_tokens:
            split_index = i + 1
            break

    if split_index == 0:
        # Edge case: first row alone is enough to exceed the target but
        # loop didn't set split_index. Summarise at least one row.
        split_index = 1

    to_summarise = rows[:split_index]

    with get_conn(db_path) as conn:
        project = conn.execute(
            "SELECT summary FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

    existing_summary = (project["summary"] or "") if project else ""

    history_text = "\n".join(
        f"[{r['role'].upper()}] {r['content']}" for r in to_summarise
    )
    summarise_prompt = (
        "You are a context compressor. Produce a concise summary of the "
        "conversation history below, incorporating any existing summary.\n\n"
        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        f"New history to incorporate:\n{history_text}\n\n"
        "Write a dense, factual summary. Preserve key decisions, outcomes, and "
        "open questions. Omit pleasantries and repetition. Output only the summary."
    )

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": summarise_prompt}],
        api_key=api_key,
    )
    new_summary: str = response.choices[0].message.content or ""

    ids_to_delete = [row["id"] for row in to_summarise]
    placeholders = ",".join("?" * len(ids_to_delete))
    now = _now()

    with get_conn(db_path) as conn:
        conn.execute(
            f"DELETE FROM history WHERE id IN ({placeholders})",
            ids_to_delete,
        )
        conn.execute(
            "UPDATE projects SET summary = ?, updated_at = ? WHERE id = ?",
            (new_summary, now, project_id),
        )
