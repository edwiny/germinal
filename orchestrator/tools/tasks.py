# Purpose: Task backlog tools: read_task_list, write_task.
# Relationships: Reads/writes the `tasks` table via storage/db.py.
#               Registered into tools/registry.py; used by task_agent and dev_agent.

import json
import uuid
from datetime import datetime, timezone

from storage.db import get_conn
from tools.registry import Tool


def make_read_task_list_tool(db_path: str) -> Tool:
    """Return a read_task_list Tool that reads the open task backlog."""

    def execute(params: dict) -> dict:
        status_filter = params.get("status", "open")
        with get_conn(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, title, description, source, priority, status,
                       created_at, updated_at
                FROM tasks
                WHERE status = ?
                ORDER BY priority ASC, created_at ASC
                """,
                (status_filter,),
            ).fetchall()
        return {"tasks": [dict(row) for row in rows], "count": len(rows)}

    return Tool(
        name="read_task_list",
        description=(
            "Read the task backlog. Returns tasks ordered by priority (1=highest). "
            "Defaults to open tasks; pass status='in_progress' or 'done' to filter."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "done", "cancelled"],
                    "description": "Filter by task status. Defaults to 'open'.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
    )


def make_write_task_tool(db_path: str) -> Tool:
    """
    Return a write_task Tool that creates or updates a task.

    If task_id is provided and the task exists, it is updated.
    If task_id is omitted or the task does not exist, a new task is created.
    """

    def execute(params: dict) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        task_id = params.get("task_id")

        with get_conn(db_path) as conn:
            if task_id:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
            else:
                row = None

            if row:
                # Update existing task. Only update fields that were provided.
                updates: list[str] = ["updated_at = ?"]
                values: list = [now]
                for field in ("title", "description", "status", "priority"):
                    if field in params:
                        updates.append(f"{field} = ?")
                        values.append(params[field])
                values.append(task_id)
                conn.execute(
                    f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
                    values,
                )
                return {"task_id": task_id, "action": "updated"}
            else:
                # Create new task.
                new_id = task_id or ("task_" + uuid.uuid4().hex[:12])
                conn.execute(
                    """
                    INSERT INTO tasks
                        (id, project_id, title, description, source, priority,
                         status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    """,
                    (
                        new_id,
                        params.get("project_id"),
                        params.get("title", "(untitled)"),
                        params.get("description", ""),
                        params.get("source", "agent"),
                        params.get("priority", 5),
                        now,
                        now,
                    ),
                )
                return {"task_id": new_id, "action": "created"}

    return Tool(
        name="write_task",
        description=(
            "Create a new task or update an existing one in the task backlog. "
            "Omit task_id to create a new task. "
            "Provide task_id to update an existing task's title, description, status, or priority."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of an existing task to update. Omit to create new.",
                },
                "title": {
                    "type": "string",
                    "description": "Task title (required for new tasks).",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed task description.",
                },
                "source": {
                    "type": "string",
                    "enum": ["user", "agent", "reflection"],
                    "description": "Who created the task.",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Priority 1 (highest) to 10 (lowest). Default 5.",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "done", "cancelled"],
                    "description": "Task status (for updates).",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project this task belongs to (for new tasks).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
    )
