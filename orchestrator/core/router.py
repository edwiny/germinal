# Purpose: Rules-based event router.
# Relationships: Called by main.py's event loop; uses config.yaml routing section.
#               Reads storage/db.py for preflight task-count queries.

# Given an event dict, the router returns a routing decision dict containing
# agent type, model key, task description, and an optional preflight callable.
# Rules are evaluated in list order; the first match wins. An event that
# matches no rule raises UnroutableEvent so the caller can log and mark
# the event failed without crashing the loop.

# Phase 1 rule set covers timer ticks only. Add rules here as new event
# sources are introduced in later phases.


class UnroutableEvent(Exception):
    """Raised when no routing rule matches the event."""


# Each rule is a dict with:
#   source      — event source to match (or None to match any)
#   type        — event type to match (or None to match any)
#   agent_type  — agent type to invoke
#   model_key   — key into config.models.list (or "default")
#   task_template — task description string passed to the agent
#
# Rules are tested in order; the first full match wins.
_ROUTING_RULES: list[dict] = [
    {
        "source": "timer",
        "type": "tick",
        "agent_type": "task_agent",
        "model_key": "default",
        # The tick task is intentionally lightweight: review open tasks and
        # notify the user if anything needs attention. Local model is preferred
        # for this classification-style workload (see config.yaml routing).
        "task_template": (
            "A scheduled timer tick has fired. "
            "Use read_task_list to check the open task backlog. "
            "If there are any open tasks, summarise them and notify the user. "
            "If there is nothing to report, use notify_user to confirm the system is idle."
        ),
    },
    {
        "source": "user",
        "type": "message",
        "agent_type": "task_agent",
        "model_key": "default",
        "task_template": "{payload[message]}",
    },
]


def route_event(event: dict, config: dict) -> dict:
    """
    Match event against routing rules. Return a routing decision dict:
        {
            "agent_type":       str,
            "model_key":        str,  # key in config.models.list, or "default"
            "task_description": str,
            "preflight":        Callable[[], bool] | None,
        }

    preflight, if present, is called by the event loop before invoking the
    LLM. If it returns False the event is marked done and the LLM call is
    skipped. This avoids burning tokens on no-op ticks (e.g. timer fires
    when there are no open tasks).

    Raises UnroutableEvent if no rule matches.
    """
    for rule in _ROUTING_RULES:
        if rule.get("source") is not None and rule["source"] != event.get("source"):
            continue
        if rule.get("type") is not None and rule["type"] != event.get("type"):
            continue

        payload = event.get("payload", {})
        if isinstance(payload, str):
            import json
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        task_description = _render_template(rule["task_template"], payload)

        preflight = None
        if event.get("source") == "timer" and event.get("type") == "tick":
            db_path = config["paths"]["db"]
            # Capture db_path in the closure — config["paths"]["db"] could
            # mutate if config is ever reloaded, so we bind the value now.
            preflight = lambda _db=db_path: _has_open_tasks(_db)

        return {
            "agent_type": rule["agent_type"],
            "model_key": rule["model_key"],
            "task_description": task_description,
            "preflight": preflight,
        }

    raise UnroutableEvent(
        f"No routing rule matched event source={event.get('source')!r} "
        f"type={event.get('type')!r}"
    )


def _has_open_tasks(db_path: str) -> bool:
    """
    Return True if there is at least one open task in the backlog.

    Used as the preflight check for timer tick events. A tick with no open
    tasks is a no-op — skipping the LLM call avoids unnecessary token spend.
    """
    from storage.db import get_conn
    with get_conn(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'open'"
        ).fetchone()[0]
    return count > 0


def _render_template(template: str, payload: dict) -> str:
    """
    Expand {payload[key]} references in the task template.

    # We use a simple manual substitution rather than str.format_map or
    # Jinja to avoid arbitrary code execution via template injection.
    # Only payload[key] references are supported; any unresolvable reference
    # is left as-is (the agent will see the literal placeholder).
    """
    import re

    def replace(match: re.Match) -> str:
        key = match.group(1)
        return str(payload.get(key, match.group(0)))

    return re.sub(r"\{payload\[(\w+)\]\}", replace, template)
