# Purpose: Rules-based event router.
# Relationships: Called by main.py's event loop; uses models.categories in config.yaml.

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
        # The tick task is intentionally lightweight: confirm the system is
        # running and notify the user. Local model is preferred for this
        # classification-style workload (see models.categories in config.yaml).
        "task_template": (
            "A scheduled timer tick has fired. "
            "Use notify_user to confirm the system is running and idle."
        ),
    },
    {
        "source": "user",
        "type": "message",
        "agent_type": "task_agent",
        "model_key": "default",
        "task_template": "{payload[message]}",
    },
    {
        # Events injected by the HTTP network adapter.
        # Agent type and model are chosen by the orchestrator, not the client.
        "source": "http",
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

    preflight is always None in the current rule set. The field is preserved
    in the return dict so callers do not need updating if a preflight is
    added in a future phase.

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
        return {
            "agent_type": rule["agent_type"],
            "model_key": rule["model_key"],
            "task_description": task_description,
            "preflight": None,
        }

    raise UnroutableEvent(
        f"No routing rule matched event source={event.get('source')!r} "
        f"type={event.get('type')!r}"
    )


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
