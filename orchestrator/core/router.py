# Purpose: Rules-based event router.
# Relationships: Called by main.py's event loop.

# Given an event dict, the router returns a routing decision dict containing
# agent type and model key. The task description is taken directly from the
# event payload ("message" field). Rules are evaluated in list order; the
# first match wins. An event that matches no rule raises UnroutableEvent so
# the caller can log and mark the event failed without crashing the loop.

# Phase 1 rule set covers user and HTTP message events. Add rules here as
# new event sources are introduced in later phases.


class UnroutableEvent(Exception):
    """Raised when no routing rule matches the event."""


# Each rule is a dict with:
#   source     — event source to match (or None to match any)
#   type       — event type to match (or None to match any)
#   agent_type — agent type to invoke
#   model_key  — key into config.models.list (or "default")
#
# The task description is always taken from payload["message"] — rules do not
# override prompt content. Rules are tested in order; the first full match wins.
_ROUTING_RULES: list[dict] = [
    {
        "source": "user",
        "type": "message",
        "agent_type": "task_agent",
        "model_key": "default",
    },
    {
        # Events injected by the HTTP network adapter.
        # Agent type and model are chosen by the orchestrator, not the client.
        "source": "http",
        "type": "message",
        "agent_type": "task_agent",
        "model_key": "default",
    },
]


def route_event(event: dict) -> dict:
    """
    Match event against routing rules. Return a routing decision dict:
        {
            "agent_type":       str,
            "model_key":        str,  # key in config.models.list, or "default"
            "task_description": str,  # taken directly from payload["message"]
        }

    Raises UnroutableEvent if no rule matches or if the matched event has no
    "message" field in its payload.
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

        message = payload.get("message", "")
        return {
            "agent_type": rule["agent_type"],
            "model_key": rule["model_key"],
            "task_description": message,
        }

    raise UnroutableEvent(
        f"No routing rule matched event source={event.get('source')!r} "
        f"type={event.get('type')!r}"
    )
