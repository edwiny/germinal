# Purpose: Notification tool — delivers messages to the user.
# Relationships: Registers into tools/registry.py via make_notify_user_tool();
#               Phase 0 transport is stdout/stderr. Later phases replace
#               the transport without changing the tool interface.

import logging

from tools.registry import Tool

logger = logging.getLogger("notify")


def make_notify_user_tool() -> Tool:
    """Return a notify_user Tool that prints to stderr in Phase 0."""

    def execute(params: dict) -> dict:
        message = params["message"]
        level = params.get("level", "info")
        # Route notification to the appropriate log level so it surfaces
        # correctly in the log stream. Later phases replace this with a
        # real delivery channel without changing the tool interface.
        log_fn = {"warning": logger.warning, "error": logger.error}.get(
            level, logger.info
        )
        log_fn("[NOTIFY] %s", message)
        return {"delivered": True, "channel": "terminal"}

    return Tool(
        name="notify_user",
        description=(
            "Send a notification message to the user. "
            "Use this to report task completion, errors, or important findings."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message text to deliver to the user.",
                },
                "level": {
                    "type": "string",
                    "enum": ["info", "warning", "error"],
                    "description": "Severity level. Defaults to 'info'.",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
    )
