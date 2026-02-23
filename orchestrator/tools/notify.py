# Purpose: Notification tool — delivers messages to the user.
# Relationships: Registers into tools/registry.py via make_notify_user_tool();
#               Phase 0 transport is stdout/stderr. Later phases replace
#               the transport without changing the tool interface.

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .registry import Tool, model_to_json_schema

logger = logging.getLogger("notify")


class NotifyUserParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="The message text to deliver to the user.")
    level: Literal["info", "warning", "error"] = Field(
        default="info",
        description="Severity level. Defaults to 'info'.",
    )


class NotifyUserResult(BaseModel):
    delivered: bool = Field(description="True if the notification was delivered.")
    channel: str = Field(description="Delivery channel used (e.g. 'terminal').")


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
        return NotifyUserResult(delivered=True, channel="terminal").model_dump()

    return Tool(
        name="notify_user",
        description=(
            "Send a notification message to the user. "
            "Use this to report task completion, errors, or important findings."
        ),
        parameters_schema=model_to_json_schema(NotifyUserParams),
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
        params_model=NotifyUserParams,
    )
