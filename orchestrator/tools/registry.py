# [SAFETY-CRITICAL] This module is the sole dispatch point for all tool
# execution. It validates parameters before any tool runs and is the
# last line of defence before a side effect reaches the system.
# Any change here requires human review regardless of risk classification.
# Do not modify as part of autonomous improvement tasks.
#
# Purpose: Tool dataclass, ToolRegistry, and the global registry instance.
# Relationships: tools/filesystem.py and tools/notify.py register into this;
#               core/agent_invoker.py dispatches through this.

from dataclasses import dataclass, field
from typing import Callable

import jsonschema


@dataclass
class Tool:
    name: str
    description: str
    parameters_schema: dict   # JSON Schema — validated before every execute()
    risk_level: str           # 'low' | 'medium' | 'high'
    allowed_agents: list[str]
    _execute: Callable[[dict], dict] = field(repr=False)

    def execute(self, parameters: dict) -> dict:
        # [INVARIANT] Parameters are always validated before execution.
        # Never call _execute directly — always go through this method.
        # Skipping validation here would allow malformed inputs to reach
        # tool implementations that assume clean data.
        try:
            jsonschema.validate(parameters, self.parameters_schema)
        except jsonschema.ValidationError as exc:
            return {"error": f"Parameter validation failed: {exc.message}"}
        return self._execute(parameters)


class ToolRegistry:
    """Registry of all tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}")
        return self._tools[name]

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schema_for_agent(self) -> list[dict]:
        """Return tool descriptions suitable for injection into agent prompts."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema,
                "risk_level": t.risk_level,
            }
            for t in self._tools.values()
        ]
