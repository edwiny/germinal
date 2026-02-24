# Purpose: task_agent type definition and registry helper.
# Relationships: Used by main.py to build the per-agent tool registry;
#               agent_type string matched against config.yaml agents section.

# The task_agent is the general-purpose agent for routine work:
# reading files and notifying the user. It deliberately has a small
# tool surface so that most of its actions are low-risk and do not
# require approval.

from ..tools.registry import ToolRegistry

AGENT_TYPE = "task_agent"


def make_registry(full_registry: ToolRegistry, config: dict, agent_type: str = AGENT_TYPE) -> ToolRegistry:
    """
    Return a ToolRegistry filtered to the tools allowed for the specified agent.

    If "*" is in the allowed_tools list, all available tools are included.
    Tools not yet registered (e.g. if a tool is in the config but its
    make_* factory was not called at startup) are silently skipped.
    This allows the config to list future tools without crashing the loop.
    """
    allowed_names: list[str] = config["agents"][agent_type]["allowed_tools"]

    # If "*" is in the allowed tools, include all tools
    if "*" in allowed_names:
        return full_registry

    filtered = ToolRegistry()
    for name in allowed_names:
        try:
            filtered.register(full_registry.get(name))
        except KeyError:
            # Tool listed in config but not registered — skip rather than crash.
            # This happens when a Phase N tool is listed in config before Phase N
            # is implemented.
            pass
    return filtered
