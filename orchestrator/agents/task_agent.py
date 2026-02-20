# Purpose: task_agent type definition and registry helper.
# Relationships: Used by main.py to build the per-agent tool registry;
#               agent_type string matched against config.yaml agents section.

# The task_agent is the general-purpose agent for routine work:
# reading tasks, reading files, and notifying the user. It deliberately
# has a small tool surface so that most of its actions are low-risk
# and do not require approval.

from tools.registry import ToolRegistry

AGENT_TYPE = "task_agent"


def make_registry(full_registry: ToolRegistry, config: dict) -> ToolRegistry:
    """
    Return a ToolRegistry filtered to the tools allowed for task_agent.

    Tools not yet registered (e.g. if a tool is in the config but its
    make_* factory was not called at startup) are silently skipped.
    This allows the config to list future tools without crashing the loop.
    """
    allowed_names: list[str] = config["agents"]["task_agent"]["allowed_tools"]
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
