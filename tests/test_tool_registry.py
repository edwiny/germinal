# Purpose: Tests for tools/registry.py.
# Covers: parameter validation before execute, unknown tool error,
#         schema_for_agent output format, allowed_agents list.

import pytest
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.tools.registry import Tool, ToolRegistry, model_to_json_schema


class _EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(description="Message to echo.")


def _make_tool(name="echo", risk_level="low", allowed_agents=None) -> Tool:
    """Return a minimal test tool backed by a Pydantic params model."""
    if allowed_agents is None:
        allowed_agents = ["task_agent"]

    def _execute(params: dict) -> dict:
        return {"echoed": params.get("message", "")}

    return Tool(
        name=name,
        description="Echo a message.",
        parameters_schema=model_to_json_schema(_EchoParams),
        risk_level=risk_level,
        allowed_agents=allowed_agents,
        _execute=_execute,
        params_model=_EchoParams,
    )


def test_execute_valid_parameters():
    """Tool.execute must call _execute when parameters match the schema."""
    tool = _make_tool()
    result = tool.execute({"message": "hello"})
    assert result == {"echoed": "hello"}


def test_execute_invalid_parameters_returns_error():
    """Tool.execute must return an error dict when parameters fail validation."""
    tool = _make_tool()
    result = tool.execute({"wrong_field": 123})
    assert "error" in result
    assert "validation" in result["error"].lower()


def test_execute_additional_properties_rejected():
    """Extra properties not in the schema must be rejected (additionalProperties: false)."""
    tool = _make_tool()
    result = tool.execute({"message": "hi", "extra": "bad"})
    assert "error" in result


def test_registry_get_registered_tool():
    """ToolRegistry.get must return a registered tool by name."""
    reg = ToolRegistry()
    tool = _make_tool()
    reg.register(tool)
    assert reg.get("echo") is tool


def test_registry_get_unknown_raises_key_error():
    """ToolRegistry.get must raise KeyError for an unknown tool name."""
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_all_tools():
    """ToolRegistry.all_tools must return all registered tools."""
    reg = ToolRegistry()
    reg.register(_make_tool("a"))
    reg.register(_make_tool("b"))
    names = {t.name for t in reg.all_tools()}
    assert names == {"a", "b"}


def test_schema_for_agent_contains_expected_fields():
    """schema_for_agent must include name, description, parameters, risk_level."""
    reg = ToolRegistry()
    reg.register(_make_tool())
    schema = reg.schema_for_agent()
    assert len(schema) == 1
    entry = schema[0]
    assert entry["name"] == "echo"
    assert "description" in entry
    assert "parameters" in entry
    assert "risk_level" in entry


def test_register_overwrites_existing_name():
    """Registering two tools with the same name keeps the last one."""
    reg = ToolRegistry()
    reg.register(_make_tool("dup"))
    tool2 = _make_tool("dup")
    reg.register(tool2)
    assert reg.get("dup") is tool2


def test_make_registry_with_wildcard():
    """make_registry with "*" in allowed_tools should return all tools."""
    from orchestrator.agents.task_agent import make_registry

    # Create a full registry with multiple tools
    full_reg = ToolRegistry()
    tool1 = _make_tool("tool1")
    tool2 = _make_tool("tool2")
    tool3 = _make_tool("tool3")
    full_reg.register(tool1)
    full_reg.register(tool2)
    full_reg.register(tool3)

    # Config with "*" wildcard
    config = {
        "agents": {
            "test_agent": {
                "allowed_tools": ["*"]
            }
        }
    }

    # Should return the full registry
    filtered_reg = make_registry(full_reg, config, "test_agent")
    assert filtered_reg is full_reg  # Should return the same registry object

    # Verify all tools are available
    assert filtered_reg.get("tool1") is tool1
    assert filtered_reg.get("tool2") is tool2
    assert filtered_reg.get("tool3") is tool3


def test_make_registry_with_specific_tools():
    """make_registry with specific tool names should filter correctly."""
    from orchestrator.agents.task_agent import make_registry

    # Create a full registry with multiple tools
    full_reg = ToolRegistry()
    tool1 = _make_tool("tool1")
    tool2 = _make_tool("tool2")
    tool3 = _make_tool("tool3")
    full_reg.register(tool1)
    full_reg.register(tool2)
    full_reg.register(tool3)

    # Config with specific tools
    config = {
        "agents": {
            "test_agent": {
                "allowed_tools": ["tool1", "tool3"]
            }
        }
    }

    # Should return filtered registry
    filtered_reg = make_registry(full_reg, config, "test_agent")
    assert filtered_reg is not full_reg  # Should be a different registry object

    # Verify only specified tools are available
    assert filtered_reg.get("tool1") is tool1
    assert filtered_reg.get("tool3") is tool3

    # tool2 should not be available
    with pytest.raises(KeyError):
        filtered_reg.get("tool2")
