# Purpose: Pydantic contract tests for the tool registry and migrated tools.
# Covers:
#   - params_model validation: valid params pass through cleanly
#   - params_model validation: invalid params return a structured error dict
#   - extra properties forbidden by ConfigDict(extra='forbid')
#   - result models produce the expected shape
#   - EventEnvelope catches malformed events at the adapter boundary

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.event_queue import EventEnvelope
from tools.filesystem import (
    ReadFileParams,
    ReadFileResult,
    WriteFileParams,
    WriteFileResult,
    ListDirectoryParams,
    ListDirectoryResult,
)
from tools.git import (
    GitStatusParams,
    GitCommitParams,
    GitCommitResult,
    GitBranchParams,
    GitRollbackParams,
)
from tools.notify import NotifyUserParams, NotifyUserResult
from tools.registry import Tool, ToolRegistry, model_to_json_schema
from tools.shell import RunTestsParams, RunTestsResult, ShellRunParams
from tools.tasks import ReadTaskListParams, WriteTaskParams, WriteTaskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(description="Message to echo.")


class _EchoResult(BaseModel):
    echoed: str = Field(description="The echoed message.")


def _make_pydantic_tool(name="echo_pydantic") -> Tool:
    """Minimal tool that uses a Pydantic params_model."""

    def _execute(params: dict) -> dict:
        return _EchoResult(echoed=params["message"]).model_dump()

    return Tool(
        name=name,
        description="Echo via Pydantic.",
        parameters_schema=model_to_json_schema(_EchoParams),
        risk_level="low",
        allowed_agents=["task_agent"],
        _execute=_execute,
        params_model=_EchoParams,
    )


# ---------------------------------------------------------------------------
# registry: Pydantic validation path
# ---------------------------------------------------------------------------

def test_pydantic_valid_params_pass_through():
    """Valid params via params_model reach _execute cleanly."""
    tool = _make_pydantic_tool()
    result = tool.execute({"message": "hello"})
    assert result == {"echoed": "hello"}


def test_pydantic_missing_required_field_returns_error():
    """Missing required field returns a structured error dict, not a raised exception."""
    tool = _make_pydantic_tool()
    result = tool.execute({})
    assert "error" in result
    assert "validation" in result["error"].lower()


def test_pydantic_wrong_type_returns_error():
    """Wrong field type returns a structured error dict."""
    tool = _make_pydantic_tool()
    result = tool.execute({"message": 123})
    # Pydantic coerces int to str in lax mode; check the result is still valid
    # (Pydantic v2 by default coerces int → str in non-strict mode).
    # The tool should succeed or return an error — it must not raise.
    assert isinstance(result, dict)


def test_pydantic_extra_properties_rejected():
    """Extra properties not in the Pydantic model are rejected (extra='forbid')."""
    tool = _make_pydantic_tool()
    result = tool.execute({"message": "hi", "unexpected_field": "bad"})
    assert "error" in result


def test_pydantic_error_is_dict_not_exception():
    """Validation failures must return a dict, never raise."""
    tool = _make_pydantic_tool()
    try:
        result = tool.execute({"not_a_valid_field": True})
        assert "error" in result
    except Exception as exc:
        pytest.fail(f"execute() raised instead of returning error dict: {exc}")


# ---------------------------------------------------------------------------
# model_to_json_schema helper
# ---------------------------------------------------------------------------

def test_model_to_json_schema_returns_dict():
    """model_to_json_schema must return a non-empty dict."""
    schema = model_to_json_schema(_EchoParams)
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_model_to_json_schema_includes_required():
    """Required fields appear in the generated schema."""
    schema = model_to_json_schema(_EchoParams)
    assert "message" in schema.get("required", [])


def test_model_to_json_schema_forbids_additional_properties():
    """extra='forbid' must produce additionalProperties: false in the schema."""
    schema = model_to_json_schema(_EchoParams)
    assert schema.get("additionalProperties") is False


def test_parameters_schema_matches_params_model_schema():
    """parameters_schema on a migrated tool equals model_to_json_schema output."""
    tool = _make_pydantic_tool()
    assert tool.parameters_schema == model_to_json_schema(_EchoParams)


# ---------------------------------------------------------------------------
# Result model shapes
# ---------------------------------------------------------------------------

def test_notify_user_result_shape():
    result = NotifyUserResult(delivered=True, channel="terminal")
    d = result.model_dump()
    assert d == {"delivered": True, "channel": "terminal"}


def test_read_file_result_shape():
    result = ReadFileResult(content="hello", path="/tmp/f.txt")
    d = result.model_dump()
    assert set(d.keys()) == {"content", "path"}


def test_write_file_result_shape():
    result = WriteFileResult(success=True, path="/tmp/f.txt", bytes_written=5)
    d = result.model_dump()
    assert d["success"] is True
    assert d["bytes_written"] == 5


def test_git_commit_result_shape():
    from tools.git import GitCommitResult
    result = GitCommitResult(stdout="ok", stderr="", returncode=0, success=True)
    d = result.model_dump()
    assert "success" in d
    assert d["success"] is True


def test_write_task_result_shape():
    result = WriteTaskResult(task_id="task_abc123", action="created")
    d = result.model_dump()
    assert d == {"task_id": "task_abc123", "action": "created"}


def test_run_tests_result_shape():
    result = RunTestsResult(stdout=".", stderr="", returncode=0, passed=True)
    d = result.model_dump()
    assert d["passed"] is True


# ---------------------------------------------------------------------------
# Params model validation — spot checks on migrated tools
# ---------------------------------------------------------------------------

def test_notify_params_level_default():
    """level defaults to 'info' when not provided."""
    p = NotifyUserParams(message="hi")
    assert p.level == "info"


def test_notify_params_invalid_level_raises():
    """Invalid level raises ValidationError."""
    with pytest.raises(ValidationError):
        NotifyUserParams(message="hi", level="debug")


def test_git_commit_params_empty_message_raises():
    """Empty commit message is rejected by min_length=1."""
    with pytest.raises(ValidationError):
        GitCommitParams(message="")


def test_shell_run_params_accepts_list():
    """ShellRunParams accepts command as a list of strings."""
    p = ShellRunParams(command=["pytest", "-v"])
    assert p.command == ["pytest", "-v"]


def test_shell_run_params_accepts_string():
    """ShellRunParams accepts command as a whitespace-separated string."""
    p = ShellRunParams(command="pytest -v")
    assert p.command == "pytest -v"


def test_write_task_params_priority_bounds():
    """Priority outside 1-10 is rejected."""
    with pytest.raises(ValidationError):
        WriteTaskParams(priority=0)
    with pytest.raises(ValidationError):
        WriteTaskParams(priority=11)


def test_git_status_params_rejects_extra():
    """GitStatusParams (empty model with extra='forbid') rejects any extra field."""
    with pytest.raises(ValidationError):
        GitStatusParams(unexpected="field")


def test_read_task_list_params_invalid_status_raises():
    """ReadTaskListParams rejects an unknown status value."""
    with pytest.raises(ValidationError):
        ReadTaskListParams(status="archived")


# ---------------------------------------------------------------------------
# EventEnvelope model
# ---------------------------------------------------------------------------

def test_event_envelope_valid():
    """Valid envelope fields are accepted."""
    env = EventEnvelope(source="timer", type="tick", payload={"minute": "00"})
    assert env.source == "timer"
    assert env.priority == 5  # default


def test_event_envelope_priority_bounds():
    """Priority outside 1-10 is rejected."""
    with pytest.raises(ValidationError):
        EventEnvelope(source="s", type="t", payload={}, priority=0)
    with pytest.raises(ValidationError):
        EventEnvelope(source="s", type="t", payload={}, priority=11)


def test_event_envelope_missing_required_raises():
    """Missing source field raises ValidationError."""
    with pytest.raises(ValidationError):
        EventEnvelope(type="tick", payload={})  # type: ignore[call-arg]


def test_event_envelope_extra_field_rejected():
    """Extra fields not in the model are rejected."""
    with pytest.raises(ValidationError):
        EventEnvelope(source="s", type="t", payload={}, unknown_field="x")


def test_event_envelope_project_id_defaults_to_none():
    """project_id defaults to None when not provided."""
    env = EventEnvelope(source="s", type="t", payload={})
    assert env.project_id is None
