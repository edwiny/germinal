# Purpose: Unit tests for tools/code_quality.py (lint and check_syntax).
# Covers:
#   - check_syntax on a valid Python file → valid=True
#   - check_syntax on a file with a syntax error → valid=False
#   - lint on a clean file → passed=True (ruff mocked)
#   - lint falls back to flake8 when ruff is not found (both mocked)
#   - lint returns error dict when neither linter is available
#   - lint with a non-existent path returns an error dict (not a raised exception)
#   - All tools return dicts, never raise

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.tools.code_quality import make_check_syntax_tool, make_lint_tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_py_file(tmp_path):
    """A syntactically valid Python file."""
    f = tmp_path / "valid.py"
    f.write_text("x = 1 + 2\n")
    return str(f)


@pytest.fixture()
def invalid_py_file(tmp_path):
    """A Python file with a syntax error."""
    f = tmp_path / "invalid.py"
    f.write_text("def broken(\n")
    return str(f)


# ---------------------------------------------------------------------------
# check_syntax
# ---------------------------------------------------------------------------

def test_check_syntax_valid_file(valid_py_file):
    """check_syntax returns valid=True for a syntactically correct file."""
    tool = make_check_syntax_tool()
    result = tool.execute({"path": valid_py_file})
    assert isinstance(result, dict)
    assert result.get("valid") is True
    assert result.get("returncode") == 0
    assert result.get("path") == valid_py_file


def test_check_syntax_invalid_file(invalid_py_file):
    """check_syntax returns valid=False for a file with a syntax error."""
    tool = make_check_syntax_tool()
    result = tool.execute({"path": invalid_py_file})
    assert isinstance(result, dict)
    assert result.get("valid") is False
    assert result.get("returncode") != 0


def test_check_syntax_result_is_dict_not_exception(valid_py_file):
    """check_syntax never raises — always returns a dict."""
    tool = make_check_syntax_tool()
    try:
        result = tool.execute({"path": valid_py_file})
        assert isinstance(result, dict)
    except Exception as exc:
        pytest.fail(f"check_syntax raised instead of returning a dict: {exc}")


def test_check_syntax_nonexistent_file():
    """check_syntax on a missing file returns an error dict, not a raised exception."""
    tool = make_check_syntax_tool()
    result = tool.execute({"path": "/nonexistent/path/file.py"})
    assert isinstance(result, dict)
    # py_compile exits non-zero on a missing file; valid must not be True.
    assert result.get("valid") is not True


# ---------------------------------------------------------------------------
# lint — mocked subprocess so tests are hermetic regardless of installed tools
# ---------------------------------------------------------------------------

def _mock_proc(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a mock CompletedProcess-like object."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_lint_clean_file_passes(valid_py_file):
    """lint returns passed=True when ruff reports no issues."""
    tool = make_lint_tool()
    with patch("orchestrator.tools.code_quality.subprocess.run", return_value=_mock_proc(0)) as mock_run:
        result = tool.execute({"path": valid_py_file})
    assert isinstance(result, dict)
    assert result.get("passed") is True
    assert result.get("tool_used") == "ruff"
    mock_run.assert_called_once()


def test_lint_issues_found_fails(valid_py_file):
    """lint returns passed=False when ruff reports issues (non-zero exit)."""
    tool = make_lint_tool()
    with patch("orchestrator.tools.code_quality.subprocess.run", return_value=_mock_proc(1, stdout="E501 line too long")):
        result = tool.execute({"path": valid_py_file})
    assert result.get("passed") is False
    assert result.get("tool_used") == "ruff"


def test_lint_falls_back_to_flake8_when_ruff_missing(valid_py_file):
    """lint uses flake8 when ruff raises FileNotFoundError."""
    tool = make_lint_tool()

    def side_effect(cmd, **kwargs):
        if cmd[0] == "ruff":
            raise FileNotFoundError
        return _mock_proc(0)

    with patch("orchestrator.tools.code_quality.subprocess.run", side_effect=side_effect):
        result = tool.execute({"path": valid_py_file})
    assert result.get("passed") is True
    assert result.get("tool_used") == "flake8"


def test_lint_error_dict_when_no_linter_available(valid_py_file):
    """lint returns an error dict (not a raised exception) when neither linter is found."""
    tool = make_lint_tool()
    with patch("orchestrator.tools.code_quality.subprocess.run", side_effect=FileNotFoundError):
        result = tool.execute({"path": valid_py_file})
    assert isinstance(result, dict)
    assert "error" in result


def test_lint_nonexistent_path_returns_dict():
    """lint with a non-existent path returns a dict — either an error or a linter report."""
    tool = make_lint_tool()
    with patch("orchestrator.tools.code_quality.subprocess.run", return_value=_mock_proc(1, stderr="No such file")):
        result = tool.execute({"path": "/nonexistent/path"})
    assert isinstance(result, dict)


def test_lint_result_is_dict_not_exception(valid_py_file):
    """lint never raises — always returns a dict."""
    tool = make_lint_tool()
    with patch("orchestrator.tools.code_quality.subprocess.run", return_value=_mock_proc(0)):
        try:
            result = tool.execute({"path": valid_py_file})
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"lint raised instead of returning a dict: {exc}")


def test_lint_fix_flag_passed_to_ruff(valid_py_file):
    """When fix=True, --fix is appended to the ruff command."""
    tool = make_lint_tool()
    with patch("orchestrator.tools.code_quality.subprocess.run", return_value=_mock_proc(0)) as mock_run:
        tool.execute({"path": valid_py_file, "fix": True})
    args = mock_run.call_args[0][0]
    assert "--fix" in args
