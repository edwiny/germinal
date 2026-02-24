# Purpose: Code quality tools: lint (ruff) and check_syntax (py_compile).
# Relationships: Registered into tools/registry.py via make_* factories;
#               called by dev_agent through core/agent_invoker.py.

# All tools use subprocess.run(..., shell=False). Neither tool modifies files
# unless fix=True is passed to lint. check_syntax is read-only by design.

import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from .registry import Tool, model_to_json_schema

_LINT_TIMEOUT = 60   # seconds; linting a large codebase should not exceed this
_SYNTAX_TIMEOUT = 30  # seconds; py_compile on a single file is always fast


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------

class LintParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File or directory to lint.")
    fix: bool = Field(
        default=False,
        description="If true, pass --fix to ruff to auto-fix safe issues.",
    )


class LintResult(BaseModel):
    stdout: str = Field(description="Standard output from the linter.")
    stderr: str = Field(description="Standard error from the linter.")
    returncode: int = Field(description="Return code from the linter.")
    passed: bool = Field(description="True if the linter reported no issues (returncode 0).")


def make_lint_tool() -> Tool:
    """
    Return a lint tool that runs ruff.

    """

    def execute(params: dict) -> dict:
        path = params["path"]
        fix = params.get("fix", False)

        # Try ruff first.
        ruff_cmd = ["ruff", "check", path]
        if fix:
            ruff_cmd.append("--fix")
        try:
            proc = subprocess.run(
                ruff_cmd,
                capture_output=True,
                text=True,
                timeout=_LINT_TIMEOUT,
                shell=False,
            )
            return LintResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                passed=proc.returncode == 0,
            ).model_dump()
        except FileNotFoundError:
            return {"error": "ruff not found in PATH"}
        except subprocess.TimeoutExpired:
            return {"error": f"Linter timed out after {_LINT_TIMEOUT}s"}
        except Exception as exc:
            return {"error": str(exc)}

    return Tool(
        name="lint",
        description=(
            "Run ruff on a file or directory. "
            "Set fix=true to auto-fix safe issues with ruff. "
            "Returns passed=true if no issues were found."
        ),
        parameters_schema=model_to_json_schema(LintParams),
        risk_level="low",
        _execute=execute,
        params_model=LintParams,
    )


# ---------------------------------------------------------------------------
# check_syntax
# ---------------------------------------------------------------------------

class CheckSyntaxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Python file to syntax-check.")


class CheckSyntaxResult(BaseModel):
    stdout: str = Field(description="Standard output from py_compile (usually empty).")
    stderr: str = Field(description="Standard error from py_compile (contains syntax errors).")
    returncode: int = Field(description="Return code from py_compile.")
    valid: bool = Field(description="True if the file has no syntax errors (returncode 0).")
    path: str = Field(description="The file that was checked.")


def make_check_syntax_tool() -> Tool:
    """
    Return a check_syntax tool that validates Python syntax via py_compile.

    # Uses sys.executable so the same interpreter that runs the orchestrator
    # is used for the check. This ensures the Python version matches.
    # Faster than run_tests — use this as a first check after writing a file.
    """

    def execute(params: dict) -> dict:
        path = params["path"]
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True,
                text=True,
                timeout=_SYNTAX_TIMEOUT,
                shell=False,
            )
            return CheckSyntaxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                valid=proc.returncode == 0,
                path=path,
            ).model_dump()
        except subprocess.TimeoutExpired:
            return {"error": f"py_compile timed out after {_SYNTAX_TIMEOUT}s", "valid": False, "path": path}
        except Exception as exc:
            return {"error": str(exc), "valid": False, "path": path}

    return Tool(
        name="check_syntax",
        description=(
            "Check a Python file for syntax errors using py_compile. "
            "Faster than run_tests — use this immediately after writing or editing a file. "
            "Returns valid=true if the file parses without errors."
        ),
        parameters_schema=model_to_json_schema(CheckSyntaxParams),
        risk_level="low",
        _execute=execute,
        params_model=CheckSyntaxParams,
    )
