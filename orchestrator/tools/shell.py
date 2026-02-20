# [SAFETY-CRITICAL] This module executes shell commands on behalf of agents.
# The allowlist check is the sole enforcement point preventing arbitrary
# command execution. Never use shell=True. Never add wildcards to the
# allowlist. Do not modify as part of autonomous improvement tasks.
#
# Purpose: shell_run (allowlisted shell execution) and run_tests (pytest wrapper).
# Relationships: Registered into tools/registry.py; allowlist from config.yaml.

import subprocess

from tools.registry import Tool

# Timeout for shell_run commands in seconds. Long-running commands should
# use a dedicated tool rather than increasing this limit.
_SHELL_TIMEOUT = 120

# Timeout for the test suite specifically. Tests can run longer than a
# normal shell command — 300 seconds gives CI-style test runs room to breathe.
_TEST_TIMEOUT = 300


def _is_allowed_command(command: list[str], allowlist: list[str]) -> bool:
    """
    Return True only if the first token of command is in the allowlist.

    # We check command[0] (the executable name only, not its path) against
    # the allowlist. This means 'pytest' is allowed but '/tmp/evil/pytest'
    # would also pass if the name is 'pytest'. To lock down further, the
    # allowlist could be changed to absolute paths — but at current threat
    # model (agents running as the user, on a trusted host) name-matching
    # is sufficient. Revisit if the system gains network-facing inputs.
    """
    if not command:
        return False
    return command[0] in allowlist


def make_shell_run_tool(allowlist: list[str]) -> Tool:
    """
    Return a shell_run Tool restricted to commands in the allowlist.

    The command parameter must be a list of strings (argv-style). This
    prevents shell injection via argument strings — we never pass
    the command to a shell interpreter.
    """

    def execute(params: dict) -> dict:
        command = params["command"]
        # Accept a list or a whitespace-split string for convenience,
        # but never pass through a shell. Both paths reach subprocess.run()
        # with shell=False.
        if isinstance(command, str):
            command = command.split()
        if not _is_allowed_command(command, allowlist):
            return {
                "error": (
                    f"Command {command[0]!r} is not in the shell allowlist. "
                    f"Allowed commands: {allowlist}"
                )
            }
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_SHELL_TIMEOUT,
                # shell=False is the default but stated explicitly as a reminder
                # that this must never be changed to shell=True.
                shell=False,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {_SHELL_TIMEOUT}s"}
        except FileNotFoundError:
            return {"error": f"Executable not found: {command[0]!r}"}
        except Exception as exc:
            return {"error": str(exc)}

    return Tool(
        name="shell_run",
        description=(
            "Run an allowlisted shell command. The command must be provided as a list "
            "of strings (e.g. ['git', 'status']). Only commands whose base name appears "
            "in the shell_allowlist config key are permitted."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "command": {
                    "oneOf": [
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Command and arguments as a list.",
                        },
                        {
                            "type": "string",
                            "description": "Command and arguments as a whitespace-separated string.",
                        },
                    ],
                    "description": "Command to run.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        risk_level="high",
        allowed_agents=["dev_agent"],
        _execute=execute,
    )


def make_run_tests_tool() -> Tool:
    """
    Return a run_tests Tool that executes the pytest test suite.

    Lower risk than shell_run because the command is fixed (pytest) and
    the agent cannot supply arbitrary arguments that change what executes.
    The optional path parameter restricts pytest to a subdirectory.
    """

    def execute(params: dict) -> dict:
        command = ["pytest"]
        if params.get("path"):
            command.append(params["path"])
        if params.get("verbose"):
            command.append("-v")
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_TEST_TIMEOUT,
                shell=False,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "passed": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Tests timed out after {_TEST_TIMEOUT}s", "passed": False}
        except FileNotFoundError:
            return {"error": "pytest not found — is the virtualenv active?", "passed": False}
        except Exception as exc:
            return {"error": str(exc), "passed": False}

    return Tool(
        name="run_tests",
        description=(
            "Run the pytest test suite and return the output. "
            "Optionally restrict to a path or enable verbose output."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional path to pass to pytest (file or directory).",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "If true, pass -v to pytest for verbose output.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
    )
