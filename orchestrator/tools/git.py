# Purpose: Git operation tools: git_status, git_commit, git_branch, git_rollback.
# Relationships: Registered into tools/registry.py via make_* factories;
#               called by dev_agent through core/agent_invoker.py.

# All tools call subprocess git with a fixed argv list (shell=False) to prevent
# injection. git_rollback is high-risk and requires human approval before execution.

import subprocess

from tools.registry import Tool

_GIT_TIMEOUT = 60  # seconds; git operations on a local repo should never take longer


def _git(args: list[str]) -> dict:
    """
    Run 'git <args>' and return stdout, stderr, and returncode.

    Uses shell=False unconditionally. Never call this with user-supplied
    strings that have not been validated by the tool's parameter schema.
    """
    try:
        proc = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            shell=False,
        )
        return {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "git timed out", "returncode": -1}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "git not found in PATH", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def make_git_status_tool() -> Tool:
    """Return a git_status tool that reports current repo state."""

    def execute(_params: dict) -> dict:
        status = _git(["status", "--short"])
        diff_stat = _git(["diff", "--stat", "HEAD"])
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        return {
            "branch": branch["stdout"],
            "status": status["stdout"],
            "diff_stat": diff_stat["stdout"],
            "returncode": status["returncode"],
        }

    return Tool(
        name="git_status",
        description=(
            "Return the current git branch, working tree status (short format), "
            "and a diff stat against HEAD."
        ),
        parameters_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        risk_level="low",
        allowed_agents=["dev_agent"],
        _execute=execute,
    )


def make_git_commit_tool() -> Tool:
    """Return a git_commit tool that commits currently staged changes."""

    def execute(params: dict) -> dict:
        message = params["message"]
        result = _git(["commit", "-m", message])
        return {
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
            "success": result["returncode"] == 0,
        }

    return Tool(
        name="git_commit",
        description=(
            "Commit currently staged changes with the given message. "
            "Stage files first with shell_run(['git', 'add', '<path>']) before calling this. "
            "Returns success=false if there is nothing staged or another error occurs."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message.",
                    "minLength": 1,
                }
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        risk_level="medium",
        allowed_agents=["dev_agent"],
        _execute=execute,
    )


def make_git_branch_tool() -> Tool:
    """Return a git_branch tool that creates or switches branches."""

    def execute(params: dict) -> dict:
        name = params["name"]
        create = params.get("create", False)

        if create:
            result = _git(["checkout", "-b", name])
        else:
            result = _git(["checkout", name])

        return {
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
            "success": result["returncode"] == 0,
        }

    return Tool(
        name="git_branch",
        description=(
            "Switch to an existing branch or create and switch to a new one. "
            "Set create=true to create a new branch from the current HEAD."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Branch name to switch to or create.",
                    "minLength": 1,
                },
                "create": {
                    "type": "boolean",
                    "description": "If true, create the branch before switching.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        risk_level="medium",
        allowed_agents=["dev_agent"],
        _execute=execute,
    )


def make_git_rollback_tool() -> Tool:
    """Return a git_rollback tool that reverts to a previous commit."""

    def execute(params: dict) -> dict:
        # [SAFETY-CRITICAL] git_rollback is high-risk. The approval gate in
        # agent_invoker.py blocks execution until the human approves.
        # Destructive: any uncommitted changes are lost after a hard reset.
        to_commit = params["to_commit"]
        reason = params.get("reason", "")

        result = _git(["reset", "--hard", to_commit])
        return {
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
            "success": result["returncode"] == 0,
            "rolled_back_to": to_commit,
            "reason": reason,
        }

    return Tool(
        name="git_rollback",
        description=(
            "Revert the working tree to a previous commit using git reset --hard. "
            "DESTRUCTIVE: uncommitted changes are lost. Always requires human approval. "
            "Provide a reason so the approval prompt is informative."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "to_commit": {
                    "type": "string",
                    "description": "Commit hash or ref to reset to.",
                    "minLength": 1,
                },
                "reason": {
                    "type": "string",
                    "description": "Why this rollback is needed (shown in the approval prompt).",
                },
            },
            "required": ["to_commit"],
            "additionalProperties": False,
        },
        risk_level="high",
        allowed_agents=["dev_agent"],
        _execute=execute,
    )
