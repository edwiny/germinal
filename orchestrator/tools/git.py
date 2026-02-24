# Purpose: Git operation tools: git_status, git_commit, git_add, git_branch,
#          git_rollback, git_diff, git_list_branches, git_log.
# Relationships: Registered into tools/registry.py via make_* factories;
#               called by dev_agent (and task_agent for git_list_branches)
#               through core/agent_invoker.py.

# All tools call subprocess git with a fixed argv list (shell=False) to prevent
# injection. git_rollback is high-risk and requires human approval before execution.

import subprocess

from pydantic import BaseModel, ConfigDict, Field

from .registry import Tool, model_to_json_schema

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


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------

class GitStatusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # No parameters — git_status always reports current repo state.


class GitStatusResult(BaseModel):
    branch: str = Field(description="Current branch name.")
    status: str = Field(description="Working tree status in short format.")
    diff_stat: str = Field(description="Diff stat against HEAD.")
    returncode: int = Field(description="Return code from git status.")


def make_git_status_tool() -> Tool:
    """Return a git_status tool that reports current repo state."""

    def execute(_params: dict) -> dict:
        status = _git(["status", "--short"])
        diff_stat = _git(["diff", "--stat", "HEAD"])
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        return GitStatusResult(
            branch=branch["stdout"],
            status=status["stdout"],
            diff_stat=diff_stat["stdout"],
            returncode=status["returncode"],
        ).model_dump()

    return Tool(
        name="git_status",
        description=(
            "Return the current git branch, working tree status (short format), "
            "and a diff stat against HEAD."
        ),
        parameters_schema=model_to_json_schema(GitStatusParams),
        risk_level="low",
        _execute=execute,
        params_model=GitStatusParams,
    )


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------

class GitCommitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, description="Commit message.")


class GitCommitResult(BaseModel):
    stdout: str = Field(description="Standard output from git commit.")
    stderr: str = Field(description="Standard error from git commit.")
    returncode: int = Field(description="Return code from git commit.")
    success: bool = Field(description="True if the commit succeeded (returncode 0).")


def make_git_commit_tool() -> Tool:
    """Return a git_commit tool that commits currently staged changes."""

    def execute(params: dict) -> dict:
        message = params["message"]
        result = _git(["commit", "-m", message])
        return GitCommitResult(
            stdout=result["stdout"],
            stderr=result["stderr"],
            returncode=result["returncode"],
            success=result["returncode"] == 0,
        ).model_dump()

    return Tool(
        name="git_commit",
        description=(
            "Commit currently staged changes with the given message. "
            "Stage files first with git_add before calling this. "
            "Returns success=false if there is nothing staged or another error occurs."
        ),
        parameters_schema=model_to_json_schema(GitCommitParams),
        risk_level="medium",
        _execute=execute,
        params_model=GitCommitParams,
    )


# ---------------------------------------------------------------------------
# git_branch
# ---------------------------------------------------------------------------

class GitBranchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Branch name to switch to or create.")
    create: bool = Field(
        default=False,
        description="If true, create the branch before switching.",
    )


class GitBranchResult(BaseModel):
    stdout: str = Field(description="Standard output from git checkout.")
    stderr: str = Field(description="Standard error from git checkout.")
    returncode: int = Field(description="Return code from git checkout.")
    success: bool = Field(description="True if the branch operation succeeded.")


def make_git_branch_tool() -> Tool:
    """Return a git_branch tool that creates or switches branches."""

    def execute(params: dict) -> dict:
        name = params["name"]
        create = params.get("create", False)

        if create:
            result = _git(["checkout", "-b", name])
        else:
            result = _git(["checkout", name])

        return GitBranchResult(
            stdout=result["stdout"],
            stderr=result["stderr"],
            returncode=result["returncode"],
            success=result["returncode"] == 0,
        ).model_dump()

    return Tool(
        name="git_branch",
        description=(
            "Switch to an existing branch or create and switch to a new one. "
            "Set create=true to create a new branch from the current HEAD."
        ),
        parameters_schema=model_to_json_schema(GitBranchParams),
        risk_level="medium",
        _execute=execute,
        params_model=GitBranchParams,
    )


# ---------------------------------------------------------------------------
# git_rollback
# ---------------------------------------------------------------------------

class GitRollbackParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_commit: str = Field(
        min_length=1,
        description="Commit hash or ref to reset to.",
    )
    reason: str = Field(
        default="",
        description="Why this rollback is needed (shown in the approval prompt).",
    )


class GitRollbackResult(BaseModel):
    stdout: str = Field(description="Standard output from git reset.")
    stderr: str = Field(description="Standard error from git reset.")
    returncode: int = Field(description="Return code from git reset.")
    success: bool = Field(description="True if the rollback succeeded.")
    rolled_back_to: str = Field(description="The commit ref that was reset to.")
    reason: str = Field(description="The stated reason for the rollback.")


def make_git_rollback_tool() -> Tool:
    """Return a git_rollback tool that reverts to a previous commit."""

    def execute(params: dict) -> dict:
        # [SAFETY-CRITICAL] git_rollback is high-risk. The approval gate in
        # agent_invoker.py blocks execution until the human approves.
        # Destructive: any uncommitted changes are lost after a hard reset.
        to_commit = params["to_commit"]
        reason = params.get("reason", "")

        result = _git(["reset", "--hard", to_commit])
        return GitRollbackResult(
            stdout=result["stdout"],
            stderr=result["stderr"],
            returncode=result["returncode"],
            success=result["returncode"] == 0,
            rolled_back_to=to_commit,
            reason=reason,
        ).model_dump()

    return Tool(
        name="git_rollback",
        description=(
            "Revert the working tree to a previous commit using git reset --hard. "
            "DESTRUCTIVE: uncommitted changes are lost. Always requires human approval. "
            "Provide a reason so the approval prompt is informative."
        ),
        parameters_schema=model_to_json_schema(GitRollbackParams),
        risk_level="high",
        _execute=execute,
        params_model=GitRollbackParams,
    )


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------

class GitDiffParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # No parameters — always diffs working tree against HEAD.


class GitDiffResult(BaseModel):
    diff: str = Field(description="Full diff output from git diff HEAD.")
    returncode: int = Field(description="Return code from git diff.")


def make_git_diff_tool() -> Tool:
    """Return a git_diff tool that shows the full diff against HEAD."""

    def execute(_params: dict) -> dict:
        result = _git(["diff", "HEAD"])
        return GitDiffResult(
            diff=result["stdout"],
            returncode=result["returncode"],
        ).model_dump()

    return Tool(
        name="git_diff",
        description="Show the full diff of working tree changes against HEAD.",
        parameters_schema=model_to_json_schema(GitDiffParams),
        risk_level="low",
        _execute=execute,
        params_model=GitDiffParams,
    )


# ---------------------------------------------------------------------------
# git_add
# ---------------------------------------------------------------------------

class GitAddParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        min_length=1,
        description="File paths to stage.",
    )


class GitAddResult(BaseModel):
    stdout: str = Field(description="Standard output from git add.")
    stderr: str = Field(description="Standard error from git add.")
    returncode: int = Field(description="Return code from git add.")
    success: bool = Field(description="True if staging succeeded (returncode 0).")


def make_git_add_tool() -> Tool:
    """Return a git_add tool that stages files for commit."""

    def execute(params: dict) -> dict:
        paths = params["paths"]
        # The -- separator prevents paths that start with '-' from being
        # mistaken for options. Required when paths come from agent input.
        result = _git(["add", "--"] + paths)
        return GitAddResult(
            stdout=result["stdout"],
            stderr=result["stderr"],
            returncode=result["returncode"],
            success=result["returncode"] == 0,
        ).model_dump()

    return Tool(
        name="git_add",
        description=(
            "Stage one or more files for the next commit. "
            "Provide at least one path. Use git_commit after staging."
        ),
        parameters_schema=model_to_json_schema(GitAddParams),
        risk_level="medium",
        _execute=execute,
        params_model=GitAddParams,
    )


# ---------------------------------------------------------------------------
# git_list_branches
# ---------------------------------------------------------------------------

class GitListBranchesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # No parameters — lists all local and remote branches.


class GitListBranchesResult(BaseModel):
    branches: list[str] = Field(
        description="All local and remote branch names, one per entry, trimmed.",
    )
    current: str = Field(description="The currently checked-out branch name.")
    returncode: int = Field(description="Return code from git branch.")


def make_git_list_branches_tool() -> Tool:
    """Return a git_list_branches tool that lists all branches."""

    def execute(_params: dict) -> dict:
        result = _git(["branch", "-a"])
        branches = []
        current = ""
        for raw_line in result["stdout"].splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("* "):
                current = line[2:].strip()
                branches.append(current)
            else:
                branches.append(line)
        return GitListBranchesResult(
            branches=branches,
            current=current,
            returncode=result["returncode"],
        ).model_dump()

    return Tool(
        name="git_list_branches",
        description=(
            "List all local and remote branches. "
            "Branches named dev-agent/* indicate work ready for human review."
        ),
        parameters_schema=model_to_json_schema(GitListBranchesParams),
        risk_level="low",
        _execute=execute,
        params_model=GitListBranchesParams,
    )


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------

class GitLogParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recent commits to return.",
    )


class GitLogResult(BaseModel):
    log: str = Field(description="Recent commit history in oneline format.")
    returncode: int = Field(description="Return code from git log.")


def make_git_log_tool() -> Tool:
    """Return a git_log tool that shows recent commit history."""

    def execute(params: dict) -> dict:
        n = params.get("n", 10)
        result = _git(["log", "--oneline", f"-{n}"])
        return GitLogResult(
            log=result["stdout"],
            returncode=result["returncode"],
        ).model_dump()

    return Tool(
        name="git_log",
        description=(
            "Return recent commit history in oneline format. "
            "Use n to control how many commits to return (1–100, default 10)."
        ),
        parameters_schema=model_to_json_schema(GitLogParams),
        risk_level="low",
        _execute=execute,
        params_model=GitLogParams,
    )
