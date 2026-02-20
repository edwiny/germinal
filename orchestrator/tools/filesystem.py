# [SAFETY-CRITICAL] This module controls all filesystem access from agents.
# The path allowlist check in _is_allowed() is the sole enforcement point
# for the allowed_read and allowed_write config constraints.
# Do not weaken or remove the path check. Do not modify as part of
# autonomous improvement tasks.
#
# Purpose: Filesystem tools — read_file (Phase 0). write_file added Phase 1+.
# Relationships: Registers tools into tools/registry.py via make_* factories;
#               allowed paths come from config.yaml loaded in main.py.

from pathlib import Path

from tools.registry import Tool


def _is_allowed(path: str, allowed_paths: list[str]) -> bool:
    """
    Return True only if path resolves to within one of allowed_paths.

    # We resolve to absolute paths to defeat directory traversal attempts
    # (e.g. "../../etc/passwd"). expanduser() handles ~ in config values.
    # Do NOT simplify this to a string prefix check — that breaks on
    # symlinks and relative paths.
    """
    resolved = Path(path).resolve()
    for allowed in allowed_paths:
        allowed_resolved = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return True
        except ValueError:
            continue
    return False


def make_read_file_tool(allowed_paths: list[str]) -> Tool:
    """Return a read_file Tool restricted to the given allowed_paths."""

    def execute(params: dict) -> dict:
        path = params["path"]
        if not _is_allowed(path, allowed_paths):
            # Return an error dict rather than raising so the agent sees
            # a structured response and can try a different path.
            return {"error": f"Path not in allowed_read list: {path!r}"}
        try:
            content = Path(path).read_text(encoding="utf-8")
            return {"content": content, "path": path}
        except FileNotFoundError:
            return {"error": f"File not found: {path!r}"}
        except Exception as exc:
            return {"error": str(exc)}

    return Tool(
        name="read_file",
        description=(
            "Read the full text content of a file. "
            "Only paths within the configured allowed_read list are accessible."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
    )
