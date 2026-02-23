# [SAFETY-CRITICAL] This module controls all filesystem access from agents.
# The path allowlist check in _is_allowed() is the sole enforcement point
# for the allowed_read and allowed_write config constraints.
# Do not weaken or remove the path check. Do not modify as part of
# autonomous improvement tasks.
#
# Purpose: Filesystem tools — read_file, write_file, list_directory.
# Relationships: Registers tools into tools/registry.py via make_* factories;
#               allowed paths come from config.yaml loaded in main.py.

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .registry import Tool, model_to_json_schema


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


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the file to read.")


class ReadFileResult(BaseModel):
    content: str = Field(description="Full UTF-8 text content of the file.")
    path: str = Field(description="Resolved path that was read.")


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
            return ReadFileResult(content=content, path=path).model_dump()
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
        parameters_schema=model_to_json_schema(ReadFileParams),
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
        params_model=ReadFileParams,
    )


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the file to write.")
    content: str = Field(description="Text content to write.")


class WriteFileResult(BaseModel):
    success: bool = Field(description="True if the file was written successfully.")
    path: str = Field(description="Path that was written.")
    bytes_written: int = Field(description="Number of bytes written.")


def make_write_file_tool(allowed_paths: list[str]) -> Tool:
    """Return a write_file Tool restricted to the given allowed_paths."""

    def execute(params: dict) -> dict:
        path = params["path"]
        if not _is_allowed(path, allowed_paths):
            return {"error": f"Path not in allowed_write list: {path!r}"}
        try:
            p = Path(path)
            # Create parent directories if they do not exist, matching the
            # behaviour a human would expect when writing to a new path.
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(params["content"], encoding="utf-8")
            return WriteFileResult(
                success=True,
                path=path,
                bytes_written=len(params["content"]),
            ).model_dump()
        except Exception as exc:
            return {"error": str(exc)}

    return Tool(
        name="write_file",
        description=(
            "Write text content to a file. Creates parent directories as needed. "
            "Only paths within the configured allowed_write list are writable. "
            "Overwrites the file if it already exists."
        ),
        parameters_schema=model_to_json_schema(WriteFileParams),
        risk_level="medium",
        allowed_agents=["dev_agent"],
        _execute=execute,
        params_model=WriteFileParams,
    )


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class ListDirectoryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the directory to list.")


class DirectoryEntry(BaseModel):
    name: str = Field(description="Entry name (filename or directory name).")
    type: str = Field(description="'file' or 'dir'.")


class ListDirectoryResult(BaseModel):
    path: str = Field(description="The directory that was listed.")
    entries: list[dict[str, Any]] = Field(
        description="Directory entries sorted by name, directories first.",
    )


def make_list_directory_tool(allowed_paths: list[str]) -> Tool:
    """Return a list_directory Tool restricted to the given allowed_paths."""

    def execute(params: dict) -> dict:
        path = params["path"]
        if not _is_allowed(path, allowed_paths):
            return {"error": f"Path not in allowed_read list: {path!r}"}
        try:
            p = Path(path)
            if not p.is_dir():
                return {"error": f"Not a directory: {path!r}"}
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
            entry_dicts = [
                DirectoryEntry(name=e.name, type="dir" if e.is_dir() else "file").model_dump()
                for e in entries
            ]
            return ListDirectoryResult(path=path, entries=entry_dicts).model_dump()
        except Exception as exc:
            return {"error": str(exc)}

    return Tool(
        name="list_directory",
        description=(
            "List the contents of a directory. "
            "Returns entry names and types (file or dir). "
            "Only paths within the configured allowed_read list are accessible."
        ),
        parameters_schema=model_to_json_schema(ListDirectoryParams),
        risk_level="low",
        allowed_agents=["task_agent", "dev_agent"],
        _execute=execute,
        params_model=ListDirectoryParams,
    )
