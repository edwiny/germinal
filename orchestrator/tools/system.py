"""
System information tools for readonly operations on the host environment.

These tools provide basic system monitoring capabilities using built-in Python
modules. For more advanced system monitoring, consider adding psutil as a
dependency.
"""

import multiprocessing
import os
import platform
import sys
from typing import Dict, Any

from pydantic import BaseModel, ConfigDict

from .registry import Tool, model_to_json_schema


# ---------------------------------------------------------------------------


class ShowOSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShowOSResult(BaseModel):
    system: str
    node: str
    release: str
    version: str
    machine: str
    processor: str
    python_version: str
    python_implementation: str


def make_show_os_tool() -> Tool:
    """
    Return a show_os Tool that returns basic operating system information.

    Uses the built-in platform module to gather system details.
    """

    def execute(params: dict) -> dict:
        try:
            return ShowOSResult(
                system=platform.system(),
                node=platform.node(),
                release=platform.release(),
                version=platform.version(),
                machine=platform.machine(),
                processor=platform.processor() or "Unknown",
                python_version=platform.python_version(),
                python_implementation=platform.python_implementation(),
            ).model_dump()
        except Exception as exc:
            return {"error": f"Failed to get OS information: {exc}"}

    return Tool(
        name="show_os",
        description=(
            "Show operating system information. "
            "Returns details about the current OS platform, version, and architecture."
        ),
        parameters_schema=model_to_json_schema(ShowOSParams),
        risk_level="low",
        _execute=execute,
        params_model=ShowOSParams,
    )


# ---------------------------------------------------------------------------


class ShowHardwareParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShowHardwareResult(BaseModel):
    cpu_count: int
    cpu_count_logical: int
    memory_total_mb: int | None
    memory_available_mb: int | None
    system_info: Dict[str, Any]


def make_show_hardware_tool() -> Tool:
    """
    Return a show_hardware Tool that returns basic hardware information.

    Uses built-in modules to gather CPU and memory information where available.
    Note: Detailed hardware monitoring requires psutil (not included).
    """

    def execute(params: dict) -> dict:
        try:
            # CPU information
            cpu_count = multiprocessing.cpu_count()
            cpu_logical = os.cpu_count()

            # Memory information (limited on some platforms)
            memory_total = None
            memory_available = None

            # Try to get memory info using os.sysconf (Unix-like systems)
            try:
                if hasattr(os, 'sysconf'):
                    # Total memory in bytes
                    total_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                    memory_total = total_bytes // (1024 * 1024)  # Convert to MB

                    # Available memory (approximate)
                    avail_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
                    memory_available = avail_bytes // (1024 * 1024)  # Convert to MB
            except (OSError, AttributeError):
                # Memory info not available on this platform
                pass

            # System information
            system_info = {
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "uname": list(platform.uname()) if hasattr(platform, 'uname') else None,
            }

            return ShowHardwareResult(
                cpu_count=cpu_count or 0,
                cpu_count_logical=cpu_logical or cpu_count or 0,
                memory_total_mb=memory_total,
                memory_available_mb=memory_available,
                system_info=system_info,
            ).model_dump()
        except Exception as exc:
            return {"error": f"Failed to get hardware information: {exc}"}

    return Tool(
        name="show_hardware",
        description=(
            "Return basic hardware information including CPU count, memory details, "
            "and system architecture. Note: Detailed hardware monitoring requires psutil."
        ),
        parameters_schema=model_to_json_schema(ShowHardwareParams),
        risk_level="low",
        _execute=execute,
        params_model=ShowHardwareParams,
    )


# ---------------------------------------------------------------------------


class ShowPSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShowPSResult(BaseModel):
    current_process: Dict[str, Any]
    parent_process: Dict[str, Any] | None
    system_info: Dict[str, Any]


def make_show_ps_tool() -> Tool:
    """
    Return a show_ps Tool that returns basic process information.

    Uses built-in modules to gather current process and basic system information.
    Note: Full process listing requires psutil (not included).
    """

    def execute(params: dict) -> dict:
        try:
            # Current process information
            current_pid = os.getpid()
            try:
                current_ppid = os.getppid() if hasattr(os, 'getppid') else None
            except AttributeError:
                # Some platforms have getppid but it raises AttributeError
                current_ppid = None

            current_process = {
                "pid": current_pid,
                "ppid": current_ppid,
                "uid": os.getuid() if hasattr(os, 'getuid') else None,
                "gid": os.getgid() if hasattr(os, 'getgid') else None,
                "cwd": os.getcwd(),
                "command_line": sys.argv,
                "python_executable": sys.executable,
                "python_version": sys.version,
            }

            # Parent process information (limited)
            parent_process = None
            if current_ppid:
                parent_process = {
                    "pid": current_ppid,
                    "note": "Limited information available with built-in modules. Use psutil for full process details."
                }

            # System process information (very basic)
            system_info = {
                "process_count_note": "Full process listing requires psutil. Only current process shown.",
                "current_process_count": 1,  # We only know about current process
                "platform_supports_process_listing": hasattr(os, 'listdir') and os.path.exists('/proc') if os.name == 'posix' else False,
            }

            return ShowPSResult(
                current_process=current_process,
                parent_process=parent_process,
                system_info=system_info,
            ).model_dump()
        except Exception as exc:
            return {"error": f"Failed to get process information: {exc}"}

    return Tool(
        name="show_ps",
        description=(
            "Return basic process information for the current process and system. "
            "Note: Full process listing and detailed process information requires psutil."
        ),
        parameters_schema=model_to_json_schema(ShowPSParams),
        risk_level="low",
        _execute=execute,
        params_model=ShowPSParams,
    )
