"""
Tests for system information tools.
"""

import platform
import sys
from unittest.mock import patch

from orchestrator.tools.system import (
    make_show_os_tool,
    make_show_hardware_tool,
    make_show_ps_tool,
)


class TestShowOSTool:
    """Test the show_os tool."""

    def test_show_os_returns_system_info(self):
        """Test that show_os returns expected system information."""
        tool = make_show_os_tool()

        result = tool.execute({})

        assert "system" in result
        assert "node" in result
        assert "release" in result
        assert "version" in result
        assert "machine" in result
        assert "processor" in result
        assert "python_version" in result
        assert "python_implementation" in result

        # Verify some basic expectations
        assert isinstance(result["system"], str)
        assert isinstance(result["python_version"], str)
        assert result["python_implementation"] in ["CPython", "PyPy", "Jython", "IronPython"]

    def test_show_os_handles_platform_errors(self):
        """Test that show_os handles platform module errors gracefully."""
        tool = make_show_os_tool()

        # Mock platform.system to raise an exception
        with patch('platform.system', side_effect=Exception("Platform error")):
            result = tool.execute({})

            assert "error" in result
            assert "Failed to get OS information" in result["error"]


class TestShowHardwareTool:
    """Test the show_hardware tool."""

    def test_show_hardware_returns_basic_info(self):
        """Test that show_hardware returns basic hardware information."""
        tool = make_show_hardware_tool()

        result = tool.execute({})

        assert "cpu_count" in result
        assert "cpu_count_logical" in result
        assert "system_info" in result

        # CPU counts should be positive integers or None
        assert result["cpu_count"] > 0 or result["cpu_count"] == 0
        assert result["cpu_count_logical"] >= result["cpu_count"] or result["cpu_count_logical"] == 0

        # System info should contain expected keys
        system_info = result["system_info"]
        assert "platform" in system_info
        assert "architecture" in system_info

    def test_show_hardware_memory_info_handling(self):
        """Test that memory info is handled appropriately on different platforms."""
        tool = make_show_hardware_tool()

        result = tool.execute({})

        # Memory info should be either None or a number
        assert result["memory_total_mb"] is None or isinstance(result["memory_total_mb"], int)
        assert result["memory_available_mb"] is None or isinstance(result["memory_available_mb"], int)

    def test_show_hardware_handles_errors(self):
        """Test that show_hardware handles errors gracefully."""
        tool = make_show_hardware_tool()

        # Mock multiprocessing.cpu_count to raise an exception
        with patch('multiprocessing.cpu_count', side_effect=Exception("CPU error")):
            result = tool.execute({})

            assert "error" in result
            assert "Failed to get hardware information" in result["error"]


class TestShowPSTool:
    """Test the show_ps tool."""

    def test_show_ps_returns_process_info(self):
        """Test that show_ps returns current process information."""
        tool = make_show_ps_tool()

        result = tool.execute({})

        assert "current_process" in result
        assert "system_info" in result

        current_process = result["current_process"]
        assert "pid" in current_process
        assert "command_line" in current_process
        assert "python_executable" in current_process
        assert "python_version" in current_process

        # PID should be a positive integer
        assert isinstance(current_process["pid"], int)
        assert current_process["pid"] > 0

        # Command line should contain the current script
        assert len(current_process["command_line"]) > 0

    def test_show_ps_parent_process_info(self):
        """Test parent process information when available."""
        tool = make_show_ps_tool()

        # Mock getppid to return a parent PID
        with patch('os.getppid', return_value=12345):
            result = tool.execute({})

            assert "parent_process" in result
            assert result["parent_process"] is not None
            assert result["parent_process"]["pid"] == 12345

    def test_show_ps_no_parent_process(self):
        """Test when parent process information is not available."""
        tool = make_show_ps_tool()

        # Mock getppid to not be available (Windows)
        with patch('os.getppid', side_effect=AttributeError("not available")):
            result = tool.execute({})

            current_process = result["current_process"]
            assert current_process["ppid"] is None

    def test_show_ps_system_info(self):
        """Test system information in show_ps output."""
        tool = make_show_ps_tool()

        result = tool.execute({})

        system_info = result["system_info"]
        assert "process_count_note" in system_info
        assert "current_process_count" in system_info
        assert "platform_supports_process_listing" in system_info

        # Should note that full process listing requires psutil
        assert "psutil" in system_info["process_count_note"]
        assert system_info["current_process_count"] == 1

    def test_show_ps_handles_errors(self):
        """Test that show_ps handles errors gracefully."""
        tool = make_show_ps_tool()

        # Mock os.getcwd to raise an exception
        with patch('os.getcwd', side_effect=Exception("CWD error")):
            result = tool.execute({})

            assert "error" in result
            assert "Failed to get process information" in result["error"]