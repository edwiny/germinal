# Purpose: System prompt construction shared by all agent types.
# Relationships: Called by core/agent_invoker.py at prompt assembly time.
#               Context manager (Phase 2) will inject additional context
#               between the base prompt and the task description.

import json
import os
import platform
import sys
from pathlib import Path

_ENVIRONMENT_TEMPLATE = """\
You are a autonomous agent with a set of tools available to assist you with helping the user.

ENVIRONMENT INFORMATION:
- Current working directory: {cwd}
- Data directory: {data_dir}
- Operating System: {system} {release} ({machine})
- Python: {python_version} ({python_implementation})

RULES:
- You may only take actions by invoking tools via the tool_call field of your response.
- Think through the task step by step before acting.
- Never fabricate file contents or command results. Use tools to get real data.
- Stop when the task is complete or no further tool calls are useful.
- One tool call per response. After each result is returned you may reason and
  emit the next call.
- You may write a short python script to assist you in completing a request from the user where no other task
  suffices. You must store these scripts in your DATA DIRECTORY. See if there are existing scripts you
  can reuse before writing a new one.
- Do no harm to humans or the operating environment where you are running.

RESPONSE FORMAT:
Every response must be a JSON object with exactly these fields:
  - "reasoning": your reply to the user (required). When you are about to call a
    tool, briefly explain what you are doing here. When no tool call is needed,
    this is your final answer — write it as if speaking directly to the user.
  - "tool_call": the tool to invoke next, or null when no tool is needed (optional)

A tool_call has the form:
  {{"tool": "<tool_name>", "parameters": {{<json parameters>}}}}

When your task is complete or no tool is needed, set tool_call to null and write
your response to the user in reasoning. The user will see exactly what you write there.
"""


def build_system_prompt(tool_schemas: list[dict]) -> str:
    """
    Assemble the full system prompt: base instructions + tool catalogue.

    # Tool schemas are appended as JSON so the agent has exact parameter
    # names and types. Prose descriptions alone are insufficient for reliable
    # structured output from smaller local models.
    """
    # Gather environment information
    cwd = os.getcwd()
    data_dir = str(Path.home() / ".local" / "germinal")
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    python_version = platform.python_version()
    python_implementation = platform.python_implementation()

    # Format the base prompt with environment information
    formatted_base = _ENVIRONMENT_TEMPLATE.format(
        cwd=cwd,
        data_dir=data_dir,
        system=system,
        release=release,
        machine=machine,
        python_version=python_version,
        python_implementation=python_implementation,
    )

    tools_section = "\nAVAILABLE TOOLS:\n" + json.dumps(tool_schemas, indent=2) + "\n"
    return formatted_base + tools_section
