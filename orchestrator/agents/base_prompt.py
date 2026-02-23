# Purpose: System prompt construction shared by all agent types.
# Relationships: Called by core/agent_invoker.py at prompt assembly time.
#               Context manager (Phase 2) will inject additional context
#               between the base prompt and the task description.

import json

_BASE = """\
You are a autonomous agent with a set of tools available to assist you with helping the user.

RULES:
- You may only take actions by invoking tools via the tool_call field of your response.
- Think through the task step by step before acting.
- Never fabricate file contents or command results. Use tools to get real data.
- Stop when the task is complete or no further tool calls are useful.
- One tool call per response. After each result is returned you may reason and
  emit the next call.

RESPONSE FORMAT:
Every response must be a JSON object with exactly these fields:
  - "reasoning": your explanation of what you are doing or have concluded (required)
  - "tool_call": the tool to invoke next, or null when no tool is needed (optional)

A tool_call has the form:
  {"tool": "<tool_name>", "parameters": {<json parameters>}}

When your task is complete, set tool_call to null and put your final answer in reasoning.
"""


def build_system_prompt(tool_schemas: list[dict]) -> str:
    """
    Assemble the full system prompt: base instructions + tool catalogue.

    # Tool schemas are appended as JSON so the agent has exact parameter
    # names and types. Prose descriptions alone are insufficient for reliable
    # structured output from smaller local models.
    """
    tools_section = "\nAVAILABLE TOOLS:\n" + json.dumps(tool_schemas, indent=2) + "\n"
    return _BASE + tools_section
