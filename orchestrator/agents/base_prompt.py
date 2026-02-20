# Purpose: System prompt construction shared by all agent types.
# Relationships: Called by core/agent_invoker.py at prompt assembly time.
#               Context manager (Phase 2) will inject additional context
#               between the base prompt and the task description.

import json

_BASE = """\
You are an autonomous agent operating within a secure orchestration system.

RULES:
- You may only take actions by emitting tool calls in the exact format shown below.
- Think through the task step by step before acting.
- Never fabricate file contents or command results. Use tools to get real data.
- Stop when the task is complete or no further tool calls are useful.
- One tool call per response. After each result is returned you may reason and
  emit the next call.

TOOL CALL FORMAT:
To invoke a tool, emit exactly this block (no other content on those lines):

<tool_call>
{"tool": "<tool_name>", "parameters": {<json parameters>}}
</tool_call>

When your task is complete, write your final answer with no tool call block.
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
