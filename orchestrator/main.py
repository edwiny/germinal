# Purpose: Phase 0 entry point. Runs a single hardcoded task to prove the
#          end-to-end pipeline works: prompt assembly → model call → tool
#          execution → DB logging.
# Relationships: Wires together storage/db.py, tools/*, core/agent_invoker.py.
#               In Phase 1 this is replaced by an event loop.

import os
import sys

import yaml

from core.agent_invoker import invoke
from storage.db import init_db
from tools.filesystem import make_read_file_tool
from tools.notify import make_notify_user_tool
from tools.registry import ToolRegistry

# Phase 0 hardcoded task. The agent must read the architecture doc and
# summarise it — this exercises the read_file tool and proves the full loop.
_TASK = (
    "Read the file ../ARCHITECTURE.md and write a concise summary of: "
    "(1) what this orchestration system is, "
    "(2) what Phase 0 is building, and "
    "(3) what the done condition for Phase 0 is. "
    "Then use notify_user to deliver the summary to the user."
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_registry(config: dict) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_read_file_tool(config["paths"]["allowed_read"]))
    registry.register(make_notify_user_tool())
    return registry


def main() -> None:
    config = load_config()

    db_path = config["paths"]["db"]
    init_db(db_path)

    registry = build_registry(config)
    model = config["models"]["default_local"]

    print(f"Model : {model}")
    print(f"DB    : {db_path}")
    print(f"Task  : {_TASK[:80]}...")
    print()

    result = invoke(
        task_description=_TASK,
        agent_type="task_agent",
        model=model,
        registry=registry,
        db_path=db_path,
    )

    print()
    print("=" * 60)
    print(f"Status       : {result['status']}")
    print(f"Invocation ID: {result['invocation_id']}")
    print(f"Tool calls   : {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"  [{tc['tool']}] params={list(tc['parameters'].keys())}")
    print()
    print("Final response:")
    print(result["response"])


if __name__ == "__main__":
    main()
