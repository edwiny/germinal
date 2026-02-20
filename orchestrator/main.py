# Purpose: Phase 1 entry point. Runs the event loop: initialises all
#          subsystems, starts the timer adapter, and continuously dequeues
#          events, routes them to agents, and drives invocations.
# Relationships: Wires together storage/db.py, tools/*, core/*, adapters/*,
#               and agents/*. This is the only place all subsystems are
#               assembled — individual modules know nothing about each other.

import os
import signal
import time

import yaml

from adapters.timer import TimerAdapter
from agents import task_agent as task_agent_mod
from core.agent_invoker import invoke
from core.approval_gate import request_approval
from core.event_queue import (
    complete_event,
    dequeue_next_event,
    fail_event,
    reset_stale_events,
)
from core.router import UnroutableEvent, route_event
from storage.db import init_db
from tools.filesystem import (
    make_list_directory_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from tools.git import (
    make_git_branch_tool,
    make_git_commit_tool,
    make_git_rollback_tool,
    make_git_status_tool,
)
from tools.notify import make_notify_user_tool
from tools.registry import ToolRegistry
from tools.shell import make_run_tests_tool, make_shell_run_tool
from tools.tasks import make_read_task_list_tool, make_write_task_tool

# Poll interval when the event queue is empty. 500ms balances responsiveness
# against unnecessary CPU spin. Do not set below ~100ms on SQLite.
_IDLE_SLEEP_SECONDS = 0.5

# Default timer tick interval in seconds. Keep short during bootstrap so
# there is something to observe quickly; tune upward in production.
_TIMER_INTERVAL_SECONDS = 60


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_full_registry(config: dict, db_path: str) -> ToolRegistry:
    """
    Register every available Phase 1 tool into a single registry.

    Agent-specific subsets are extracted from this registry at dispatch time
    using the per-agent make_registry() helpers. Registering everything here
    means main.py is the only place that knows about tool factory functions;
    agents and agent_invoker see only the ToolRegistry interface.
    """
    allowed_read = config["paths"]["allowed_read"]
    allowed_write = config["paths"]["allowed_write"]
    shell_allowlist = config["tools"]["shell_allowlist"]

    registry = ToolRegistry()
    registry.register(make_read_file_tool(allowed_read))
    registry.register(make_write_file_tool(allowed_write))
    registry.register(make_list_directory_tool(allowed_read))
    registry.register(make_notify_user_tool())
    registry.register(make_shell_run_tool(shell_allowlist))
    registry.register(make_run_tests_tool())
    registry.register(make_git_status_tool())
    registry.register(make_git_commit_tool())
    registry.register(make_git_branch_tool())
    registry.register(make_git_rollback_tool())
    registry.register(make_read_task_list_tool(db_path))
    registry.register(make_write_task_tool(db_path))
    return registry


def _select_model(config: dict, model_key: str) -> tuple[str, str | None]:
    """
    Resolve model_key → (litellm model string, api_key | None).

    "default" is resolved via ORCHESTRATOR_MODEL env var then config default.
    Any other key is looked up directly in the model list by name.
    """
    models_cfg = config["models"]

    if model_key == "default":
        resolved_name = os.environ.get("ORCHESTRATOR_MODEL", models_cfg["default"])
    else:
        resolved_name = model_key

    index = {m["name"]: m for m in models_cfg["list"]}
    if resolved_name not in index:
        raise ValueError(
            f"Unknown model name {resolved_name!r}. Valid names: {list(index)}"
        )

    entry = index[resolved_name]
    api_key_env = entry.get("api_key_env", "")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    return entry["model"], api_key


def _make_approval_gate(db_path: str):
    """
    Return a closure that matches the approval_gate callable signature
    expected by core/agent_invoker.py.

    The db_path is captured in the closure so the caller only passes
    tool-call-specific parameters. The transport (terminal in Phase 1)
    is encapsulated in core/approval_gate.py; this closure does not
    need to know about it.
    """
    def gate(
        tool_name: str,
        parameters: dict,
        agent_type: str,
        project_id: str | None,
        tool_call_id: str,
    ) -> bool:
        return request_approval(
            tool_name=tool_name,
            parameters=parameters,
            agent_type=agent_type,
            project_id=project_id,
            tool_call_id=tool_call_id,
            db_path=db_path,
        )

    return gate


def _agent_registry(agent_type: str, full_registry: ToolRegistry, config: dict) -> ToolRegistry:
    """
    Return a ToolRegistry scoped to the tools allowed for agent_type.

    Only task_agent has a dedicated make_registry() helper in Phase 1.
    Unknown agent types fall back to the full registry rather than crashing
    the loop — a future phase will add per-agent builders for dev_agent etc.
    """
    if agent_type == task_agent_mod.AGENT_TYPE:
        return task_agent_mod.make_registry(full_registry, config)
    # Fallback: unknown agent gets the full registry.
    # [DO NOT REMOVE] This allows new agent types to work immediately when
    # their routing rules are added, before their registry helpers are written.
    return full_registry


def main() -> None:
    config = load_config()
    db_path = config["paths"]["db"]
    debug = config.get("debug", {}).get("print_prompts", False)

    init_db(db_path)

    stale = reset_stale_events(db_path)
    if stale:
        print(f"[startup] Reset {stale} stale event(s) to 'pending'.", flush=True)

    full_registry = build_full_registry(config, db_path)
    approval_gate = _make_approval_gate(db_path)

    timer = TimerAdapter(db_path=db_path, interval_seconds=_TIMER_INTERVAL_SECONDS)
    timer.start()
    print(
        f"[startup] Timer adapter started ({_TIMER_INTERVAL_SECONDS}s interval).",
        flush=True,
    )

    # Graceful shutdown: flip _running on SIGINT / SIGTERM so the loop exits
    # cleanly after the current event finishes. The timer thread is a daemon
    # and exits automatically, but we stop it explicitly for clean logging.
    _running = [True]

    def _handle_signal(sig, frame):
        print(f"\n[shutdown] Signal {sig} received — stopping after current event.",
              flush=True)
        _running[0] = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("[startup] Event loop running. Press Ctrl-C to stop.", flush=True)

    while _running[0]:
        event = dequeue_next_event(db_path)
        if event is None:
            time.sleep(_IDLE_SLEEP_SECONDS)
            continue

        event_id = event["id"]
        print(
            f"[event] source={event['source']!r} type={event['type']!r} id={event_id}",
            flush=True,
        )

        try:
            routing = route_event(event, config)
        except UnroutableEvent as exc:
            print(f"[event] Unroutable — {exc}", flush=True)
            fail_event(db_path, event_id)
            continue

        preflight = routing.get("preflight")
        if preflight is not None and not preflight():
            print(f"[event] preflight skipped — no action needed", flush=True)
            complete_event(db_path, event_id)
            continue

        agent_type = routing["agent_type"]
        model, api_key = _select_model(config, routing["model_key"])
        agent_reg = _agent_registry(agent_type, full_registry, config)
        max_iter = config.get("agents", {}).get(agent_type, {}).get("max_iterations", 10)

        try:
            result = invoke(
                task_description=routing["task_description"],
                agent_type=agent_type,
                model=model,
                api_key=api_key,
                registry=agent_reg,
                event_id=event_id,
                db_path=db_path,
                debug_print_prompts=debug,
                approval_gate=approval_gate,
                max_iterations=max_iter,
            )
            print(
                f"[event] done — invocation={result['invocation_id']} "
                f"status={result['status']} tools={len(result['tool_calls'])}",
                flush=True,
            )
            complete_event(db_path, event_id)
        except Exception as exc:
            print(f"[event] Invocation raised unexpectedly: {exc}", flush=True)
            fail_event(db_path, event_id)

    timer.stop()
    timer.join(timeout=5.0)
    print("[shutdown] Done.", flush=True)


if __name__ == "__main__":
    main()
