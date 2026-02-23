# Purpose: Entry point. Initialises all subsystems, starts the timer and
#          (optionally) the network adapter as asyncio tasks, then runs the
#          async event loop — dequeuing events, routing them, and driving
#          agent invocations.
#
# Relationships: Wires together storage/db.py, tools/*, core/*, adapters/*,
#               and agents/*. This is the only place all subsystems are
#               assembled — individual modules know nothing about each other.
#
# Async architecture: the main event loop is a single asyncio coroutine.
# SQLite calls remain synchronous (they are sub-ms local operations and do
# not starve the event loop). LLM calls use litellm.acompletion() which
# yields during network I/O. asyncio.sleep() in the idle branch yields to
# allow the network adapter to service HTTP connections between events.
# No threads are used. The timer runs as an asyncio.create_task() coroutine.

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from .adapters import timer as timer_adapter
from .adapters.network import NetworkAdapter
from .agents import task_agent as task_agent_mod
from .core.agent_invoker import invoke
from .core.approval_gate import request_approval
from .core.context_manager import ensure_project
from .core.event_queue import (
    complete_event,
    dequeue_next_event,
    fail_event,
    reset_stale_events,
)
from .core.router import UnroutableEvent, route_event
from .storage.db import init_db
from .tools.filesystem import (
    make_list_directory_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from .tools.code_quality import make_check_syntax_tool, make_lint_tool
from .tools.git import (
    make_git_add_tool,
    make_git_branch_tool,
    make_git_commit_tool,
    make_git_diff_tool,
    make_git_list_branches_tool,
    make_git_log_tool,
    make_git_rollback_tool,
    make_git_status_tool,
)
from .tools.notify import make_notify_user_tool
from .tools.registry import ToolRegistry
from .tools.shell import make_run_tests_tool, make_shell_run_tool

# Poll interval when the event queue is empty. 500ms balances responsiveness
# against unnecessary CPU spin. Do not set below ~100ms on SQLite.
_IDLE_SLEEP_SECONDS = 0.5


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(level=level.upper(), handlers=[handler])


_DEFAULT_CONFIG = Path.home() / ".config" / "germinal" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    # Resolve config path: explicit arg > ~/.config/germinal/config.yaml > local fallback.
    # The local fallback exists so 'python -m orchestrator' works from the repo
    # root during development without requiring a prior 'germ' run to seed the
    # user config.
    resolved = Path(path) if path else _DEFAULT_CONFIG
    if not resolved.exists():
        resolved = Path("config.yaml")
    with open(resolved) as f:
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
    registry.register(make_git_add_tool())
    registry.register(make_git_branch_tool())
    registry.register(make_git_rollback_tool())
    registry.register(make_git_diff_tool())
    registry.register(make_git_list_branches_tool())
    registry.register(make_git_log_tool())
    registry.register(make_lint_tool())
    registry.register(make_check_syntax_tool())
    return registry


def select_model(config: dict, model_key: str) -> tuple[str, str | None, int | None]:
    """
    Resolve model_key → (litellm model string, api_key | None, max_tokens | None).

    model_key is first looked up in models.categories by category name.
    If not found as a category, it is treated as a direct model name in
    models.list. The ORCHESTRATOR_MODEL env var overrides the resolved name
    when model_key is "default", allowing the active model to be changed at
    runtime without editing config.yaml.

    max_tokens is taken from the model entry in config; None means no explicit
    limit is passed to LiteLLM (the provider's default applies). Set this per
    model to prevent token-limit truncation of large tool call responses.
    """
    models_cfg = config["models"]
    categories = {c["category"]: c for c in models_cfg.get("categories", [])}

    if model_key in categories:
        resolved_name = categories[model_key]["model"]
    else:
        resolved_name = model_key

    # Allow runtime override of the default category's model via env var.
    if model_key == "default":
        resolved_name = os.environ.get("ORCHESTRATOR_MODEL", resolved_name)

    index = {m["name"]: m for m in models_cfg["list"]}
    if resolved_name not in index:
        raise ValueError(
            f"Unknown model name {resolved_name!r}. Valid names: {list(index)}"
        )

    entry = index[resolved_name]
    api_key_env = entry.get("api_key_env", "")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    max_tokens: int | None = entry.get("max_tokens")
    return entry["model"], api_key, max_tokens


def make_approval_gate(db_path: str):
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


def agent_registry_for(agent_type: str, full_registry: ToolRegistry, config: dict) -> ToolRegistry:
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


async def _event_loop(
    db_path: str,
    config: dict,
    full_registry: ToolRegistry,
    approval_gate,
    pending_http: dict,
    stop_event: asyncio.Event,
) -> None:
    """
    Core event loop: dequeue, route, and invoke, until stop_event is set.

    pending_http maps event_id → asyncio.Future for events that originated
    from HTTP requests. After invoke() completes, we resolve the future so
    the HTTP handler can return the response to the client.
    """
    logger_main = logging.getLogger("main")
    logger_event = logging.getLogger("event")

    while not stop_event.is_set():
        event = dequeue_next_event(db_path)
        if event is None:
            # Yield to the asyncio event loop so the network adapter can
            # accept and service connections while the queue is empty.
            await asyncio.sleep(_IDLE_SLEEP_SECONDS)
            continue

        event_id = event["id"]
        logger_event.info(
            "source=%r type=%r id=%s", event["source"], event["type"], event_id
        )

        try:
            routing = route_event(event)
        except UnroutableEvent as exc:
            logger_event.warning("Unroutable — %s", exc)
            fail_event(db_path, event_id)
            # Resolve any waiting HTTP future with an error result so the
            # client gets a response instead of hanging until timeout.
            _resolve_pending(pending_http, event_id, {"status": "failed", "response": str(exc), "tool_calls": [], "invocation_id": ""})
            continue

        # Resolve project_id: event payload takes priority, then config default.
        project_id = event.get("project_id") or config.get("projects", {}).get(
            "default_project_id"
        )
        if project_id:
            ensure_project(
                project_id,
                config.get("projects", {}).get("default_project_name", project_id),
                db_path,
            )

        # HTTP events carry the intended agent_type in the payload so the
        # router can honour it. The router rule for source="http" should
        # use {payload[agent_type]} in its task_template and set agent_type
        # from the payload; until that rule exists, fall back to routing result.
        agent_type = routing["agent_type"]
        model, api_key, max_tokens = select_model(config, routing["model_key"])
        agent_reg = agent_registry_for(agent_type, full_registry, config)
        max_iter = config.get("agents", {}).get(agent_type, {}).get("max_iterations", 10)

        try:
            result = await invoke(
                task_description=routing["task_description"],
                agent_type=agent_type,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                registry=agent_reg,
                project_id=project_id,
                event_id=event_id,
                db_path=db_path,
                approval_gate=approval_gate,
                max_iterations=max_iter,
                config=config,
            )
            logger_event.info(
                "done — invocation=%s status=%s tools=%d",
                result["invocation_id"], result["status"], len(result["tool_calls"]),
            )
            complete_event(db_path, event_id)
            _resolve_pending(pending_http, event_id, result)
        except Exception as exc:
            logger_event.error("Invocation raised unexpectedly: %s", exc)
            fail_event(db_path, event_id)
            _resolve_pending(pending_http, event_id, {"status": "failed", "response": str(exc), "tool_calls": [], "invocation_id": ""})


def _resolve_pending(pending_http: dict, event_id: str, result: dict) -> None:
    """
    If event_id has a waiting HTTP future, resolve it.

    Called after every event completes (success or failure) so HTTP clients
    always get a response rather than waiting until timeout.
    """
    future = pending_http.pop(event_id, None)
    if future is not None and not future.done():
        future.set_result(result)


async def main() -> None:
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    logger_main = logging.getLogger("main")

    db_path = config["paths"]["db"]

    init_db(db_path)

    stale = reset_stale_events(db_path)
    if stale:
        logger_main.info("Reset %d stale event(s) to 'pending'.", stale)

    full_registry = build_full_registry(config, db_path)
    approval_gate = make_approval_gate(db_path)

    # Shared dict: event_id → asyncio.Future. The network adapter writes futures
    # here; the event loop resolves them after invoke() completes.
    pending_http: dict[str, asyncio.Future] = {}

    stop_event = asyncio.Event()

    # Graceful shutdown: signal handlers set stop_event so the event loop
    # and timer task exit cleanly after the current operation finishes.
    # asyncio signal handlers run in the event loop thread — safe to call
    # asyncio.Event.set() directly.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: (
                logger_main.info("Signal %s received — stopping.", s),
                stop_event.set(),
            ),
        )

    # Start network adapter if enabled.
    network_cfg = config.get("network", {})
    net_adapter: NetworkAdapter | None = None
    if network_cfg.get("enabled", False):
        net_adapter = NetworkAdapter(
            config=config,
            db_path=db_path,
            pending=pending_http,
        )
        await net_adapter.start()

    logger_main.info("Event loop running. Press Ctrl-C to stop.")

    try:
        await _event_loop(
            db_path=db_path,
            config=config,
            full_registry=full_registry,
            approval_gate=approval_gate,
            pending_http=pending_http,
            stop_event=stop_event,
        )
    finally:
        if net_adapter:
            await net_adapter.stop()
        logger_main.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
