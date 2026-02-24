# Purpose: Interactive mode entry point — one-shot and REPL.
#          Reuses the same setup helpers as main_loop.py (load_config,
#          init_db, build_full_registry, make_approval_gate, select_model,
#          agent_registry_for, ensure_project) but does NOT start the network
#          adapter or timer; those are daemon-only concerns.
#
# Logging is directed to stderr so that stdout carries only the agent's
# response text. This matters when the caller pipes germ's output to another
# process — log noise on stdout would corrupt the response.
#
# Relationships: Called from __main__.py when --daemon is not passed.
#               Imports helpers from main_loop.py to avoid duplication.

import asyncio
import logging
import sys

from .core.agent_invoker import invoke
from .core.context_manager import ensure_project
from .main_loop import (
    agent_registry_for,
    build_full_registry,
    load_config,
    make_approval_gate,
    select_model,
    setup_logging,
)
from .storage.db import init_db

# Default agent type for interactive sessions. Routing rules are not consulted
# because there is no event — the user is the event source.
_INTERACTIVE_AGENT_TYPE = "task_agent"

# Route all interactive sessions through the "default" model category so
# the user can override via ORCHESTRATOR_MODEL without editing config.yaml.
_INTERACTIVE_MODEL_KEY = "default"


async def run_interactive(prompt: str | None) -> None:
    """
    Drive the agent from stdin/stdout.

    prompt is not None  → one-shot: invoke once, print response, return.
    prompt is None      → REPL: loop reading lines until EOF or Ctrl-D.

    Logging goes to stderr so stdout carries only the agent's response.
    """
    config = load_config()

    # Redirect logging to stderr so the agent's stdout response is clean.
    # setup_logging in main_loop writes to sys.stdout by default; we override
    # the handler here before any logger is used.
    log_level = config.get("logging", {}).get("level", "INFO")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(level=log_level.upper(), handlers=[stderr_handler], force=True)

    db_path: str = config["paths"]["db"]
    init_db(db_path)

    full_registry = build_full_registry(config, db_path)
    approval_gate = make_approval_gate(db_path)

    model, api_key, max_tokens = select_model(config, _INTERACTIVE_MODEL_KEY)

    project_id: str | None = config.get("projects", {}).get("default_project_id")
    if project_id:
        ensure_project(
            project_id,
            config.get("projects", {}).get("default_project_name", project_id),
            db_path,
        )

    agent_reg = agent_registry_for(_INTERACTIVE_AGENT_TYPE, full_registry, config)
    max_iter: int = (
        config.get("agents", {})
        .get(_INTERACTIVE_AGENT_TYPE, {})
        .get("max_iterations", 10)
    )

    if prompt is not None:
        # One-shot mode: run once, print response, done.
        await _invoke_and_print(
            task=prompt,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            agent_reg=agent_reg,
            project_id=project_id,
            db_path=db_path,
            approval_gate=approval_gate,
            max_iter=max_iter,
            exit_on_failure=True,
            config=config,
        )
        return

    # REPL mode: prompt the user repeatedly until EOF or Ctrl-D.
    print("Germinal interactive mode. Type your prompt and press Enter. Ctrl-D to exit.",
          file=sys.stderr)
    while True:
        try:
            user_input = input(" > ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            break

        if not user_input.strip():
            continue

        await _invoke_and_print(
            task=user_input,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            agent_reg=agent_reg,
            project_id=project_id,
            db_path=db_path,
            approval_gate=approval_gate,
            max_iter=max_iter,
            exit_on_failure=False,
            config=config,
        )


async def _invoke_and_print(
    *,
    task: str,
    model: str,
    api_key: str | None,
    max_tokens: int | None,
    agent_reg,
    project_id: str | None,
    db_path: str,
    approval_gate,
    max_iter: int,
    exit_on_failure: bool,
    config: dict,
) -> None:
    """
    Call invoke(), print the response to stdout, and handle errors.

    exit_on_failure=True causes SystemExit(1) on a failed invocation (one-shot).
    exit_on_failure=False prints the error to stderr and continues (REPL).
    """
    try:
        result = await invoke(
            task_description=task,
            agent_type=_INTERACTIVE_AGENT_TYPE,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            registry=agent_reg,
            project_id=project_id,
            event_id=None,
            db_path=db_path,
            approval_gate=approval_gate,
            max_iterations=max_iter,
            config=config,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exit_on_failure:
            raise SystemExit(1) from exc
        return

    if result["status"] == "failed":
        print(f"error: {result['response']}", file=sys.stderr)
        if exit_on_failure:
            raise SystemExit(1)
        return

    print(result["response"])
