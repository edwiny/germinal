#!/usr/bin/env python3
# Purpose: Control plane CLI for the Germinal orchestrator.
#          Reads and searches state stored in the orchestrator's SQLite database.
#          All operations are read-only — no mutations are made.
#
# Usage:
#   python germctl.py [--db PATH] [--json] <command> [options]
#   ./germctl.py <command> [options]         (after chmod +x)
#
# Environment:
#   ORCHESTRATOR_DB  Override the path to the SQLite database.
#
# Commands:
#   events       List events from the event queue
#   invocations  List agent invocations
#   tools        List tool calls
#   projects     List projects and their context summaries
#   history      Show conversation history for a project
#   approvals    List human-approval requests
#   show         Show full detail for a specific record

import argparse
import json
import os
import sqlite3
import sys
import textwrap
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

# Search these paths in order when --db is not provided.
_DEFAULT_DB_CANDIDATES = [
    os.environ.get("ORCHESTRATOR_DB", ""),
    os.path.join(os.path.dirname(__file__), "..", "orchestrator", "storage", "orchestrator.db"),
    os.path.join(os.getcwd(), "orchestrator", "storage", "orchestrator.db"),
    os.path.join(os.getcwd(), "storage", "orchestrator.db"),
]


def _find_db() -> str:
    for path in _DEFAULT_DB_CANDIDATES:
        if path and os.path.isfile(path):
            return os.path.normpath(path)
    return ""


@contextmanager
def _connect(db_path: str):
    """Open a read-only SQLite connection (WAL mode, Row factory)."""
    if not os.path.isfile(db_path):
        _die(f"Database not found: {db_path!r}\n"
             "Use --db to specify the path or set ORCHESTRATOR_DB.")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _status_color(status: str | None) -> str:
    if not _supports_color() or not status:
        return status or ""
    mapping = {
        "done": _GREEN,
        "executed": _GREEN,
        "approved": _GREEN,
        "open": _CYAN,
        "pending": _YELLOW,
        "processing": _YELLOW,
        "running": _YELLOW,
        "in_progress": _YELLOW,
        "failed": _RED,
        "denied": _RED,
        "cancelled": _DIM,
    }
    color = mapping.get(status.lower(), "")
    return f"{color}{status}{_RESET}" if color else status


def _trunc(value, width: int) -> str:
    """Truncate a value to width, appending … if truncated."""
    if value is None:
        return ""
    s = str(value)
    if len(s) > width:
        return s[: width - 1] + "…"
    return s


def _die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def _print_table(
    rows: list[dict],
    columns: list[tuple[str, str, int]],
    *,
    color_field: str | None = None,
    json_mode: bool = False,
) -> None:
    """
    Print rows as a fixed-width table.

    columns: list of (field_name, header_label, column_width)
    color_field: if provided, apply status colour to that column's values.
    """
    if json_mode:
        print(json.dumps(rows, indent=2, default=str))
        return

    if not rows:
        print("(no results)")
        return

    sep = "  "
    use_color = _supports_color()

    # Header line
    if use_color:
        header = sep.join(
            (_BOLD + h + _RESET).ljust(w + len(_BOLD) + len(_RESET))
            for _, h, w in columns
        )
    else:
        header = sep.join(h.ljust(w) for _, h, w in columns)
    print(header)
    print(sep.join("-" * w for _, _, w in columns))

    for row in rows:
        parts = []
        for field, _, width in columns:
            raw = row.get(field)
            cell = _trunc(raw, width)
            if use_color and field == color_field:
                colored = _status_color(cell)
                # Pad to width using raw (uncolored) length
                padding = width - len(cell)
                parts.append(colored + " " * padding)
            else:
                parts.append(cell.ljust(width))
        print(sep.join(parts))


def _print_detail(record: dict, *, json_mode: bool = False) -> None:
    """Print all fields of a single record, pretty-printing JSON values."""
    if json_mode:
        print(json.dumps(record, indent=2, default=str))
        return

    use_color = _supports_color()
    for key, value in record.items():
        if value is None:
            continue
        label = (_BOLD + key + _RESET) if use_color else key
        value_str = str(value)
        # If the value is long or multiline or looks like JSON, pretty-print it.
        is_long = len(value_str) > 100 or "\n" in value_str
        looks_like_json = value_str.strip().startswith(("{", "["))
        if is_long or looks_like_json:
            print(f"{label}:")
            try:
                parsed = json.loads(value_str)
                pretty = json.dumps(parsed, indent=2)
            except Exception:
                pretty = value_str
            print(textwrap.indent(pretty, "  "))
        else:
            print(f"{label}: {value_str}")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_events(args, conn) -> None:
    """List events from the event queue."""
    q = """
        SELECT id, source, type, project_id, priority, status, created_at
        FROM events
        WHERE 1=1
    """
    params: list = []

    if args.status:
        q += " AND status = ?"
        params.append(args.status)
    if args.source:
        q += " AND source = ?"
        params.append(args.source)
    if args.project:
        q += " AND project_id = ?"
        params.append(args.project)
    if args.search:
        q += " AND (payload LIKE ? OR id LIKE ?)"
        term = f"%{args.search}%"
        params.extend([term, term])

    q += " ORDER BY created_at DESC LIMIT ?"
    params.append(args.limit)

    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    _print_table(
        rows,
        [
            ("id", "ID", 18),
            ("source", "SOURCE", 8),
            ("type", "TYPE", 12),
            ("project_id", "PROJECT", 12),
            ("priority", "PRI", 4),
            ("status", "STATUS", 12),
            ("created_at", "CREATED", 19),
        ],
        color_field="status",
        json_mode=args.json,
    )
    if not args.json:
        print(f"\n{len(rows)} row(s) shown (limit {args.limit})")


def _cmd_invocations(args, conn) -> None:
    """List agent invocations."""
    q = """
        SELECT id, event_id, agent_type, project_id, model, status,
               started_at, finished_at, response
        FROM invocations
        WHERE 1=1
    """
    params: list = []

    if args.status:
        q += " AND status = ?"
        params.append(args.status)
    if args.agent_type:
        q += " AND agent_type = ?"
        params.append(args.agent_type)
    if args.project:
        q += " AND project_id = ?"
        params.append(args.project)
    if args.search:
        q += " AND (response LIKE ? OR id LIKE ?)"
        term = f"%{args.search}%"
        params.extend([term, term])

    q += " ORDER BY started_at DESC LIMIT ?"
    params.append(args.limit)

    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    _print_table(
        rows,
        [
            ("id", "ID", 18),
            ("agent_type", "AGENT", 12),
            ("project_id", "PROJECT", 12),
            ("model", "MODEL", 22),
            ("status", "STATUS", 8),
            ("started_at", "STARTED", 19),
            ("response", "RESPONSE", 45),
        ],
        color_field="status",
        json_mode=args.json,
    )
    if not args.json:
        print(f"\n{len(rows)} row(s) shown (limit {args.limit})")


def _cmd_tools(args, conn) -> None:
    """List tool calls."""
    q = """
        SELECT tc.id, tc.invocation_id, tc.tool_name, tc.risk_level,
               tc.status, tc.created_at, tc.executed_at, tc.parameters
        FROM tool_calls tc
        WHERE 1=1
    """
    params: list = []

    if args.status:
        q += " AND tc.status = ?"
        params.append(args.status)
    if args.tool_name:
        q += " AND tc.tool_name = ?"
        params.append(args.tool_name)
    if args.invocation:
        q += " AND tc.invocation_id = ?"
        params.append(args.invocation)
    if args.search:
        q += " AND (tc.parameters LIKE ? OR tc.result LIKE ?)"
        term = f"%{args.search}%"
        params.extend([term, term])

    q += " ORDER BY tc.created_at DESC LIMIT ?"
    params.append(args.limit)

    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    _print_table(
        rows,
        [
            ("id", "ID", 18),
            ("tool_name", "TOOL", 18),
            ("risk_level", "RISK", 8),
            ("status", "STATUS", 10),
            ("created_at", "CREATED", 19),
            ("parameters", "PARAMETERS", 40),
        ],
        color_field="status",
        json_mode=args.json,
    )
    if not args.json:
        print(f"\n{len(rows)} row(s) shown (limit {args.limit})")


def _cmd_projects(args, conn) -> None:
    """List projects."""
    q = """
        SELECT id, name, description, created_at, updated_at,
               brief, summary
        FROM projects
        WHERE 1=1
    """
    params: list = []

    if args.search:
        q += " AND (name LIKE ? OR description LIKE ? OR id LIKE ?)"
        term = f"%{args.search}%"
        params.extend([term, term, term])

    q += " ORDER BY updated_at DESC LIMIT ?"
    params.append(args.limit)

    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    _print_table(
        rows,
        [
            ("id", "ID", 16),
            ("name", "NAME", 24),
            ("description", "DESCRIPTION", 40),
            ("created_at", "CREATED", 19),
            ("updated_at", "UPDATED", 19),
        ],
        json_mode=args.json,
    )
    if not args.json:
        print(f"\n{len(rows)} row(s) shown (limit {args.limit})")

def _cmd_history(args, conn) -> None:
    """Show conversation history for a project."""
    q = """
        SELECT h.id, h.project_id, h.role, h.content, h.created_at
        FROM history h
        WHERE 1=1
    """
    params: list = []

    if args.project:
        q += " AND h.project_id = ?"
        params.append(args.project)
    if args.role:
        q += " AND h.role = ?"
        params.append(args.role)
    if args.search:
        q += " AND h.content LIKE ?"
        params.append(f"%{args.search}%")

    q += " ORDER BY h.created_at DESC LIMIT ?"
    params.append(args.limit)

    rows = [dict(r) for r in conn.execute(q, params).fetchall()]

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return

    if not rows:
        print("(no results)")
        return

    # For history, show a richer format since content matters.
    use_color = _supports_color()
    role_colors = {
        "user": _CYAN,
        "agent": _GREEN,
        "tool": _YELLOW,
    }
    for row in rows:
        role = row["role"]
        ts = row["created_at"]
        project = row.get("project_id", "")
        color = role_colors.get(role, "") if use_color else ""
        reset = _RESET if use_color else ""
        label = f"{color}[{role.upper()}]{reset} {_DIM if use_color else ''}{ts} ({project}){reset}"
        print(label)
        content = row["content"] or ""
        print(textwrap.indent(textwrap.fill(content, width=100), "  "))
        print()

    print(f"{len(rows)} entry/entries shown (limit {args.limit})")


def _cmd_approvals(args, conn) -> None:
    """List human-approval requests."""
    q = """
        SELECT a.id, a.tool_call_id, a.response, a.created_at, a.responded_at,
               tc.tool_name, tc.risk_level
        FROM approvals a
        LEFT JOIN tool_calls tc ON tc.id = a.tool_call_id
        WHERE 1=1
    """
    params: list = []

    if args.pending:
        q += " AND a.response IS NULL"
    if args.search:
        q += " AND (a.prompt LIKE ? OR tc.tool_name LIKE ?)"
        term = f"%{args.search}%"
        params.extend([term, term])

    q += " ORDER BY a.created_at DESC LIMIT ?"
    params.append(args.limit)

    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    _print_table(
        rows,
        [
            ("id", "ID", 18),
            ("tool_name", "TOOL", 18),
            ("risk_level", "RISK", 8),
            ("response", "RESPONSE", 10),
            ("created_at", "CREATED", 19),
            ("responded_at", "RESPONDED", 19),
        ],
        color_field="response",
        json_mode=args.json,
    )
    if not args.json:
        print(f"\n{len(rows)} row(s) shown (limit {args.limit})")


def _cmd_show(args, conn) -> None:
    """Show full detail for a specific record."""
    table_map = {
        "event": "events",
        "events": "events",
        "invocation": "invocations",
        "invocations": "invocations",
        "tool": "tool_calls",
        "tools": "tool_calls",
        "tool_call": "tool_calls",
        "tool_calls": "tool_calls",
        "project": "projects",
        "projects": "projects",
        "history": "history",
        "approval": "approvals",
        "approvals": "approvals",
    }
    table = table_map.get(args.table.lower())
    if not table:
        _die(
            f"Unknown table {args.table!r}. "
            f"Valid values: {', '.join(sorted(set(table_map)))}"
        )

    # Check the table exists (protects against SQL injection via args.table,
    # though the table_map already ensures only known names reach here).
    row = conn.execute(
        f"SELECT * FROM {table} WHERE id = ?", (args.id,)  # noqa: S608
    ).fetchone()

    if row is None:
        _die(f"No {table} record with id {args.id!r}")

    _print_detail(dict(row), json_mode=args.json)


def _cmd_stats(args, conn) -> None:
    """Print a summary count of rows in every table."""
    tables = ["events", "invocations", "tool_calls", "approvals", "projects", "history"]
    stats = {}
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        stats[table] = count

    if args.json:
        print(json.dumps(stats, indent=2))
        return

    use_color = _supports_color()
    print((_BOLD + "TABLE" + _RESET if use_color else "TABLE").ljust(16) + "  ROWS")
    print("-" * 16 + "  " + "-" * 8)
    for table, count in stats.items():
        print(table.ljust(16) + f"  {count}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="germctl",
        description="Control plane CLI for the Germinal orchestrator. Read-only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  germctl events --status pending
  germctl events --source http --limit 20
  germctl invocations --status done --project default
  germctl invocations --search "error"
  germctl tools --tool-name read_file
  germctl history --project default --limit 50
  germctl approvals --pending
  germctl show events <id>
  germctl show invocations <id>
  germctl stats
        """,
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default="",
        help="Path to the orchestrator SQLite database "
             "(default: auto-detected from ORCHESTRATOR_DB env var or repo layout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON instead of a formatted table",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- events ------------------------------------------------------------
    p_ev = sub.add_parser("events", help="List events from the event queue")
    p_ev.add_argument("--status", choices=["pending", "processing", "done", "failed"])
    p_ev.add_argument("--source", help="Filter by source (e.g. http, timer)")
    p_ev.add_argument("--project", metavar="ID", help="Filter by project_id")
    p_ev.add_argument("--search", metavar="TEXT", help="Search payload and id")
    p_ev.add_argument("--limit", type=int, default=50, metavar="N")

    # ---- invocations -------------------------------------------------------
    p_inv = sub.add_parser("invocations", help="List agent invocations")
    p_inv.add_argument("--status", choices=["running", "done", "failed"])
    p_inv.add_argument("--agent-type", dest="agent_type", help="Filter by agent type")
    p_inv.add_argument("--project", metavar="ID", help="Filter by project_id")
    p_inv.add_argument("--search", metavar="TEXT", help="Search response text and id")
    p_inv.add_argument("--limit", type=int, default=20, metavar="N")

    # ---- tools -------------------------------------------------------------
    p_tc = sub.add_parser("tools", help="List tool calls")
    p_tc.add_argument(
        "--status",
        choices=["pending", "approved", "denied", "executed", "failed"],
    )
    p_tc.add_argument("--tool-name", dest="tool_name", help="Filter by tool name")
    p_tc.add_argument("--invocation", metavar="ID", help="Filter by invocation_id")
    p_tc.add_argument("--search", metavar="TEXT", help="Search parameters and result")
    p_tc.add_argument("--limit", type=int, default=50, metavar="N")

    # ---- projects ----------------------------------------------------------
    p_proj = sub.add_parser("projects", help="List projects")
    p_proj.add_argument("--search", metavar="TEXT", help="Search name and description")
    p_proj.add_argument("--limit", type=int, default=50, metavar="N")

    # ---- history -----------------------------------------------------------
    p_hist = sub.add_parser("history", help="Show conversation history for a project")
    p_hist.add_argument("--project", metavar="ID", help="Filter by project_id")
    p_hist.add_argument("--role", choices=["user", "agent", "tool"])
    p_hist.add_argument("--search", metavar="TEXT", help="Search content")
    p_hist.add_argument("--limit", type=int, default=30, metavar="N")

    # ---- approvals ---------------------------------------------------------
    p_appr = sub.add_parser("approvals", help="List human-approval requests")
    p_appr.add_argument("--pending", action="store_true", help="Show only unanswered approvals")
    p_appr.add_argument("--search", metavar="TEXT", help="Search prompt and tool name")
    p_appr.add_argument("--limit", type=int, default=50, metavar="N")

    # ---- show --------------------------------------------------------------
    p_show = sub.add_parser("show", help="Show full detail for a specific record")
    p_show.add_argument(
        "table",
        metavar="TABLE",
        help="Table name: events, invocations, tools, projects, history, approvals",
    )
    p_show.add_argument("id", metavar="ID", help="Record id")

    # ---- stats -------------------------------------------------------------
    sub.add_parser("stats", help="Show row counts for all tables")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMAND_MAP = {
    "events": _cmd_events,
    "invocations": _cmd_invocations,
    "tools": _cmd_tools,
    "projects": _cmd_projects,
    "history": _cmd_history,
    "approvals": _cmd_approvals,
    "show": _cmd_show,
    "stats": _cmd_stats,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    db_path = args.db or _find_db()
    if not db_path:
        _die(
            "Cannot find the orchestrator database.\n"
            "Run from the repo root, or pass --db PATH, or set ORCHESTRATOR_DB."
        )

    handler = _COMMAND_MAP[args.command]
    with _connect(db_path) as conn:
        handler(args, conn)


if __name__ == "__main__":
    main()
