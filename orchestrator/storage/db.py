# Purpose: SQLite connection pool and helpers.
# Relationships: Used by core/agent_invoker.py and any module that needs DB access.
#               schema.sql lives alongside this file and is loaded by init_db().

import os
import sqlite3
from contextlib import contextmanager

# Default DB path; overridden by config.yaml or tests via explicit argument.
DB_PATH = os.environ.get("ORCHESTRATOR_DB", "./storage/orchestrator.db")


def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables from schema.sql. Safe to call on an existing DB."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        schema = f.read()
    # Ensure the directory exists before trying to open the DB file.
    db_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(db_dir, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(schema)


@contextmanager
def get_conn(db_path: str | None = None):
    """
    Yield an open SQLite connection with WAL journal mode and Row factory.

    # We use WAL mode here because it allows concurrent reads during writes,
    # which matters once the event loop and adapter threads are running in
    # Phase 1. If the DB is ever replaced with PostgreSQL, remove this pragma.

    Commits on clean exit, rolls back on exception, always closes.
    """
    path = db_path if db_path is not None else DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
