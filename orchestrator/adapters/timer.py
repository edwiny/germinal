# Purpose: Cron-like timer event adapter.
# Relationships: Pushes events into core/event_queue.py; started by main.py as an asyncio task.

# Generates a 'tick' event on a fixed interval using an asyncio coroutine.
# There are no threads — the coroutine sleeps with asyncio.sleep(), yielding
# to the event loop between ticks. A stop asyncio.Event is used for clean shutdown.

# The adapter does not guarantee exactly-once delivery. If the main loop
# is slow and multiple ticks accumulate, the deduplication in push_event
# (INSERT OR IGNORE on the hash-based ID) means only one tick per hour
# is enqueued. For sub-hourly scheduling, include the minute in the payload
# so each tick gets a unique ID.

import asyncio
import logging
from datetime import datetime, timezone

from core.event_queue import push_event

logger = logging.getLogger("timer")


async def run(
    db_path: str,
    interval_seconds: int = 60,
    stop_event: asyncio.Event | None = None,
    project_id: str | None = None,
) -> None:
    """
    Asyncio coroutine: push a 'tick' event every interval_seconds until stop_event is set.

    Usage in main.py:
        stop = asyncio.Event()
        asyncio.create_task(timer.run(db_path, interval_seconds=60, stop_event=stop))
        # ... on shutdown:
        stop.set()
    """
    if stop_event is None:
        stop_event = asyncio.Event()

    while not stop_event.is_set():
        _push_tick(db_path, project_id)
        try:
            # Wait for interval_seconds, but wake immediately if stop is signalled.
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            # Normal case: interval elapsed without a stop signal.
            pass


def _push_tick(db_path: str, project_id: str | None) -> None:
    """Push one tick event. Logs but does not raise on failure."""
    # Include the current minute in the payload so each tick gets a
    # unique hash-based event ID (see event_queue._event_id comment).
    minute_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    try:
        push_event(
            db_path=db_path,
            source="timer",
            type="tick",
            payload={"minute": minute_str},
            project_id=project_id,
            priority=8,  # Timer ticks are low-urgency background work.
        )
    except Exception as exc:
        # Log but keep the coroutine alive. A transient DB error should not kill the adapter.
        logger.error("push_event failed: %s", exc)
