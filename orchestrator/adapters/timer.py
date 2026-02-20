# Purpose: Cron-like timer event adapter.
# Relationships: Pushes events into core/event_queue.py; started by main.py.

# Generates a 'tick' event on a fixed interval by running a background
# daemon thread. The thread is self-healing: a push_event failure is
# logged to stderr and the thread continues rather than crashing.

# The adapter does not guarantee exactly-once delivery. If the main loop
# is slow and multiple ticks accumulate, the deduplication in push_event
# (INSERT OR IGNORE on the hash-based ID) means only one tick per hour
# is enqueued. For sub-hourly scheduling, include the minute in the payload
# so each tick gets a unique ID.

import logging
import threading
from datetime import datetime, timezone

from core.event_queue import push_event

logger = logging.getLogger("timer")


class TimerAdapter:
    """
    Background thread that pushes a 'tick' event every interval_seconds.

    Usage:
        adapter = TimerAdapter(db_path="./storage/orchestrator.db", interval_seconds=60)
        adapter.start()
        # ... main loop ...
        adapter.stop()
    """

    def __init__(
        self,
        db_path: str,
        interval_seconds: int = 60,
        project_id: str | None = None,
    ) -> None:
        self._db_path = db_path
        self._interval = interval_seconds
        self._project_id = project_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background timer thread. Returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="timer-adapter",
            # Daemon thread exits automatically when the main process exits,
            # so we do not need explicit cleanup on normal shutdown.
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the timer thread to stop. Does not block.

        The thread will exit after at most interval_seconds more seconds.
        Call join() on the thread if you need to wait for it to finish.
        """
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background thread to finish (for clean shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    def _run(self) -> None:
        """Thread body: push a tick event every interval_seconds."""
        while not self._stop_event.is_set():
            self._push_tick()
            # Use Event.wait() instead of time.sleep() so that stop() can
            # interrupt the sleep immediately rather than waiting a full interval.
            self._stop_event.wait(timeout=self._interval)

    def _push_tick(self) -> None:
        """Push one tick event. Logs but does not raise on failure."""
        # Include the current minute in the payload so each tick gets a
        # unique hash-based event ID (see event_queue._event_id comment).
        minute_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        try:
            push_event(
                db_path=self._db_path,
                source="timer",
                type="tick",
                payload={"minute": minute_str},
                project_id=self._project_id,
                priority=8,  # Timer ticks are low-urgency background work.
            )
        except Exception as exc:
            # Log but keep the thread alive. A transient DB error should not kill the adapter.
            logger.error("push_event failed: %s", exc)
