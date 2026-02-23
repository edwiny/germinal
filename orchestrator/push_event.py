import time

from .storage.db import init_db
from .core.event_queue import push_event

init_db('./storage/orchestrator.db')
eid = push_event(
    './storage/orchestrator.db',
    source='user',
    type='message',
    payload={
        'message': 'Use list_directory to list the current directory, then notify me what you find.',
        # Millisecond timestamp ensures a unique hash on every run so the
        # deduplication window (one hour) does not suppress manual test events.
        '_ts': time.time_ns() // 1_000_000,
    },
)
print('Pushed:', eid)
