# Tests

## Setup

Install dependencies (from the `orchestrator/` directory):

```sh
pip install -r requirements.txt
```

Or, if using the local venv:

```sh
.venv/bin/pip install -r requirements.txt
```

## Running the tests

All commands below are run from the `orchestrator/` directory.

**Run everything:**

```sh
pytest tests/
```

**Run a single file:**

```sh
pytest tests/test_event_queue.py
pytest tests/integration/test_http_e2e.py
```

**Run a single test by name:**

```sh
pytest tests/ -k test_dequeue_priority_ordering
```

**Verbose output (shows each test name as it runs):**

```sh
pytest tests/ -v
```

**Stop at the first failure:**

```sh
pytest tests/ -x
```

## Test layout

```
tests/
├── test_agent_invoker.py    # core/agent_invoker.py — prompt assembly, tool loop, iteration cap
├── test_context_manager.py  # core/context_manager.py — three-tier context, summarisation
├── test_event_queue.py      # core/event_queue.py — push, dequeue, priority, deduplication
├── test_router.py           # core/router.py — rule matching, template rendering
├── test_tool_registry.py    # tools/registry.py — registration, schema validation
└── integration/
    ├── test_end_to_end.py   # invoke() + real tools + mocked LLM (no HTTP)
    └── test_http_e2e.py     # full HTTP pipeline: network adapter + event loop + mocked LLM
```

## How the HTTP end-to-end tests work

`test_http_e2e.py` runs the entire stack in a single asyncio event loop:

- A clean SQLite database is created per test in a `tmp_path` directory.
- `_event_loop` (from `main.py`) runs as a background `asyncio.Task`, polling for events and calling `invoke()`.
- An `aiohttp.test_utils.TestServer` wraps the `NetworkAdapter`'s app on an ephemeral port — no port conflicts between tests.
- `litellm.acompletion` is replaced with `AsyncMock` so no real model is needed.
- All other components (event queue, router, tool registry, approval gate, context manager) are real.

This means these tests exercise the complete request path:

```
HTTP client (TestClient)
  → NetworkAdapter (aiohttp)
    → push_event() → SQLite
      → _event_loop dequeues → route_event() → invoke()
        → litellm.acompletion() [mocked]
        → tool.execute() [real, if any tool calls]
      → asyncio.Future resolved
    → SSE or JSON response built
  → assertion on response + DB state
```

## Notes

- `asyncio_mode = auto` is set in `pytest.ini`, so `async def` test functions run automatically in the asyncio event loop without needing `@pytest.mark.asyncio`.
- LLM calls are always mocked. No API keys or running models are required.
- The HTTP end-to-end tests take a few seconds each because `_event_loop` polls SQLite every 500 ms; that latency is intentional and reflects real system behaviour.
