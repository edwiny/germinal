# Purpose: Tests for core/router.py.
# Covers: user message routing, HTTP message routing,
#         UnroutableEvent for unmatched events, payload extraction.

import pytest

from core.router import UnroutableEvent, route_event


@pytest.fixture()
def config():
    return {
        "models": {
            "list": [{"name": "kimi-k2", "model": "openrouter/moonshotai/kimi-k2.5", "api_key_env": "OPENROUTER_API_KEY"}],
            "categories": [{"category": "default", "model": "kimi-k2", "used_for": []}],
        },
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_user_message_routes_to_task_agent():
    """User message events must route to task_agent."""
    event = {"source": "user", "type": "message", "payload": '{"message": "hello"}'}
    result = route_event(event)
    assert result["agent_type"] == "task_agent"


def test_http_message_routes_to_task_agent():
    """HTTP message events must route to task_agent."""
    event = {"source": "http", "type": "message", "payload": '{"message": "hello"}'}
    result = route_event(event)
    assert result["agent_type"] == "task_agent"


def test_user_message_extracted_as_task_description():
    """payload["message"] is used directly as the task description."""
    event = {"source": "user", "type": "message", "payload": '{"message": "do the thing"}'}
    result = route_event(event)
    assert result["task_description"] == "do the thing"


def test_payload_as_dict_is_also_accepted():
    """route_event must handle payload already parsed as a dict (not just a JSON string)."""
    event = {"source": "user", "type": "message", "payload": {"message": "hello dict"}}
    result = route_event(event)
    assert result["task_description"] == "hello dict"


def test_missing_message_field_yields_empty_string():
    """An event with no 'message' in the payload produces an empty task description."""
    event = {"source": "user", "type": "message", "payload": "{}"}
    result = route_event(event)
    assert result["task_description"] == ""


def test_unroutable_event_raises():
    """An event with no matching rule must raise UnroutableEvent."""
    event = {"source": "unknown_source", "type": "unknown_type", "payload": "{}"}
    with pytest.raises(UnroutableEvent):
        route_event(event)


def test_timer_tick_is_unroutable():
    """Timer tick events no longer have a routing rule and must raise UnroutableEvent."""
    event = {"source": "timer", "type": "tick", "payload": '{"minute": "2026-01-01T00:00"}'}
    with pytest.raises(UnroutableEvent):
        route_event(event)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_no_preflight_in_routing_result():
    """Routing result must not include a preflight key."""
    event = {"source": "user", "type": "message", "payload": '{"message": "hi"}'}
    result = route_event(event)
    assert "preflight" not in result
