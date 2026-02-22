# Purpose: Tests for core/router.py.
# Covers: timer tick routing, user message routing, template expansion,
#         UnroutableEvent for unmatched events, preflight callable behaviour.

import pytest

from core.router import UnroutableEvent, route_event
from storage.db import init_db


# Config must include paths.db because the timer tick rule creates a
# preflight closure that captures config["paths"]["db"].
@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


@pytest.fixture()
def config(tmp_db):
    return {
        "models": {
            "list": [{"name": "kimi-k2", "model": "openrouter/moonshotai/kimi-k2.5", "api_key_env": "OPENROUTER_API_KEY"}],
            "categories": [{"category": "default", "model": "kimi-k2", "used_for": []}],
        },
        "paths": {"db": tmp_db},
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_timer_tick_routes_to_task_agent(config):
    """Timer tick events must route to task_agent."""
    event = {"source": "timer", "type": "tick", "payload": '{"minute": "2026-01-01T00:00"}'}
    result = route_event(event, config)
    assert result["agent_type"] == "task_agent"


def test_timer_tick_task_description_is_non_empty(config):
    """Routed task description for a tick must be a non-empty string."""
    event = {"source": "timer", "type": "tick", "payload": '{"minute": "2026-01-01T00:00"}'}
    result = route_event(event, config)
    assert isinstance(result["task_description"], str)
    assert len(result["task_description"]) > 0


def test_user_message_routes_to_task_agent(config):
    """User message events must route to task_agent."""
    event = {"source": "user", "type": "message", "payload": '{"message": "hello"}'}
    result = route_event(event, config)
    assert result["agent_type"] == "task_agent"


def test_user_message_payload_expanded_into_task(config):
    """The {payload[message]} template must be replaced with the actual message."""
    event = {"source": "user", "type": "message", "payload": '{"message": "do the thing"}'}
    result = route_event(event, config)
    assert result["task_description"] == "do the thing"


def test_unroutable_event_raises(config):
    """An event with no matching rule must raise UnroutableEvent."""
    event = {"source": "unknown_source", "type": "unknown_type", "payload": "{}"}
    with pytest.raises(UnroutableEvent):
        route_event(event, config)


def test_template_missing_key_left_as_literal():
    """A template referencing a missing payload key is left as-is (not crashing)."""
    from core.router import _render_template
    result = _render_template("{payload[missing_key]}", {})
    assert result == "{payload[missing_key]}"


def test_payload_as_dict_is_also_accepted(config):
    """route_event must handle payload already parsed as a dict (not just a JSON string)."""
    event = {"source": "user", "type": "message", "payload": {"message": "hello dict"}}
    result = route_event(event, config)
    assert result["task_description"] == "hello dict"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_timer_tick_has_no_preflight(config):
    """Timer tick routing must have preflight=None (no task backlog to check)."""
    event = {"source": "timer", "type": "tick", "payload": '{"minute": "2026-01-01T00:00"}'}
    result = route_event(event, config)
    assert result["preflight"] is None


def test_user_message_has_no_preflight(config):
    """User message routing must not include a preflight (never skip on user input)."""
    event = {"source": "user", "type": "message", "payload": '{"message": "hi"}'}
    result = route_event(event, config)
    assert result["preflight"] is None
