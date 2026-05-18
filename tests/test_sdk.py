"""Tests for agents_trace SDK — AgentTracer, decorator, and LangGraph handler."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agents_trace import AgentTracer, trace_agent
from agents_trace.langgraph_handler import AgentTraceCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracer() -> tuple[AgentTracer, MagicMock]:
    """Return (tracer, mock_redis) with Redis patched out."""
    with patch("agents_trace.tracer.Redis") as MockRedis:
        mock_client = MagicMock()
        MockRedis.from_url.return_value = mock_client
        tracer = AgentTracer(redis_url="redis://localhost:6379", root_agent="test_agent")
    tracer._redis = mock_client  # re-attach mock after __init__
    return tracer, mock_client


def _last_published(mock_client: MagicMock) -> dict:
    """Return the payload dict of the last xadd call."""
    call_args = mock_client.xadd.call_args
    data_str = call_args[0][1]["data"]
    return json.loads(data_str)


# ---------------------------------------------------------------------------
# AgentTracer tests
# ---------------------------------------------------------------------------


def test_start_run_emits_run_start_event() -> None:
    tracer, mock_client = _make_tracer()
    tracer.start_run()

    assert mock_client.xadd.called
    payload = _last_published(mock_client)
    assert payload["event_type"] == "run_start"
    assert payload["run_id"] == str(tracer.run_id)
    assert payload["root_agent"] == "test_agent"


def test_end_run_emits_run_end_event() -> None:
    tracer, mock_client = _make_tracer()
    tracer.end_run(status="success", total_tokens=42)

    payload = _last_published(mock_client)
    assert payload["event_type"] == "run_end"
    assert payload["status"] == "success"
    assert payload["total_tokens"] == 42


def test_start_span_returns_uuid_and_emits_event() -> None:
    tracer, mock_client = _make_tracer()
    span_id = tracer.start_span(agent_name="planner", span_type="agent")

    assert isinstance(span_id, uuid.UUID)
    payload = _last_published(mock_client)
    assert payload["event_type"] == "span_start"
    assert payload["agent_name"] == "planner"
    assert payload["span_type"] == "agent"


def test_end_span_emits_span_end_event() -> None:
    tracer, mock_client = _make_tracer()
    span_id = tracer.start_span(agent_name="summariser")
    mock_client.reset_mock()

    tracer.end_span(span_id, output="Done", status="ok", prompt_tokens=10)

    payload = _last_published(mock_client)
    assert payload["event_type"] == "span_end"
    assert payload["output"] == "Done"
    assert payload["prompt_tokens"] == 10


def test_record_tool_call_emits_event() -> None:
    tracer, mock_client = _make_tracer()
    span_id = tracer.start_span(agent_name="tool_user")
    mock_client.reset_mock()

    tc_id = tracer.record_tool_call(
        span_id, tool_name="web_search", arguments={"q": "AI news"}, result="found"
    )

    assert isinstance(tc_id, uuid.UUID)
    payload = _last_published(mock_client)
    assert payload["event_type"] == "tool_call"
    assert payload["tool_name"] == "web_search"
    assert payload["arguments"]["q"] == "AI news"


def test_record_llm_call_emits_event() -> None:
    tracer, mock_client = _make_tracer()
    span_id = tracer.start_span(agent_name="llm_node", span_type="llm")
    mock_client.reset_mock()

    tracer.record_llm_call(
        span_id, model="gpt-4o", prompt_tokens=100, completion_tokens=50, cost_usd=0.002
    )

    payload = _last_published(mock_client)
    assert payload["event_type"] == "llm_call"
    assert payload["model"] == "gpt-4o"
    assert payload["cost_usd"] == pytest.approx(0.002)


def test_context_manager_emits_run_start_and_end() -> None:
    tracer, mock_client = _make_tracer()
    calls: list[str] = []
    mock_client.xadd.side_effect = lambda *a, **kw: calls.append(
        json.loads(a[1]["data"])["event_type"]
    )

    with tracer:
        pass

    assert calls[0] == "run_start"
    assert calls[-1] == "run_end"


def test_span_context_manager_emits_start_and_end() -> None:
    tracer, mock_client = _make_tracer()
    calls: list[str] = []
    mock_client.xadd.side_effect = lambda *a, **kw: calls.append(
        json.loads(a[1]["data"])["event_type"]
    )

    with tracer.span("researcher"):
        pass

    assert "span_start" in calls
    assert "span_end" in calls


def test_redis_failure_does_not_propagate() -> None:
    tracer, mock_client = _make_tracer()
    mock_client.xadd.side_effect = ConnectionError("Redis down")

    # Should log but not raise
    tracer.start_run()  # no exception


# ---------------------------------------------------------------------------
# @trace_agent decorator tests
# ---------------------------------------------------------------------------


def test_trace_agent_decorator_calls_wrapped_function() -> None:
    tracer, mock_client = _make_tracer()

    @trace_agent(tracer, agent_name="adder")
    def add(x: int, y: int) -> int:
        return x + y

    result = add(2, 3)
    assert result == 5


def test_trace_agent_emits_span_events() -> None:
    tracer, mock_client = _make_tracer()
    event_types: list[str] = []
    mock_client.xadd.side_effect = lambda *a, **kw: event_types.append(
        json.loads(a[1]["data"])["event_type"]
    )

    @trace_agent(tracer, agent_name="greeter")
    def greet(name: str) -> str:
        return f"Hello, {name}"

    greet("world")
    assert "span_start" in event_types
    assert "span_end" in event_types


# ---------------------------------------------------------------------------
# LangGraph callback handler tests
# ---------------------------------------------------------------------------


def test_langgraph_handler_chain_start_emits_span() -> None:
    tracer, mock_client = _make_tracer()
    cb = AgentTraceCallback(tracer)

    run_id = uuid.uuid4()
    cb.on_chain_start(
        {"name": "PlannerAgent"},
        {"input": "Research AI"},
        run_id=run_id,
    )

    payload = _last_published(mock_client)
    assert payload["event_type"] == "span_start"
    assert payload["agent_name"] == "PlannerAgent"
    assert run_id in cb._span_map


def test_langgraph_handler_chain_end_closes_span() -> None:
    tracer, mock_client = _make_tracer()
    cb = AgentTraceCallback(tracer)

    run_id = uuid.uuid4()
    cb.on_chain_start({"name": "Summariser"}, {}, run_id=run_id)
    mock_client.reset_mock()
    cb.on_chain_end({"output": "done"}, run_id=run_id)

    payload = _last_published(mock_client)
    assert payload["event_type"] == "span_end"
    assert payload["status"] == "ok"
    assert run_id not in cb._span_map


def test_langgraph_handler_chain_error_marks_span_error() -> None:
    tracer, mock_client = _make_tracer()
    cb = AgentTraceCallback(tracer)

    run_id = uuid.uuid4()
    cb.on_chain_start({"name": "Worker"}, {}, run_id=run_id)
    mock_client.reset_mock()
    cb.on_chain_error(ValueError("boom"), run_id=run_id)

    payload = _last_published(mock_client)
    assert payload["event_type"] == "span_end"
    assert payload["status"] == "error"
    assert "boom" in payload["error_msg"]


def test_langgraph_handler_parent_span_linking() -> None:
    tracer, mock_client = _make_tracer()
    cb = AgentTraceCallback(tracer)

    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    cb.on_chain_start({"name": "Parent"}, {}, run_id=parent_id)
    cb.on_chain_start({"name": "Child"}, {}, run_id=child_id, parent_run_id=parent_id)

    # last event should be the child span_start
    payload = _last_published(mock_client)
    assert payload["event_type"] == "span_start"
    assert payload["agent_name"] == "Child"
    # parent_span_id should link to parent
    parent_span_id = str(cb._span_map[parent_id])
    assert payload["parent_span_id"] == parent_span_id
