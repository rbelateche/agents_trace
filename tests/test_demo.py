"""Tests for the demo research workflow."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from demo.tools import reset_call_counts, web_search, retrieve_papers


# ---------------------------------------------------------------------------
# Tools tests
# ---------------------------------------------------------------------------


def test_web_search_first_call_returns_invalid_json() -> None:
    reset_call_counts()
    result = web_search("test query")
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)


def test_web_search_second_call_returns_valid_json() -> None:
    reset_call_counts()
    web_search("q1")  # first call — malformed
    result = web_search("q2")  # second call — valid
    data = json.loads(result)
    assert "results" in data
    assert len(data["results"]) > 0


def test_retrieve_papers_returns_valid_json() -> None:
    reset_call_counts()
    result = retrieve_papers("AI safety")
    data = json.loads(result)
    assert "papers" in data
    assert all("title" in p for p in data["papers"])


def test_reset_call_counts_resets_failure_behaviour() -> None:
    reset_call_counts()
    web_search("q")  # first = bad
    reset_call_counts()
    result = web_search("q")  # should be bad again (call count reset)
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)


# ---------------------------------------------------------------------------
# Workflow integration tests (Redis mocked out)
# ---------------------------------------------------------------------------


def _mock_redis_tracer() -> MagicMock:
    """Return a mock Redis client that swallows all xadd calls."""
    mock = MagicMock()
    mock.xadd.return_value = "0-0"
    return mock


def test_run_workflow_returns_summary_and_critique() -> None:
    """Full workflow end-to-end with Redis mocked."""
    from agents_trace.tracer import AgentTracer

    with patch("agents_trace.tracer.Redis") as MockRedis:
        MockRedis.from_url.return_value = _mock_redis_tracer()
        reset_call_counts()
        from demo.research_workflow import run_workflow

        result = run_workflow(topic="AI safety", redis_url="redis://localhost")

    assert "summary" in result
    assert "critique" in result
    assert "AI safety" in result["summary"]
    assert result["error_count"] >= 1  # at least one web_search failure


def test_run_workflow_produces_sub_questions() -> None:
    from agents_trace.tracer import AgentTracer

    with patch("agents_trace.tracer.Redis") as MockRedis:
        MockRedis.from_url.return_value = _mock_redis_tracer()
        reset_call_counts()
        from demo.research_workflow import run_workflow

        result = run_workflow(topic="robotics")

    assert len(result.get("sub_questions", [])) == 3
    for q in result["sub_questions"]:
        assert "robotics" in q


def test_run_workflow_emits_redis_events() -> None:
    """Verify that events are published to Redis during the workflow."""
    mock_client = _mock_redis_tracer()

    with patch("agents_trace.tracer.Redis") as MockRedis:
        MockRedis.from_url.return_value = mock_client
        reset_call_counts()
        from demo.research_workflow import run_workflow

        run_workflow(topic="test topic", redis_url="redis://localhost")

    # At least: run_start, 4× span_start, 4× span_end, tool_calls, run_end
    assert mock_client.xadd.call_count >= 10

    event_types = []
    for call in mock_client.xadd.call_args_list:
        payload = json.loads(call[0][1]["data"])
        event_types.append(payload["event_type"])

    assert "run_start" in event_types
    assert "run_end" in event_types
    assert "span_start" in event_types
    assert "span_end" in event_types
    assert "tool_call" in event_types
