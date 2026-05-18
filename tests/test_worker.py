"""Tests for the Redis Streams worker and event schemas."""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.events import (
    REDIS_STREAM_KEY,
    RunStartEvent,
    SpanEndEvent,
    SpanStartEvent,
    ToolCallEvent,
    EventType,
)
from app.worker import _process_event


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Event schema round-trip tests
# ---------------------------------------------------------------------------


def test_run_start_event_serialises() -> None:
    run_id = uuid.uuid4()
    event = RunStartEvent(
        run_id=run_id,
        root_agent="planner",
        started_at=_utcnow(),
    )
    assert event.event_type == EventType.run_start
    data = json.loads(event.model_dump_json())
    assert data["run_id"] == str(run_id)


def test_span_start_event_defaults() -> None:
    event = SpanStartEvent(
        span_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        agent_name="researcher",
        span_type="agent",
        started_at=_utcnow(),
    )
    assert event.parent_span_id is None
    assert event.metadata == {}


def test_tool_call_event_serialises() -> None:
    tc = ToolCallEvent(
        tool_call_id=uuid.uuid4(),
        span_id=uuid.uuid4(),
        tool_name="web_search",
        arguments={"query": "AI news"},
        started_at=_utcnow(),
    )
    assert tc.status == "success"
    assert tc.event_type == EventType.tool_call


# ---------------------------------------------------------------------------
# _process_event dispatch tests (mocked session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_event_run_start_adds_run() -> None:
    session = AsyncMock()
    run_id = uuid.uuid4()
    payload = RunStartEvent(
        run_id=run_id,
        root_agent="planner",
        started_at=_utcnow(),
    ).model_dump_json()

    fields: dict[str, str] = {"data": payload}
    await _process_event(session, fields)  # type: ignore[arg-type]
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_process_event_invalid_json_does_not_raise() -> None:
    session = AsyncMock()
    fields: dict[str, str] = {"data": "not-json{{{"}
    # Should log error but not raise
    await _process_event(session, fields)  # type: ignore[arg-type]
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_process_event_span_start_adds_span() -> None:
    session = AsyncMock()
    payload = SpanStartEvent(
        span_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        agent_name="researcher",
        span_type="agent",
        started_at=_utcnow(),
    ).model_dump_json()

    await _process_event(session, {"data": payload})  # type: ignore[arg-type]
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_process_event_span_start_with_prompt_adds_two_rows() -> None:
    session = AsyncMock()
    payload = SpanStartEvent(
        span_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        agent_name="researcher",
        span_type="llm",
        started_at=_utcnow(),
        system_prompt="You are a helpful assistant.",
        user_prompt="Summarise AI trends.",
    ).model_dump_json()

    await _process_event(session, {"data": payload})  # type: ignore[arg-type]
    # One Span + one PromptSnapshot
    assert session.add.call_count == 2


@pytest.mark.asyncio
async def test_process_event_span_end_updates_span() -> None:
    session = AsyncMock()
    span_id = uuid.uuid4()
    mock_span = MagicMock()
    session.get = AsyncMock(return_value=mock_span)

    payload = SpanEndEvent(
        span_id=span_id,
        run_id=uuid.uuid4(),
        ended_at=_utcnow(),
        output="Done",
        status="ok",
    ).model_dump_json()

    await _process_event(session, {"data": payload})  # type: ignore[arg-type]
    assert mock_span.output == "Done"
