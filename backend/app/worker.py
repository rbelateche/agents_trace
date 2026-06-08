"""Redis Streams consumer worker.

Reads events from the `agent_trace:events` stream and persists them to Postgres.
Runs as an async background task wired into the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import cast

from pydantic import TypeAdapter, ValidationError
from redis.asyncio import Redis
from redis.asyncio import ResponseError as RedisResponseError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.events import (
    REDIS_CONSUMER_GROUP,
    REDIS_CONSUMER_NAME,
    REDIS_STREAM_KEY,
    AnyEvent,
    LLMCallEvent,
    RunEndEvent,
    RunStartEvent,
    SpanEndEvent,
    SpanStartEvent,
    ToolCallEvent,
)
from app.models import (
    PromptSnapshot,
    Run,
    RunStatus,
    Span,
    SpanStatus,
    SpanType,
    ToolCall,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)

_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


async def _ensure_consumer_group(client: Redis) -> None:
    try:
        await client.xgroup_create(REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP, id="0", mkstream=True)
    except RedisResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def _handle_run_start(session: AsyncSession, event: RunStartEvent) -> None:
    run = Run(
        id=event.run_id,
        root_agent=event.root_agent,
        status=RunStatus.running,
        metadata_=event.metadata or None,
    )
    session.add(run)


async def _handle_run_end(session: AsyncSession, event: RunEndEvent) -> None:
    run = await session.get(Run, event.run_id)
    if run is None:
        logger.warning("run_end for unknown run %s", event.run_id)
        return
    run.status = RunStatus.success if event.status == "success" else RunStatus.error
    run.total_tokens = event.total_tokens
    run.total_cost_usd = event.total_cost_usd
    if run.created_at:
        delta = event.ended_at - run.created_at
        run.duration_ms = int(delta.total_seconds() * 1000)


async def _handle_span_start(session: AsyncSession, event: SpanStartEvent) -> None:
    span = Span(
        id=event.span_id,
        run_id=event.run_id,
        parent_span_id=event.parent_span_id,
        agent_name=event.agent_name,
        span_type=SpanType(event.span_type),
        started_at=event.started_at,
        input=event.input,
        model=event.model,
        status=SpanStatus.ok,
        metadata_=event.metadata or None,
    )
    session.add(span)

    if event.system_prompt or event.user_prompt:
        snapshot = PromptSnapshot(
            span_id=event.span_id,
            system_prompt=event.system_prompt,
            user_prompt=event.user_prompt,
            retrieved_context=event.retrieved_context,
            model=event.model,
        )
        session.add(snapshot)


async def _handle_span_end(session: AsyncSession, event: SpanEndEvent) -> None:
    span = await session.get(Span, event.span_id)
    if span is None:
        logger.warning("span_end for unknown span %s", event.span_id)
        return
    span.ended_at = event.ended_at
    span.output = event.output
    span.status = SpanStatus(event.status)
    span.error_msg = event.error_msg
    span.prompt_tokens = event.prompt_tokens
    span.completion_tokens = event.completion_tokens
    span.cost_usd = event.cost_usd


async def _handle_tool_call(session: AsyncSession, event: ToolCallEvent) -> None:
    tool_call = ToolCall(
        id=event.tool_call_id,
        span_id=event.span_id,
        tool_name=event.tool_name,
        arguments=event.arguments,
        result=event.result,
        started_at=event.started_at,
        ended_at=event.ended_at,
        status=ToolCallStatus(event.status),
        error=event.error,
        latency_ms=event.latency_ms,
    )
    session.add(tool_call)


async def _handle_llm_call(session: AsyncSession, event: LLMCallEvent) -> None:
    span = await session.get(Span, event.span_id)
    if span is None:
        return
    span.model = event.model
    span.prompt_tokens += event.prompt_tokens
    span.completion_tokens += event.completion_tokens
    span.cost_usd += event.cost_usd


async def _process_event(session: AsyncSession, raw: dict[str, str]) -> None:
    payload = raw.get("data", "{}")
    try:
        data = json.loads(payload)
        event = _event_adapter.validate_python(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("Failed to parse event: %s — %s", payload[:200], exc)
        return

    if isinstance(event, RunStartEvent):
        await _handle_run_start(session, event)
    elif isinstance(event, RunEndEvent):
        await _handle_run_end(session, event)
    elif isinstance(event, SpanStartEvent):
        await _handle_span_start(session, event)
    elif isinstance(event, SpanEndEvent):
        await _handle_span_end(session, event)
    elif isinstance(event, ToolCallEvent):
        await _handle_tool_call(session, event)
    elif isinstance(event, LLMCallEvent):
        await _handle_llm_call(session, event)


async def consume_forever(stop_event: asyncio.Event) -> None:
    """Main consumer loop. Reads from Redis Streams, writes to Postgres."""
    client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await _ensure_consumer_group(client)
    logger.info("Redis Streams worker started on %s", REDIS_STREAM_KEY)

    while not stop_event.is_set():
        try:
            raw_results = await client.xreadgroup(
                groupname=REDIS_CONSUMER_GROUP,
                consumername=REDIS_CONSUMER_NAME,
                streams={REDIS_STREAM_KEY: ">"},
                count=50,
                block=1000,  # ms — yields control when idle
            )
            if not raw_results:
                continue

            # redis-py's async stubs type this loosely; with decode_responses=True
            # the concrete shape is list[(stream, list[(msg_id, fields)])].
            results = cast("list[tuple[str, list[tuple[str, dict[str, str]]]]]", raw_results)
            async with AsyncSessionLocal() as session:
                for _stream, messages in results:
                    for msg_id, fields in messages:
                        await _process_event(session, fields)
                        await client.xack(REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP, msg_id)
                await session.commit()

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Worker error — retrying in 2s")
            await asyncio.sleep(2)

    await client.aclose()
    logger.info("Redis Streams worker stopped")
