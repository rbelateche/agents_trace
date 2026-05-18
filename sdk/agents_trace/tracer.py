"""AgentTracer — publishes observability events to Redis Streams."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from redis import Redis

logger = logging.getLogger(__name__)

_STREAM_KEY = "agent_trace:events"
_MAXLEN = 100_000  # approximate cap on stream length


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentTracer:
    """Thin client that serialises events and pushes them to Redis Streams.

    All methods are synchronous so they can be called from both sync and
    async contexts without requiring an event loop in the SDK.  The worker
    on the backend side consumes the stream asynchronously.

    Parameters
    ----------
    redis_url:
        Redis connection URL (e.g. ``redis://localhost:6379``).
    run_id:
        UUID for the top-level agent run.  Generated automatically if omitted.
    root_agent:
        Human-readable name of the root agent (e.g. ``"research_workflow"``).
    metadata:
        Arbitrary key/value pairs attached to the run (e.g. user ID, version).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        run_id: uuid.UUID | None = None,
        root_agent: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id: uuid.UUID = run_id or uuid.uuid4()
        self.root_agent = root_agent
        self.metadata: dict[str, Any] = metadata or {}
        self._redis: Redis = Redis.from_url(redis_url, decode_responses=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(self, payload: dict[str, Any]) -> None:
        try:
            self._redis.xadd(
                _STREAM_KEY,
                {"data": json.dumps(payload, default=str)},
                maxlen=_MAXLEN,
                approximate=True,
            )
        except Exception:
            logger.exception("AgentTracer: failed to publish event %s", payload.get("event_type"))

    @staticmethod
    def _ts(dt: datetime | None = None) -> str:
        return (dt or _utcnow()).isoformat()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Emit a RunStartEvent for this tracer's run."""
        self._publish(
            {
                "event_type": "run_start",
                "run_id": str(self.run_id),
                "root_agent": self.root_agent,
                "started_at": self._ts(),
                "metadata": self.metadata,
            }
        )

    def end_run(
        self,
        *,
        status: str = "success",
        total_tokens: int = 0,
        total_cost_usd: float = 0.0,
        error_msg: str | None = None,
    ) -> None:
        """Emit a RunEndEvent."""
        self._publish(
            {
                "event_type": "run_end",
                "run_id": str(self.run_id),
                "status": status,
                "ended_at": self._ts(),
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "error_msg": error_msg,
            }
        )

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        *,
        span_id: uuid.UUID | None = None,
        parent_span_id: uuid.UUID | None = None,
        agent_name: str,
        span_type: str = "agent",
        input: str | None = None,  # noqa: A002
        model: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        retrieved_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Emit a SpanStartEvent and return the span_id."""
        sid = span_id or uuid.uuid4()
        self._publish(
            {
                "event_type": "span_start",
                "span_id": str(sid),
                "run_id": str(self.run_id),
                "parent_span_id": str(parent_span_id) if parent_span_id else None,
                "agent_name": agent_name,
                "span_type": span_type,
                "started_at": self._ts(),
                "input": input,
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "retrieved_context": retrieved_context,
                "metadata": metadata or {},
            }
        )
        return sid

    def end_span(
        self,
        span_id: uuid.UUID,
        *,
        output: str | None = None,
        status: str = "ok",
        error_msg: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Emit a SpanEndEvent."""
        self._publish(
            {
                "event_type": "span_end",
                "span_id": str(span_id),
                "run_id": str(self.run_id),
                "ended_at": self._ts(),
                "output": output,
                "status": status,
                "error_msg": error_msg,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            }
        )

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def record_tool_call(
        self,
        span_id: uuid.UUID,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        result: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        status: str = "success",
        error: str | None = None,
        latency_ms: int | None = None,
    ) -> uuid.UUID:
        """Emit a ToolCallEvent and return the tool_call_id."""
        tc_id = uuid.uuid4()
        self._publish(
            {
                "event_type": "tool_call",
                "tool_call_id": str(tc_id),
                "span_id": str(span_id),
                "tool_name": tool_name,
                "arguments": arguments or {},
                "result": result,
                "started_at": self._ts(started_at),
                "ended_at": self._ts(ended_at) if ended_at else None,
                "status": status,
                "error": error,
                "latency_ms": latency_ms,
            }
        )
        return tc_id

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        span_id: uuid.UUID,
        *,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        """Emit an LLMCallEvent."""
        self._publish(
            {
                "event_type": "llm_call",
                "span_id": str(span_id),
                "run_id": str(self.run_id),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "started_at": self._ts(started_at),
                "ended_at": self._ts(ended_at) if ended_at else None,
            }
        )

    # ------------------------------------------------------------------
    # Context manager helpers for spans
    # ------------------------------------------------------------------

    def span(
        self,
        agent_name: str,
        span_type: str = "agent",
        parent_span_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> _SpanContext:
        """Return a context manager that wraps a span's lifecycle."""
        return _SpanContext(
            self,
            agent_name=agent_name,
            span_type=span_type,
            parent_span_id=parent_span_id,
            **kwargs,
        )

    def close(self) -> None:
        """Close the underlying Redis connection."""
        self._redis.close()

    def __enter__(self) -> AgentTracer:
        self.start_run()
        return self

    def __exit__(self, exc_type: type | None, *_: object) -> None:
        status = "error" if exc_type else "success"
        self.end_run(status=status)
        self.close()


class _SpanContext:
    """Context manager for a single span."""

    def __init__(
        self,
        tracer: AgentTracer,
        agent_name: str,
        span_type: str = "agent",
        parent_span_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tracer = tracer
        self._agent_name = agent_name
        self._span_type = span_type
        self._parent_span_id = parent_span_id
        self._kwargs = kwargs
        self.span_id: uuid.UUID = uuid.uuid4()

    def __enter__(self) -> _SpanContext:
        self.span_id = self._tracer.start_span(
            span_id=self.span_id,
            parent_span_id=self._parent_span_id,
            agent_name=self._agent_name,
            span_type=self._span_type,
            **self._kwargs,
        )
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, _: object) -> None:
        status = "error" if exc_type else "ok"
        error_msg = str(exc_val) if exc_val else None
        self._tracer.end_span(self.span_id, status=status, error_msg=error_msg)
