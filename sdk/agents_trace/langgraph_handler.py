"""LangGraph CallbackHandler for agents_trace.

Instruments a LangGraph graph by listening to chain/tool/LLM events
and emitting the corresponding agents_trace span events.

Usage::

    from agents_trace import AgentTracer
    from agents_trace.langgraph_handler import AgentTraceCallback

    tracer = AgentTracer(root_agent="my_graph")
    callback = AgentTraceCallback(tracer)

    graph.invoke(state, config={"callbacks": [callback]})
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from agents_trace.tracer import AgentTracer

# LangChain's callback base is optional — only needed when LangGraph is installed.
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object
    LLMResult = Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentTraceCallback(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain/LangGraph callback that streams events to agents_trace."""

    def __init__(self, tracer: AgentTracer) -> None:
        super().__init__()
        self._tracer = tracer
        # Maps langchain run_id → agents_trace span_id
        self._span_map: dict[UUID, uuid.UUID] = {}
        self._start_times: dict[UUID, datetime] = {}

    # ------------------------------------------------------------------
    # Chain (agent node) events
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        agent_name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        parent_span_id = self._span_map.get(parent_run_id) if parent_run_id else None
        span_id = self._tracer.start_span(
            agent_name=str(agent_name),
            span_type="agent",
            parent_span_id=parent_span_id,
            input=str(inputs)[:2000],
            metadata=metadata or {},
        )
        self._span_map[run_id] = span_id
        self._start_times[run_id] = _utcnow()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span_id = self._span_map.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span_id:
            self._tracer.end_span(span_id, output=str(outputs)[:2000], status="ok")

    def on_chain_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span_id = self._span_map.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span_id:
            self._tracer.end_span(span_id, status="error", error_msg=str(error)[:500])

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        parent_span_id = self._span_map.get(parent_run_id) if parent_run_id else None
        span_id = self._tracer.start_span(
            agent_name=str(tool_name),
            span_type="tool",
            parent_span_id=parent_span_id,
            input=input_str[:2000],
        )
        self._span_map[run_id] = span_id
        self._start_times[run_id] = _utcnow()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span_id = self._span_map.pop(run_id, None)
        started = self._start_times.pop(run_id, None)
        if span_id:
            latency = int((_utcnow() - started).total_seconds() * 1000) if started else None
            self._tracer.end_span(span_id, output=str(output)[:2000], status="ok")
            self._tracer.record_tool_call(
                span_id,
                tool_name="tool",
                result=str(output)[:2000],
                latency_ms=latency,
            )

    def on_tool_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span_id = self._span_map.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span_id:
            self._tracer.end_span(span_id, status="error", error_msg=str(error)[:500])

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = (invocation_params or {}).get("model_name", serialized.get("name", "unknown"))
        parent_span_id = self._span_map.get(parent_run_id) if parent_run_id else None
        span_id = self._tracer.start_span(
            agent_name=str(model),
            span_type="llm",
            parent_span_id=parent_span_id,
            model=str(model),
            user_prompt=(prompts[0] if prompts else None),
        )
        self._span_map[run_id] = span_id
        self._start_times[run_id] = _utcnow()

    def on_llm_end(
        self,
        response: Any,  # LLMResult
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span_id = self._span_map.pop(run_id, None)
        started = self._start_times.pop(run_id, None)
        if span_id is None:
            return

        # Extract token usage if available
        token_usage: dict[str, Any] = {}
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
        prompt_tokens: int = token_usage.get("prompt_tokens", 0)
        completion_tokens: int = token_usage.get("completion_tokens", 0)

        self._tracer.end_span(
            span_id,
            status="ok",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._tracer.record_llm_call(
            span_id,
            model="unknown",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            started_at=started,
            ended_at=_utcnow(),
        )

    def on_llm_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span_id = self._span_map.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span_id:
            self._tracer.end_span(span_id, status="error", error_msg=str(error)[:500])
