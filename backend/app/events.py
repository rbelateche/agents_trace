"""Pydantic event schemas emitted by the SDK and consumed by the worker."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class EventType(StrEnum):
    span_start = "span_start"
    span_end = "span_end"
    tool_call = "tool_call"
    llm_call = "llm_call"
    run_start = "run_start"
    run_end = "run_end"


class RunStartEvent(BaseModel):
    event_type: Literal[EventType.run_start] = EventType.run_start
    run_id: uuid.UUID
    root_agent: str
    started_at: datetime
    metadata: dict[str, object] = Field(default_factory=dict)


class RunEndEvent(BaseModel):
    event_type: Literal[EventType.run_end] = EventType.run_end
    run_id: uuid.UUID
    status: str  # "success" | "error"
    ended_at: datetime
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_msg: str | None = None


class SpanStartEvent(BaseModel):
    event_type: Literal[EventType.span_start] = EventType.span_start
    span_id: uuid.UUID
    run_id: uuid.UUID
    parent_span_id: uuid.UUID | None = None
    agent_name: str
    span_type: str  # "agent" | "tool" | "llm"
    started_at: datetime
    input: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    retrieved_context: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class SpanEndEvent(BaseModel):
    event_type: Literal[EventType.span_end] = EventType.span_end
    span_id: uuid.UUID
    run_id: uuid.UUID
    ended_at: datetime
    output: str | None = None
    status: str = "ok"  # "ok" | "error" | "timeout"
    error_msg: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class ToolCallEvent(BaseModel):
    event_type: Literal[EventType.tool_call] = EventType.tool_call
    tool_call_id: uuid.UUID
    span_id: uuid.UUID
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "success"  # "success" | "error"
    error: str | None = None
    latency_ms: int | None = None


class LLMCallEvent(BaseModel):
    event_type: Literal[EventType.llm_call] = EventType.llm_call
    span_id: uuid.UUID
    run_id: uuid.UUID
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    started_at: datetime
    ended_at: datetime | None = None


# Discriminated union for parsing any event from Redis
AnyEvent = Annotated[
    RunStartEvent | RunEndEvent | SpanStartEvent | SpanEndEvent | ToolCallEvent | LLMCallEvent,
    Field(discriminator="event_type"),
]

REDIS_STREAM_KEY = "agent_trace:events"
REDIS_CONSUMER_GROUP = "backend-workers"
REDIS_CONSUMER_NAME = "worker-1"
