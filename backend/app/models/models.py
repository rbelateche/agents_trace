import enum
import uuid
from datetime import datetime

from app.database import Base
from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class RunStatus(enum.StrEnum):
    running = "running"
    success = "success"
    error = "error"


class SpanType(enum.StrEnum):
    agent = "agent"
    tool = "tool"
    llm = "llm"


class SpanStatus(enum.StrEnum):
    ok = "ok"
    error = "error"
    timeout = "timeout"


class ToolCallStatus(enum.StrEnum):
    success = "success"
    error = "error"


class FailureCategory(enum.StrEnum):
    hallucinated_tool = "hallucinated_tool"
    invalid_json = "invalid_json"
    timeout = "timeout"
    loop = "loop"
    context_overflow = "context_overflow"
    retrieval_failure = "retrieval_failure"
    permission = "permission"
    unknown = "unknown"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.running)
    root_agent: Mapped[str | None] = mapped_column(String(255))
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=True
    )
    metadata_: Mapped[dict[str, object] | None] = mapped_column("metadata", JSONB)

    spans: Mapped[list["Span"]] = relationship(
        "Span", back_populates="run", cascade="all, delete-orphan"
    )


class Span(Base):
    __tablename__ = "spans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    parent_span_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spans.id"), nullable=True
    )
    agent_name: Mapped[str] = mapped_column(String(255))
    span_type: Mapped[SpanType] = mapped_column(Enum(SpanType))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    input: Mapped[str | None] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(255))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[SpanStatus] = mapped_column(Enum(SpanStatus), default=SpanStatus.ok)
    error_msg: Mapped[str | None] = mapped_column(Text)
    failure_category: Mapped[FailureCategory | None] = mapped_column(Enum(FailureCategory))
    metadata_: Mapped[dict[str, object] | None] = mapped_column("metadata", JSONB)

    run: Mapped["Run"] = relationship("Run", back_populates="spans")
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        "ToolCall", back_populates="span", cascade="all, delete-orphan"
    )
    prompt_snapshots: Mapped[list["PromptSnapshot"]] = relationship(
        "PromptSnapshot", back_populates="span", cascade="all, delete-orphan"
    )
    children: Mapped[list["Span"]] = relationship("Span", back_populates="parent")
    parent: Mapped["Span | None"] = relationship(
        "Span", back_populates="children", remote_side=[id]
    )


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    span_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spans.id"), nullable=False
    )
    tool_name: Mapped[str] = mapped_column(String(255))
    arguments: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    result: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[ToolCallStatus] = mapped_column(Enum(ToolCallStatus))
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    span: Mapped["Span"] = relationship("Span", back_populates="tool_calls")


class PromptSnapshot(Base):
    __tablename__ = "prompt_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    span_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spans.id"), nullable=False
    )
    system_prompt: Mapped[str | None] = mapped_column(Text)
    user_prompt: Mapped[str | None] = mapped_column(Text)
    retrieved_context: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(255))
    temperature: Mapped[float | None] = mapped_column(Float)
    version: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    span: Mapped["Span"] = relationship("Span", back_populates="prompt_snapshots")
