"""initial schema: runs, spans, tool_calls, prompt_snapshots

Revision ID: 4e85d2a27327
Revises:
Create Date: 2026-05-09 17:05:17.873409

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql  # noqa: F401
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4e85d2a27327"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "status",
            sa.Enum("running", "success", "error", name="runstatus"),
            nullable=False,
            server_default="running",
        ),
        sa.Column("root_agent", sa.String(255), nullable=True),
        sa.Column("total_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "parent_run_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id"),
            nullable=True,
        ),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, nullable=True),
    )

    op.create_table(
        "spans",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id"),
            nullable=False,
        ),
        sa.Column(
            "parent_span_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spans.id"),
            nullable=True,
        ),
        sa.Column("agent_name", sa.String(255), nullable=False),
        sa.Column("span_type", sa.Enum("agent", "tool", "llm", name="spantype"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("input", sa.Text, nullable=True),
        sa.Column("output", sa.Text, nullable=True),
        sa.Column("model", sa.String(255), nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column(
            "status",
            sa.Enum("ok", "error", "timeout", name="spanstatus"),
            nullable=False,
            server_default="ok",
        ),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.Column(
            "failure_category",
            sa.Enum(
                "hallucinated_tool",
                "invalid_json",
                "timeout",
                "loop",
                "context_overflow",
                "retrieval_failure",
                "permission",
                "unknown",
                name="failurecategory",
            ),
            nullable=True,
        ),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, nullable=True),
    )

    op.create_table(
        "tool_calls",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "span_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spans.id"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("arguments", sa.dialects.postgresql.JSONB, nullable=True),
        sa.Column("result", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Enum("success", "error", name="toolcallstatus"), nullable=False),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
    )

    op.create_table(
        "prompt_snapshots",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "span_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spans.id"),
            nullable=False,
        ),
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column("user_prompt", sa.Text, nullable=True),
        sa.Column("retrieved_context", sa.Text, nullable=True),
        sa.Column("model", sa.String(255), nullable=True),
        sa.Column("temperature", sa.Float, nullable=True),
        sa.Column("version", sa.String(50), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Indexes for common query patterns
    op.create_index("ix_spans_run_id", "spans", ["run_id"])
    op.create_index("ix_spans_parent_span_id", "spans", ["parent_span_id"])
    op.create_index("ix_tool_calls_span_id", "tool_calls", ["span_id"])
    op.create_index("ix_prompt_snapshots_span_id", "prompt_snapshots", ["span_id"])
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_index("ix_runs_created_at", "runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_runs_created_at", table_name="runs")
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_prompt_snapshots_span_id", table_name="prompt_snapshots")
    op.drop_index("ix_tool_calls_span_id", table_name="tool_calls")
    op.drop_index("ix_spans_parent_span_id", table_name="spans")
    op.drop_index("ix_spans_run_id", table_name="spans")
    op.drop_table("prompt_snapshots")
    op.drop_table("tool_calls")
    op.drop_table("spans")
    op.drop_table("runs")
    op.execute("DROP TYPE IF EXISTS failurecategory")
    op.execute("DROP TYPE IF EXISTS spanstatus")
    op.execute("DROP TYPE IF EXISTS spantype")
    op.execute("DROP TYPE IF EXISTS toolcallstatus")
    op.execute("DROP TYPE IF EXISTS runstatus")
