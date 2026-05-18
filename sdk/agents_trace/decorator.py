"""@trace_agent decorator — wraps a callable to auto-emit span events."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

from agents_trace.tracer import AgentTracer

F = TypeVar("F", bound=Callable[..., Any])


def trace_agent(
    tracer: AgentTracer,
    *,
    agent_name: str | None = None,
    span_type: str = "agent",
    parent_span_id: uuid.UUID | None = None,
) -> Callable[[F], F]:
    """Decorator that wraps a function in a span.

    Usage::

        tracer = AgentTracer(root_agent="pipeline")

        @trace_agent(tracer, agent_name="summariser")
        def summarise(text: str) -> str:
            ...
    """

    def decorator(fn: F) -> F:
        name = agent_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(name, span_type=span_type, parent_span_id=parent_span_id):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
