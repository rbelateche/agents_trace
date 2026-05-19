"""Multi-agent research workflow built with LangGraph + agents_trace SDK.

Architecture
------------
    user_input
        │
        ▼
  PlannerAgent  ─── decomposes the topic into sub-questions
        │
        ▼
  ResearchAgent ─── web_search + retrieve_papers (first call fails → retry)
        │
        ▼
 SummarizerAgent ── synthesises findings into a structured report
        │
        ▼
   CriticAgent  ─── scores the report and flags gaps
        │
        ▼
   final_output

Every node emits spans via AgentTracer.  The web_search tool failure on the
first call is intentional: it demonstrates the platform's failure detection.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents_trace.langgraph_handler import AgentTraceCallback
from agents_trace.tracer import AgentTracer
from demo.tools import reset_call_counts, retrieve_papers, web_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    """Shared state passed between nodes."""

    topic: str
    sub_questions: list[str]
    raw_research: list[str]
    summary: str
    critique: str
    error_count: int


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def planner_node(state: ResearchState, tracer: AgentTracer) -> ResearchState:
    """Decompose the topic into 3 focused sub-questions."""
    with tracer.span("PlannerAgent", span_type="agent", input=state["topic"]):
        # In production this would call an LLM.  For the demo we use a
        # deterministic decomposition so the workflow runs without API keys.
        topic = state["topic"]
        sub_questions = [
            f"What are the latest advances in {topic}?",
            f"What are the main challenges in {topic}?",
            f"What is the future outlook for {topic}?",
        ]
        logger.info("[PlannerAgent] Generated %d sub-questions", len(sub_questions))
    return {**state, "sub_questions": sub_questions}


def research_node(state: ResearchState, tracer: AgentTracer) -> ResearchState:
    """Execute tool calls to gather raw research material."""
    results: list[str] = []
    error_count = state.get("error_count", 0)

    with tracer.span("ResearchAgent", span_type="agent") as ctx:
        for question in state["sub_questions"]:
            # web_search — first call will return malformed JSON
            t_start = time.monotonic()
            raw = web_search(question)
            try:
                parsed = json.loads(raw)
                result_text = json.dumps(parsed)
                tc_status = "success"
                tc_error = None
            except json.JSONDecodeError as e:
                logger.warning("[ResearchAgent] web_search returned malformed JSON: %s", e)
                error_count += 1
                tc_status = "error"
                tc_error = str(e)
                # Retry once
                raw = web_search(question)
                result_text = raw
            latency = int((time.monotonic() - t_start) * 1000)
            tracer.record_tool_call(
                ctx.span_id,
                tool_name="web_search",
                arguments={"query": question},
                result=result_text[:500],
                status=tc_status,
                error=tc_error,
                latency_ms=latency,
            )
            results.append(result_text)

            # retrieve_papers
            papers_raw = retrieve_papers(question)
            tracer.record_tool_call(
                ctx.span_id,
                tool_name="retrieve_papers",
                arguments={"topic": question},
                result=papers_raw[:500],
                status="success",
            )
            results.append(papers_raw)

    return {**state, "raw_research": results, "error_count": error_count}


def summarizer_node(state: ResearchState, tracer: AgentTracer) -> ResearchState:
    """Synthesise raw research into a structured markdown report."""
    with tracer.span("SummarizerAgent", span_type="llm") as ctx:
        combined = "\n\n".join(state["raw_research"][:2000])
        summary = (
            f"# Research Summary: {state['topic']}\n\n"
            f"## Key Findings\n"
            f"- Large language models are rapidly advancing across all domains.\n"
            f"- Multi-agent architectures decompose complex tasks effectively.\n"
            f"- Open challenges include alignment, cost, and latency.\n\n"
            f"## Sources Consulted\n"
            f"{combined[:300]}..."
        )
        tracer.record_llm_call(
            ctx.span_id,
            model="gpt-4o-mini",
            prompt_tokens=len(combined) // 4,
            completion_tokens=len(summary) // 4,
            cost_usd=0.0002,
        )
    return {**state, "summary": summary}


def critic_node(state: ResearchState, tracer: AgentTracer) -> ResearchState:
    """Score the summary and identify gaps."""
    with tracer.span("CriticAgent", span_type="llm") as ctx:
        critique = (
            "**Score: 7/10**\n\n"
            "Strengths: Good breadth of coverage, clear structure.\n"
            "Gaps: Missing quantitative benchmarks; no primary sources cited.\n"
            f"Error rate during research: {state['error_count']} tool failure(s) encountered."
        )
        tracer.record_llm_call(
            ctx.span_id,
            model="gpt-4o-mini",
            prompt_tokens=len(state["summary"]) // 4,
            completion_tokens=len(critique) // 4,
            cost_usd=0.0001,
        )
    return {**state, "critique": critique}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(tracer: AgentTracer) -> Any:
    """Assemble the LangGraph StateGraph with tracer injected via closures."""

    def _planner(state: ResearchState) -> ResearchState:
        return planner_node(state, tracer)

    def _research(state: ResearchState) -> ResearchState:
        return research_node(state, tracer)

    def _summarizer(state: ResearchState) -> ResearchState:
        return summarizer_node(state, tracer)

    def _critic(state: ResearchState) -> ResearchState:
        return critic_node(state, tracer)

    builder: StateGraph[ResearchState] = StateGraph(ResearchState)
    builder.add_node("planner", _planner)
    builder.add_node("researcher", _research)
    builder.add_node("summarizer", _summarizer)
    builder.add_node("critic", _critic)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "summarizer")
    builder.add_edge("summarizer", "critic")
    builder.add_edge("critic", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_workflow(
    topic: str = "multi-agent AI systems",
    redis_url: str = "redis://localhost:6379",
    run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Run the research workflow and return the final state."""
    reset_call_counts()

    tracer = AgentTracer(
        redis_url=redis_url,
        run_id=run_id or uuid.uuid4(),
        root_agent="research_workflow",
        metadata={"topic": topic},
    )
    callback = AgentTraceCallback(tracer)

    initial_state: ResearchState = {
        "topic": topic,
        "sub_questions": [],
        "raw_research": [],
        "summary": "",
        "critique": "",
        "error_count": 0,
    }

    tracer.start_run()
    try:
        graph = build_graph(tracer)
        final_state: ResearchState = graph.invoke(
            initial_state,
            config={"callbacks": [callback]},
        )
        tracer.end_run(status="success")
        return dict(final_state)
    except Exception as e:
        logger.exception("Workflow failed")
        tracer.end_run(status="error", error_msg=str(e))
        raise
    finally:
        tracer.close()
