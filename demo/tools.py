"""Mock tools used by the demo research workflow.

Design: ``web_search`` fails with malformed JSON on the first call
(simulating a real-world hallucination / invalid-output error) then
succeeds on subsequent calls.  This gives the observability platform
something interesting to display.
"""

from __future__ import annotations

import json
import time
from threading import Lock

_call_counts: dict[str, int] = {}
_lock = Lock()


def _increment(tool: str) -> int:
    with _lock:
        _call_counts[tool] = _call_counts.get(tool, 0) + 1
        return _call_counts[tool]


def web_search(query: str) -> str:
    """Simulate a web search.  Fails with malformed JSON on first call."""
    call_n = _increment("web_search")
    time.sleep(0.05)  # simulate latency

    if call_n == 1:
        # Scripted failure: return invalid JSON so the caller's parser breaks
        return '{"results": [{"title": "AI Trends 2024", "snippet": INVALID'

    return json.dumps(
        {
            "results": [
                {
                    "title": "AI Trends 2024",
                    "snippet": "Large language models continue to reshape software engineering.",
                    "url": "https://example.com/ai-trends",
                },
                {
                    "title": "Agentic AI Systems",
                    "snippet": "Multi-agent architectures enable complex task decomposition.",
                    "url": "https://example.com/agentic-ai",
                },
            ]
        }
    )


def retrieve_papers(topic: str) -> str:
    """Simulate a RAG retrieval over academic papers."""
    _increment("retrieve_papers")
    time.sleep(0.03)
    return json.dumps(
        {
            "papers": [
                {
                    "title": f"Survey of {topic}",
                    "abstract": f"A comprehensive overview of {topic} in 2024.",
                    "year": 2024,
                },
                {
                    "title": f"{topic}: Open Problems",
                    "abstract": "We identify key open challenges and future directions.",
                    "year": 2024,
                },
            ]
        }
    )


def reset_call_counts() -> None:
    """Reset tool call counters (for testing)."""
    with _lock:
        _call_counts.clear()
