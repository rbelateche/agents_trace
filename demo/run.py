"""CLI entrypoint for the demo research workflow.

Usage::

    python -m demo.run [TOPIC] [--redis-url REDIS_URL]

Examples::

    python -m demo.run "multi-agent AI systems"
    python -m demo.run "LLM alignment" --redis-url redis://localhost:6379
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the demo research workflow")
    parser.add_argument(
        "topic",
        nargs="?",
        default="multi-agent AI systems",
        help="Research topic (default: 'multi-agent AI systems')",
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis URL (default: redis://localhost:6379)",
    )
    args = parser.parse_args()

    from demo.research_workflow import run_workflow

    logger.info("Starting research workflow — topic: %r", args.topic)
    result = run_workflow(topic=args.topic, redis_url=args.redis_url)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result.get("summary", "(no summary)"))
    print("\n" + "=" * 60)
    print("CRITIQUE")
    print("=" * 60)
    print(result.get("critique", "(no critique)"))
    print(f"\nTool errors encountered: {result.get('error_count', 0)}")


if __name__ == "__main__":
    main()
