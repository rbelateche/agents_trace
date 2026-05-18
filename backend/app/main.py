"""FastAPI application entrypoint."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.database import engine
from app.worker import consume_forever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(consume_forever(stop_event))
    yield
    stop_event.set()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()


app = FastAPI(
    title="agents-trace",
    description="Multi-Agent Observability Platform",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
