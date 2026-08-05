"""
Cache Event Capture
===================

Provides a scoped context manager for capturing cache hit/miss/disabled events
during LLM requests. This allows external code to track exactly which requests
hit the cache, missed the cache, or had caching disabled entirely.

Uses Python's contextvars for thread-safety and async-safety.

Usage:
    from unillm import capture_cache_events

    # Sync usage
    with capture_cache_events() as events:
        client.generate(messages=[...])
    print(events[0]["cache_status"])  # "hit", "miss", or "disabled"

    # Async usage
    async with acapture_cache_events() as events:
        await client.generate(messages=[...])
    print(events[0]["cache_status"])  # "hit", "miss", or "disabled"
"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from typing import Any, Generator, AsyncGenerator, List, Literal, TypedDict


class CacheEvent(TypedDict):
    """Event emitted when a cache decision is made during an LLM request.

    Attributes:
        cache_status: Whether the request was a cache "hit", "miss", or
            "disabled" (cache reading was not attempted, e.g. cache=False).
        endpoint: The endpoint string (e.g., "openai/gpt-4o@openrouter").
        request_kw: The full request kwargs sent to the LLM (model, messages, etc.).
    """

    cache_status: Literal["hit", "miss", "disabled", "pending", "error"]
    endpoint: str
    request_kw: dict[str, Any]


# Context variable for the current event sink (thread-safe and async-safe)
_cache_event_sink: ContextVar[List[CacheEvent] | None] = ContextVar(
    "cache_event_sink",
    default=None,
)


@contextmanager
def capture_cache_events() -> Generator[List[CacheEvent], None, None]:
    """Context manager to capture cache events within a scope.

    All cache events from LLM calls made within this context will be
    appended to the returned list. Events from other threads/tasks
    or outside this context are not captured.

    Yields:
        A list that will be populated with CacheEvent dicts as LLM calls complete.

    Example:
        with capture_cache_events() as events:
            client.generate(messages=[{"role": "user", "content": "Hi"}])

        if events:
            print(f"Cache status: {events[0]['cache_status']}")
    """
    events: List[CacheEvent] = []
    token = _cache_event_sink.set(events)
    try:
        yield events
    finally:
        _cache_event_sink.reset(token)


@asynccontextmanager
async def acapture_cache_events() -> AsyncGenerator[List[CacheEvent], None]:
    """Async context manager to capture cache events within a scope.

    All cache events from LLM calls made within this context will be
    appended to the returned list. Events from other threads/tasks
    or outside this context are not captured.

    Yields:
        A list that will be populated with CacheEvent dicts as LLM calls complete.

    Example:
        async with acapture_cache_events() as events:
            await client.generate(messages=[{"role": "user", "content": "Hi"}])

        if events:
            print(f"Cache status: {events[0]['cache_status']}")
    """
    events: List[CacheEvent] = []
    token = _cache_event_sink.set(events)
    try:
        yield events
    finally:
        _cache_event_sink.reset(token)


def _emit_cache_event(event: CacheEvent) -> None:
    """Emit a cache event to the current context's sink, if any.

    This is an internal function called by the LLM clients. If no
    capture_cache_events() context is active, the event is silently dropped.

    Args:
        event: The cache event to emit.
    """
    sink = _cache_event_sink.get()
    if sink is not None:
        sink.append(event)
