"""
Cost Event Capture
==================

Provides scoped context managers for capturing cost events during LLM requests.
This allows external code to track exactly how much each request costs, broken
down by model, token counts, and cache status.

Uses Python's contextvars for thread-safety and async-safety. The pattern
mirrors ``cache_events.py`` -- a separate ContextVar sink that never interferes
with the LLM event hook system or cache event capture.

Usage:
    from unillm import capture_costs

    # Sync usage
    with capture_costs() as events:
        client.generate(messages=[...])
    for event in events:
        print(f"Cost: ${event.provider_cost:.6f} ({event.cache_status})")

    # Async usage
    async with acapture_costs() as events:
        await client.generate(messages=[...])
    for event in events:
        print(f"Cost: ${event.provider_cost:.6f} ({event.cache_status})")
"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generator, AsyncGenerator, List


@dataclass
class CostEvent:
    """Event emitted after an LLM request with cost information.

    Attributes:
        model: The model identifier used for the request (e.g., "gpt-4o").
        provider_cost: The raw cost charged by the LLM provider (in USD).
            0.0 for cache hits or when cost cannot be determined.
        billed_cost: The cost charged to the user (provider_cost x margin, in USD).
            0.0 for cache hits or when cost cannot be determined.
        prompt_tokens: Number of input/prompt tokens used. 0 for cache hits.
        completion_tokens: Number of output/completion tokens used. 0 for cache hits.
        cache_status: Whether the request was a cache "hit", "miss", or "disabled".
    """

    model: str
    provider_cost: float = 0.0
    billed_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_status: str = "disabled"


# Context variable for the current cost event sink (thread-safe and async-safe)
_cost_event_sink: ContextVar[List[CostEvent] | None] = ContextVar(
    "cost_event_sink",
    default=None,
)


@contextmanager
def capture_costs() -> Generator[List[CostEvent], None, None]:
    """Context manager to capture cost events within a scope.

    All cost events from LLM calls made within this context will be
    appended to the returned list. Events from other threads/tasks
    or outside this context are not captured.

    Yields:
        A list that will be populated with CostEvent objects as LLM calls complete.

    Example:
        with capture_costs() as events:
            client.generate(messages=[{"role": "user", "content": "Hi"}])

        for event in events:
            print(f"Cost: ${event.provider_cost:.6f} ({event.cache_status})")
    """
    events: List[CostEvent] = []
    token = _cost_event_sink.set(events)
    try:
        yield events
    finally:
        _cost_event_sink.reset(token)


@asynccontextmanager
async def acapture_costs() -> AsyncGenerator[List[CostEvent], None]:
    """Async context manager to capture cost events within a scope.

    All cost events from LLM calls made within this context will be
    appended to the returned list. Events from other threads/tasks
    or outside this context are not captured.

    Yields:
        A list that will be populated with CostEvent objects as LLM calls complete.

    Example:
        async with acapture_costs() as events:
            await client.generate(messages=[{"role": "user", "content": "Hi"}])

        for event in events:
            print(f"Cost: ${event.provider_cost:.6f} ({event.cache_status})")
    """
    events: List[CostEvent] = []
    token = _cost_event_sink.set(events)
    try:
        yield events
    finally:
        _cost_event_sink.reset(token)


def _emit_cost_event(event: CostEvent) -> None:
    """Emit a cost event to the current context's sink, if any.

    This is an internal function called by the LLM clients. If no
    capture_costs() context is active, the event is silently dropped.

    Args:
        event: The cost event to emit.
    """
    sink = _cost_event_sink.get()
    if sink is not None:
        sink.append(event)
