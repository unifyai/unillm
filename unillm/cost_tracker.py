"""
Cost Event Capture
==================

Provides scoped context managers for capturing cost events during LLM requests.
This allows external code to track exactly how much each request costs, broken
down by model, token counts, and cache status.

Uses Python's contextvars for threasignificant d-safety and async-safety. The pattern
mirrors ``cache_events.py`` -- a separate ContextVar sink that never interferes
with the LLM event hook system or cache event capture.

Usage:
    from unillm import capture_costs, summarize_costs

    # Sync usage
    with capture_costs() as events:
        client.generate(messages=[...])
    summary = summarize_costs(events)
    print(f"Total cost: ${summary.total_provider_cost:.6f}")

    # Async usage
    async with acapture_costs() as events:
        await client.generate(messages=[...])
    summary = summarize_costs(events)
    print(f"Total cost: ${summary.total_provider_cost:.6f}")
"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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


@dataclass
class CostSummary:
    """Aggregated cost summary from a collection of CostEvents.

    Attributes:
        total_provider_cost: Sum of all provider costs (in USD).
        total_billed_cost: Sum of all billed costs (in USD).
        total_prompt_tokens: Sum of all prompt tokens.
        total_completion_tokens: Sum of all completion tokens.
        total_requests: Total number of LLM requests.
        cache_hits: Number of requests that were cache hits.
        cache_misses: Number of requests that were cache misses.
        events: The raw list of CostEvent objects used to build this summary.
    """

    total_provider_cost: float = 0.0
    total_billed_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    events: List[CostEvent] = field(default_factory=list)


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

        if events:
            summary = summarize_costs(events)
            print(f"Provider cost: ${summary.total_provider_cost:.6f}")
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

        if events:
            summary = summarize_costs(events)
            print(f"Provider cost: ${summary.total_provider_cost:.6f}")
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


def summarize_costs(events: List[CostEvent]) -> CostSummary:
    """Aggregate a list of CostEvents into a CostSummary.

    Args:
        events: A list of CostEvent objects (typically from capture_costs()).

    Returns:
        A CostSummary with totals computed from all events.

    Example:
        with capture_costs() as events:
            client.generate(messages=[...])
            client.generate(messages=[...])

        summary = summarize_costs(events)
        print(f"Total: ${summary.total_provider_cost:.6f} over {summary.total_requests} requests")
    """
    summary = CostSummary(events=list(events))
    for event in events:
        summary.total_provider_cost += event.provider_cost
        summary.total_billed_cost += event.billed_cost
        summary.total_prompt_tokens += event.prompt_tokens
        summary.total_completion_tokens += event.completion_tokens
        summary.total_requests += 1
        if event.cache_status == "hit":
            summary.cache_hits += 1
        elif event.cache_status == "miss":
            summary.cache_misses += 1
    return summary
