"""
LLM Event Hooks
===============

Provides a hook mechanism for external integrations to receive LLM request/response
events. This allows downstream consumers (like Unity's EventBus) to capture and
log all LLM activity without coupling unillm to specific logging implementations.

The pattern mirrors cache_events.py - using a ContextVar for thread-safety and
async-safety, with a simple callback interface.

Usage:
    from unillm import set_llm_event_hook, LLMEvent

    def my_hook(event: LLMEvent) -> None:
        print(f"LLM {event.phase}: {event.endpoint}")

    set_llm_event_hook(my_hook)

    # Now all LLM calls will trigger the hook
    client = AsyncUnify("gpt-4o@openai")
    await client.generate(messages=[...])  # Hook called twice: request + response
"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generator, AsyncGenerator, Optional


@dataclass
class LLMEvent:
    """Event emitted before and after LLM requests.

    Attributes:
        phase: Event phase - "request" (before call) or "response" (after call).
        endpoint: The endpoint string (e.g., "gpt-4o@openai").
        model: The model name extracted from the endpoint.
        provider: The provider name extracted from the endpoint.
        request_kw: The full request kwargs sent to the LLM (model, messages, tools, etc.).
        response: The ChatCompletion response object (only on "response" phase).
        cache_status: "hit" or "miss" (only on "response" phase for non-streaming).
        error: The exception if the LLM call failed (only on "response" phase).
        stream: Whether this is a streaming request.
    """

    phase: str  # "request" or "response"
    endpoint: str
    model: str
    provider: str
    request_kw: dict[str, Any]
    response: Optional[Any] = None
    cache_status: Optional[str] = None
    error: Optional[Exception] = None
    stream: bool = False


# Context variable for the current event hook (thread-safe and async-safe)
_llm_event_hook: ContextVar[Callable[[LLMEvent], None] | None] = ContextVar(
    "llm_event_hook",
    default=None,
)


def set_llm_event_hook(hook: Callable[[LLMEvent], None] | None) -> None:
    """Set a hook to receive LLM request/response events.

    The hook will be called twice per LLM call:
    1. Before the request (phase="request")
    2. After the response (phase="response")

    The hook is stored in a ContextVar, so it's automatically inherited by
    child tasks/threads but isolated from unrelated code paths.

    Args:
        hook: A callable that receives an LLMEvent. Pass None to clear the hook.

    Example:
        def my_logger(event: LLMEvent) -> None:
            if event.phase == "response":
                print(f"LLM call to {event.endpoint}: {event.cache_status}")

        set_llm_event_hook(my_logger)
    """
    _llm_event_hook.set(hook)


def get_llm_event_hook() -> Callable[[LLMEvent], None] | None:
    """Get the currently active LLM event hook, if any.

    Returns:
        The current hook callable, or None if no hook is set.
    """
    return _llm_event_hook.get()


@contextmanager
def llm_event_hook_scope(
    hook: Callable[[LLMEvent], None],
) -> Generator[None, None, None]:
    """Context manager to temporarily set an LLM event hook.

    The hook is active only within the context manager scope and is
    automatically restored to the previous value on exit.

    Args:
        hook: The hook to use within the scope.

    Yields:
        None

    Example:
        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with llm_event_hook_scope(capture_hook):
            client.generate(messages=[...])
        # Hook is now restored to previous value
    """
    token = _llm_event_hook.set(hook)
    try:
        yield
    finally:
        _llm_event_hook.reset(token)


@asynccontextmanager
async def allm_event_hook_scope(
    hook: Callable[[LLMEvent], None],
) -> AsyncGenerator[None, None]:
    """Async context manager to temporarily set an LLM event hook.

    The hook is active only within the context manager scope and is
    automatically restored to the previous value on exit.

    Args:
        hook: The hook to use within the scope.

    Yields:
        None

    Example:
        async def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        async with allm_event_hook_scope(capture_hook):
            await client.generate(messages=[...])
        # Hook is now restored to previous value
    """
    token = _llm_event_hook.set(hook)
    try:
        yield
    finally:
        _llm_event_hook.reset(token)


def _emit_llm_event(event: LLMEvent) -> None:
    """Emit an LLM event to the current hook, if any.

    This is an internal function called by the LLM clients. If no hook is
    set, the event is silently dropped.

    The hook is called synchronously but wrapped in a try/except to ensure
    hook failures never break LLM calls.

    Args:
        event: The LLM event to emit.
    """
    hook = _llm_event_hook.get()
    if hook is not None:
        try:
            hook(event)
        except Exception:
            # Never let hook failures break LLM calls
            pass
