"""
LLM Event Hooks
===============

Provides a hook mechanism for external integrations to receive LLM completion
events. This allows downstream consumers to capture and
log all LLM activity without coupling unillm to specific logging implementations.

The pattern mirrors cache_events.py - using a ContextVar for thread-safety and
async-safety, with a simple callback interface.

Usage:
    from unillm import set_llm_event_hook, LLMEvent

    def my_hook(event: LLMEvent) -> None:
        print(f"LLM call: {event.request.get('model')}")

    set_llm_event_hook(my_hook)

    # Now all LLM calls will trigger the hook (once per call, after completion)
    client = AsyncUnify("openai/gpt-4o@openrouter")
    await client.generate(messages=[...])  # Hook called once with full event
"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generator, AsyncGenerator, Optional


@dataclass
class LLMEvent:
    """Event emitted after LLM requests complete.

    A single event is emitted per LLM call, containing the full request and
    response data. This mirrors what gets logged to logs/unillm/ files.

    Attributes:
        request: The full request dict sent to the LLM (model, messages, tools, etc.).
        response: The full response dict from the LLM (serialized ChatCompletion).
            None for streaming requests or errors.
        provider_cost: What the call cost, as charged by the LLM provider
            (in USD). None for cache hits, streaming, or errors.
        origin: Optional user-supplied tag identifying the call origin
            (e.g. ``"ConversationManager.decide"``). ``None`` when not set.
    """

    request: dict[str, Any]
    response: Optional[dict[str, Any]] = None
    provider_cost: Optional[float] = None
    origin: Optional[str] = None


# Context variable for the current event hook (context-local, for scoped captures)
_llm_event_hook: ContextVar[Callable[[LLMEvent], None] | None] = ContextVar(
    "llm_event_hook",
    default=None,
)

# Module-level global hook (process-wide fallback when no context-specific hook is set)
_global_llm_event_hook: Callable[[LLMEvent], None] | None = None


def set_global_llm_event_hook(hook: Callable[[LLMEvent], None] | None) -> None:
    """Set a process-global hook that applies to all threads and async contexts.

    Unlike set_llm_event_hook() which uses a ContextVar (context-local), this
    sets a module-level global that will be used as a fallback when no
    context-specific hook is set.

    This is the preferred way to install a hook at application startup that
    should capture all LLM calls across all threads. The hook will be called
    for any LLM call where no context-specific hook has been set.

    If both a context-specific hook (via set_llm_event_hook or llm_event_hook_scope)
    and a global hook are set, the context-specific hook takes precedence.

    Args:
        hook: A callable that receives an LLMEvent. Pass None to clear the hook.

    Example:
        def my_logger(event: LLMEvent) -> None:
            print(f"LLM call to {event.request.get('model')}")

        # Install once at startup - works across all threads
        set_global_llm_event_hook(my_logger)
    """
    global _global_llm_event_hook
    _global_llm_event_hook = hook


def get_global_llm_event_hook() -> Callable[[LLMEvent], None] | None:
    """Get the currently active global LLM event hook, if any.

    Returns:
        The current global hook callable, or None if no global hook is set.
    """
    return _global_llm_event_hook


def set_llm_event_hook(hook: Callable[[LLMEvent], None] | None) -> None:
    """Set a hook to receive LLM completion events.

    The hook will be called once per LLM call, after the request completes.
    The event contains the full request and response dicts, plus cost info.

    The hook is stored in a ContextVar, so it's automatically inherited by
    child tasks/threads but isolated from unrelated code paths.

    Args:
        hook: A callable that receives an LLMEvent. Pass None to clear the hook.

    Example:
        def my_logger(event: LLMEvent) -> None:
            model = event.request.get("model", "unknown")
            print(f"LLM call to {model}, cost: ${event.provider_cost or 0:.4f}")

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

    The hook resolution order is:
    1. Context-specific hook (set via set_llm_event_hook or llm_event_hook_scope)
    2. Global hook (set via set_global_llm_event_hook)

    This ensures scoped captures in tests take precedence, while the global
    hook catches all other LLM calls across threads.

    The hook is called synchronously but wrapped in a try/except to ensure
    hook failures never break LLM calls.

    Args:
        event: The LLM event to emit.
    """
    # First try context-specific hook (for scoped captures in tests)
    hook = _llm_event_hook.get()
    # Fall back to global hook if no context-specific hook
    if hook is None:
        hook = _global_llm_event_hook
    if hook is not None:
        try:
            hook(event)
        except Exception:
            # Never let hook failures break LLM calls
            pass
