"""
LLM Event Hooks
===============

Provides a hook mechanism for external integrations to receive LLM completion
events. This allows downstream consumers to capture and
log all LLM activity without coupling unillm to specific logging implementations.

There are two ways to receive events, and they compose:

- **Listeners** (``add_llm_event_listener``) are the supported way to meter a
  process. Registration is *additive* — every listener receives every event,
  from every thread and every event loop — so one consumer can never displace
  another, and installation order does not matter.
- **Scoped hooks** (``set_llm_event_hook`` / ``llm_event_hook_scope``) are a
  ContextVar-based capture for a single call path, useful in tests. A scoped
  hook receives events *in addition to* the registered listeners; it does not
  suppress them.

Usage:
    from unillm import add_llm_event_listener, LLMEvent

    def my_hook(event: LLMEvent) -> None:
        print(f"LLM call: {event.request.get('model')}")

    listener = add_llm_event_listener(my_hook)

    # Now all LLM calls will trigger the hook (once per call, after completion)
    client = AsyncUnify("openai/gpt-4o@openrouter")
    await client.generate(messages=[...])  # Hook called once with full event

    # A consumer that reports totals must check it actually received them,
    # because a listener that raises is isolated, not fatal:
    assert listener.healthy, listener.last_error
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generator, AsyncGenerator, Optional

_LOGGER = logging.getLogger("unillm")


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


class LLMEventListener:
    """A registered recipient of LLM events, carrying its own delivery health.

    A listener's callback runs inside the LLM client, so it must never be able
    to fail a call — an exception from one listener is isolated and the
    remaining listeners still receive the event. That isolation is also a trap
    for anything that *counts* events: a callback which raises on every event
    records nothing, and a zero total reads as "nothing happened" rather than
    "measurement broke". ``failed``/``last_error`` exist so a consumer can tell
    those two apart, and every failure is logged at ERROR as well.
    """

    def __init__(self, callback: Callable[[LLMEvent], None]) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self._delivered = 0
        self._failed = 0
        self._last_error: BaseException | None = None

    @property
    def callback(self) -> Callable[[LLMEvent], None]:
        return self._callback

    @property
    def delivered(self) -> int:
        """Events the callback accepted without raising."""
        with self._lock:
            return self._delivered

    @property
    def failed(self) -> int:
        """Events the callback raised on, and therefore did not record."""
        with self._lock:
            return self._failed

    @property
    def last_error(self) -> BaseException | None:
        with self._lock:
            return self._last_error

    @property
    def healthy(self) -> bool:
        """True when the callback has never raised.

        Check this before trusting any total derived from the events, so a
        broken callback surfaces as an error instead of a zero.
        """
        with self._lock:
            return self._failed == 0

    def remove(self) -> None:
        """Deregister this listener. Idempotent."""
        remove_llm_event_listener(self)

    def _deliver(self, event: LLMEvent) -> None:
        try:
            self._callback(event)
        except Exception as exc:
            with self._lock:
                self._failed += 1
                self._last_error = exc
                failed = self._failed
            _LOGGER.error(
                "LLM event listener %s raised; %d event(s) have now been lost "
                "to it. Totals derived from this listener are incomplete.",
                _describe(self._callback),
                failed,
                exc_info=True,
            )
        else:
            with self._lock:
                self._delivered += 1

    def __repr__(self) -> str:
        return (
            f"LLMEventListener({_describe(self._callback)}, "
            f"delivered={self.delivered}, failed={self.failed})"
        )


def _describe(callback: Callable[[LLMEvent], None]) -> str:
    return getattr(callback, "__qualname__", None) or repr(callback)


# Registered listeners, in registration order. A plain module-level list, so it
# is genuinely process-global: shared across threads, event loops and contexts.
_listeners: list[LLMEventListener] = []
_listeners_lock = threading.Lock()

# Context variable for a scoped hook, layered on top of the listeners above.
_llm_event_hook: ContextVar[Callable[[LLMEvent], None] | None] = ContextVar(
    "llm_event_hook",
    default=None,
)


def add_llm_event_listener(
    callback: Callable[[LLMEvent], None],
) -> LLMEventListener:
    """Register a callback to receive every LLM event in this process.

    This is the supported way to meter LLM activity. Registration is additive
    and process-global: the callback is invoked for every completed LLM call
    regardless of which thread, task or event loop made it, and registering a
    second listener does not displace the first. Installation order therefore
    does not matter, and no caller has to chain to a predecessor.

    The callback runs synchronously inside the LLM client, from the calling
    thread, so it should be cheap and must not block. If it raises, the
    exception is logged and counted on the returned handle rather than
    propagating into the LLM call.

    Args:
        callback: A callable that receives an LLMEvent.

    Returns:
        A handle used to deregister the listener and to check its delivery
        health (``delivered`` / ``failed`` / ``last_error`` / ``healthy``).

    Example:
        listener = add_llm_event_listener(my_logger)
        ...
        if not listener.healthy:
            raise RuntimeError(f"metering broke: {listener.last_error!r}")
        listener.remove()
    """
    listener = LLMEventListener(callback)
    with _listeners_lock:
        _listeners.append(listener)
    return listener


def remove_llm_event_listener(listener: LLMEventListener) -> None:
    """Deregister a listener previously added by add_llm_event_listener.

    Idempotent: removing an already-removed listener is a no-op.
    """
    with _listeners_lock:
        if listener in _listeners:
            _listeners.remove(listener)


def llm_event_listeners() -> tuple[LLMEventListener, ...]:
    """The currently registered listeners, in registration order."""
    with _listeners_lock:
        return tuple(_listeners)


def clear_llm_event_listeners() -> None:
    """Deregister every listener. Intended for test teardown."""
    with _listeners_lock:
        _listeners.clear()


def set_llm_event_hook(hook: Callable[[LLMEvent], None] | None) -> None:
    """Set a context-local hook to receive LLM completion events.

    The hook will be called once per LLM call, after the request completes.
    The event contains the full request and response dicts, plus cost info.

    The hook is stored in a ContextVar, so it is inherited by child
    tasks/threads but isolated from unrelated call paths. It is delivered to
    *in addition to* any registered listeners — setting it never suppresses
    process-wide metering. To meter a whole process, use
    ``add_llm_event_listener`` instead.

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
    """Get the currently active context-local LLM event hook, if any.

    Returns:
        The current hook callable, or None if no hook is set.
    """
    return _llm_event_hook.get()


@contextmanager
def llm_event_hook_scope(
    hook: Callable[[LLMEvent], None],
) -> Generator[None, None, None]:
    """Context manager to temporarily set a context-local LLM event hook.

    The hook is active only within the context manager scope and is
    automatically restored to the previous value on exit. Registered listeners
    keep receiving events throughout.

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
    """Async context manager to temporarily set a context-local LLM event hook.

    The hook is active only within the context manager scope and is
    automatically restored to the previous value on exit. Registered listeners
    keep receiving events throughout.

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
    """Emit an LLM event to the scoped hook and to every registered listener.

    This is an internal function called by the LLM clients. Delivery is
    additive: the context-local hook (if set) and all registered listeners
    each receive the event, so no recipient can displace another.

    Each recipient is called synchronously and independently. A recipient that
    raises is logged and skipped, so neither the LLM call nor the other
    recipients are affected.

    Args:
        event: The LLM event to emit.
    """
    hook = _llm_event_hook.get()
    if hook is not None:
        try:
            hook(event)
        except Exception:
            _LOGGER.error(
                "Scoped LLM event hook %s raised; the event was not delivered "
                "to it.",
                _describe(hook),
                exc_info=True,
            )
    for listener in llm_event_listeners():
        listener._deliver(event)
