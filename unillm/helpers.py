import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

import openai

from .settings import SETTINGS

T = TypeVar("T")

_LOGGER = logging.getLogger("unillm.retry")


class _UnsetSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        # Stable textual form to avoid process-specific object addresses
        return "<UNSET>"


UNSET = _UnsetSentinel()


def _default(value: Any, default_value: Any) -> Any:
    return value if value is not None else default_value


# Global seed for reproducibility
_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: Optional[int]) -> None:
    """Set the global seed for reproducibility."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed


def get_seed() -> Optional[int]:
    """Get the global seed, or None if not set."""
    return _GLOBAL_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Transient Error Retry Logic
# ─────────────────────────────────────────────────────────────────────────────
# This is the only layer that retries an LLM call. OpenRouter retries among its
# own upstreams but cannot retry its own reply to us; LiteLLM retries only when
# passed ``num_retries``/``max_retries``, which we never do; and the OpenAI SDK
# is not in the transport path for provider calls routed through LiteLLM's HTTP
# handler. So whatever this layer declines is never attempted again — the
# caller's whole trajectory ends on it.
#
# That makes the default the important decision, and the default is to retry.
# Classification is by *fault*, read off the response status:
#
# - **4xx** — we sent something wrong, so a retry repeats the mistake. Declined,
#   except for the handful of providers that report their own faults with a
#   client status (see ``_PROVIDER_FAULT_PATTERNS``), and except 408/429, which
#   describe timing rather than request content.
# - **Everything else** — 5xx, and 2xx carrying a body we could not read: the
#   provider's side of the exchange failed. Retried.
# - **No status at all** — a semantic error raised without a response (length or
#   content-filter finish reasons, bad client configuration). A settled fact, not
#   a blip. Declined.
#
# The asymmetry is what settles the default: a wrong "transient" verdict costs
# seconds of latency and no provider charge (4xx are not billed), while a wrong
# "permanent" verdict costs the caller's entire run. Recognising failures by name
# gets that backwards — an OpenRouter 200 whose body is whitespace maps to
# APIError and matches no message pattern, so a name-based policy declines it
# without a single attempt.
#
# References:
# - LiteLLM GitHub issue #12503: 400 errors don't trigger channel fallback
#   https://github.com/BerriAI/litellm/issues/12503
# - OpenAI community forum: intermittent "something went wrong" errors
#   https://community.openai.com/t/error-something-went-wrong-if-this-issue-persists/200411
# - LiteLLM exception mapping: https://docs.litellm.ai/docs/exception_mapping
# ─────────────────────────────────────────────────────────────────────────────

# Message substrings that identify a provider-side fault reported with a client
# error status. "unable to get json response" is LiteLLM's wording when a
# response body could not be parsed as JSON at all — every provider
# transformation raises it with that prefix, so one entry covers OpenRouter,
# OpenAI, Anthropic, Fireworks, Vertex and Databricks alike.
_PROVIDER_FAULT_PATTERNS = (
    "something went wrong reading your request",
    "service temporarily unavailable",
    "temporarily unavailable, please try again",
    "unable to get json response",
)

# Client-error statuses that describe timing rather than request content, so
# the same request can succeed unchanged: 408 request timeout, 429 rate limit.
_TRANSIENT_CLIENT_STATUSES = frozenset({408, 429})

# Base backoff delay in seconds (doubles each attempt). With
# UNILLM_TRANSIENT_RETRY_COUNT=6 this yields 1/2/4/8/16/32s between attempts.
_BACKOFF_BASE_SECONDS = 1.0

# Root of every error LiteLLM's exception mapping produces for a provider call.
# Catching the root rather than a list of subclasses is what lets an unfamiliar
# failure reach the classifier at all.
_PROVIDER_EXCEPTIONS = (openai.OpenAIError,)


def _is_provider_fault_message(exc: BaseException) -> bool:
    """Check whether a client-status error is really the provider's own fault."""
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _PROVIDER_FAULT_PATTERNS)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception leaves room for the same request to succeed."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return False
    if 400 <= status < 500 and status not in _TRANSIENT_CLIENT_STATUSES:
        return _is_provider_fault_message(exc)
    return True


def _describe(exc: BaseException) -> str:
    """One-line identity of a failure, for the retry log lines.

    Whitespace is collapsed because the bodies that provoke these lines are
    often whitespace themselves, and a log entry has to stay one entry.
    """
    status = getattr(exc, "status_code", None)
    detail = " ".join(str(exc).split())[:200]
    return f"{type(exc).__name__}(status={status}): {detail}"


def retry_transient_llm_sync(fn: Callable[[], T]) -> T:
    """
    Execute a sync provider call, retrying failures that may yet succeed.

    Retries up to UNILLM_TRANSIENT_RETRY_COUNT times with exponential backoff
    doubling from ``_BACKOFF_BASE_SECONDS`` (default schedule 1/2/4/8/16/32s).
    See the module comment above for which failures are retried and why.

    Every terminal outcome is logged, so a failure that reaches the caller
    always says whether it was retried and how often.
    """
    import time

    max_retries = SETTINGS.UNILLM_TRANSIENT_RETRY_COUNT

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except _PROVIDER_EXCEPTIONS as e:
            if not _is_retryable(e):
                _LOGGER.warning(f"not retried, treated as permanent: {_describe(e)}")
                raise
            if attempt >= max_retries:
                _LOGGER.warning(
                    f"retries exhausted after {attempt + 1} attempts: {_describe(e)}",
                )
                raise
            _LOGGER.debug(
                f"retrying (attempt {attempt + 1}/{max_retries + 1}): {_describe(e)}",
            )
            time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

    raise AssertionError("unreachable: the loop either returns or raises")


async def retry_transient_llm_async(fn: Callable[[], Awaitable[T]]) -> T:
    """
    Execute an async provider call, retrying failures that may yet succeed.

    Retries up to UNILLM_TRANSIENT_RETRY_COUNT times with exponential backoff
    doubling from ``_BACKOFF_BASE_SECONDS`` (default schedule 1/2/4/8/16/32s).
    See the module comment above for which failures are retried and why.

    Every terminal outcome is logged, so a failure that reaches the caller
    always says whether it was retried and how often.
    """
    max_retries = SETTINGS.UNILLM_TRANSIENT_RETRY_COUNT

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except _PROVIDER_EXCEPTIONS as e:
            if not _is_retryable(e):
                _LOGGER.warning(f"not retried, treated as permanent: {_describe(e)}")
                raise
            if attempt >= max_retries:
                _LOGGER.warning(
                    f"retries exhausted after {attempt + 1} attempts: {_describe(e)}",
                )
                raise
            _LOGGER.debug(
                f"retrying (attempt {attempt + 1}/{max_retries + 1}): {_describe(e)}",
            )
            await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

    raise AssertionError("unreachable: the loop either returns or raises")
