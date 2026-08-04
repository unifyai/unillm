import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

import litellm

from .settings import SETTINGS

T = TypeVar("T")


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
# LLM provider APIs are subject to transient failures that should be retried:
#
# 1. **Transient 400s** — OpenAI occasionally returns HTTP 400 BadRequest for
#    valid requests (e.g. "something went wrong reading your request"). Neither
#    the OpenAI SDK (retries only 5xx) nor LiteLLM (treats 400 as permanent)
#    will retry these.
#
# 2. **503 ServiceUnavailableError** — Upstream overload or connection resets
#    (e.g. Anthropic "upstream connect error … reset reason: overflow").
#
# 3. **500 InternalServerError** — Transient server-side failures.
#
# 4. **429 RateLimitError** — Temporary rate limiting from providers.
#
# References:
# - LiteLLM GitHub issue #12503: 400 errors don't trigger channel fallback
#   https://github.com/BerriAI/litellm/issues/12503
# - OpenAI community forum: intermittent "something went wrong" errors
#   https://community.openai.com/t/error-something-went-wrong-if-this-issue-persists/200411
# - LiteLLM exception mapping: https://docs.litellm.ai/docs/exception_mapping
# ─────────────────────────────────────────────────────────────────────────────

# Error message substrings that indicate transient server-side issues
# despite being returned as HTTP 400 BadRequest.
_TRANSIENT_400_ERROR_PATTERNS = (
    "something went wrong reading your request",
    "service temporarily unavailable",
    "temporarily unavailable, please try again",
)

# Exception types that are inherently transient (server-side) and always
# worth retrying regardless of the error message.
_TRANSIENT_SERVER_EXCEPTIONS = (
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.RateLimitError,
    litellm.APIConnectionError,
)

# Base backoff delay in seconds for transient server errors (doubles each attempt).
# With UNILLM_TRANSIENT_RETRY_COUNT=6 this yields 1/2/4/8/16/32s between attempts.
_BACKOFF_BASE_SECONDS = 1.0


def _is_transient_400_error(exc: BaseException) -> bool:
    """
    Check if an exception is a known transient error masquerading as a 400.

    These are server-side processing failures that upstream providers
    incorrectly return as HTTP 400 BadRequest instead of 5xx.
    """
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _TRANSIENT_400_ERROR_PATTERNS)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception represents a transient failure worth retrying."""
    if isinstance(exc, _TRANSIENT_SERVER_EXCEPTIONS):
        return True
    if isinstance(exc, (litellm.BadRequestError, litellm.exceptions.APIError)):
        return _is_transient_400_error(exc)
    return False


def retry_transient_400_sync(fn: Callable[[], T]) -> T:
    """
    Execute a sync function with retry logic for transient LLM errors.

    Retries up to UNILLM_TRANSIENT_RETRY_COUNT times when encountering:
    - BadRequestError / APIError with known transient message patterns (400)
    - ServiceUnavailableError (503)
    - InternalServerError (500)
    - RateLimitError (429)

    Uses exponential backoff for server-side errors (503/500/429),
    doubling from ``_BACKOFF_BASE_SECONDS`` (default schedule 1/2/4/8/16/32s).
    """
    import time

    max_retries = SETTINGS.UNILLM_TRANSIENT_RETRY_COUNT
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (
            litellm.BadRequestError,
            litellm.exceptions.APIError,
            *_TRANSIENT_SERVER_EXCEPTIONS,
        ) as e:
            if _is_retryable(e) and attempt < max_retries:
                last_exc = e
                if isinstance(e, _TRANSIENT_SERVER_EXCEPTIONS):
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            raise

    # Should never reach here, but satisfies type checker
    assert last_exc is not None
    raise last_exc


async def retry_transient_400_async(fn: Callable[[], Awaitable[T]]) -> T:
    """
    Execute an async function with retry logic for transient LLM errors.

    Retries up to UNILLM_TRANSIENT_RETRY_COUNT times when encountering:
    - BadRequestError / APIError with known transient message patterns (400)
    - ServiceUnavailableError (503)
    - InternalServerError (500)
    - RateLimitError (429)

    Uses exponential backoff for server-side errors (503/500/429),
    doubling from ``_BACKOFF_BASE_SECONDS`` (default schedule 1/2/4/8/16/32s).
    """
    max_retries = SETTINGS.UNILLM_TRANSIENT_RETRY_COUNT
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except (
            litellm.BadRequestError,
            litellm.exceptions.APIError,
            *_TRANSIENT_SERVER_EXCEPTIONS,
        ) as e:
            if _is_retryable(e) and attempt < max_retries:
                last_exc = e
                if isinstance(e, _TRANSIENT_SERVER_EXCEPTIONS):
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            raise

    # Should never reach here, but satisfies type checker
    assert last_exc is not None
    raise last_exc
