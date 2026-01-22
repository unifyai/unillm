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
# OpenAI (and potentially other providers) occasionally return HTTP 400
# BadRequest for valid requests due to transient server-side processing issues.
# These errors are incorrectly classified as client errors, so neither the
# OpenAI SDK (which only retries 5xx) nor LiteLLM (which treats 400 as
# permanent) will retry them.
#
# Known transient error patterns that return as 400:
# - "something went wrong reading your request" (OpenAI)
# - "Service temporarily unavailable, please try again later" (various)
#
# References:
# - LiteLLM GitHub issue #12503: 400 errors don't trigger channel fallback
#   even when the message indicates a transient condition.
#   https://github.com/BerriAI/litellm/issues/12503
# - OpenAI community forum threads documenting intermittent "something went
#   wrong" errors that resolve on retry:
#   https://community.openai.com/t/error-something-went-wrong-if-this-issue-persists/200411
# - LiteLLM exception mapping docs confirm 400 -> BadRequestError (not retried):
#   https://docs.litellm.ai/docs/exception_mapping
#
# This workaround adds targeted retry logic for these specific error patterns.
# ─────────────────────────────────────────────────────────────────────────────

# Error message substrings that indicate transient server-side issues
# despite being returned as HTTP 400 BadRequest.
_TRANSIENT_400_ERROR_PATTERNS = (
    "something went wrong reading your request",
    "service temporarily unavailable",
    "temporarily unavailable, please try again",
)


def _is_transient_400_error(exc: BaseException) -> bool:
    """
    Check if an exception is a known transient error masquerading as a 400.

    These are server-side processing failures that upstream providers
    incorrectly return as HTTP 400 BadRequest instead of 5xx.
    """
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _TRANSIENT_400_ERROR_PATTERNS)


def retry_transient_400_sync(fn: Callable[[], T]) -> T:
    """
    Execute a sync function with retry logic for transient 400 errors.

    Retries up to UNILLM_TRANSIENT_RETRY_COUNT times when encountering
    BadRequestError with known transient error message patterns.
    """
    max_retries = SETTINGS.UNILLM_TRANSIENT_RETRY_COUNT
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (litellm.BadRequestError, litellm.exceptions.APIError) as e:
            if _is_transient_400_error(e) and attempt < max_retries:
                last_exc = e
                continue
            raise

    # Should never reach here, but satisfies type checker
    assert last_exc is not None
    raise last_exc


async def retry_transient_400_async(fn: Callable[[], Awaitable[T]]) -> T:
    """
    Execute an async function with retry logic for transient 400 errors.

    Retries up to UNILLM_TRANSIENT_RETRY_COUNT times when encountering
    BadRequestError with known transient error message patterns.
    """
    max_retries = SETTINGS.UNILLM_TRANSIENT_RETRY_COUNT
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except (litellm.BadRequestError, litellm.exceptions.APIError) as e:
            if _is_transient_400_error(e) and attempt < max_retries:
                last_exc = e
                continue
            raise

    # Should never reach here, but satisfies type checker
    assert last_exc is not None
    raise last_exc
