"""
Limit check hooks for UniLLM.

Provides a callback mechanism for host applications (like Unity) to implement
spending limit checks. UniLLM invokes the registered callback before each LLM
call and uses the response to decide whether to proceed.

This follows the same pattern as the existing LLM event hook - UniLLM is
agnostic to how limits are checked; it just invokes the callback.

Usage:
    # In host application (e.g., Unity):
    async def check_limits(request: LimitCheckRequest) -> LimitCheckResponse:
        # Check limits against your backend
        return LimitCheckResponse(allowed=True)

    unillm.set_limit_check_hook(check_limits)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

_LOGGER = logging.getLogger("unillm.limit_hooks")


# ---------------------------------------------------------------------------
# Data classes for hook interface
# ---------------------------------------------------------------------------


class LimitType(Enum):
    """Type of spending limit that was exceeded."""

    ASSISTANT = "assistant"
    USER = "user"  # Personal context
    MEMBER = "member"  # Organization context
    ORGANIZATION = "organization"


@dataclass
class LimitCheckRequest:
    """Request sent to the limit check callback before each LLM call.

    Contains information about the pending LLM call that the callback
    can use to make limit decisions.
    """

    model: str
    endpoint: str
    estimated_input_tokens: Optional[int] = None
    estimated_output_tokens: Optional[int] = None


@dataclass
class LimitCheckResponse:
    """Response from the limit check callback.

    Attributes:
        allowed: If True, proceed with the LLM call. If False, block it.
        reason: Human-readable reason for blocking (shown in error message).
        limit_type: Which limit was exceeded (if any).
        limit_value: The limit value that was exceeded.
        current_spend: Current spending amount.
        entity_id: ID of the entity whose limit was exceeded.
        entity_name: Name of the entity whose limit was exceeded.
    """

    allowed: bool
    reason: Optional[str] = None
    limit_type: Optional[LimitType] = None
    limit_value: Optional[float] = None
    current_spend: Optional[float] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None

    @property
    def percent_used(self) -> Optional[float]:
        """Calculate percentage of limit used."""
        if self.limit_value and self.limit_value > 0 and self.current_spend is not None:
            return (self.current_spend / self.limit_value) * 100
        return None


# Type for the limit check callback
LimitCheckHook = Callable[[LimitCheckRequest], Awaitable[LimitCheckResponse]]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class SpendingLimitExceededError(Exception):
    """Exception raised when a spending limit is exceeded."""

    def __init__(self, response: LimitCheckResponse):
        self.response = response
        message = response.reason or self._default_message(response)
        super().__init__(message)

    @staticmethod
    def _default_message(response: LimitCheckResponse) -> str:
        """Generate default error message from response."""
        limit_type = response.limit_type.value if response.limit_type else "unknown"
        current = f"${response.current_spend:.2f}" if response.current_spend else "unknown"
        limit = f"${response.limit_value:.2f}" if response.limit_value else "unknown"
        return f"Monthly spending limit exceeded: {limit_type} limit of {limit} reached (current: {current})"


# ---------------------------------------------------------------------------
# Global hook management
# ---------------------------------------------------------------------------

# The global limit check hook
_LIMIT_CHECK_HOOK: Optional[LimitCheckHook] = None

# Lock for thread-safe hook access
_HOOK_LOCK = asyncio.Lock()


def set_limit_check_hook(hook: Optional[LimitCheckHook]) -> None:
    """Set the global limit check hook.

    The hook is called before each LLM call to determine if the call should
    proceed. If the hook is not set, all calls are allowed.

    Args:
        hook: Async function that takes LimitCheckRequest and returns
            LimitCheckResponse. Pass None to disable limit checking.
    """
    global _LIMIT_CHECK_HOOK
    _LIMIT_CHECK_HOOK = hook
    if hook:
        _LOGGER.debug("Limit check hook installed")
    else:
        _LOGGER.debug("Limit check hook cleared")


def get_limit_check_hook() -> Optional[LimitCheckHook]:
    """Get the current limit check hook (or None if not set)."""
    return _LIMIT_CHECK_HOOK


def is_limit_check_enabled() -> bool:
    """Check if limit checking is enabled (hook is registered)."""
    return _LIMIT_CHECK_HOOK is not None


async def check_limits(request: LimitCheckRequest) -> LimitCheckResponse:
    """Invoke the limit check hook.

    If no hook is registered, returns an allowed response.

    Args:
        request: Information about the pending LLM call.

    Returns:
        LimitCheckResponse indicating whether to proceed.
    """
    hook = _LIMIT_CHECK_HOOK
    if hook is None:
        return LimitCheckResponse(allowed=True)

    try:
        return await hook(request)
    except Exception as e:
        # Fail open: if the hook fails, allow the call
        _LOGGER.warning(f"Limit check hook failed, allowing call: {e}")
        return LimitCheckResponse(allowed=True)


def check_limits_sync(request: LimitCheckRequest) -> LimitCheckResponse:
    """Synchronous version of check_limits.

    Creates a new event loop if needed.
    """
    hook = _LIMIT_CHECK_HOOK
    if hook is None:
        return LimitCheckResponse(allowed=True)

    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, we can't use asyncio.run()
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(check_limits(request), loop)
        return future.result(timeout=10.0)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(check_limits(request))
    except Exception as e:
        _LOGGER.warning(f"Limit check hook failed, allowing call: {e}")
        return LimitCheckResponse(allowed=True)


def clear_limit_check_hook() -> None:
    """Clear the limit check hook (disable limit checking)."""
    set_limit_check_hook(None)


