"""
Tests for transient HTTP error retry logic.

Covers 503 ServiceUnavailableError, 500 InternalServerError, and 429
RateLimitError (plus mixed sequences). Sleeps are patched so exhaustion
tests stay fast under the default 1/2/4/8/16/32s backoff schedule.
"""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from unillm.helpers import (
    retry_transient_400_async,
    retry_transient_400_sync,
)


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Skip real exponential backoff delays in this module."""
    with patch("time.sleep"), patch("asyncio.sleep", new_callable=AsyncMock):
        yield


# ─────────────────────────────────────────────────────────────────────────────
#  Exception factories
# ─────────────────────────────────────────────────────────────────────────────


def _make_503() -> litellm.ServiceUnavailableError:
    """Reproduce the exact Anthropic overflow error that killed the CI run."""
    return litellm.ServiceUnavailableError(
        message=(
            "AnthropicException - upstream connect error or disconnect/reset "
            "before headers. reset reason: overflow"
        ),
        model="claude-4.6-opus",
        llm_provider="anthropic",
    )


def _make_500() -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message="Internal server error",
        model="gpt-5.2",
        llm_provider="openai",
    )


def _make_429() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="Rate limit exceeded. Please retry after 1 second.",
        model="gpt-5.2",
        llm_provider="openai",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  503 — ServiceUnavailableError
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryServiceUnavailable:

    def test_sync_recovers_after_single_503(self):
        """One 503 followed by success should be retried transparently."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_503()
            return "ok"

        assert retry_transient_400_sync(fn) == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_recovers_after_single_503(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_503()
            return "ok"

        assert await retry_transient_400_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_503(self):
        """If all retries fail, the original exception must propagate."""

        def fn():
            raise _make_503()

        with pytest.raises(litellm.ServiceUnavailableError):
            retry_transient_400_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_503(self):
        async def fn():
            raise _make_503()

        with pytest.raises(litellm.ServiceUnavailableError):
            await retry_transient_400_async(fn)


# ─────────────────────────────────────────────────────────────────────────────
#  500 — InternalServerError
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryInternalServerError:

    def test_sync_recovers_after_single_500(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_500()
            return "ok"

        assert retry_transient_400_sync(fn) == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_recovers_after_single_500(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_500()
            return "ok"

        assert await retry_transient_400_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_500(self):
        def fn():
            raise _make_500()

        with pytest.raises(litellm.InternalServerError):
            retry_transient_400_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_500(self):
        async def fn():
            raise _make_500()

        with pytest.raises(litellm.InternalServerError):
            await retry_transient_400_async(fn)


# ─────────────────────────────────────────────────────────────────────────────
#  429 — RateLimitError
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryRateLimitError:

    def test_sync_recovers_after_single_429(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_429()
            return "ok"

        assert retry_transient_400_sync(fn) == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_recovers_after_single_429(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_429()
            return "ok"

        assert await retry_transient_400_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_429(self):
        def fn():
            raise _make_429()

        with pytest.raises(litellm.RateLimitError):
            retry_transient_400_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_429(self):
        async def fn():
            raise _make_429()

        with pytest.raises(litellm.RateLimitError):
            await retry_transient_400_async(fn)


# ─────────────────────────────────────────────────────────────────────────────
#  Mixed / multi-failure sequences
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryMultipleTransientFailures:

    def test_sync_recovers_after_two_consecutive_503s(self):
        """Two back-to-back 503s followed by success (within retry budget)."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise _make_503()
            return "ok"

        assert retry_transient_400_sync(fn) == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_recovers_after_two_consecutive_503s(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise _make_503()
            return "ok"

        assert await retry_transient_400_async(fn) == "ok"
        assert call_count == 3

    def test_sync_mixed_transient_errors_retried(self):
        """A 429 followed by a 503 followed by success should all be retried."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_429()
            if call_count == 2:
                raise _make_503()
            return "ok"

        assert retry_transient_400_sync(fn) == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_mixed_transient_errors_retried(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_429()
            if call_count == 2:
                raise _make_503()
            return "ok"

        assert await retry_transient_400_async(fn) == "ok"
        assert call_count == 3
