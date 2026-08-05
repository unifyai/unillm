"""
Tests for provider-call retry logic.

This is the only layer that retries an LLM call, so what it declines is never
attempted again. Two things therefore need covering: that the familiar
transients still recover (503, 500, 429, unparseable bodies, mixed sequences),
and that the *default* for an unfamiliar failure is to retry rather than to end
the caller's trajectory.

Sleeps are patched so exhaustion tests stay fast under the default
1/2/4/8/16/32s backoff schedule.
"""

import logging
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import openai
import pytest

from unillm.helpers import (
    retry_transient_llm_async,
    retry_transient_llm_sync,
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


def _make_unparseable_body() -> litellm.APIError:
    """Reproduce the OpenRouter 200-with-whitespace body that killed two runs.

    LiteLLM maps every provider's "the body was not JSON" failure to an
    ``APIError`` carrying the response status, which is 200 here — so nothing
    but the message distinguishes it from a permanent request error.
    """
    return litellm.APIError(
        message=(
            "APIError: OpenrouterException - Unable to get json response - "
            "Expecting value: line 127 column 1 (char 693), Original "
            "Response: " + "\n" * 40
        ),
        model="openai/gpt-5.6-sol",
        llm_provider="openrouter",
        status_code=200,
    )


def _make_permanent_400() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message="Invalid value for 'tool_choice': no such tool.",
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

        assert retry_transient_llm_sync(fn) == "ok"
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

        assert await retry_transient_llm_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_503(self):
        """If all retries fail, the original exception must propagate."""

        def fn():
            raise _make_503()

        with pytest.raises(litellm.ServiceUnavailableError):
            retry_transient_llm_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_503(self):
        async def fn():
            raise _make_503()

        with pytest.raises(litellm.ServiceUnavailableError):
            await retry_transient_llm_async(fn)


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

        assert retry_transient_llm_sync(fn) == "ok"
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

        assert await retry_transient_llm_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_500(self):
        def fn():
            raise _make_500()

        with pytest.raises(litellm.InternalServerError):
            retry_transient_llm_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_500(self):
        async def fn():
            raise _make_500()

        with pytest.raises(litellm.InternalServerError):
            await retry_transient_llm_async(fn)


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

        assert retry_transient_llm_sync(fn) == "ok"
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

        assert await retry_transient_llm_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_429(self):
        def fn():
            raise _make_429()

        with pytest.raises(litellm.RateLimitError):
            retry_transient_llm_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_429(self):
        async def fn():
            raise _make_429()

        with pytest.raises(litellm.RateLimitError):
            await retry_transient_llm_async(fn)


# ─────────────────────────────────────────────────────────────────────────────
#  Unparseable response body (200 with a non-JSON payload)
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryUnparseableBody:

    def test_sync_recovers_after_single_unparseable_body(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_unparseable_body()
            return "ok"

        assert retry_transient_llm_sync(fn) == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_recovers_after_single_unparseable_body(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_unparseable_body()
            return "ok"

        assert await retry_transient_llm_async(fn) == "ok"
        assert call_count == 2

    def test_sync_exhausted_retries_raises_unparseable_body(self):
        def fn():
            raise _make_unparseable_body()

        with pytest.raises(litellm.APIError):
            retry_transient_llm_sync(fn)

    @pytest.mark.asyncio
    async def test_async_exhausted_retries_raises_unparseable_body(self):
        async def fn():
            raise _make_unparseable_body()

        with pytest.raises(litellm.APIError):
            await retry_transient_llm_async(fn)

    def test_unparseable_body_backs_off_between_attempts(self):
        """The upstream was still queueing, so retries must not hammer it."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise _make_unparseable_body()
            return "ok"

        with patch("time.sleep") as sleep:
            assert retry_transient_llm_sync(fn) == "ok"

        assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]

    def test_permanent_400_is_not_retried(self):
        """A genuinely malformed request must still fail on the first attempt."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise _make_permanent_400()

        with pytest.raises(litellm.BadRequestError):
            retry_transient_llm_sync(fn)

        assert call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Unfamiliar failures — the default decides whether a run survives
# ─────────────────────────────────────────────────────────────────────────────


def _count_attempts(exc: BaseException, *, succeed_on: int = 2):
    """Return a callable that raises *exc* until its *succeed_on*-th call."""
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] < succeed_on:
            raise exc
        return "ok"

    return fn, state


class TestUnfamiliarFailuresRetryByDefault:
    """No message pattern matches these, and none is a known transient type.

    Under an allowlist each would get exactly one attempt and end the caller's
    trajectory; the fault-based default retries them because the status says the
    provider's side of the exchange failed, not ours.
    """

    def test_unrecognised_5xx_is_retried(self):
        exc = litellm.APIError(
            message="ProviderException - something we have never seen before",
            model="gpt-5.2",
            llm_provider="openai",
            status_code=502,
        )
        fn, state = _count_attempts(exc)

        assert retry_transient_llm_sync(fn) == "ok"
        assert state["calls"] == 2

    def test_unrecognised_2xx_failure_is_retried(self):
        """The incident's shape, minus the message we now recognise."""
        exc = litellm.APIError(
            message="OpenrouterException - upstream disconnected mid-body",
            model="openai/gpt-5.6-sol",
            llm_provider="openrouter",
            status_code=200,
        )
        fn, state = _count_attempts(exc)

        assert retry_transient_llm_sync(fn) == "ok"
        assert state["calls"] == 2

    def test_request_timeout_is_retried(self):
        """408 is a 4xx by convention but describes timing, not content."""
        fn, state = _count_attempts(
            litellm.Timeout(
                message="Request timed out",
                model="gpt-5.2",
                llm_provider="openai",
            ),
        )

        assert retry_transient_llm_sync(fn) == "ok"
        assert state["calls"] == 2

    def test_malformed_provider_payload_is_retried(self):
        fn, state = _count_attempts(
            litellm.APIResponseValidationError(
                message="Invalid response object from provider",
                model="gpt-5.2",
                llm_provider="openai",
            ),
        )

        assert retry_transient_llm_sync(fn) == "ok"
        assert state["calls"] == 2

    @pytest.mark.asyncio
    async def test_unrecognised_2xx_failure_is_retried_async(self):
        exc = litellm.APIError(
            message="OpenrouterException - upstream disconnected mid-body",
            model="openai/gpt-5.6-sol",
            llm_provider="openrouter",
            status_code=200,
        )
        state = {"calls": 0}

        async def fn():
            state["calls"] += 1
            if state["calls"] < 2:
                raise exc
            return "ok"

        assert await retry_transient_llm_async(fn) == "ok"
        assert state["calls"] == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Client-fault failures — retrying repeats our own mistake
# ─────────────────────────────────────────────────────────────────────────────


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(status, request=request)


_CLIENT_FAULTS = [
    litellm.AuthenticationError(
        message="Incorrect API key provided",
        model="gpt-5.2",
        llm_provider="openai",
    ),
    litellm.PermissionDeniedError(
        message="Project does not have access to this model",
        model="gpt-5.2",
        llm_provider="openai",
        response=_response(403),
    ),
    litellm.NotFoundError(
        message="The model does not exist",
        model="gpt-5.2",
        llm_provider="openai",
    ),
    litellm.ContextWindowExceededError(
        message="This model's maximum context length is 400000 tokens",
        model="gpt-5.2",
        llm_provider="openai",
    ),
    litellm.ContentPolicyViolationError(
        message="Your request was rejected by our safety system",
        model="gpt-5.2",
        llm_provider="openai",
    ),
    litellm.UnprocessableEntityError(
        message="Unprocessable entity",
        model="gpt-5.2",
        llm_provider="openai",
        response=_response(422),
    ),
    openai.OpenAIError("Finish reason was length, no parsed content available"),
]


class TestClientFaultsAreNotRetried:

    @pytest.mark.parametrize(
        "exc",
        _CLIENT_FAULTS,
        ids=lambda exc: type(exc).__name__,
    )
    def test_one_attempt_only(self, exc: BaseException):
        state = {"calls": 0}

        def fn():
            state["calls"] += 1
            raise exc

        with pytest.raises(type(exc)):
            retry_transient_llm_sync(fn)

        assert state["calls"] == 1


class TestNonProviderFailuresPropagate:
    """Failures that never touched a provider are not this layer's business."""

    def test_budget_exceeded_is_not_caught(self):
        state = {"calls": 0}

        def fn():
            state["calls"] += 1
            raise litellm.BudgetExceededError(current_cost=10.0, max_budget=1.0)

        with pytest.raises(litellm.BudgetExceededError):
            retry_transient_llm_sync(fn)

        assert state["calls"] == 1

    def test_local_programming_error_is_not_caught(self):
        state = {"calls": 0}

        def fn():
            state["calls"] += 1
            raise TypeError("got an unexpected keyword argument")

        with pytest.raises(TypeError):
            retry_transient_llm_sync(fn)

        assert state["calls"] == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Observability — a failure must say whether it was ever retried
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryOutcomesAreLogged:

    def test_declined_failure_says_so(self, caplog: pytest.LogCaptureFixture):
        def fn():
            raise _make_permanent_400()

        with caplog.at_level(logging.WARNING, logger="unillm.retry"):
            with pytest.raises(litellm.BadRequestError):
                retry_transient_llm_sync(fn)

        assert "not retried, treated as permanent" in caplog.text
        assert "BadRequestError(status=400)" in caplog.text

    def test_exhausted_failure_reports_the_attempt_count(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        def fn():
            raise _make_unparseable_body()

        with caplog.at_level(logging.WARNING, logger="unillm.retry"):
            with pytest.raises(litellm.APIError):
                retry_transient_llm_sync(fn)

        assert "retries exhausted after 7 attempts" in caplog.text

    def test_recovered_failure_is_logged_below_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        """A recovery is routine, so it must not read as a problem."""
        fn, _ = _count_attempts(_make_503())

        with caplog.at_level(logging.DEBUG, logger="unillm.retry"):
            assert retry_transient_llm_sync(fn) == "ok"

        assert "retrying (attempt 1/7)" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


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

        assert retry_transient_llm_sync(fn) == "ok"
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

        assert await retry_transient_llm_async(fn) == "ok"
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

        assert retry_transient_llm_sync(fn) == "ok"
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

        assert await retry_transient_llm_async(fn) == "ok"
        assert call_count == 3
