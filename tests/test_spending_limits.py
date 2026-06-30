"""
Tests for spending limit callback functionality in UniLLM.

These tests verify:
1. LimitCheckRequest and LimitCheckResponse data classes
2. Callback hook registration and invocation
3. Error handling (callback failures fail open)
4. SpendingLimitExceededError exception

The actual limit checking logic (HTTP calls to Orchestra) is implemented in Unify.
UniLLM simply invokes a registered callback hook and respects its response.
"""

import asyncio

import pytest

from unillm.limit_hooks import (
    LimitCheckRequest,
    LimitCheckResponse,
    LimitType,
    SpendingLimitExceededError,
    check_limits,
    check_limits_sync,
    clear_limit_check_hook,
    get_limit_check_hook,
    is_limit_check_enabled,
    set_limit_check_hook,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_hook():
    """Reset the limit check hook before and after each test."""
    clear_limit_check_hook()
    yield
    clear_limit_check_hook()


# ---------------------------------------------------------------------------
# Test LimitCheckRequest
# ---------------------------------------------------------------------------


class TestLimitCheckRequest:
    """Tests for LimitCheckRequest data class."""

    def test_minimal_request(self):
        """Create request with minimal fields."""
        request = LimitCheckRequest(
            model="gpt-4",
            endpoint="openai/gpt-4",
        )
        assert request.model == "gpt-4"
        assert request.endpoint == "openai/gpt-4"
        assert request.estimated_input_tokens is None
        assert request.estimated_output_tokens is None

    def test_full_request(self):
        """Create request with all fields."""
        request = LimitCheckRequest(
            model="gpt-4",
            endpoint="openai/gpt-4",
            estimated_input_tokens=100,
            estimated_output_tokens=500,
        )
        assert request.estimated_input_tokens == 100
        assert request.estimated_output_tokens == 500


# ---------------------------------------------------------------------------
# Test LimitCheckResponse
# ---------------------------------------------------------------------------


class TestLimitCheckResponse:
    """Tests for LimitCheckResponse data class."""

    def test_allowed_response(self):
        """Response indicating call is allowed."""
        response = LimitCheckResponse(allowed=True)
        assert response.allowed is True
        assert response.reason is None
        assert response.limit_type is None
        assert response.percent_used is None

    def test_denied_response(self):
        """Response indicating call is denied."""
        response = LimitCheckResponse(
            allowed=False,
            reason="Monthly limit exceeded",
            limit_type=LimitType.ASSISTANT,
            limit_value=100.0,
            current_spend=105.0,
            entity_id="agent_123",
            entity_name="TestAssistant",
        )
        assert response.allowed is False
        assert response.reason == "Monthly limit exceeded"
        assert response.limit_type == LimitType.ASSISTANT
        assert response.percent_used == 105.0

    def test_percent_used_calculation(self):
        """Percent used is calculated correctly."""
        response = LimitCheckResponse(
            allowed=True,
            limit_value=200.0,
            current_spend=50.0,
        )
        assert response.percent_used == 25.0

    def test_percent_used_no_limit(self):
        """Percent used is None when no limit set."""
        response = LimitCheckResponse(allowed=True, limit_value=None)
        assert response.percent_used is None

    def test_percent_used_zero_limit(self):
        """Percent used is None when limit is zero."""
        response = LimitCheckResponse(
            allowed=False,
            limit_value=0.0,
            current_spend=50.0,
        )
        assert response.percent_used is None

    def test_percent_used_zero_spend(self):
        """Percent used is 0 when spend is zero."""
        response = LimitCheckResponse(
            allowed=True,
            limit_value=100.0,
            current_spend=0.0,
        )
        assert response.percent_used == 0.0


# ---------------------------------------------------------------------------
# Test LimitType
# ---------------------------------------------------------------------------


class TestLimitType:
    """Tests for LimitType enum."""

    def test_all_limit_types(self):
        """Verify all limit types have correct values."""
        assert LimitType.ASSISTANT.value == "assistant"
        assert LimitType.USER.value == "user"
        assert LimitType.MEMBER.value == "member"
        assert LimitType.ORGANIZATION.value == "organization"


# ---------------------------------------------------------------------------
# Test SpendingLimitExceededError
# ---------------------------------------------------------------------------


class TestSpendingLimitExceededError:
    """Tests for SpendingLimitExceededError exception."""

    def test_default_message(self):
        """Default error message is generated correctly."""
        response = LimitCheckResponse(
            allowed=False,
            limit_type=LimitType.USER,
            limit_value=500.0,
            current_spend=520.0,
        )
        error = SpendingLimitExceededError(response)
        assert "user" in str(error).lower()
        assert "$500.00" in str(error)
        assert "$520.00" in str(error)

    def test_custom_message(self):
        """Custom reason from response is used."""
        response = LimitCheckResponse(
            allowed=False,
            reason="Custom error message",
        )
        error = SpendingLimitExceededError(response)
        assert str(error) == "Custom error message"

    def test_response_attribute(self):
        """Error stores the response."""
        response = LimitCheckResponse(allowed=False)
        error = SpendingLimitExceededError(response)
        assert error.response is response


# ---------------------------------------------------------------------------
# Test Hook Registration
# ---------------------------------------------------------------------------


class TestHookRegistration:
    """Tests for limit check hook registration."""

    def test_no_hook_by_default(self):
        """No hook is registered by default."""
        assert get_limit_check_hook() is None
        assert is_limit_check_enabled() is False

    def test_set_hook(self):
        """Setting a hook enables limit checking."""

        async def my_hook(request):
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(my_hook)
        assert get_limit_check_hook() is my_hook
        assert is_limit_check_enabled() is True

    def test_clear_hook(self):
        """Clearing hook disables limit checking."""

        async def my_hook(request):
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(my_hook)
        clear_limit_check_hook()
        assert get_limit_check_hook() is None
        assert is_limit_check_enabled() is False

    def test_set_hook_to_none(self):
        """Setting hook to None clears it."""

        async def my_hook(request):
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(my_hook)
        set_limit_check_hook(None)
        assert get_limit_check_hook() is None
        assert is_limit_check_enabled() is False


# ---------------------------------------------------------------------------
# Test Async check_limits
# ---------------------------------------------------------------------------


class TestCheckLimits:
    """Tests for async check_limits function."""

    @pytest.mark.asyncio
    async def test_no_hook_returns_allowed(self):
        """When no hook is set, requests are allowed."""
        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = await check_limits(request)
        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_hook_called_with_request(self):
        """Hook is called with the request."""
        received_request = None

        async def my_hook(request):
            nonlocal received_request
            received_request = request
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(my_hook)

        request = LimitCheckRequest(
            model="gpt-4",
            endpoint="openai/gpt-4",
            estimated_input_tokens=100,
        )
        await check_limits(request)

        assert received_request is request

    @pytest.mark.asyncio
    async def test_hook_returns_allowed(self):
        """Hook that allows the request."""

        async def allow_hook(request):
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(allow_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = await check_limits(request)
        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_hook_returns_denied(self):
        """Hook that denies the request."""

        async def deny_hook(request):
            return LimitCheckResponse(
                allowed=False,
                reason="Limit exceeded",
                limit_type=LimitType.ASSISTANT,
            )

        set_limit_check_hook(deny_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = await check_limits(request)
        assert response.allowed is False
        assert response.reason == "Limit exceeded"
        assert response.limit_type == LimitType.ASSISTANT

    @pytest.mark.asyncio
    async def test_hook_exception_fails_open(self):
        """Hook that raises exception fails open."""

        async def failing_hook(request):
            raise RuntimeError("Hook failed!")

        set_limit_check_hook(failing_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = await check_limits(request)
        # Should fail open (allow the request)
        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_async_hook_execution(self):
        """Verify hook is actually called asynchronously."""
        call_times = []

        async def slow_hook(request):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)  # Simulate async work
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(slow_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = await check_limits(request)

        assert response.allowed is True
        assert len(call_times) == 1


# ---------------------------------------------------------------------------
# Test Sync check_limits_sync
# ---------------------------------------------------------------------------


class TestCheckLimitsSync:
    """Tests for sync check_limits_sync function."""

    def test_no_hook_returns_allowed(self):
        """When no hook is set, requests are allowed."""
        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = check_limits_sync(request)
        assert response.allowed is True

    def test_hook_returns_allowed(self):
        """Hook that allows the request (sync call)."""

        async def allow_hook(request):
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(allow_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = check_limits_sync(request)
        assert response.allowed is True

    def test_hook_returns_denied(self):
        """Hook that denies the request (sync call)."""

        async def deny_hook(request):
            return LimitCheckResponse(
                allowed=False,
                reason="Over limit",
            )

        set_limit_check_hook(deny_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = check_limits_sync(request)
        assert response.allowed is False
        assert response.reason == "Over limit"

    def test_hook_exception_fails_open(self):
        """Hook that raises exception fails open (sync call)."""

        async def failing_hook(request):
            raise ValueError("Oops!")

        set_limit_check_hook(failing_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = check_limits_sync(request)
        assert response.allowed is True


# ---------------------------------------------------------------------------
# Test Hook with Different Responses
# ---------------------------------------------------------------------------


class TestHookResponses:
    """Tests for different hook response scenarios."""

    @pytest.mark.asyncio
    async def test_hook_with_full_response_data(self):
        """Hook returns response with all fields populated."""

        async def detailed_hook(request):
            return LimitCheckResponse(
                allowed=False,
                reason="Monthly spending limit exceeded",
                limit_type=LimitType.ORGANIZATION,
                limit_value=1000.0,
                current_spend=1050.0,
                entity_id="org_123",
                entity_name="Acme Corp",
            )

        set_limit_check_hook(detailed_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        response = await check_limits(request)

        assert response.allowed is False
        assert response.limit_type == LimitType.ORGANIZATION
        assert response.limit_value == 1000.0
        assert response.current_spend == 1050.0
        assert response.entity_id == "org_123"
        assert response.entity_name == "Acme Corp"
        assert response.percent_used == 105.0

    @pytest.mark.asyncio
    async def test_hook_based_on_model(self):
        """Hook can make decisions based on model."""

        async def model_based_hook(request):
            if "gpt-4" in request.model:
                return LimitCheckResponse(allowed=False, reason="GPT-4 disabled")
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(model_based_hook)

        gpt4_request = LimitCheckRequest(model="gpt-4", endpoint="test")
        gpt4_response = await check_limits(gpt4_request)
        assert gpt4_response.allowed is False

        gpt35_request = LimitCheckRequest(model="gpt-3.5-turbo", endpoint="test")
        gpt35_response = await check_limits(gpt35_request)
        assert gpt35_response.allowed is True

    @pytest.mark.asyncio
    async def test_hook_based_on_endpoint(self):
        """Hook can make decisions based on endpoint."""

        async def endpoint_based_hook(request):
            if "expensive" in request.endpoint:
                return LimitCheckResponse(
                    allowed=False,
                    reason="Expensive endpoint blocked",
                )
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(endpoint_based_hook)

        cheap_request = LimitCheckRequest(model="test", endpoint="cheap/model")
        cheap_response = await check_limits(cheap_request)
        assert cheap_response.allowed is True

        expensive_request = LimitCheckRequest(model="test", endpoint="expensive/model")
        expensive_response = await check_limits(expensive_request)
        assert expensive_response.allowed is False


# ---------------------------------------------------------------------------
# Test Multiple Hooks (replacement behavior)
# ---------------------------------------------------------------------------


class TestHookReplacement:
    """Tests for hook replacement behavior."""

    @pytest.mark.asyncio
    async def test_new_hook_replaces_old(self):
        """Setting a new hook replaces the old one."""

        async def first_hook(request):
            return LimitCheckResponse(allowed=True, reason="first")

        async def second_hook(request):
            return LimitCheckResponse(allowed=False, reason="second")

        set_limit_check_hook(first_hook)
        set_limit_check_hook(second_hook)

        request = LimitCheckRequest(model="test", endpoint="test")
        response = await check_limits(request)

        # Should use second hook
        assert response.allowed is False
        assert response.reason == "second"


# ---------------------------------------------------------------------------
# Test Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_hook_returns_wrong_type_fails_open(self):
        """Hook returning wrong type should fail open."""

        async def bad_hook(request):
            return "not a response"  # Wrong type

        set_limit_check_hook(bad_hook)

        request = LimitCheckRequest(model="gpt-4", endpoint="test")
        # This should not raise, but behavior depends on implementation
        # At minimum, it shouldn't crash the caller

    @pytest.mark.asyncio
    async def test_empty_model_and_endpoint(self):
        """Request with empty model and endpoint."""

        async def echo_hook(request):
            return LimitCheckResponse(
                allowed=True,
                reason=f"model={request.model}, endpoint={request.endpoint}",
            )

        set_limit_check_hook(echo_hook)

        request = LimitCheckRequest(model="", endpoint="")
        response = await check_limits(request)
        assert response.allowed is True
        assert "model=, endpoint=" in response.reason

    @pytest.mark.asyncio
    async def test_concurrent_checks(self):
        """Multiple concurrent limit checks work correctly."""
        call_count = 0

        async def counting_hook(request):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return LimitCheckResponse(allowed=True)

        set_limit_check_hook(counting_hook)

        # Make 5 concurrent checks
        requests = [
            LimitCheckRequest(model=f"model-{i}", endpoint="test") for i in range(5)
        ]
        responses = await asyncio.gather(*[check_limits(r) for r in requests])

        assert call_count == 5
        assert all(r.allowed for r in responses)


# ---------------------------------------------------------------------------
# Test Limit Boundary
# ---------------------------------------------------------------------------


class TestLimitBoundary:
    """Tests for limit boundary conditions."""

    @pytest.mark.asyncio
    async def test_at_limit_is_denied(self):
        """Spend exactly at limit should be denied (design decision)."""

        async def boundary_hook(request):
            # Simulating exactly at limit
            return LimitCheckResponse(
                allowed=False,
                limit_value=100.0,
                current_spend=100.0,
                limit_type=LimitType.USER,
            )

        set_limit_check_hook(boundary_hook)

        request = LimitCheckRequest(model="test", endpoint="test")
        response = await check_limits(request)

        assert response.allowed is False
        assert response.percent_used == 100.0

    @pytest.mark.asyncio
    async def test_just_under_limit_is_allowed(self):
        """Spend just under limit should be allowed."""

        async def boundary_hook(request):
            return LimitCheckResponse(
                allowed=True,
                limit_value=100.0,
                current_spend=99.99,
            )

        set_limit_check_hook(boundary_hook)

        request = LimitCheckRequest(model="test", endpoint="test")
        response = await check_limits(request)

        assert response.allowed is True
        assert response.percent_used == pytest.approx(99.99, rel=0.01)
