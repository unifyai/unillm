"""Tests for billing calls whose caller stopped waiting for the answer."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import unillm
from unillm.limit_hooks import SpendingLimitExceededError


def _response(cost: float) -> ModelResponse:
    response = ModelResponse(
        id="chatcmpl_abandoned",
        choices=[
            Choices(
                message=Message(content="an answer nobody read"),
                finish_reason="stop",
                index=0,
            ),
        ],
    )
    response.model = "openai/gpt-4o-mini"
    usage = Usage(prompt_tokens=200_000, completion_tokens=4_000, total_tokens=204_000)
    usage.cost = cost
    response.usage = usage
    return response


async def _abandon_mid_call(deduct_patch_target: str, delay: float = 0.05):
    """Dispatch a call, drop it while in flight, and return the deduct mock."""
    response = _response(4.25)

    async def slow_completion(*args, **kwargs):
        await asyncio.sleep(delay)
        return response

    with (
        patch(
            "unillm.clients.uni_llm._acompletion_with_empty_responses_fallback",
            side_effect=slow_completion,
        ),
        patch(deduct_patch_target) as deduct,
    ):
        client = unillm.AsyncUnify(
            "openai/gpt-4o-mini@openrouter",
            api_key="test-key",
            cache=False,
        )
        call = asyncio.create_task(
            client.generate(messages=[{"role": "user", "content": "hi"}]),
        )
        # Let the request reach the provider, then walk away from it.
        await asyncio.sleep(delay / 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        # The generation was already running, so it lands shortly after.
        await asyncio.sleep(delay * 4)
        return deduct


@pytest.mark.asyncio
async def test_abandoned_call_is_charged_at_the_reported_cost():
    """A cancelled call still bills, at the amount the provider reports."""
    deduct = await _abandon_mid_call("unillm.clients.uni_llm._safe_deduct_credits")

    deduct.assert_called_once()
    assert deduct.call_args.args[0] == 4.25
    assert deduct.call_args.kwargs["abandoned"] is True
    assert deduct.call_args.kwargs["model"] == "openrouter/openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_abandoned_call_reports_its_cost_to_spending_limits():
    """The charge reaches the LLM event, which is what spending limits read."""
    captured: list = []
    response = _response(1.5)

    async def slow_completion(*args, **kwargs):
        await asyncio.sleep(0.05)
        return response

    with (
        patch(
            "unillm.clients.uni_llm._acompletion_with_empty_responses_fallback",
            side_effect=slow_completion,
        ),
        patch("unillm.clients.uni_llm._safe_deduct_credits"),
    ):
        unillm.set_llm_event_hook(captured.append)
        try:
            client = unillm.AsyncUnify(
                "openai/gpt-4o-mini@openrouter",
                api_key="test-key",
                cache=False,
            )
            call = asyncio.create_task(
                client.generate(messages=[{"role": "user", "content": "hi"}]),
            )
            await asyncio.sleep(0.01)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            await asyncio.sleep(0.2)
        finally:
            unillm.set_llm_event_hook(None)

    priced = [e for e in captured if e.provider_cost]
    assert len(priced) == 1
    assert priced[0].provider_cost == 1.5


@pytest.mark.asyncio
async def test_call_denied_by_a_spending_limit_is_not_charged():
    """A limit denial cancels the call itself; the account owes nothing."""
    from unillm.limit_hooks import LimitCheckResponse

    async def slow_completion(*args, **kwargs):
        await asyncio.sleep(0.05)
        return _response(9.99)

    async def denied(*args, **kwargs):
        return LimitCheckResponse(allowed=False, reason="over budget")

    with (
        patch(
            "unillm.clients.uni_llm._acompletion_with_empty_responses_fallback",
            side_effect=slow_completion,
        ),
        patch("unillm.clients.uni_llm.is_limit_check_enabled", return_value=True),
        patch("unillm.clients.uni_llm.check_limits", side_effect=denied),
        patch("unillm.clients.uni_llm._safe_deduct_credits") as deduct,
    ):
        client = unillm.AsyncUnify(
            "openai/gpt-4o-mini@openrouter",
            api_key="test-key",
            cache=False,
        )
        with pytest.raises(SpendingLimitExceededError):
            await client.generate(messages=[{"role": "user", "content": "hi"}])
        await asyncio.sleep(0.2)

    deduct.assert_not_called()
