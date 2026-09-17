"""Tests for accounting calls whose caller stopped waiting for the answer."""

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


@pytest.mark.asyncio
async def test_abandoned_call_reports_its_cost_to_spending_limits():
    """The cost reaches the LLM event, which is what spending limits read."""
    captured: list = []
    response = _response(1.5)

    async def slow_completion(*args, **kwargs):
        await asyncio.sleep(0.05)
        return response

    with patch(
        "unillm.clients.uni_llm._acompletion_with_transient_retry",
        side_effect=slow_completion,
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
async def test_call_denied_by_a_spending_limit_reports_no_cost():
    """A limit denial cancels the call itself; nothing was spent."""
    from unillm.limit_hooks import LimitCheckResponse

    captured: list = []

    async def slow_completion(*args, **kwargs):
        await asyncio.sleep(0.05)
        return _response(9.99)

    async def denied(*args, **kwargs):
        return LimitCheckResponse(allowed=False, reason="over budget")

    with (
        patch(
            "unillm.clients.uni_llm._acompletion_with_transient_retry",
            side_effect=slow_completion,
        ),
        patch("unillm.clients.uni_llm.is_limit_check_enabled", return_value=True),
        patch("unillm.clients.uni_llm.check_limits", side_effect=denied),
    ):
        unillm.set_llm_event_hook(captured.append)
        try:
            client = unillm.AsyncUnify(
                "openai/gpt-4o-mini@openrouter",
                api_key="test-key",
                cache=False,
            )
            with pytest.raises(SpendingLimitExceededError):
                await client.generate(messages=[{"role": "user", "content": "hi"}])
            await asyncio.sleep(0.2)
        finally:
            unillm.set_llm_event_hook(None)

    assert [e for e in captured if e.provider_cost] == []
