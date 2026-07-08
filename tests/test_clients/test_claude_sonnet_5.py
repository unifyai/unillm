import pytest

import unillm
from unillm.clients.provider_preprocessing import apply_provider_preprocessing
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    list_models,
)

SONNET_5_ENDPOINT = "claude-sonnet-5@anthropic"
SONNET_5_PROVIDER_MODEL = "anthropic/claude-sonnet-5"


def test_claude_sonnet_5_alias_registered() -> None:
    assert get_model_alias(SONNET_5_ENDPOINT) == SONNET_5_PROVIDER_MODEL
    assert "claude-sonnet-5" in list_models("anthropic")


def test_claude_sonnet_5_model_info_registered() -> None:
    info = get_model_info(SONNET_5_ENDPOINT)
    assert info["max_input_tokens"] == 1_000_000
    assert info["max_output_tokens"] == 128_000
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_claude_sonnet_5_strips_rejected_sampling_params() -> None:
    kw = {
        "model": SONNET_5_PROVIDER_MODEL,
        "temperature": 0,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": "hello"}],
    }

    apply_provider_preprocessing(kw, "anthropic")

    assert "temperature" not in kw
    assert "top_p" not in kw


def test_claude_sonnet_5_uses_adaptive_thinking_payload() -> None:
    kw = {
        "model": SONNET_5_PROVIDER_MODEL,
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "hello"}],
    }

    apply_provider_preprocessing(kw, "anthropic")

    assert "reasoning_effort" not in kw
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}


def test_sync_claude_sonnet_5_simple_message() -> None:
    client = unillm.Unify(SONNET_5_ENDPOINT)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
async def test_async_claude_sonnet_5_simple_message() -> None:
    client = unillm.AsyncUnify(SONNET_5_ENDPOINT)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()
