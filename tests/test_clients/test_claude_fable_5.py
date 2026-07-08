import pytest

import unillm
from unillm.clients.provider_preprocessing import apply_provider_preprocessing
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    list_models,
)

FABLE_5_ENDPOINT = "claude-fable-5@anthropic"
FABLE_5_PROVIDER_MODEL = "anthropic/claude-fable-5"


def test_claude_fable_5_alias_registered() -> None:
    assert get_model_alias(FABLE_5_ENDPOINT) == FABLE_5_PROVIDER_MODEL
    assert "claude-fable-5" in list_models("anthropic")


def test_claude_fable_5_model_info_registered() -> None:
    info = get_model_info(FABLE_5_ENDPOINT)
    assert info["max_input_tokens"] == 1_000_000
    assert info["max_output_tokens"] == 128_000
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_claude_fable_5_strips_rejected_sampling_params() -> None:
    kw = {
        "model": FABLE_5_PROVIDER_MODEL,
        "temperature": 0,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": "hello"}],
    }

    apply_provider_preprocessing(kw, "anthropic")

    assert "temperature" not in kw
    assert "top_p" not in kw


def test_claude_fable_5_uses_adaptive_thinking_payload() -> None:
    kw = {
        "model": FABLE_5_PROVIDER_MODEL,
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "hello"}],
    }

    apply_provider_preprocessing(kw, "anthropic")

    assert "reasoning_effort" not in kw
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}


def test_sync_claude_fable_5_simple_message() -> None:
    client = unillm.Unify(FABLE_5_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
async def test_async_claude_fable_5_simple_message() -> None:
    client = unillm.AsyncUnify(FABLE_5_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()
