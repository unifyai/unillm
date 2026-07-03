import pytest

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)

MINIMAX_V3_ENDPOINT = "minimax-v3@minimax"
MINIMAX_V3_PROVIDER_MODEL = "minimax/MiniMax-M3"


def test_minimax_v3_alias_registered() -> None:
    assert get_model_alias(MINIMAX_V3_ENDPOINT) == MINIMAX_V3_PROVIDER_MODEL
    assert "minimax-v3" in list_models("minimax")


def test_minimax_v3_model_info_registered() -> None:
    info = get_model_info(MINIMAX_V3_ENDPOINT)
    assert info["max_input_tokens"] == 1_000_000
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_minimax_request_uses_default_api_base() -> None:
    kw = {
        "model": MINIMAX_V3_PROVIDER_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="minimax", stream=False)

    assert kw["api_base"] == "https://api.minimax.io/v1"


def test_minimax_openrouter_transport_skips_direct_api_base() -> None:
    transport_model = get_transport_model_alias(MINIMAX_V3_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="minimax", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "api_base" not in kw


def test_sync_minimax_v3_simple_message() -> None:
    client = unillm.Unify(MINIMAX_V3_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
async def test_async_minimax_v3_simple_message() -> None:
    client = unillm.AsyncUnify(MINIMAX_V3_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()
