import pytest

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)

QWEN_37_PLUS_ENDPOINT = "qwen3.7-plus@qwen"
QWEN_37_PLUS_PROVIDER_MODEL = "qwen/qwen3.7-plus"


def test_qwen_37_plus_alias_registered() -> None:
    assert get_model_alias(QWEN_37_PLUS_ENDPOINT) == QWEN_37_PLUS_PROVIDER_MODEL
    assert "qwen3.7-plus" in list_models("qwen")


def test_qwen_37_plus_model_info_registered() -> None:
    info = get_model_info(QWEN_37_PLUS_ENDPOINT)
    assert info["max_input_tokens"] == 1_000_000
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_qwen_37_plus_openrouter_transport_has_no_direct_api_base() -> None:
    transport_model = get_transport_model_alias(QWEN_37_PLUS_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="qwen", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "api_base" not in kw


def test_sync_qwen_37_plus_simple_message() -> None:
    client = unillm.Unify(QWEN_37_PLUS_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
async def test_async_qwen_37_plus_simple_message() -> None:
    client = unillm.AsyncUnify(QWEN_37_PLUS_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()
