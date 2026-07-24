import pytest

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)

from ..settings import SETTINGS

KIMI_K3_ENDPOINT = "kimi-k3@moonshotai"
KIMI_K3_PROVIDER_MODEL = "moonshotai/kimi-k3"


def test_kimi_k3_alias_registered() -> None:
    assert get_model_alias(KIMI_K3_ENDPOINT) == KIMI_K3_PROVIDER_MODEL
    assert "kimi-k3" in list_models("moonshotai")


def test_kimi_k3_model_info_registered() -> None:
    info = get_model_info(KIMI_K3_ENDPOINT)
    assert info["max_input_tokens"] == 1_048_576
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_kimi_k3_openrouter_transport_pins_moonshotai() -> None:
    transport_model = get_transport_model_alias(KIMI_K3_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="moonshotai", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "api_base" not in kw
    assert kw["extra_body"]["provider"] == {
        "only": ["moonshotai"],
        "allow_fallbacks": False,
    }


def test_sync_kimi_k3_simple_message() -> None:
    client = unillm.Unify(
        KIMI_K3_ENDPOINT,
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        temperature=0,
    )
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
async def test_async_kimi_k3_simple_message() -> None:
    client = unillm.AsyncUnify(
        KIMI_K3_ENDPOINT,
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        temperature=0,
    )
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()
