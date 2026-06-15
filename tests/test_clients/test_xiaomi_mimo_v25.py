import os

import pytest

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw
from unillm.endpoints.utils import get_model_alias, get_model_info, list_models

MIMO_V25_ENDPOINT = "mimo-v2.5@xiaomi-mimo"
MIMO_V25_PROVIDER_MODEL = "xiaomi_mimo/mimo-v2.5"
MIMO_V25_PRO_ENDPOINT = "mimo-v2.5-pro@xiaomi-mimo"
MIMO_V25_PRO_PROVIDER_MODEL = "xiaomi_mimo/mimo-v2.5-pro"
_HAS_XIAOMI_MIMO_API_KEY = bool(os.environ.get("XIAOMI_MIMO_API_KEY"))


def test_mimo_v25_alias_registered() -> None:
    assert get_model_alias(MIMO_V25_ENDPOINT) == MIMO_V25_PROVIDER_MODEL
    assert "mimo-v2.5" in list_models("xiaomi-mimo")


def test_mimo_v25_pro_alias_registered() -> None:
    assert get_model_alias(MIMO_V25_PRO_ENDPOINT) == MIMO_V25_PRO_PROVIDER_MODEL
    assert "mimo-v2.5-pro" in list_models("xiaomi-mimo")


def test_mimo_v25_model_info_registered() -> None:
    info = get_model_info(MIMO_V25_ENDPOINT)
    assert info["max_input_tokens"] == 1_000_000
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_mimo_v25_pro_model_info_registered() -> None:
    info = get_model_info(MIMO_V25_PRO_ENDPOINT)
    assert info["max_input_tokens"] == 1_000_000
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_mimo_token_plan_key_uses_regional_api_base(monkeypatch) -> None:
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "tp-sgp-test")
    kw = {
        "model": MIMO_V25_PROVIDER_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="xiaomi-mimo", stream=False)

    assert kw["api_base"] == "https://token-plan-sgp.xiaomimimo.com/v1"


def test_mimo_tool_requests_disable_thinking(monkeypatch) -> None:
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "tp-sgp-test")
    kw = {
        "model": MIMO_V25_PROVIDER_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_id", "parameters": {"type": "object"}},
            },
        ],
    }

    _prepare_provider_request_kw(kw=kw, provider="xiaomi-mimo", stream=False)

    assert set(kw["allowed_openai_params"]) >= {"tools", "tool_choice"}
    assert kw["tool_choice"] == "auto"
    assert kw["extra_body"]["thinking"] == {"type": "disabled"}


@pytest.mark.skipif(
    not _HAS_XIAOMI_MIMO_API_KEY,
    reason="No Xiaomi MiMo API key available",
)
def test_sync_mimo_v25_simple_message() -> None:
    client = unillm.Unify(MIMO_V25_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=500,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _HAS_XIAOMI_MIMO_API_KEY,
    reason="No Xiaomi MiMo API key available",
)
async def test_async_mimo_v25_simple_message() -> None:
    client = unillm.AsyncUnify(MIMO_V25_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=500,
    )

    assert "paris" in response.lower()
