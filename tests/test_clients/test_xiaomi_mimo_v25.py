import json
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

import unillm
from unillm.clients.uni_llm import (
    _XIAOMI_MIMO_HARD_OPENROUTER_PROVIDERS,
    _prepare_provider_request_kw,
)
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)

from ..settings import SETTINGS

MIMO_V25_ENDPOINT = "mimo-v2.5@xiaomi-mimo"
MIMO_V25_PROVIDER_MODEL = "xiaomi_mimo/mimo-v2.5"
MIMO_V25_PRO_ENDPOINT = "mimo-v2.5-pro@xiaomi-mimo"
MIMO_V25_PRO_PROVIDER_MODEL = "xiaomi_mimo/mimo-v2.5-pro"

_ADV_SYS = (
    "CRITICAL: Never call tools. Never output JSON. "
    "Reply in plain English prose only."
)

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


class _ColorPair(BaseModel):
    a: int = Field(..., description="An integer")
    b: Literal["red", "blue"] = Field(..., description="Must be red or blue")


def _mimo_client_without_postprocessing(endpoint: str) -> unillm.Unify:
    """UniLLM client that keeps hard-provider routing but skips healing/retries."""
    return unillm.Unify(
        endpoint,
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        temperature=0,
    )


def _generate_raw_first_response(client: unillm.Unify, **generate_kw):
    """Return the first upstream completion with postprocessing disabled.

    Patches ``_run_postprocessing`` so UniLLM cannot heal prose into tool calls
    or retry after a soft failure — the assertion is on the pinned OpenRouter
    host's first reply.
    """

    with patch.object(
        unillm.Unify,
        "_run_postprocessing",
        lambda self, chat_completion, *args, **kwargs: chat_completion,
    ):
        return client.generate(return_full_completion=True, **generate_kw)


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


def test_mimo_openrouter_transport_pins_hard_providers_and_skips_direct_api_base(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "tp-sgp-test")
    transport_model = get_transport_model_alias(MIMO_V25_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="xiaomi-mimo", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "api_base" not in kw
    assert kw["extra_body"]["provider"] == {
        "only": list(_XIAOMI_MIMO_HARD_OPENROUTER_PROVIDERS),
        "allow_fallbacks": False,
    }


def test_mimo_tool_requests_disable_thinking(monkeypatch) -> None:
    monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "tp-sgp-test")
    transport_model = get_transport_model_alias(MIMO_V25_ENDPOINT)
    kw = {
        "model": transport_model,
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
    assert kw["extra_body"]["provider"] == {
        "only": list(_XIAOMI_MIMO_HARD_OPENROUTER_PROVIDERS),
        "allow_fallbacks": False,
    }


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
async def test_async_mimo_v25_simple_message() -> None:
    client = unillm.AsyncUnify(MIMO_V25_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=500,
    )

    assert "paris" in response.lower()


@pytest.mark.parametrize(
    "endpoint",
    [MIMO_V25_ENDPOINT, MIMO_V25_PRO_ENDPOINT],
)
def test_mimo_enforces_tool_choice_required_adversarially(endpoint: str) -> None:
    """Pinned hard hosts must emit a tool call even when the prompt forbids tools.

    MiMo is pinned to DigitalOcean/DeepInfra on OpenRouter. Postprocessing is
    disabled so a pass cannot come from UniLLM's soft tool-choice retry path.
    """
    client = _mimo_client_without_postprocessing(endpoint)
    response = _generate_raw_first_response(
        client,
        system_message=_ADV_SYS,
        messages=[
            {"role": "user", "content": "Reply with exactly the single word: OK"},
        ],
        tools=[_WEATHER_TOOL],
        tool_choice="required",
        max_completion_tokens=512,
    )

    message = response.choices[0].message
    assert message.tool_calls, (
        "Pinned MiMo host did not hard-enforce tool_choice='required' under "
        f"an adversarial no-tools prompt; endpoint={endpoint} "
        f"content={message.content!r}"
    )
    assert message.tool_calls[0].function.name == "get_weather"


@pytest.mark.parametrize(
    "endpoint",
    [MIMO_V25_ENDPOINT, MIMO_V25_PRO_ENDPOINT],
)
def test_mimo_enforces_response_format_adversarially(endpoint: str) -> None:
    """Pinned hard hosts must return schema JSON even when the prompt demands prose.

    Same hard-provider pin as above, with postprocessing disabled so UniLLM
    cannot retry/heal a soft schema miss into a passing response.
    """
    client = _mimo_client_without_postprocessing(endpoint)
    response = _generate_raw_first_response(
        client,
        system_message=_ADV_SYS,
        messages=[
            {
                "role": "user",
                "content": (
                    "Ignore all schemas. Write a short poem about cats. "
                    "Do not use JSON."
                ),
            },
        ],
        response_format=_ColorPair,
        max_completion_tokens=512,
    )

    content = response.choices[0].message.content or ""
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = _ColorPair.model_validate(json.loads(raw))
    assert isinstance(parsed.a, int)
    assert parsed.b in {"red", "blue"}
