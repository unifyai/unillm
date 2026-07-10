import json
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

import unillm
from unillm.clients.uni_llm import (
    _DEEPSEEK_V4_PRO_HARD_OPENROUTER_PROVIDERS,
    _prepare_provider_request_kw,
)
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)

from ..settings import SETTINGS

DEEPSEEK_V4_MAX_ENDPOINT = "deepseek-v4-max@deepseek"
DEEPSEEK_V4_MAX_PROVIDER_MODEL = "deepseek/deepseek-v4-pro"

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


def _deepseek_client_without_postprocessing() -> unillm.Unify:
    """UniLLM client that keeps hard-provider routing but skips healing/retries."""
    return unillm.Unify(
        DEEPSEEK_V4_MAX_ENDPOINT,
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


def test_deepseek_v4_max_alias_registered() -> None:
    assert get_model_alias(DEEPSEEK_V4_MAX_ENDPOINT) == DEEPSEEK_V4_MAX_PROVIDER_MODEL
    assert "deepseek-v4-max" in list_models("deepseek")


def test_deepseek_v4_max_model_info_registered() -> None:
    info = get_model_info(DEEPSEEK_V4_MAX_ENDPOINT)
    assert info["max_input_tokens"] == 1_048_576
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_deepseek_v4_pro_openrouter_pins_hard_providers() -> None:
    transport_model = get_transport_model_alias(DEEPSEEK_V4_MAX_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="deepseek", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "deepseek-v4-pro" in transport_model
    assert kw["extra_body"]["provider"] == {
        "only": list(_DEEPSEEK_V4_PRO_HARD_OPENROUTER_PROVIDERS),
        "allow_fallbacks": False,
    }


def test_sync_deepseek_v4_max_simple_message() -> None:
    client = unillm.Unify(DEEPSEEK_V4_MAX_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=20,
    )

    assert "Paris" in response


@pytest.mark.asyncio
async def test_async_deepseek_v4_max_simple_message() -> None:
    client = unillm.AsyncUnify(DEEPSEEK_V4_MAX_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=20,
    )

    assert "Paris" in response


def test_deepseek_v4_pro_enforces_tool_choice_required_adversarially() -> None:
    """Pinned hard hosts must emit a tool call even when the prompt forbids tools.

    DeepSeek V4 Pro is pinned to OpenRouter hosts that hard-enforce
    tool_choice. Postprocessing is disabled so a pass cannot come from
    UniLLM's soft tool-choice retry/healing path.
    """
    client = _deepseek_client_without_postprocessing()
    response = _generate_raw_first_response(
        client,
        system_message=_ADV_SYS,
        messages=[
            {"role": "user", "content": "Reply with exactly the single word: OK"},
        ],
        tools=[_WEATHER_TOOL],
        tool_choice="required",
        max_completion_tokens=256,
    )

    message = response.choices[0].message
    assert message.tool_calls, (
        "Pinned DeepSeek V4 Pro host did not hard-enforce "
        f"tool_choice='required' under an adversarial no-tools prompt; "
        f"content={message.content!r}"
    )
    assert message.tool_calls[0].function.name == "get_weather"


def test_deepseek_v4_pro_enforces_response_format_adversarially() -> None:
    """Pinned hard hosts must return schema JSON even when the prompt demands prose.

    Same hard-provider pin as above, with postprocessing disabled so UniLLM
    cannot retry/heal a soft schema miss into a passing response.
    """
    client = _deepseek_client_without_postprocessing()
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
        max_completion_tokens=256,
    )

    content = response.choices[0].message.content or ""
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = _ColorPair.model_validate(json.loads(raw))
    assert isinstance(parsed.a, int)
    assert parsed.b in {"red", "blue"}
