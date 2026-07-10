import json
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)

from ..settings import SETTINGS

MINIMAX_V3_ENDPOINT = "minimax-v3@minimax"
MINIMAX_V3_PROVIDER_MODEL = "minimax/MiniMax-M3"

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


def _minimax_client_without_postprocessing() -> unillm.Unify:
    """UniLLM client that keeps Together transport but skips healing/retries."""
    return unillm.Unify(
        MINIMAX_V3_ENDPOINT,
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        temperature=0,
    )


def _generate_raw_first_response(client: unillm.Unify, **generate_kw):
    """Return the first upstream completion with postprocessing disabled.

    Patches ``_run_postprocessing`` so UniLLM cannot heal prose into tool calls
    or retry after a soft failure — the assertion is on Together's first reply.
    """

    with patch.object(
        unillm.Unify,
        "_run_postprocessing",
        lambda self, chat_completion, *args, **kwargs: chat_completion,
    ):
        return client.generate(return_full_completion=True, **generate_kw)


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
    assert kw["extra_body"]["provider"] == {
        "only": ["together"],
        "allow_fallbacks": False,
    }


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


def test_minimax_together_enforces_tool_choice_required_adversarially() -> None:
    """Together must emit a tool call even when the prompt forbids tools.

    MiniMax-M3 is pinned to Together on OpenRouter. This hits that upstream
    through UniLLM transport with postprocessing disabled, so a pass cannot
    come from UniLLM's soft tool-choice retry/healing path.
    """
    client = _minimax_client_without_postprocessing()
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
        "Together did not hard-enforce tool_choice='required' under an "
        f"adversarial no-tools prompt; content={message.content!r}"
    )
    assert message.tool_calls[0].function.name == "get_weather"


def test_minimax_together_enforces_response_format_adversarially() -> None:
    """Together must return schema JSON even when the prompt demands prose.

    Same Together-pinned transport as above, with postprocessing disabled so
    UniLLM cannot retry/heal a soft schema miss into a passing response.
    """
    client = _minimax_client_without_postprocessing()
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
