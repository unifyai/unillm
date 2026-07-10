import json
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

import unillm
from unillm.clients.uni_llm import (
    _ZAI_GLM_HARD_OPENROUTER_PROVIDERS,
    _prepare_provider_request_kw,
)
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)
from tests.test_clients.vision_probe_helpers import assert_native_image_input_rejected

from ..settings import SETTINGS

GLM_52_ENDPOINT = "glm-5.2@zai"
GLM_52_PROVIDER_MODEL = "z-ai/glm-5.2"

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


def _glm_client_without_postprocessing() -> unillm.Unify:
    """UniLLM client that keeps hard-provider routing but skips healing/retries."""
    return unillm.Unify(
        GLM_52_ENDPOINT,
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


def test_glm_52_alias_registered() -> None:
    assert get_model_alias(GLM_52_ENDPOINT) == GLM_52_PROVIDER_MODEL
    assert "glm-5.2" in list_models("zai")


def test_glm_52_model_info_registered() -> None:
    info = get_model_info(GLM_52_ENDPOINT)
    assert info["max_input_tokens"] == 1_048_576
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_glm_52_openrouter_transport_pins_hard_providers() -> None:
    transport_model = get_transport_model_alias(GLM_52_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="zai", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "api_base" not in kw
    assert kw["extra_body"]["provider"] == {
        "order": list(_ZAI_GLM_HARD_OPENROUTER_PROVIDERS),
        "allow_fallbacks": False,
    }


def test_sync_glm_52_simple_message() -> None:
    client = unillm.Unify(GLM_52_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
async def test_async_glm_52_simple_message() -> None:
    client = unillm.AsyncUnify(GLM_52_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


def test_glm_52_rejects_native_image_input_on_openrouter() -> None:
    assert_native_image_input_rejected(GLM_52_ENDPOINT)


def test_glm_52_enforces_tool_choice_required_adversarially() -> None:
    """Pinned hard hosts must emit a tool call even when the prompt forbids tools.

    GLM-5.2 is pinned to OpenRouter hosts that hard-enforce tool_choice.
    Postprocessing is disabled so a pass cannot come from UniLLM retries.
    """
    client = _glm_client_without_postprocessing()
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
        "Pinned GLM-5.2 host did not hard-enforce tool_choice='required' "
        f"under an adversarial no-tools prompt; content={message.content!r}"
    )
    assert message.tool_calls[0].function.name == "get_weather"


def test_glm_52_enforces_response_format_adversarially() -> None:
    """Pinned hard hosts must return schema JSON even when the prompt demands prose.

    Same hard-provider pin as above, with postprocessing disabled so UniLLM
    cannot retry/heal a soft schema miss into a passing response.
    """
    client = _glm_client_without_postprocessing()
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
        max_completion_tokens=1024,
    )

    content = response.choices[0].message.content or ""
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = _ColorPair.model_validate(json.loads(raw))
    assert isinstance(parsed.a, int)
    assert parsed.b in {"red", "blue"}
