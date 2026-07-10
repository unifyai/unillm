import json
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

import unillm
from unillm.clients.uni_llm import (
    _MISTRAL_LARGE_HARD_OPENROUTER_PROVIDERS,
    _MISTRAL_MEDIUM_31_HARD_OPENROUTER_PROVIDERS,
    _MISTRAL_MEDIUM_3_HARD_OPENROUTER_PROVIDERS,
    _prepare_provider_request_kw,
)
from unillm.endpoints.utils import get_transport_model_alias, list_models

from ..settings import SETTINGS

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


_PINNED = (
    (
        "mistral-large@mistral",
        "mistralai/mistral-large",
        "mistral",
        _MISTRAL_LARGE_HARD_OPENROUTER_PROVIDERS,
        "mistral-large",
    ),
    (
        "mistral-medium@mistral",
        "mistralai/mistral-medium-3.1",
        "mistral",
        _MISTRAL_MEDIUM_31_HARD_OPENROUTER_PROVIDERS,
        "mistral-medium",
    ),
    (
        "mistral-medium@vertex-ai",
        "mistralai/mistral-medium-3",
        "vertex-ai",
        _MISTRAL_MEDIUM_3_HARD_OPENROUTER_PROVIDERS,
        "mistral-medium",
    ),
)


def _client_without_postprocessing(endpoint: str) -> unillm.Unify:
    return unillm.Unify(
        endpoint,
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        temperature=0,
    )


def _generate_raw_first_response(client: unillm.Unify, **generate_kw):
    with patch.object(
        unillm.Unify,
        "_run_postprocessing",
        lambda self, chat_completion, *args, **kwargs: chat_completion,
    ):
        return client.generate(return_full_completion=True, **generate_kw)


@pytest.mark.parametrize(
    "endpoint,catalog_id,provider,hard_order,model_name",
    _PINNED,
)
def test_mistral_alias_registered(
    endpoint,
    catalog_id,
    provider,
    hard_order,
    model_name,
) -> None:
    assert model_name in list_models(provider)


@pytest.mark.parametrize(
    "endpoint,catalog_id,provider,hard_order,model_name",
    _PINNED,
)
def test_mistral_openrouter_pins_hard_providers(
    endpoint,
    catalog_id,
    provider,
    hard_order,
    model_name,
) -> None:
    transport_model = get_transport_model_alias(endpoint)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider=provider, stream=False)

    assert transport_model == f"openrouter/{catalog_id}"
    assert kw["extra_body"]["provider"] == {
        "only": list(hard_order),
        "allow_fallbacks": False,
    }


@pytest.mark.parametrize(
    "endpoint,catalog_id,provider,hard_order,model_name",
    _PINNED,
)
def test_mistral_enforces_tool_choice_required_adversarially(
    endpoint,
    catalog_id,
    provider,
    hard_order,
    model_name,
) -> None:
    client = _client_without_postprocessing(endpoint)
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
        f"Pinned {catalog_id} host did not hard-enforce tool_choice='required'; "
        f"content={message.content!r}"
    )
    assert message.tool_calls[0].function.name == "get_weather"


@pytest.mark.parametrize(
    "endpoint,catalog_id,provider,hard_order,model_name",
    _PINNED,
)
def test_mistral_enforces_response_format_adversarially(
    endpoint,
    catalog_id,
    provider,
    hard_order,
    model_name,
) -> None:
    client = _client_without_postprocessing(endpoint)
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
