"""OpenRouter model-info overrides must advertise tool/reasoning capabilities.

LiteLLM runs with ``drop_params=True``. OpenRouter entries that only register
pricing (and omit capability flags) silently strip ``reasoning_effort`` and can
lose tool controls that the native OpenAI backend keeps.
"""

from __future__ import annotations

from unittest.mock import patch

import litellm
from litellm.utils import get_optional_params

from unillm.endpoints.utils import _OPENROUTER_REGISTERED, get_transport_model_alias


def test_openrouter_gpt_55_keeps_reasoning_effort_and_tool_flags() -> None:
    transport = get_transport_model_alias("openai/gpt-5.5@openrouter")
    assert transport == "openrouter/openai/gpt-5.5"

    info = litellm.get_model_info(transport)
    assert info.get("supports_reasoning") is True
    assert info.get("supports_function_calling") is True
    # LiteLLM's get_model_info may omit parallel; model_cost is authoritative.
    assert (
        litellm.model_cost[transport].get("supports_parallel_function_calling") is True
    )
    assert "reasoning_effort" in (info.get("supported_openai_params") or [])

    optional = get_optional_params(
        model=transport,
        custom_llm_provider="openrouter",
        reasoning_effort="high",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "short_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    )
    assert optional.get("reasoning_effort") == "high"
    assert optional.get("parallel_tool_calls") is True
    assert optional.get("tool_choice") == "auto"


def test_native_openai_gpt_55_transport_is_not_openrouter() -> None:
    transport = get_transport_model_alias("gpt-5.5@openai")
    assert transport == "gpt-5.5"
    assert not transport.startswith("openrouter/")


def test_catalog_supplies_capabilities_for_unregistered_openrouter_model() -> None:
    """A catalog model the pinned LiteLLM release lacks keeps effort control.

    Without this, every newly released OpenRouter model needs a hand-written
    entry in ``openrouter_overrides`` before ``reasoning_effort`` survives
    ``drop_params``.
    """

    model_id = "someorg/some-new-reasoning-model"
    transport = f"openrouter/{model_id}"
    _OPENROUTER_REGISTERED.discard(model_id)
    litellm.model_cost.pop(transport, None)

    catalog_entry = {
        "id": model_id,
        "context_length": 200_000,
        "input_cost_per_token": 1.00 / 1_000_000,
        "output_cost_per_token": 4.00 / 1_000_000,
        "supports_reasoning": True,
        "supports_tools": True,
        "supports_image_input": True,
    }
    with patch(
        "unillm.openrouter_catalog.get_openrouter_model_info",
        return_value=catalog_entry,
    ):
        assert get_transport_model_alias(f"{model_id}@openrouter") == transport

    info = litellm.get_model_info(transport)
    assert info.get("supports_reasoning") is True
    assert info.get("supports_function_calling") is True
    assert info.get("supports_vision") is True

    optional = get_optional_params(
        model=transport,
        custom_llm_provider="openrouter",
        reasoning_effort="high",
    )
    assert optional.get("reasoning_effort") == "high"

    _OPENROUTER_REGISTERED.discard(model_id)
    litellm.model_cost.pop(transport, None)
