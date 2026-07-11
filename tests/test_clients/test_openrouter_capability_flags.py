"""OpenRouter model-info overrides must advertise tool/reasoning capabilities.

LiteLLM runs with ``drop_params=True``. OpenRouter entries that only register
pricing (and omit capability flags) silently strip ``reasoning_effort`` and can
lose tool controls that the native OpenAI backend keeps.
"""

from __future__ import annotations

import litellm
from litellm.utils import get_optional_params

from unillm.endpoints.utils import get_transport_model_alias


def test_openrouter_gpt_55_keeps_reasoning_effort_and_tool_flags() -> None:
    transport = get_transport_model_alias("gpt-5.5@openai")
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
