"""Tests for dynamic *@openrouter endpoints and usage.cost billing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import unillm
from unillm.costs import (
    compute_cost_from_response,
    extract_openrouter_usage_cost,
)
from unillm.endpoints import list_endpoints, list_providers
from unillm.endpoints.utils import get_model_alias, get_transport_model_alias


def test_dynamic_openrouter_endpoint_resolves_transport() -> None:
    endpoint = "meta-llama/llama-3.3-70b-instruct@openrouter"
    assert get_transport_model_alias(endpoint) == (
        "openrouter/meta-llama/llama-3.3-70b-instruct"
    )
    assert get_model_alias(endpoint) == "openrouter/meta-llama/llama-3.3-70b-instruct"


def test_async_unify_accepts_dynamic_openrouter_endpoint() -> None:
    client = unillm.AsyncUnify(
        "google/gemini-2.5-flash@openrouter",
        api_key="test-key",
    )
    assert client.endpoint == "google/gemini-2.5-flash@openrouter"
    assert client._provider == "openrouter"
    assert client._model == "google/gemini-2.5-flash"


def test_openrouter_provider_is_listed() -> None:
    assert "openrouter" in list_providers()
    assert "openai/gpt-5.6-sol@openrouter" in list_endpoints("openrouter")


def test_extract_openrouter_usage_cost_from_object_and_dict() -> None:
    assert extract_openrouter_usage_cost(SimpleNamespace(cost=0.0123)) == 0.0123
    assert extract_openrouter_usage_cost({"cost": "0.004"}) == 0.004
    assert extract_openrouter_usage_cost({"cost": None}) is None
    assert extract_openrouter_usage_cost(SimpleNamespace(prompt_tokens=10)) is None


def test_compute_cost_from_response_prefers_openrouter_usage_cost() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            cost=0.042,
        ),
    )
    assert compute_cost_from_response("openai/gpt-5.5@openrouter", response) == 0.042
    assert compute_cost_from_response("openrouter/openai/gpt-5.5", response) == 0.042


def test_compute_cost_from_response_ignores_usage_cost_off_openrouter() -> None:
    """Only OpenRouter reports an authoritative usage.cost; others price tokens."""
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            cost=999.0,
        ),
    )
    cost = compute_cost_from_response("claude-opus-5@anthropic", response)
    assert cost is not None
    assert cost != 999.0
    assert cost > 0


def test_sync_openrouter_generate_bills_usage_cost() -> None:
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    response = ModelResponse(
        id="chatcmpl_or_cost",
        choices=[
            Choices(
                message=Message(content="ok"),
                finish_reason="stop",
                index=0,
            ),
        ],
    )
    response.model = "openai/gpt-4o-mini"
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    usage.cost = 0.0015
    response.usage = usage

    with (
        patch(
            "unillm.clients.uni_llm.litellm.completion",
            return_value=response,
        ),
        patch("unillm.clients.uni_llm._safe_deduct_credits") as deduct,
    ):
        client = unillm.Unify(
            "openai/gpt-4o-mini@openrouter",
            api_key="test-key",
            cache=False,
        )
        with unillm.capture_costs() as events:
            client.generate(
                messages=[{"role": "user", "content": "hi"}],
                return_full_completion=True,
            )

    assert len(events) == 1
    assert events[0].provider_cost == 0.0015
    assert events[0].billed_cost == 0.0015
    deduct.assert_called_once()
    assert deduct.call_args.args[0] == 0.0015
