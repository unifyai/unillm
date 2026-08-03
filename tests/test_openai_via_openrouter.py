"""OpenAI catalog models are reachable only through OpenRouter.

The company's direct OpenAI account is inactive, so ``<model>@openai`` is not a
routable endpoint. OpenAI models carry the ``openai/<id>@openrouter`` form.
"""

import pytest

import unillm
from unillm.endpoints import list_endpoints, list_providers
from unillm.endpoints.utils import (
    get_model_alias,
    get_transport_model_alias,
    list_models,
)

_GPT_56_FAMILY = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")


def test_gpt_5_6_family_resolves_through_openrouter() -> None:
    openrouter_models = list_models("openrouter")
    for model in _GPT_56_FAMILY:
        endpoint = f"openai/{model}@openrouter"
        transport = f"openrouter/openai/{model}"
        assert get_model_alias(endpoint) == transport
        assert get_transport_model_alias(endpoint) == transport
        assert f"openai/{model}" in openrouter_models


def test_native_openai_provider_is_not_registered() -> None:
    assert "openai" not in list_providers()
    assert list_models("openai") == []
    assert list_endpoints("openai") == []


@pytest.mark.parametrize("model", _GPT_56_FAMILY)
def test_native_openai_endpoint_is_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="not found"):
        get_model_alias(f"{model}@openai")


def test_async_unify_accepts_openrouter_gpt_endpoint() -> None:
    endpoint = "openai/gpt-5.6-sol@openrouter"
    client = unillm.AsyncUnify(endpoint, api_key="test-key")

    assert client.endpoint == endpoint
    assert client._provider == "openrouter"


def test_endpoint_catalog_lists_openrouter_gpt_endpoints() -> None:
    endpoints = list_endpoints()

    assert "openrouter" in list_providers()
    for model in _GPT_56_FAMILY:
        assert f"openai/{model}@openrouter" in endpoints


def test_endpoint_catalog_can_filter_by_provider() -> None:
    openrouter_endpoints = list_endpoints("openrouter")

    for model in _GPT_56_FAMILY:
        assert f"openai/{model}@openrouter" in openrouter_endpoints
    assert all(endpoint.endswith("@openrouter") for endpoint in openrouter_endpoints)
