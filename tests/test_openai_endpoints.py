"""Tests for OpenAI endpoint registrations."""

import unillm
from unillm.endpoints import list_endpoints, list_providers
from unillm.endpoints.utils import get_model_alias, list_models

_GPT_56_FAMILY = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")


def test_gpt_5_6_family_is_available_through_openai_endpoint() -> None:
    """GPT 5.6 Sol/Terra/Luna should be routable through the endpoint registry."""
    openai_models = list_models("openai")
    for model in _GPT_56_FAMILY:
        assert get_model_alias(f"{model}@openai") == model
        assert model in openai_models


def test_async_unify_accepts_gpt_5_6_sol_endpoint() -> None:
    """AsyncUnify should construct with the public GPT 5.6 Sol endpoint."""
    client = unillm.AsyncUnify("gpt-5.6-sol@openai", api_key="test-key")

    assert client.endpoint == "gpt-5.6-sol@openai"


def test_endpoint_catalog_lists_full_model_provider_endpoints() -> None:
    endpoints = list_endpoints()

    assert "openai" in list_providers()
    assert "gpt-4.1-nano@openai" in endpoints
    for model in _GPT_56_FAMILY:
        assert f"{model}@openai" in endpoints


def test_endpoint_catalog_can_filter_by_provider() -> None:
    openai_endpoints = list_endpoints("openai")

    for model in _GPT_56_FAMILY:
        assert f"{model}@openai" in openai_endpoints
    assert all(endpoint.endswith("@openai") for endpoint in openai_endpoints)
