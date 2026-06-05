"""Tests for OpenAI endpoint registrations."""

import unillm
from unillm.endpoints import list_endpoints, list_providers
from unillm.endpoints.utils import get_model_alias, list_models


def test_gpt_5_5_is_available_through_openai_endpoint() -> None:
    """GPT 5.5 should be routable through the endpoint registry."""
    assert get_model_alias("gpt-5.5@openai") == "gpt-5.5"
    assert "gpt-5.5" in list_models("openai")


def test_async_unify_accepts_gpt_5_5_endpoint() -> None:
    """AsyncUnify should construct with the public GPT 5.5 endpoint."""
    client = unillm.AsyncUnify("gpt-5.5@openai", api_key="test-key")

    assert client.endpoint == "gpt-5.5@openai"


def test_endpoint_catalog_lists_full_model_provider_endpoints() -> None:
    endpoints = list_endpoints()

    assert "openai" in list_providers()
    assert "gpt-4.1-nano@openai" in endpoints
    assert "gpt-5.5@openai" in endpoints


def test_endpoint_catalog_can_filter_by_provider() -> None:
    openai_endpoints = list_endpoints("openai")

    assert "gpt-5.5@openai" in openai_endpoints
    assert all(endpoint.endswith("@openai") for endpoint in openai_endpoints)
