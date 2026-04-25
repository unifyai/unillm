"""Tests for OpenAI endpoint registrations."""

import unillm
from unillm.endpoints.utils import get_model_alias, list_models


def test_gpt_5_5_is_available_through_openai_endpoint() -> None:
    """GPT 5.5 should be routable through the endpoint registry."""
    assert get_model_alias("gpt-5.5@openai") == "gpt-5.5"
    assert "gpt-5.5" in list_models("openai")


def test_async_unify_accepts_gpt_5_5_endpoint() -> None:
    """AsyncUnify should construct with the public GPT 5.5 endpoint."""
    client = unillm.AsyncUnify("gpt-5.5@openai", api_key="test-key")

    assert client.endpoint == "gpt-5.5@openai"
