"""Tests for how an endpoint string resolves to a provider."""

import pytest

import unillm
from unillm.endpoints.utils import get_model_alias


class TestEveryUsableEndpointNamesItsProvider:
    """An endpoint reaches a provider only by naming it.

    Aliases are registered as ``model@provider``, so a bare model name
    resolves to nothing and is rejected before any credential is used. Which
    provider a call reaches, and whose model it runs, can therefore always be
    read off the endpoint string. A default-provider fallback would quietly
    break that.
    """

    def test_a_bare_model_name_resolves_to_nothing(self):
        with pytest.raises(ValueError, match="not found"):
            get_model_alias("claude-fable-5")

    def test_a_bare_model_name_cannot_construct_a_client(self):
        """Rejected at construction, before any request is sent."""
        with pytest.raises(ValueError, match="not found"):
            unillm.AsyncUnify("claude-fable-5", api_key="unused")

    def test_both_routes_to_a_vendor_keep_the_vendor_visible(self):
        """Neither form can reach Anthropic without saying so in the string."""
        assert get_model_alias("claude-fable-5@anthropic") == "anthropic/claude-fable-5"
        assert (
            get_model_alias("anthropic/claude-opus-4.8@openrouter")
            == "openrouter/anthropic/claude-opus-4.8"
        )
