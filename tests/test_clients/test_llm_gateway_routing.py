"""Tests for LLM-gateway routing in the OpenRouter path.

When ``UNILLM_LLM_GATEWAY_URL`` and ``UNILLM_LLM_GATEWAY_KEY`` are both set,
OpenRouter traffic is redirected to the gateway via ``api_base``/``api_key``.
Everything is default-off and must not touch other providers.
"""

import pytest

from unillm.clients.uni_llm import _llm_gateway, _prepare_provider_request_kw

_GATEWAY = "https://gateway.example/v0/llm"


def _enable_gateway(monkeypatch, url=_GATEWAY + "/", key="gateway-key"):
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", url)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", key)


class TestGatewayRouting:
    def test_inactive_by_default(self, monkeypatch):
        """With no gateway configured the call goes straight to the provider.

        The provider credential travels with every request, so the gateway is
        distinguished by ``api_base`` and by the key not being the gateway's.
        """
        monkeypatch.delenv("UNILLM_LLM_GATEWAY_URL", raising=False)
        monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", "gateway-key")
        assert _llm_gateway() is None
        kw = {"model": "openrouter/openai/gpt-5.6-sol"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        assert kw.get("api_base") is None
        assert kw.get("api_key") != "gateway-key"

    def test_active_requires_both_url_and_key(self, monkeypatch):
        monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", _GATEWAY)
        monkeypatch.delenv("UNILLM_LLM_GATEWAY_KEY", raising=False)
        assert _llm_gateway() is None

    def test_openrouter_redirected_when_active(self, monkeypatch):
        _enable_gateway(monkeypatch)
        kw = {"model": "openrouter/openai/gpt-5.6-sol"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        # Trailing slash trimmed.
        assert kw["api_base"] == _GATEWAY
        assert kw["api_key"] == "gateway-key"

    @pytest.mark.parametrize(
        ("model", "provider"),
        [("claude-opus-5", "anthropic"), ("deepseek-chat@deepseek", "deepseek")],
    )
    def test_other_providers_go_direct_when_active(
        self,
        monkeypatch,
        model,
        provider,
    ):
        """The gateway speaks OpenRouter's API, so only OpenRouter calls go to it."""
        _enable_gateway(monkeypatch)
        kw = {"model": model}
        _prepare_provider_request_kw(kw=kw, provider=provider, stream=False)
        assert kw.get("api_base") is None
        assert kw.get("api_key") != "gateway-key"

    def test_existing_api_base_not_overridden(self, monkeypatch):
        _enable_gateway(monkeypatch)
        kw = {"model": "openrouter/openai/gpt-5.6-sol", "api_base": "https://x"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        assert kw["api_base"] == "https://x"
