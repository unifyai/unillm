"""Tests for LLM-gateway routing (Orchestra broker) in the OpenRouter path.

When ``UNILLM_LLM_GATEWAY_URL`` (+ an auth key) is set, OpenRouter traffic is
redirected to the gateway via ``api_base``/``api_key`` and billing is skipped
client-side (the gateway settles it). Everything is default-off and must not
touch non-OpenRouter providers.
"""

import unisdk

from unillm.clients.uni_llm import (
    _llm_gateway_active,
    _prepare_provider_request_kw,
    _safe_deduct_credits,
)

_GATEWAY = "https://api.staging.internal.saas.unify.ai/v0/llm"


def _enable_gateway(monkeypatch, url=_GATEWAY + "/", key="unify-key"):
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", url)
    monkeypatch.setenv("UNIFY_KEY", key)


class TestGatewayRouting:
    def test_inactive_by_default(self, monkeypatch):
        monkeypatch.delenv("UNILLM_LLM_GATEWAY_URL", raising=False)
        assert _llm_gateway_active() is False
        kw = {"model": "openrouter/openai/gpt-5.6-sol"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        assert kw.get("api_base") is None
        assert kw.get("api_key") is None

    def test_active_requires_both_url_and_key(self, monkeypatch):
        monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", _GATEWAY)
        monkeypatch.delenv("UNIFY_KEY", raising=False)
        monkeypatch.delenv("UNILLM_LLM_GATEWAY_KEY", raising=False)
        assert _llm_gateway_active() is False

    def test_openrouter_redirected_when_active(self, monkeypatch):
        _enable_gateway(monkeypatch)
        kw = {"model": "openrouter/openai/gpt-5.6-sol"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        # Trailing slash trimmed; key taken from UNIFY_KEY.
        assert kw["api_base"] == _GATEWAY
        assert kw["api_key"] == "unify-key"

    def test_non_openrouter_not_redirected_when_active(self, monkeypatch):
        _enable_gateway(monkeypatch)
        kw = {"model": "claude-4.5-sonnet@anthropic"}
        _prepare_provider_request_kw(kw=kw, provider="anthropic", stream=False)
        assert kw.get("api_base") is None
        assert kw.get("api_key") is None

    def test_existing_api_base_not_overridden(self, monkeypatch):
        _enable_gateway(monkeypatch)
        kw = {"model": "openrouter/openai/gpt-5.6-sol", "api_base": "https://x"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        assert kw["api_base"] == "https://x"

    def test_dedicated_gateway_key_env(self, monkeypatch):
        monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", _GATEWAY)
        monkeypatch.delenv("UNIFY_KEY", raising=False)
        monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", "dedicated")
        kw = {"model": "openrouter/openai/gpt-5.6-sol"}
        _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
        assert kw["api_key"] == "dedicated"


class TestDeductSkip:
    def test_skips_openrouter_deduct_when_active(self, monkeypatch):
        _enable_gateway(monkeypatch)
        calls = []
        monkeypatch.setattr(
            unisdk,
            "deduct_credits",
            lambda *a, **k: calls.append(k),
            raising=False,
        )
        _safe_deduct_credits(1.23, model="openrouter/openai/gpt-5.6-sol")
        assert calls == []

    def test_deducts_openrouter_when_inactive(self, monkeypatch):
        monkeypatch.delenv("UNILLM_LLM_GATEWAY_URL", raising=False)
        calls = []
        monkeypatch.setattr(
            unisdk,
            "deduct_credits",
            lambda *a, **k: calls.append(k),
            raising=False,
        )
        _safe_deduct_credits(1.23, model="openrouter/openai/gpt-5.6-sol")
        assert len(calls) == 1

    def test_deducts_non_openrouter_even_when_active(self, monkeypatch):
        _enable_gateway(monkeypatch)
        calls = []
        monkeypatch.setattr(
            unisdk,
            "deduct_credits",
            lambda *a, **k: calls.append(k),
            raising=False,
        )
        _safe_deduct_credits(1.23, model="claude-4.5-sonnet@anthropic")
        assert len(calls) == 1
