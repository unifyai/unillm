"""Tests for LLM-gateway routing (Orchestra broker) in the OpenRouter path.

When ``UNILLM_LLM_GATEWAY_URL`` (+ an auth key) is set, OpenRouter traffic is
redirected to the gateway via ``api_base``/``api_key`` and billing is skipped
client-side (the gateway settles it). Everything is default-off and must not
touch non-OpenRouter providers.
"""

import unisdk

from unillm.clients.uni_llm import (
    _gateway_brokers_model,
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

    def test_an_unbrokered_provider_is_not_redirected_when_active(self, monkeypatch):
        """The gateway carries OpenRouter and Anthropic; others go direct.

        This previously asserted Anthropic was left alone, which was true
        while the gateway had no Anthropic leg. It has one now, so the case
        is made with a provider the gateway genuinely does not carry --
        otherwise nothing pins that enabling the gateway stops being
        opt-in per provider.
        """
        _enable_gateway(monkeypatch)
        kw = {"model": "deepseek-chat@deepseek"}
        _prepare_provider_request_kw(kw=kw, provider="deepseek", stream=False)
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

    def test_deducts_an_unbrokered_provider_even_when_active(self, monkeypatch):
        """A provider the gateway does not carry is still charged here.

        Was asserted with Anthropic, which the gateway now brokers and
        settles server-side; charging it client-side too would double-bill.
        Restated against a provider that still goes direct, so the case it
        was written for is still covered.
        """
        _enable_gateway(monkeypatch)
        calls = []
        monkeypatch.setattr(
            unisdk,
            "deduct_credits",
            lambda *a, **k: calls.append(k),
            raising=False,
        )
        _safe_deduct_credits(1.23, model="deepseek-chat@deepseek")
        assert len(calls) == 1

    def test_a_brokered_anthropic_call_is_not_charged_twice(self, monkeypatch):
        """The gateway settled it; charging here would bill the same call twice."""
        _enable_gateway(monkeypatch)
        calls = []
        monkeypatch.setattr(
            unisdk,
            "deduct_credits",
            lambda *a, **k: calls.append(k),
            raising=False,
        )
        _safe_deduct_credits(1.23, model="claude-opus-5@anthropic")
        assert calls == []


class TestAnthropicRouting:
    """Anthropic is brokered over its own protocol, not the OpenAI-shaped one."""

    def test_anthropic_addresses_the_messages_route_directly(self, monkeypatch):
        """LiteLLM uses api_base verbatim here, so the full path is set."""
        _enable_gateway(monkeypatch)
        kw = {"model": "claude-opus-5"}
        _prepare_provider_request_kw(kw=kw, provider="anthropic", stream=False)
        assert kw["api_base"] == _GATEWAY + "/anthropic/v1/messages"

    def test_the_key_is_sent_on_both_headers(self, monkeypatch):
        """LiteLLM sends x-api-key; Orchestra reads Authorization: Bearer.

        Sending only what the provider expects would reach the gateway
        unauthenticated, which reads as a broken credential rather than a
        missing header.
        """
        _enable_gateway(monkeypatch)
        kw = {"model": "claude-opus-5"}
        _prepare_provider_request_kw(kw=kw, provider="anthropic", stream=False)
        assert kw["api_key"] == "unify-key"
        assert kw["extra_headers"]["Authorization"] == "Bearer unify-key"

    def test_anthropic_untouched_when_the_gateway_is_off(self, monkeypatch):
        monkeypatch.delenv("UNILLM_LLM_GATEWAY_URL", raising=False)
        kw = {"model": "claude-opus-5"}
        _prepare_provider_request_kw(kw=kw, provider="anthropic", stream=False)
        assert kw.get("api_base") is None
        assert kw.get("extra_headers") is None

    def test_an_explicit_api_base_still_wins(self, monkeypatch):
        _enable_gateway(monkeypatch)
        kw = {"model": "claude-opus-5", "api_base": "https://x"}
        _prepare_provider_request_kw(kw=kw, provider="anthropic", stream=False)
        assert kw["api_base"] == "https://x"


class TestBrokeredSetMatchesTheRedirects:
    """The billing skip and the redirects must cover the same providers.

    A provider routed to the gateway but missing from the skip is charged
    twice -- once server-side, once here -- and neither charge looks wrong
    on its own, so the drift is invisible until the numbers are compared.
    """

    def test_both_spellings_of_each_brokered_provider_are_covered(self):
        for model in (
            "openrouter/openai/gpt-5.6-sol",
            "openai/gpt-5.6-sol@openrouter",
            "anthropic/claude-opus-4.8",
            "claude-opus-5@anthropic",
        ):
            assert _gateway_brokers_model(model) is True

    def test_providers_the_gateway_does_not_carry_still_deduct(self):
        for model in ("gpt-4", "deepseek/deepseek-chat", "kimi-k3@moonshotai"):
            assert _gateway_brokers_model(model) is False
