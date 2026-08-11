"""OpenRouter generations must be costed from the provider's own reporting."""

from pydantic import SecretStr

from unillm.clients.uni_llm import _request_openrouter_usage_accounting
from unillm.costs import compute_cost_from_response


def test_openrouter_requests_opt_into_usage_accounting():
    kw: dict = {}
    _request_openrouter_usage_accounting(kw, "openrouter/anthropic/some-model")
    assert kw["extra_body"]["usage"] == {"include": True}


def test_non_openrouter_requests_are_left_alone():
    kw: dict = {}
    _request_openrouter_usage_accounting(kw, "anthropic/some-model")
    assert kw == {}


def test_existing_extra_body_is_preserved():
    kw = {"extra_body": {"provider": {"only": ["Anthropic"]}}}
    _request_openrouter_usage_accounting(kw, "openrouter/anthropic/some-model")
    assert kw["extra_body"]["provider"] == {"only": ["Anthropic"]}
    assert kw["extra_body"]["usage"] == {"include": True}


def test_caller_supplied_usage_options_win():
    kw = {"extra_body": {"usage": {"include": False}}}
    _request_openrouter_usage_accounting(kw, "openrouter/anthropic/some-model")
    assert kw["extra_body"]["usage"] == {"include": False}


def test_reported_cost_is_used_for_a_model_with_no_local_pricing():
    """The provider's figure must cost a model local pricing data lacks.

    A model nobody can price locally is still charged by the provider, so
    falling back to "no cost" would write off real spend.
    """
    response = {
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 1,
            "cost": 0.0123,
        },
    }
    cost = compute_cost_from_response(
        "openrouter/vendor/model-that-does-not-exist",
        response,
    )
    assert cost == 0.0123


def test_unpriceable_generation_reports_no_cost_rather_than_guessing():
    """Without a provider figure an unknown model yields no cost, and says so."""
    response = {"usage": {"prompt_tokens": 16, "completion_tokens": 1}}
    cost = compute_cost_from_response(
        "openrouter/vendor/model-that-does-not-exist",
        response,
    )
    assert cost is None


class TestCredentialTravelsWithTheRequest:
    """The provider credential is sent with the call, not read from the env.

    A host that runs untrusted code in the same process can only withhold a
    credential from that code if inference does not depend on it still being
    in ``os.environ``.
    """

    def test_openrouter_credential_is_attached(self, monkeypatch):
        from unillm.clients import uni_llm
        from unillm.settings import SETTINGS

        monkeypatch.setattr(
            SETTINGS,
            "OPENROUTER_API_KEY",
            SecretStr("sk-or-test"),  # pragma: allowlist secret
        )
        kw: dict = {}
        uni_llm._pass_provider_credential_explicitly(
            kw,
            "openrouter/anthropic/some-model",
            "openrouter",
        )
        assert kw["api_key"] == "sk-or-test"

    def test_anthropic_credential_is_attached(self, monkeypatch):
        from unillm.clients import uni_llm
        from unillm.settings import SETTINGS

        monkeypatch.setattr(
            SETTINGS,
            "ANTHROPIC_API_KEY",
            SecretStr("sk-ant-test"),  # pragma: allowlist secret
        )
        kw: dict = {}
        uni_llm._pass_provider_credential_explicitly(kw, "claude-x", "anthropic")
        assert kw["api_key"] == "sk-ant-test"

    def test_an_existing_key_is_never_overridden(self, monkeypatch):
        """The gateway sets its own credential and must keep it."""
        from unillm.clients import uni_llm
        from unillm.settings import SETTINGS

        monkeypatch.setattr(
            SETTINGS,
            "OPENROUTER_API_KEY",
            SecretStr("sk-or-test"),  # pragma: allowlist secret
        )
        kw = {"api_key": "gateway-key"}
        uni_llm._pass_provider_credential_explicitly(
            kw,
            "openrouter/anthropic/some-model",
            "openrouter",
        )
        assert kw["api_key"] == "gateway-key"

    def test_unrecognised_provider_keeps_litellm_resolution(self):
        from unillm.clients import uni_llm

        kw: dict = {}
        uni_llm._pass_provider_credential_explicitly(kw, "some/model", "mystery")
        assert "api_key" not in kw

    def test_absent_credential_is_not_invented(self, monkeypatch):
        from unillm.clients import uni_llm
        from unillm.settings import SETTINGS

        monkeypatch.setattr(SETTINGS, "OPENROUTER_API_KEY", SecretStr(""))
        kw: dict = {}
        uni_llm._pass_provider_credential_explicitly(
            kw,
            "openrouter/x/y",
            "openrouter",
        )
        assert "api_key" not in kw
