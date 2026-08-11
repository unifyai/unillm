"""OpenRouter generations must be costed from the provider's own reporting."""

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
