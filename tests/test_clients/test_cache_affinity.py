"""A client's ``cache_affinity`` key reaches each transport as its routing hint.

Requests are captured at the HTTP boundary by a fake transport, so the
assertions are about the bytes a provider would receive. No network call is
made and no real key is used.
"""

import json
from pathlib import Path
from unittest.mock import patch

import litellm
import pytest

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw

from .fake_transport import ENDPOINTS, PROMPT_CACHING_VARIANTS, captured_requests, send

KEY = "run-7f3a"
_RECORDED = json.loads(
    (Path(__file__).parent / "recorded" / "off_path_requests.json").read_text(),
)

# Behind OpenRouter with no fixed host list, so OpenRouter chooses the
# provider and can keep choosing the one holding the conversation's cache.
_UNPINNED_OPENROUTER = (
    "openai/gpt-5.6-sol@openrouter",
    "anthropic/claude-sonnet-4.6@openrouter",
    "claude-4.5-sonnet@bedrock",
    "grok-4@xai",
    "gemini-2.5-pro@vertex-ai",
)
# Pinned to an explicit OpenRouter host list for tool and schema enforcement.
_PINNED_OPENROUTER = (
    "gpt-oss-120b@togetherai",
    "deepseek-v4-max@deepseek",
    "kimi-k3@moonshotai",
    "glm-5.2@zai",
    "minimax-v3@minimax",
    "mimo-v2.5@xiaomi-mimo",
    "llama-3.3-70b-chat@groq",
    "mistral-large@mistral",
)


def _recorded(endpoint: str, variant: str = "no-markers") -> list[dict]:
    return _RECORDED[f"{endpoint}|{variant}"]


def test_the_endpoint_groups_cover_every_recorded_endpoint():
    assert sorted(
        (*_UNPINNED_OPENROUTER, *_PINNED_OPENROUTER, "claude-opus-5@anthropic"),
    ) == sorted(ENDPOINTS)


class TestWithoutAKeyRequestsAreUnchanged:
    """The recording was taken before the key existed, from the same turn."""

    @pytest.mark.parametrize("variant", PROMPT_CACHING_VARIANTS)
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_request_matches_the_recording(self, endpoint, variant):
        sent = send(endpoint, prompt_caching=PROMPT_CACHING_VARIANTS[variant])
        assert sent == _recorded(endpoint, variant)


class TestTheKeyIsClientState:
    def test_constructor_and_setter(self):
        client = unillm.Unify("openai/gpt-5.6-sol@openrouter", cache_affinity=KEY)
        assert client.cache_affinity == KEY
        assert client.set_cache_affinity("other") is client
        assert client.cache_affinity == "other"
        assert unillm.Unify("openai/gpt-5.6-sol@openrouter").cache_affinity is None

    def test_copy_and_json_keep_it(self):
        client = unillm.Unify("openai/gpt-5.6-sol@openrouter")
        client.set_cache_affinity(KEY)
        assert client.copy().cache_affinity == KEY
        assert client.json()["cache_affinity"] == KEY

    def test_a_key_set_after_construction_is_sent(self):
        client = unillm.Unify("openai/gpt-5.6-sol@openrouter", cache=False)
        client.set_cache_affinity(KEY)
        with captured_requests() as sent:
            client.generate(user_message="hi")
        assert sent[0]["body"]["session_id"] == KEY


class TestOpenRouterStickiness:
    @pytest.mark.parametrize(
        "endpoint",
        [e for e in _UNPINNED_OPENROUTER if not e.startswith("openai/")],
    )
    def test_unpinned_models_carry_session_id_and_nothing_else(self, endpoint):
        (sent,) = send(endpoint, cache_affinity=KEY)
        (recorded,) = _recorded(endpoint)
        assert sent["body"].pop("session_id") == KEY
        assert sent == recorded

    def test_openai_models_also_carry_openai_prompt_cache_key(self):
        endpoint = "openai/gpt-5.6-sol@openrouter"
        (sent,) = send(endpoint, cache_affinity=KEY)
        (recorded,) = _recorded(endpoint)
        assert sent["body"].pop("session_id") == KEY
        assert sent["body"].pop("prompt_cache_key") == KEY
        assert sent == recorded

    @pytest.mark.parametrize("endpoint", _PINNED_OPENROUTER)
    def test_pinned_models_keep_the_pin_and_send_no_session_id(self, endpoint):
        """The pin decides the host, and OpenRouter does not stick on top of it."""
        sent = send(endpoint, cache_affinity=KEY)
        assert sent == _recorded(endpoint)
        assert sent[0]["body"]["provider"]["only"]

    def test_direct_anthropic_has_no_routing_hint_to_send(self):
        assert send("claude-opus-5@anthropic", cache_affinity=KEY) == _recorded(
            "claude-opus-5@anthropic",
        )


class TestOpenRouterRequestPreparation:
    MODEL = "openrouter/google/gemini-2.5-pro"

    def _prepare(self, kw: dict, key: str | None = KEY) -> dict:
        _prepare_provider_request_kw(
            kw=kw,
            provider="openrouter",
            stream=False,
            cache_affinity=key,
        )
        return kw

    def test_no_key_adds_nothing(self):
        kw = self._prepare({"model": self.MODEL}, key=None)
        assert "session_id" not in kw["extra_body"]

    def test_keys_the_caller_chose_win(self):
        chosen = {"session_id": "caller", "prompt_cache_key": "caller"}
        kw = self._prepare(
            {"model": "openrouter/openai/gpt-5.6-sol", "extra_body": dict(chosen)},
        )
        assert {name: kw["extra_body"][name] for name in chosen} == chosen

    def test_only_openai_models_get_prompt_cache_key(self):
        assert (
            "prompt_cache_key" not in self._prepare({"model": self.MODEL})["extra_body"]
        )
        kw = self._prepare({"model": "openrouter/openai/gpt-5.6-sol"})
        assert kw["extra_body"]["prompt_cache_key"] == KEY

    @pytest.mark.parametrize("hosts", ["only", "order"])
    def test_a_caller_host_list_suppresses_the_key(self, hosts):
        provider = {hosts: ["google-vertex"]}
        kw = self._prepare({"model": self.MODEL, "extra_body": {"provider": provider}})
        assert "session_id" not in kw["extra_body"]
        assert kw["extra_body"]["provider"] == provider

    def test_provider_preferences_without_a_host_list_keep_the_key(self):
        kw = self._prepare(
            {"model": self.MODEL, "extra_body": {"provider": {"sort": "price"}}},
        )
        assert kw["extra_body"]["session_id"] == KEY

    def test_other_transports_are_left_alone(self):
        kw = {"model": "claude-opus-5"}
        _prepare_provider_request_kw(
            kw=kw,
            provider="anthropic",
            stream=False,
            cache_affinity=KEY,
        )
        assert "extra_body" not in kw


def test_the_response_cache_key_ignores_the_routing_hint():
    """A replayed response must still be found once a run sets a key."""
    seen = []

    def lookup(**kw):
        seen.append(kw["kw"])
        return None, None

    for key in (None, KEY):
        client = unillm.Unify(
            "openai/gpt-5.6-sol@openrouter",
            cache=True,
            cache_affinity=key,
        )
        with (
            patch("unillm.clients.uni_llm._get_cache", side_effect=lookup),
            patch("unillm.clients.uni_llm._write_to_cache"),
            captured_requests(),
        ):
            client.generate(user_message="hi")
    assert seen[0] == seen[1]
    assert "session_id" not in json.dumps(seen[1], default=str)


async def test_the_async_client_sends_the_key():
    calls = []

    async def acompletion(**kw):
        calls.append(kw)
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    client = unillm.AsyncUnify(
        "gemini-2.5-pro@vertex-ai",
        cache=False,
        cache_affinity=KEY,
    )
    with patch.object(litellm, "acompletion", acompletion), captured_requests():
        await client.generate(user_message="hi")
    assert calls[0]["extra_body"]["session_id"] == KEY
