"""With an affinity key, Claude behind OpenRouter gets Anthropic's breakpoints.

Claude caches only up to an explicit ``cache_control`` breakpoint, so without
one a Claude call through OpenRouter reads nothing from cache however stable
its prefix. The breakpoints ride on the ``cache_affinity`` key, because
callers pass ``prompt_caching`` by default and a client without a key must
send what it always did. Requests are captured at the HTTP boundary by a
fake transport: no network call is made and no real key is used.
"""

import json

import pytest

from .fake_transport import CLAUDE_ON_OPENROUTER, recorded, send

KEY = "run-7f3a"
EPHEMERAL = {"type": "ephemeral"}
SYSTEM = "You answer weather questions briefly."
USER = "What is the weather in Paris?"


def _marked(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": EPHEMERAL}]


def _without_messages_and_tools(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in ("messages", "tools")}


@pytest.mark.parametrize("endpoint", CLAUDE_ON_OPENROUTER)
class TestBreakpointsReachTheWire:
    def test_every_requested_breakpoint_is_sent(self, endpoint):
        (sent,) = send(
            endpoint,
            prompt_caching=["system", "tools", "messages"],
            cache_affinity=KEY,
        )
        (before,) = recorded(endpoint, "all-markers")
        body = sent["body"]
        assert body.pop("session_id") == KEY

        assert body["messages"] == [
            {"role": "system", "content": _marked(SYSTEM)},
            {"role": "user", "content": _marked(USER)},
        ]
        assert body["tools"] == [
            {**before["body"]["tools"][0], "cache_control": EPHEMERAL},
        ]
        assert _without_messages_and_tools(body) == _without_messages_and_tools(
            before["body"],
        )
        assert sent["url"] == before["url"]

    def test_only_the_requested_breakpoints_are_sent(self, endpoint):
        (sent,) = send(endpoint, prompt_caching=["system"], cache_affinity=KEY)
        (before,) = recorded(endpoint)
        body = sent["body"]

        assert body["messages"] == [
            {"role": "system", "content": _marked(SYSTEM)},
            {"role": "user", "content": USER},
        ]
        assert body["tools"] == before["body"]["tools"]

    def test_no_breakpoints_without_prompt_caching(self, endpoint):
        (sent,) = send(endpoint, cache_affinity=KEY)
        assert sent["body"].pop("session_id") == KEY
        assert [sent] == recorded(endpoint)

    def test_no_breakpoints_without_a_key(self, endpoint):
        sent = send(endpoint, prompt_caching=["system", "tools", "messages"])
        assert sent == recorded(endpoint, "all-markers")


def test_other_openrouter_families_ignore_a_marker_request():
    """Gemini caches implicitly; a breakpoint there is not Claude's to add."""
    endpoint = "gemini-2.5-pro@vertex-ai"
    (sent,) = send(
        endpoint,
        prompt_caching=["system", "tools", "messages"],
        cache_affinity=KEY,
    )
    assert sent["body"].pop("session_id") == KEY
    assert [sent] == recorded(endpoint, "all-markers") == recorded(endpoint)


def test_direct_anthropic_breakpoints_do_not_need_a_key():
    """The direct API keeps its upstream behaviour exactly."""
    endpoint = "claude-opus-5@anthropic"
    assert send(endpoint, prompt_caching=["system", "tools", "messages"]) == (
        recorded(endpoint, "all-markers")
    )
    assert "cache_control" in json.dumps(recorded(endpoint, "all-markers"))
