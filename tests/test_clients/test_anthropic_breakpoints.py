"""Claude gets Anthropic's ``cache_control`` breakpoints by default.

Claude reads from its prompt cache only up to an explicit breakpoint, so a
Claude call without one reads nothing from cache however stable its prefix,
directly or through OpenRouter. Requests are captured at the HTTP boundary by
a fake transport: no network call is made and no real key is used.
"""

import copy
import json
from typing import Any

import pytest

import unillm

from .fake_transport import (
    CLAUDE_ON_OPENROUTER,
    WEATHER_TOOL,
    captured_requests,
    recorded,
    send,
)

KEY = "run-7f3a"
ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}
CLAUDE = (*CLAUDE_ON_OPENROUTER, "claude-opus-5@anthropic")
SYSTEM = "You answer weather questions briefly."


def _without_breakpoints(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _without_breakpoints(v) for k, v in node.items() if k != "cache_control"
        }
    if isinstance(node, list):
        return [_without_breakpoints(v) for v in node]
    return node


def _breakpoints(body: dict) -> list[str]:
    """Where *body* carries a breakpoint, each asserted to be the one-hour one."""
    found = []
    for i, tool in enumerate(body.get("tools") or []):
        if "cache_control" in tool:
            assert tool["cache_control"] == ONE_HOUR
            found.append(f"tools[{i}]")
    messages = body.get("messages") or []
    if isinstance(body.get("system"), list):
        messages = [{"role": "system", "content": body["system"]}, *messages]
    for i, msg in enumerate(messages):
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        for j, block in enumerate(blocks):
            if isinstance(block, dict) and "cache_control" in block:
                assert block["cache_control"] == ONE_HOUR
                found.append(f"{msg['role']}[{j}]")
        if "cache_control" in msg:
            assert msg["cache_control"] == ONE_HOUR
            found.append(msg["role"])
    return found


def _send_messages(endpoint: str, messages: list, **client_kw: Any) -> dict:
    client = unillm.Unify(endpoint, cache=False, **client_kw)
    with captured_requests() as sent:
        client.generate(
            messages=copy.deepcopy(messages),
            tools=[copy.deepcopy(WEATHER_TOOL)],
            tool_choice="auto",
        )
    (request,) = sent
    return request["body"]


@pytest.mark.parametrize("endpoint", CLAUDE)
class TestClaudeBreakpoints:
    def test_a_default_client_marks_tools_system_and_the_last_message(
        self,
        endpoint,
    ):
        (sent,) = send(endpoint)
        assert sorted(_breakpoints(sent["body"])) == [
            "system[0]",
            "tools[0]",
            "user[0]",
        ]
        (before,) = recorded(endpoint, "all-markers")
        prompt = ("messages", "system")
        assert sent["url"] == before["url"]
        assert _without_breakpoints(
            {k: v for k, v in sent["body"].items() if k not in prompt},
        ) == _without_breakpoints(
            {k: v for k, v in before["body"].items() if k not in prompt},
        )

    def test_only_the_requested_breakpoints_are_sent(self, endpoint):
        (sent,) = send(endpoint, prompt_caching=["system"])
        assert _breakpoints(sent["body"]) == ["system[0]"]

    @pytest.mark.parametrize("off", [None, []])
    def test_turning_caching_off_sends_what_was_recorded(self, endpoint, off):
        assert send(endpoint, prompt_caching=off) == recorded(endpoint)

    def test_an_affinity_key_leaves_the_breakpoints_alone(self, endpoint):
        (with_key,) = send(endpoint, cache_affinity=KEY)
        (without,) = send(endpoint)
        with_key["body"].pop("session_id", None)
        assert with_key == without

    def test_a_caller_marked_block_gets_a_breakpoint(self, endpoint):
        document = {"type": "text", "text": "The forecast.", "_cache_breakpoint": True}
        body = _send_messages(
            endpoint,
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": [document, {"type": "text", "text": "Paris?"}],
                },
            ],
        )
        assert sorted(_breakpoints(body)) == [
            "system[0]",
            "tools[0]",
            "user[0]",
            "user[1]",
        ]
        assert "_cache_breakpoint" not in json.dumps(body)

    def test_the_earliest_caller_marks_give_way_to_the_limit_of_four(
        self,
        endpoint,
    ):
        def turn(text: str) -> dict:
            return {
                "role": "user",
                "content": [{"type": "text", "text": text, "_cache_breakpoint": True}],
            }

        body = _send_messages(
            endpoint,
            [
                {"role": "system", "content": SYSTEM},
                turn("first"),
                {"role": "assistant", "content": "ok"},
                turn("second"),
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Paris?"},
            ],
        )
        marked = [
            msg["content"][0]["text"]
            for msg in body["messages"]
            if isinstance(msg.get("content"), list)
            and msg["role"] == "user"
            and "cache_control" in msg["content"][0]
        ]
        assert len(_breakpoints(body)) == 4
        assert "first" not in marked and "second" in marked


def test_other_families_get_no_breakpoints_and_no_marks():
    """Gemini caches a repeated prefix on its own."""
    endpoint = "gemini-2.5-pro@vertex-ai"
    assert send(endpoint) == recorded(endpoint)
    body = _send_messages(
        endpoint,
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Paris?", "_cache_breakpoint": True},
                ],
            },
        ],
    )
    assert "cache_control" not in json.dumps(body)
    assert "_cache_breakpoint" not in json.dumps(body)


def test_static_system_blocks_end_where_the_first_dynamic_one_begins():
    body = _send_messages(
        "claude-opus-5@anthropic",
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Rules."},
                    {"type": "text", "text": "Reference."},
                    {"type": "text", "text": "The time is 18:30.", "_static": False},
                ],
            },
            {"role": "user", "content": "Paris?"},
        ],
        prompt_caching=["system"],
    )
    assert _breakpoints(body) == ["system[1]"]


def test_a_body_posted_outside_a_client_gets_its_breakpoints_on_blocks():
    """OpenRouter honours a breakpoint only on a content block."""
    tools = [copy.deepcopy(WEATHER_TOOL)]
    body = {
        "model": "anthropic/claude-opus-5.5",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Paris?"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "user", "content": "And Rome?"},
        ],
        "tools": tools,
    }
    unillm.place_cache_breakpoints(body)
    assert sorted(_breakpoints(body)) == ["system[0]", "tools[0]", "user[0]"]
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": SYSTEM, "cache_control": ONE_HOUR},
    ]
    assert body["messages"][2]["content"] == ""
    assert "cache_control" not in json.dumps(tools)


@pytest.mark.parametrize("endpoint", CLAUDE)
def test_a_callers_own_breakpoints_count_toward_the_four_and_last_an_hour(endpoint):
    """Anthropic rejects a fifth breakpoint, and a one-hour one after a five-minute one."""
    five_minutes = {"type": "ephemeral"}

    def turn(text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "text", "text": text, "cache_control": five_minutes}],
        }

    client = unillm.Unify(endpoint, cache=False)
    tools = [
        {**copy.deepcopy(WEATHER_TOOL), "cache_control": five_minutes},
        {
            "type": "function",
            "function": {"name": "get_time", "parameters": {"type": "object"}},
        },
    ]
    with captured_requests() as sent:
        client.generate(
            messages=[
                {"role": "system", "content": SYSTEM},
                turn("first"),
                {"role": "assistant", "content": "ok"},
                turn("second"),
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Paris?"},
            ],
            tools=tools,
            tool_choice="auto",
        )
    (request,) = sent
    body = request["body"]
    assert len(_breakpoints(body)) == 4
    assert [("cache_control" in tool) for tool in body["tools"]] == [False, True]
    assert "first" not in json.dumps(
        [m for m in body["messages"] if "cache_control" in json.dumps(m)],
    )
    assert tools[0]["cache_control"] == five_minutes
