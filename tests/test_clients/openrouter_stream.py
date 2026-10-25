"""A synthetic OpenRouter chat-completion stream, as server-sent events.

Written by hand after OpenRouter's documented streaming format, not captured
from a real call; a recorded real stream will replace it or join it.

``events`` is one reply that reasons, says a few words and calls a tool:

- ``reasoning`` text arrives in pieces, each with a ``reasoning_details``
  piece of the same item (``index`` 0);
- an encrypted reasoning item (``index`` 1) arrives whole: with the first
  piece of content, in a delta of its own (no content and no reasoning
  text), or in the finish chunk's delta;
- the tool call's arguments arrive in pieces;
- the last event has no choices and carries usage: the charged ``cost``,
  cache read and write tokens and reasoning tokens.

``SUMMARY_EVENTS`` is a reply whose reasoning is a summary item in pieces
and an encrypted item, and ``SUMMARY_REPLY`` the same reply unstreamed.

It holds no credential and no key-shaped string.
"""

import json
from typing import Any, Optional

MODEL = "openai/gpt-5.6-sol"
COST = 0.0123
REASONING = "The user wants the weather in Paris. Call the tool."
CONTENT = "Checking now."
ARGUMENTS = '{"city": "Paris"}'
FORMAT = "openai-responses-v1"
ENCRYPTED = {
    "type": "reasoning.encrypted",
    "data": "opaque-reasoning-state-" * 4,
    "id": "rs_1",
    "format": FORMAT,
    "index": 1,
}
USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 40,
    "total_tokens": 1240,
    "cost": COST,
    "is_byok": False,
    "prompt_tokens_details": {
        "cached_tokens": 1000,
        "cache_write_tokens": 150,
        "audio_tokens": 0,
    },
    "completion_tokens_details": {"reasoning_tokens": 25, "image_tokens": 0},
}


def _event(
    delta: Optional[dict] = None,
    finish_reason: Optional[str] = None,
    **top: Any,
) -> dict:
    choices = (
        []
        if delta is None
        else [
            {
                "index": 0,
                "delta": {"role": "assistant", **delta},
                "finish_reason": finish_reason,
                "native_finish_reason": finish_reason,
                "logprobs": None,
            },
        ]
    )
    return {
        "id": "gen-stream",
        "provider": "OpenAI",
        "model": MODEL,
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "choices": choices,
        **top,
    }


def _reasoning(text: str) -> dict:
    piece = {"type": "reasoning.text", "text": text, "format": FORMAT, "index": 0}
    return _event({"content": "", "reasoning": text, "reasoning_details": [piece]})


def _arguments(text: str, **call: Any) -> dict:
    return _event(
        {
            "content": "",
            "tool_calls": [{"index": 0, **call, "function": {"arguments": text}}],
        },
    )


def events(encrypted_in: str = "content") -> list[dict]:
    """The reply's events, with the encrypted item in the first ``content``
    piece's delta, ``alone`` in a delta of its own, or in the ``finish``
    chunk's delta."""

    def encrypted(where: str) -> dict:
        return {"reasoning_details": [ENCRYPTED]} if encrypted_in == where else {}

    alone = _event({"content": "", **encrypted("alone")})
    return [
        _reasoning("The user wants "),
        _reasoning("the weather in Paris. "),
        _reasoning("Call the tool."),
        *([alone] if encrypted_in == "alone" else []),
        _event({"content": "Checking ", **encrypted("content")}),
        _event({"content": "now."}),
        _event(
            {
                "content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    },
                ],
            },
        ),
        _arguments('{"city": '),
        _arguments('"Paris"}'),
        _event({"content": "", **encrypted("finish")}, finish_reason="tool_calls"),
        _event(usage=USAGE),
    ]


EVENTS = events()

SUMMARY = "Look up the weather, then answer."
SUMMARY_ITEM = {
    "type": "reasoning.summary",
    "summary": SUMMARY,
    "format": FORMAT,
    "index": 0,
}


def _summary(text: str) -> dict:
    piece = {**SUMMARY_ITEM, "summary": text}
    return _event({"content": "", "reasoning": text, "reasoning_details": [piece]})


SUMMARY_EVENTS = [
    _summary("Look up the weather, "),
    _summary("then answer."),
    _event({"content": "", "reasoning_details": [ENCRYPTED]}),
    _event({"content": "Sunny."}),
    _event({"content": ""}, finish_reason="stop"),
    _event(usage=USAGE),
]

SUMMARY_REPLY = {
    "id": "gen-whole",
    "provider": "OpenAI",
    "model": MODEL,
    "object": "chat.completion",
    "created": 1760000000,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Sunny.",
                "reasoning": SUMMARY,
                "reasoning_details": [SUMMARY_ITEM, ENCRYPTED],
            },
            "finish_reason": "stop",
            "native_finish_reason": "stop",
            "logprobs": None,
        },
    ],
    "usage": USAGE,
}


def error_event(code: int) -> dict:
    """An error OpenRouter reports part-way through a stream."""
    return _event(
        {"content": ""},
        finish_reason="error",
        error={"code": code, "message": "Provider disconnected unexpectedly"},
    )


def sse(events: list[dict], *, done: bool = True) -> bytes:
    """*events* as the bytes of an event stream."""
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()
