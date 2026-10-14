"""Single-completion requests sent as streams and reassembled.

With ``stream_transport`` a request for one completion goes out as a stream,
so bytes arrive while the model generates, and is read whole and rebuilt into
the completion the same reply gives unstreamed. The provider is replaced at
LiteLLM's HTTP handler, so the event stream (``openrouter_stream``) passes
through LiteLLM's own parsing, and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import logging
from json import dumps, loads
from types import SimpleNamespace
from typing import Callable, Iterator
from unittest.mock import patch

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from pydantic import SecretStr

import unillm
from unillm.clients import uni_llm
from unillm.cost_tracker import acapture_costs
from unillm.settings import SETTINGS

from .fake_transport import PLACEHOLDER_KEY, WEATHER_TOOL
from .openrouter_stream import (
    ARGUMENTS,
    CONTENT,
    COST,
    ENCRYPTED,
    EVENTS,
    FORMAT,
    REASONING,
    SUMMARY_EVENTS,
    SUMMARY_ITEM,
    SUMMARY_REPLY,
    USAGE,
    error_event,
    events,
    sse,
)

ENDPOINT = "openai/gpt-5.6-sol@openrouter"
Reply = Callable[[str], httpx.Response]

WHOLE = {
    "id": "gen-whole",
    "object": "chat.completion",
    "created": 0,
    "model": "openai/gpt-5.6-sol",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Sunny."},
            "finish_reason": "stop",
        },
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
}


def whole(body: dict) -> Reply:
    return lambda url: httpx.Response(
        200,
        json=body,
        request=httpx.Request("POST", url),
    )


def streamed(body: bytes) -> Reply:
    return lambda url: httpx.Response(
        200,
        content=body,
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", url),
    )


class _Dropped(httpx.AsyncByteStream):
    """An event stream whose connection drops after *body*."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body
        raise httpx.ReadError("connection dropped")


def dropped(body: bytes) -> Reply:
    return lambda url: httpx.Response(
        200,
        stream=_Dropped(body),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", url),
    )


class _KeptAlive(httpx.AsyncByteStream):
    """An event stream that sends only OpenRouter's keep-alive comments."""

    async def __aiter__(self):
        while True:
            yield b": OPENROUTER PROCESSING\n\n"
            await asyncio.sleep(0.05)


def kept_alive(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        stream=_KeptAlive(),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", url),
    )


@contextlib.contextmanager
def provider(*replies: Reply, retries: int = 1) -> Iterator[list[dict]]:
    """Answer each request with the next of *replies*; yield the bodies sent.

    *retries* transient retries, with no backoff, so a retried failure costs
    no time.
    """
    sent: list[dict] = []
    queue = list(replies)

    def answer(url, data, json):  # noqa: A002
        sent.append(loads(data) if data is not None else json)
        return queue.pop(0)(url)

    async def post(self, url, data=None, json=None, **_):  # noqa: A002
        reply = answer(url, data, json)
        return await reply if inspect.isawaitable(reply) else reply

    def sync_post(self, url, data=None, json=None, **_):  # noqa: A002
        return answer(url, data, json)

    with (
        patch.object(SETTINGS, "OPENROUTER_API_KEY", SecretStr(PLACEHOLDER_KEY)),
        patch.object(SETTINGS, "UNILLM_TRANSIENT_RETRY_COUNT", retries),
        patch("unillm.helpers._BACKOFF_BASE_SECONDS", 0),
        patch.object(AsyncHTTPHandler, "post", post),
        patch.object(HTTPHandler, "post", sync_post),
    ):
        yield sent


@pytest.fixture(autouse=True)
def _no_gateway(monkeypatch):
    monkeypatch.delenv("UNILLM_LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("UNILLM_LLM_GATEWAY_KEY", raising=False)
    monkeypatch.delenv(uni_llm.STREAM_IDLE_ENV, raising=False)


@pytest.fixture
def llm_events():
    captured: list = []
    unillm.set_llm_event_hook(captured.append)
    yield captured
    unillm.set_llm_event_hook(None)


async def ask(*, stream_transport: bool = True):
    client = unillm.AsyncUnify(
        ENDPOINT,
        cache=False,
        stream_transport=stream_transport,
    )
    return await client.generate(
        user_message="What is the weather in Paris?",
        tools=[copy.deepcopy(WEATHER_TOOL)],
    )


async def test_text_and_tool_calls_are_assembled():
    with provider(streamed(sse(EVENTS))) as sent:
        reply = await ask()

    assert sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    message = reply.choices[0].message
    assert message.content == CONTENT
    [call] = message.tool_calls
    assert (call.id, call.function.name, call.function.arguments) == (
        "call_1",
        "get_weather",
        ARGUMENTS,
    )
    assert reply.choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize("encrypted_in", ["content", "alone", "finish"])
async def test_reasoning_details_are_joined_in_order(encrypted_in):
    """Text pieces of one item are joined; the encrypted item is kept whole.

    LiteLLM's stream wrapper passes on no delta whose only payload is
    ``reasoning_details`` (``alone``) and empties the finish chunk's delta
    (``finish``), so the item must come from the provider's parsed chunks.
    """
    with provider(streamed(sse(events(encrypted_in)))):
        reply = await ask()

    message = reply.choices[0].message
    assert message.content == CONTENT
    assert message.reasoning_content == REASONING
    fields = message.provider_specific_fields
    assert fields["reasoning"] == REASONING
    assert fields["reasoning_details"] == [
        {"type": "reasoning.text", "text": REASONING, "format": FORMAT, "index": 0},
        ENCRYPTED,
    ]


async def test_reasoning_details_match_the_unstreamed_reply():
    """A summary in pieces and an encrypted item alone, streamed or not."""
    with provider(streamed(sse(SUMMARY_EVENTS))):
        streamed_reply = await ask()
    with provider(whole(SUMMARY_REPLY)):
        unstreamed_reply = await ask(stream_transport=False)

    streamed_details, unstreamed_details = (
        reply.choices[0].message.provider_specific_fields["reasoning_details"]
        for reply in (streamed_reply, unstreamed_reply)
    )
    assert unstreamed_details == [SUMMARY_ITEM, ENCRYPTED]
    assert streamed_details == unstreamed_details


def test_a_stream_with_no_provider_stream_to_record_is_refused():
    """The parsed chunks are read from LiteLLM's ``completion_stream``.

    A wrapper without one fails loudly instead of losing the keys outside
    LiteLLM's model: one LiteLLM opens lazily is refused by name, and a
    LiteLLM that renamed the attribute raises ``AttributeError``.
    """
    lazy = SimpleNamespace(completion_stream=None, custom_llm_provider="bedrock")
    with pytest.raises(TypeError, match="bedrock"):
        uni_llm._record_parsed_chunks(lazy)
    with pytest.raises(AttributeError):
        uni_llm._record_parsed_chunks(object())


async def test_usage_is_the_one_the_provider_reported(llm_events):
    """Cost, cache and reasoning tokens survive, so the call is priced as unstreamed."""
    with provider(streamed(sse(EVENTS))):
        reply = await ask()

    assert reply.usage.model_dump() == Usage(**USAGE).model_dump()
    assert [event.provider_cost for event in llm_events] == [COST]


@pytest.mark.parametrize(
    "code, error, attempts",
    [(502, Exception, 2), (400, litellm.BadRequestError, 1)],
)
async def test_an_error_in_the_stream_fails_as_an_unstreamed_call_does(
    code,
    error,
    attempts,
):
    """A provider error sent part-way through is raised as it is unstreamed.

    5xx is retried (once here) and 4xx is not, and the caller gets the same
    exception a failed unstreamed call gives: a plain ``Exception`` carrying
    the message of LiteLLM's ``APIError``, or the mapped 4xx error.
    """
    body = sse([*EVENTS[:3], error_event(code)], done=False)
    with provider(*[streamed(body)] * attempts) as sent:
        with pytest.raises(Exception) as caught:
            await ask()

    assert len(sent) == attempts
    assert type(caught.value) is error
    assert "Provider disconnected unexpectedly" in str(caught.value)


async def test_a_dropped_stream_is_retried_and_only_the_finished_call_is_costed(
    llm_events,
):
    """The dropped attempt's charge is unknown to unillm, as for any failed call."""
    with provider(
        dropped(sse(EVENTS[:4], done=False)),
        streamed(sse(EVENTS)),
    ) as sent:
        reply = await ask()

    assert len(sent) == 2
    assert reply.choices[0].message.content == CONTENT
    assert [event.provider_cost for event in llm_events] == [COST]


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setattr(uni_llm, "STREAM_CALL_DEADLINE_S", 0.3)


async def test_a_stream_past_its_deadline_is_tried_once_more(
    short_deadline,
    llm_events,
):
    """Keep-alive comments alone never end a stream; the deadline does."""
    with provider(kept_alive, streamed(sse(EVENTS))) as sent:
        reply = await ask()

    assert len(sent) == 2
    assert reply.choices[0].message.content == CONTENT
    assert [event.provider_cost for event in llm_events] == [COST]


async def test_a_close_that_fails_past_the_deadline_still_tries_once_more(
    short_deadline,
    monkeypatch,
):
    """Closing the stuck stream may fail or hang; the expiry still gets its one more try."""

    async def failing_close(self):
        raise RuntimeError("close failed")

    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    monkeypatch.setattr(CustomStreamWrapper, "aclose", failing_close)
    with provider(kept_alive, streamed(sse(EVENTS))) as sent:
        reply = await ask()

    assert len(sent) == 2
    assert reply.choices[0].message.content == CONTENT


async def test_a_second_expiry_raises_a_timeout_to_the_caller(short_deadline):
    """One more try only, however many the transient policy would allow."""
    with provider(kept_alive, kept_alive, kept_alive, retries=3) as sent:
        with pytest.raises(litellm.exceptions.Timeout):
            await ask()

    assert len(sent) == 2


async def test_an_expired_call_is_costed_as_unknown(short_deadline, llm_events):
    """Neither the LLM event nor the cost event says it cost nothing."""
    async with acapture_costs() as costs:
        with provider(kept_alive, kept_alive):
            with pytest.raises(litellm.exceptions.Timeout):
                await ask()

    assert [event.provider_cost for event in llm_events] == [None]
    assert [event.provider_cost for event in costs] == [None]


async def test_without_the_flag_the_request_is_not_streamed():
    with provider(whole(WHOLE)) as sent:
        reply = await ask(stream_transport=False)

    assert not sent[0].get("stream")
    assert "stream_options" not in sent[0]
    assert reply.choices[0].message.content == "Sunny."


async def test_a_transport_returning_a_whole_reply_is_used_as_is(monkeypatch):
    """A replaced transport is asked to stream, and its reply passes through."""
    asked: list[dict] = []

    async def transport(*, shared_session=None, client=None, **kw):
        asked.append(kw)
        return ModelResponse(
            id="scripted",
            choices=[
                Choices(
                    message=Message(content="Sunny."),
                    finish_reason="stop",
                    index=0,
                ),
            ],
        )

    monkeypatch.setattr(uni_llm, "_acompletion_with_transient_retry", transport)
    client = unillm.AsyncUnify(ENDPOINT, cache=False, stream_transport=True)

    assert await client.generate(user_message="Weather in Paris?") == "Sunny."
    assert asked[0]["stream"] is True
    assert asked[0]["stream_options"] == {"include_usage": True}


def test_the_flag_is_off_by_default_and_copies_keep_it():
    assert unillm.AsyncUnify(ENDPOINT, cache=False).stream_transport is False
    client = unillm.AsyncUnify(ENDPOINT, cache=False, stream_transport=True)
    assert client.copy().stream_transport is True


# ── silence: no byte for UNILLM_STREAM_IDLE_S ─────────────────────────────────

IDLE_S = 0.25
KEEP_ALIVE = b": OPENROUTER PROCESSING\n\n"


class _Paced(httpx.AsyncByteStream):
    """An event stream sent piece by piece, *gap* seconds apart.

    *keep_alive_for* seconds of keep-alive comments come first; with
    *stall_after* the connection then goes silent, for good, after that
    many pieces.
    """

    def __init__(
        self,
        pieces: list[bytes],
        *,
        gap: float = 0.05,
        keep_alive_for: float = 0.0,
        stall_after: int | None = None,
    ) -> None:
        self._pieces = pieces
        self._gap = gap
        self._keep_alive_for = keep_alive_for
        self._stall_after = stall_after

    async def __aiter__(self):
        loop = asyncio.get_running_loop()
        until = loop.time() + self._keep_alive_for
        while loop.time() < until:
            yield KEEP_ALIVE
            await asyncio.sleep(self._gap)
        for n, piece in enumerate(self._pieces):
            if n == self._stall_after:
                await asyncio.sleep(3600)
            yield piece
            await asyncio.sleep(self._gap)


def paced(**options) -> Reply:
    pieces = [sse([event], done=False) for event in EVENTS] + [b"data: [DONE]\n\n"]
    return lambda url: httpx.Response(
        200,
        stream=_Paced(pieces, **options),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", url),
    )


def unanswered(url: str):
    """A provider that never answers the request: not even its headers."""

    async def wait() -> httpx.Response:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    return wait()


@pytest.fixture
def idle_limit(monkeypatch):
    monkeypatch.setenv(uni_llm.STREAM_IDLE_ENV, str(IDLE_S))


async def test_a_stream_that_goes_silent_after_its_first_bytes_is_aborted_and_retried(
    idle_limit,
    llm_events,
    caplog,
):
    """Silent past the limit: the attempt is closed and retried in the same call."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    with caplog.at_level(logging.WARNING, logger="unillm"):
        with provider(paced(stall_after=2), paced()) as sent:
            reply = await ask()

    assert len(sent) == 2
    assert reply.choices[0].message.content == CONTENT
    assert [event.provider_cost for event in llm_events] == [COST]
    assert loop.time() - started < 10 * IDLE_S
    assert "streamed call aborted, no byte arrived for" in caplog.text


async def test_keep_alive_comments_alone_longer_than_the_limit_are_not_silence(
    idle_limit,
    llm_events,
):
    """A provider still working sends comments only; every byte counts."""
    with provider(paced(keep_alive_for=4 * IDLE_S)) as sent:
        reply = await ask()

    assert len(sent) == 1
    assert reply.choices[0].message.content == CONTENT
    assert [event.provider_cost for event in llm_events] == [COST]


async def test_a_healthy_stream_with_gaps_under_the_limit_is_unchanged(idle_limit):
    with provider(paced(gap=IDLE_S / 3)) as sent:
        paced_reply = await ask()
    with provider(streamed(sse(EVENTS))):
        whole_reply = await ask()

    assert len(sent) == 1
    assert paced_reply.choices[0].message == whole_reply.choices[0].message
    assert paced_reply.usage.model_dump() == whole_reply.usage.model_dump()


async def test_a_request_never_answered_is_silence_too(idle_limit):
    with provider(unanswered, paced()) as sent:
        reply = await ask()

    assert len(sent) == 2
    assert reply.choices[0].message.content == CONTENT


async def test_silence_on_every_attempt_raises_a_timeout_costed_as_unknown(
    idle_limit,
    llm_events,
):
    """The transient policy's retries, then the timeout an unstreamed call raises."""
    async with acapture_costs() as costs:
        with provider(paced(stall_after=1), paced(stall_after=0), retries=1) as sent:
            with pytest.raises(litellm.exceptions.Timeout, match="silent"):
                await ask()

    assert len(sent) == 2
    assert [event.provider_cost for event in llm_events] == [None]
    assert [event.provider_cost for event in costs] == [None]


async def test_with_the_limit_off_only_the_deadline_ends_a_silent_stream(
    monkeypatch,
    short_deadline,
):
    monkeypatch.setenv(uni_llm.STREAM_IDLE_ENV, "off")
    with provider(paced(stall_after=1), paced(gap=0.0)) as sent:
        reply = await ask()

    assert len(sent) == 2
    assert reply.choices[0].message.content == CONTENT


def test_the_limit_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv(uni_llm.STREAM_IDLE_ENV, raising=False)
    assert uni_llm._stream_idle_s() == uni_llm.STREAM_IDLE_DEFAULT_S == 120.0
    for raw, value in [("90", 90.0), (" 0.5 ", 0.5), ("0", None), ("OFF", None), ("", 120.0)]:
        monkeypatch.setenv(uni_llm.STREAM_IDLE_ENV, raw)
        assert uni_llm._stream_idle_s() == value
    for raw in ["-1", "soon", "nan", "inf"]:
        monkeypatch.setenv(uni_llm.STREAM_IDLE_ENV, raw)
        with pytest.raises(ValueError, match=uni_llm.STREAM_IDLE_ENV):
            uni_llm._stream_idle_s()


# ── the reply's reasoning, sent back on the next request ─────────────────────

# The joined items of ``EVENTS``: its reasoning text pieces as one item, then
# the encrypted item as it came.
EVENTS_ITEMS = [
    {"type": "reasoning.text", "text": REASONING, "format": FORMAT, "index": 0},
    ENCRYPTED,
]
SUMMARY_ITEMS = [SUMMARY_ITEM, ENCRYPTED]

# A reply whose only reasoning is the encrypted item: no reasoning text, so
# LiteLLM gives its message no ``reasoning_content`` and, streamed, no
# ``provider_specific_fields`` of its own.
ENCRYPTED_ONLY_EVENTS = SUMMARY_EVENTS[2:]
ENCRYPTED_ONLY_REPLY = {
    **SUMMARY_REPLY,
    "choices": [
        {
            **SUMMARY_REPLY["choices"][0],
            "message": {
                "role": "assistant",
                "content": "Sunny.",
                "reasoning_details": [ENCRYPTED],
            },
        },
    ],
}

# A reply with no reasoning at all.
PLAIN_EVENTS = SUMMARY_EVENTS[3:]

FIRST, SECOND = "What is the weather in Paris?", "And tomorrow?"


def second(stream_transport: bool) -> Reply:
    """The second turn's reply, in the form the request asks for."""
    return streamed(sse(PLAIN_EVENTS)) if stream_transport else whole(WHOLE)


async def two_turns(first: Reply, *, stream_transport: bool = True):
    """One stateful conversation: *first* answers the first turn.

    Returns the first reply and the bodies of both requests.
    """
    client = unillm.AsyncUnify(
        ENDPOINT,
        cache=False,
        stateful=True,
        stream_transport=stream_transport,
    )
    with provider(first, second(stream_transport)) as sent:
        reply = await client.generate(
            user_message=FIRST,
            tools=[copy.deepcopy(WEATHER_TOOL)],
        )
        await client.generate(user_message=SECOND, tools=[copy.deepcopy(WEATHER_TOOL)])
    return reply, sent


def two_turns_sync(first: Reply, *, stream_transport: bool = True):
    """``two_turns`` with the synchronous client."""
    client = unillm.Unify(
        ENDPOINT,
        cache=False,
        stateful=True,
        stream_transport=stream_transport,
    )
    with provider(first, second(stream_transport)) as sent:
        reply = client.generate(user_message=FIRST, tools=[copy.deepcopy(WEATHER_TOOL)])
        client.generate(user_message=SECOND, tools=[copy.deepcopy(WEATHER_TOOL)])
    return reply, sent


def sent_back(body: dict) -> dict:
    """The one assistant message of the second request's *body*."""
    [message] = [m for m in body["messages"] if m["role"] == "assistant"]
    assert body["messages"][-1] == {"role": "user", "content": SECOND}
    return message


def as_sent(message) -> dict:
    """What *message* went out as before its reasoning was sent back: its
    dump, as JSON, without the keys LiteLLM drops for being ``None``."""
    dumped = loads(dumps(message.model_dump(warnings=False), default=str))
    return {key: value for key, value in dumped.items() if value is not None}


TWO_TURNS_WITH_REASONING = [
    pytest.param(streamed(sse(EVENTS)), True, EVENTS_ITEMS, id="streamed-tool-call"),
    pytest.param(
        streamed(sse(events("alone"))),
        True,
        EVENTS_ITEMS,
        id="streamed-encrypted-alone",
    ),
    pytest.param(
        streamed(sse(events("finish"))),
        True,
        EVENTS_ITEMS,
        id="streamed-encrypted-at-finish",
    ),
    pytest.param(streamed(sse(SUMMARY_EVENTS)), True, SUMMARY_ITEMS, id="streamed-summary"),
    pytest.param(whole(SUMMARY_REPLY), False, SUMMARY_ITEMS, id="unstreamed-summary"),
]


@pytest.mark.parametrize("first, stream_transport, items", TWO_TURNS_WITH_REASONING)
async def test_the_next_request_sends_the_reasoning_details_back_at_the_top_level(
    first,
    stream_transport,
    items,
):
    """The provider reads them back from the message itself, item for item."""
    _reply, sent = await two_turns(first, stream_transport=stream_transport)

    message = sent_back(sent[1])
    assert dumps(message["reasoning_details"]) == dumps(items)
    assert "reasoning_details" not in (message.get("provider_specific_fields") or {})


@pytest.mark.parametrize("first, stream_transport, items", TWO_TURNS_WITH_REASONING)
def test_the_sync_client_sends_them_back_too(first, stream_transport, items):
    _reply, sent = two_turns_sync(first, stream_transport=stream_transport)

    assert dumps(sent_back(sent[1])["reasoning_details"]) == dumps(items)


@pytest.mark.parametrize("first, stream_transport, items", TWO_TURNS_WITH_REASONING)
async def test_everything_else_in_the_message_goes_back_as_before(
    first,
    stream_transport,
    items,
):
    """Content, tool calls and the other provider fields are not touched."""
    reply, sent = await two_turns(first, stream_transport=stream_transport)

    before = as_sent(reply.choices[0].message)
    fields = before.pop("provider_specific_fields")
    assert fields.pop("reasoning_details") == items
    message = sent_back(sent[1])
    assert message.pop("reasoning_details") == items
    assert message.pop("provider_specific_fields", None) == (fields or None)
    assert message == before
    assert message["content"] == reply.choices[0].message.content


async def test_streamed_and_unstreamed_replies_are_sent_back_the_same():
    """The equality of the assembled items holds on the next request too."""
    _reply, streamed_sent = await two_turns(streamed(sse(SUMMARY_EVENTS)))
    _reply, unstreamed_sent = await two_turns(
        whole(SUMMARY_REPLY),
        stream_transport=False,
    )

    streamed_message, unstreamed_message = (
        sent_back(sent[1]) for sent in (streamed_sent, unstreamed_sent)
    )
    assert dumps(streamed_message["reasoning_details"]) == dumps(
        unstreamed_message["reasoning_details"],
    )
    assert streamed_message["content"] == unstreamed_message["content"] == "Sunny."


@pytest.mark.parametrize(
    "first, stream_transport",
    [
        pytest.param(streamed(sse(ENCRYPTED_ONLY_EVENTS)), True, id="streamed"),
        pytest.param(whole(ENCRYPTED_ONLY_REPLY), False, id="unstreamed"),
    ],
)
async def test_reasoning_with_no_other_provider_field_is_sent_back(
    first,
    stream_transport,
):
    """An encrypted item alone, with no reasoning text beside it, still goes back."""
    _reply, sent = await two_turns(first, stream_transport=stream_transport)

    message = sent_back(sent[1])
    assert dumps(message["reasoning_details"]) == dumps([ENCRYPTED])
    assert message["content"] == "Sunny."
    assert "reasoning_details" not in (message.get("provider_specific_fields") or {})


@pytest.mark.parametrize(
    "first, stream_transport",
    [
        pytest.param(streamed(sse(PLAIN_EVENTS)), True, id="streamed"),
        pytest.param(whole(WHOLE), False, id="unstreamed"),
    ],
)
async def test_a_reply_without_reasoning_goes_back_exactly_as_before(
    first,
    stream_transport,
):
    reply, sent = await two_turns(first, stream_transport=stream_transport)

    message = reply.choices[0].message
    assert uni_llm._history_message(message) == message.model_dump(warnings=False)
    assert sent_back(sent[1]) == as_sent(message) == {
        "role": "assistant",
        "content": "Sunny.",
    }


THIRD = "And the day after?"


@pytest.mark.parametrize(
    "first, stream_transport",
    [
        *(pytest.param(*p.values[:2], id=p.id) for p in TWO_TURNS_WITH_REASONING),
        pytest.param(streamed(sse(PLAIN_EVENTS)), True, id="streamed-plain"),
        pytest.param(whole(WHOLE), False, id="unstreamed-plain"),
    ],
)
async def test_a_message_goes_out_the_same_on_every_later_request(
    first,
    stream_transport,
):
    """Sending a request leaves the history it read unchanged: the first
    reply's message goes out byte for byte the same on the third request as
    on the second, its reasoning still at the top level."""
    client = unillm.AsyncUnify(
        ENDPOINT,
        cache=False,
        stateful=True,
        stream_transport=stream_transport,
    )
    later = second(stream_transport), second(stream_transport)
    with provider(first, *later) as sent:
        for text in (FIRST, SECOND, THIRD):
            await client.generate(user_message=text, tools=[copy.deepcopy(WEATHER_TOOL)])

    on_second, on_third = (
        [m for m in body["messages"] if m["role"] == "assistant"][0] for body in sent[1:]
    )
    assert dumps(on_third, sort_keys=True) == dumps(on_second, sort_keys=True)
    assert body_prefix(sent[2], len(sent[1]["messages"]) - 1) == body_prefix(
        sent[1],
        len(sent[1]["messages"]) - 1,
    )


def body_prefix(body: dict, n: int) -> str:
    """The first *n* messages of a request *body*, as JSON."""
    return dumps(body["messages"][:n], sort_keys=True)
