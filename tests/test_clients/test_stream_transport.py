"""Single-completion requests sent as streams and reassembled.

With ``stream_transport`` a request for one completion goes out as a stream,
so bytes arrive while the model generates, and is read whole and rebuilt into
the completion the same reply gives unstreamed. The provider is replaced at
LiteLLM's HTTP handler, so the event stream (``openrouter_stream``) passes
through LiteLLM's own parsing, and nothing leaves the process.
"""

from __future__ import annotations

import contextlib
import copy
from json import loads
from typing import Callable, Iterator
from unittest.mock import patch

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from pydantic import SecretStr

import unillm
from unillm.clients import uni_llm
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
    USAGE,
    error_event,
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


@contextlib.contextmanager
def provider(*replies: Reply) -> Iterator[list[dict]]:
    """Answer each request with the next of *replies*; yield the bodies sent.

    One transient retry, with no backoff, so a retried failure costs no time.
    """
    sent: list[dict] = []
    queue = list(replies)

    async def post(self, url, data=None, json=None, **_):  # noqa: A002
        sent.append(loads(data) if data is not None else json)
        return queue.pop(0)(url)

    with (
        patch.object(SETTINGS, "OPENROUTER_API_KEY", SecretStr(PLACEHOLDER_KEY)),
        patch.object(SETTINGS, "UNILLM_TRANSIENT_RETRY_COUNT", 1),
        patch("unillm.helpers._BACKOFF_BASE_SECONDS", 0),
        patch.object(AsyncHTTPHandler, "post", post),
    ):
        yield sent


@pytest.fixture(autouse=True)
def _no_gateway(monkeypatch):
    monkeypatch.delenv("UNILLM_LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("UNILLM_LLM_GATEWAY_KEY", raising=False)


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


async def test_reasoning_details_are_joined_in_order():
    """Text pieces of one item are joined; the encrypted item is kept whole."""
    with provider(streamed(sse(EVENTS))):
        reply = await ask()

    message = reply.choices[0].message
    assert message.reasoning_content == REASONING
    fields = message.provider_specific_fields
    assert fields["reasoning"] == REASONING
    assert fields["reasoning_details"] == [
        {"type": "reasoning.text", "text": REASONING, "format": FORMAT, "index": 0},
        ENCRYPTED,
    ]


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
