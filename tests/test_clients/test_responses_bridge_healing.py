"""Tests for Responses→Chat Completions bridge choice collapse."""

from __future__ import annotations

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from unillm.clients.completion_mutator import inject_tool_call
from unillm.clients.provider_postprocessing import apply_postprocessing_pipeline
from unillm.clients.responses_bridge_healing import (
    maybe_collapse_responses_bridge_choices,
)


def _choice(
    *,
    index: int,
    content: str | None,
    tool_calls=None,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
) -> Choice:
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    if reasoning_content is not None and hasattr(message, "reasoning_content"):
        message.reasoning_content = reasoning_content
    return Choice(index=index, message=message, finish_reason=finish_reason)


def _completion(choices: list[Choice]) -> ChatCompletion:
    return ChatCompletion(
        id="test-bridge",
        choices=choices,
        created=1234567890,
        model="gpt-5.6-terra",
        object="chat.completion",
    )


def _make_call_tool():
    return [
        {
            "id": "call_ZE449sg7gFKjfbLDRCo6vmci",
            "type": "function",
            "function": {
                "name": "make_call",
                "arguments": '{"contact_id":2,"opener":"Hi Alice!"}',
            },
        },
    ]


def test_collapse_text_then_tool_choices_when_n_absent():
    completion = _completion(
        [
            _choice(
                index=0,
                content="I'll call you now with a joke.",
                finish_reason="stop",
                reasoning_content="Announce before calling.",
            ),
            _choice(
                index=1,
                content=None,
                tool_calls=_make_call_tool(),
                finish_reason="tool_calls",
            ),
        ],
    )

    healed = maybe_collapse_responses_bridge_choices(completion, request_kw={})

    assert len(healed.choices) == 1
    msg = healed.choices[0].message
    assert msg.content == "I'll call you now with a joke."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "make_call"
    assert healed.choices[0].finish_reason == "tool_calls"


def test_collapse_when_n_is_one():
    completion = _completion(
        [
            _choice(index=0, content="Calling now.", finish_reason="stop"),
            _choice(
                index=1,
                content=None,
                tool_calls=_make_call_tool(),
                finish_reason="tool_calls",
            ),
        ],
    )
    healed = maybe_collapse_responses_bridge_choices(
        completion,
        request_kw={"n": 1},
    )
    assert len(healed.choices) == 1
    assert healed.choices[0].message.tool_calls[0].function.name == "make_call"


def test_does_not_collapse_when_n_greater_than_one():
    completion = _completion(
        [
            _choice(index=0, content="Calling now.", finish_reason="stop"),
            _choice(
                index=1,
                content=None,
                tool_calls=_make_call_tool(),
                finish_reason="tool_calls",
            ),
        ],
    )
    healed = maybe_collapse_responses_bridge_choices(
        completion,
        request_kw={"n": 2},
    )
    assert len(healed.choices) == 2


def test_does_not_collapse_text_only_multi_choice():
    completion = _completion(
        [
            _choice(index=0, content="Hello.", finish_reason="stop"),
            _choice(index=1, content="Hello again.", finish_reason="stop"),
        ],
    )
    healed = maybe_collapse_responses_bridge_choices(completion, request_kw={})
    assert len(healed.choices) == 2


def test_does_not_collapse_tool_only_multi_choice():
    """Tool-only multi-choice is outside this heal's signature."""
    other_tool = [
        {
            "id": "call_other",
            "type": "function",
            "function": {"name": "send_sms", "arguments": "{}"},
        },
    ]
    completion = _completion(
        [
            _choice(
                index=0,
                content=None,
                tool_calls=_make_call_tool(),
                finish_reason="tool_calls",
            ),
            _choice(
                index=1,
                content=None,
                tool_calls=other_tool,
                finish_reason="tool_calls",
            ),
        ],
    )
    healed = maybe_collapse_responses_bridge_choices(completion, request_kw={})
    assert len(healed.choices) == 2


def test_pipeline_collapse_prevents_prose_mutator_from_eating_tools():
    """Regression for email_to_phone_call: announce text + make_call split."""
    completion = _completion(
        [
            _choice(
                index=0,
                content="I'll call you now with a joke.",
                finish_reason="stop",
            ),
            _choice(
                index=1,
                content=None,
                tool_calls=_make_call_tool(),
                finish_reason="tool_calls",
            ),
        ],
    )
    mutator_ran = {"count": 0}

    def prose_mutator(comp, context):
        del context
        msg = comp.choices[0].message
        if msg.tool_calls:
            return comp
        mutator_ran["count"] += 1
        return inject_tool_call(
            comp,
            tool_name="send_email",
            arguments={"body": msg.content, "subject": msg.content},
        )

    result = apply_postprocessing_pipeline(
        completion,
        kw={"n": 1, "tools": [{"type": "function", "function": {"name": "make_call"}}]},
        provider="openai",
        original_tool_choice="required",
        reasoning_effort="high",
        execute_retry=lambda *_a, **_k: completion,
        completion_mutator=prose_mutator,
    )

    assert mutator_ran["count"] == 0
    assert len(result.choices) == 1
    assert result.choices[0].message.tool_calls[0].function.name == "make_call"
    assert result.choices[0].message.content == "I'll call you now with a joke."
