import json

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel, ConfigDict

from unillm.clients.json_tool_call_normalization import (
    JSON_TOOL_CALL_NAME,
    normalize_json_tool_call_wrappers,
)
from unillm.clients.provider_postprocessing import (
    RETRY_REASON_INVALID_TOOL_NAME,
    apply_postprocessing_pipeline,
    check_needs_postprocessing,
)
from unillm.clients.response_format import (
    RESPONSE_FORMAT_SPEC_KEY,
    canonicalize_response_format,
)

GREET_TOOL = {
    "type": "function",
    "function": {
        "name": "greet",
        "description": "Greet someone.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}

WAIT_TOOL = {
    "type": "function",
    "function": {
        "name": "wait",
        "description": "Wait.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class TextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thoughts: str


def _completion(
    content: str | None,
    *,
    tool_calls=None,
    finish_reason="stop",
):
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="claude-sonnet-4-20250514",
        object="chat.completion",
    )


def _json_tool_call(arguments: dict | str, *, call_id: str = "call_wrapper") -> dict:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=JSON_TOOL_CALL_NAME, arguments=arguments),
    ).model_dump(warnings=False)


def _tool_names(completion: ChatCompletion) -> list[str]:
    tool_calls = completion.choices[0].message.tool_calls or []
    names: list[str] = []
    for call in tool_calls:
        if isinstance(call, dict):
            names.append(call["function"]["name"])
        else:
            names.append(call.function.name)
    return names


def test_structured_only_wrapper_promotes_to_content():
    spec = canonicalize_response_format(TextResponse)
    payload = {"thoughts": "Need to greet Bob."}
    completion = _completion(
        None,
        tool_calls=[_json_tool_call(payload)],
        finish_reason="tool_calls",
    )

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=spec,
        tools=[GREET_TOOL],
    )

    assert result.choices[0].message.content == json.dumps(payload)
    assert result.choices[0].message.tool_calls is None
    assert result.choices[0].finish_reason == "stop"


def test_wrapper_with_inner_single_tool():
    completion = _completion(
        None,
        tool_calls=[
            _json_tool_call(
                {"name": "greet", "arguments": {"name": "Bob"}},
            ),
        ],
        finish_reason="tool_calls",
    )

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=None,
        tools=[GREET_TOOL],
    )

    assert _tool_names(result) == ["greet"]
    assert JSON_TOOL_CALL_NAME not in _tool_names(result)
    assert result.choices[0].finish_reason == "tool_calls"


def test_wrapper_with_tool_calls_array_name_shape():
    completion = _completion(
        None,
        tool_calls=[
            _json_tool_call(
                {
                    "tool_calls": [
                        {"name": "greet", "arguments": {"name": "Alice"}},
                        {"tool_name": "wait", "tool_args": {}},
                    ],
                },
            ),
        ],
        finish_reason="tool_calls",
    )

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=None,
        tools=[GREET_TOOL, WAIT_TOOL],
    )

    assert _tool_names(result) == ["greet", "wait"]


def test_wrapper_with_openai_shaped_inner_tool_calls():
    inner = {
        "type": "function",
        "id": "call_inner",
        "function": {"name": "greet", "arguments": '{"name": "Carol"}'},
    }
    completion = _completion(
        None,
        tool_calls=[_json_tool_call({"tool_calls": [inner]})],
        finish_reason="tool_calls",
    )

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=None,
        tools=[GREET_TOOL],
    )

    assert _tool_names(result) == ["greet"]
    promoted = result.choices[0].message.tool_calls[0]
    if isinstance(promoted, dict):
        assert promoted["function"]["name"] == "greet"
    else:
        assert promoted.function.name == "greet"


def test_wrapper_coexists_with_normal_tool_call():
    normal = ChatCompletionMessageToolCall(
        id="call_normal",
        type="function",
        function=Function(name="wait", arguments="{}"),
    ).model_dump(warnings=False)
    completion = _completion(
        None,
        tool_calls=[
            normal,
            _json_tool_call({"name": "greet", "arguments": {"name": "Dan"}}),
        ],
        finish_reason="tool_calls",
    )

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=None,
        tools=[GREET_TOOL, WAIT_TOOL],
    )

    assert _tool_names(result) == ["wait", "greet"]


def test_non_wrapper_response_unchanged():
    completion = _completion(
        '{"thoughts": "hello"}',
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_greet",
                type="function",
                function=Function(name="greet", arguments='{"name": "Eve"}'),
            ).model_dump(warnings=False),
        ],
        finish_reason="tool_calls",
    )
    before = json.dumps(completion.model_dump(warnings=False))

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=canonicalize_response_format(TextResponse),
        tools=[GREET_TOOL],
    )

    assert json.dumps(result.model_dump(warnings=False)) == before


def test_pipeline_does_not_retry_on_json_tool_call_only():
    spec = canonicalize_response_format(TextResponse)
    payload = {"thoughts": "Structured only."}
    completion = _completion(
        None,
        tool_calls=[_json_tool_call(payload)],
        finish_reason="tool_calls",
    )
    kw = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [GREET_TOOL],
        RESPONSE_FORMAT_SPEC_KEY: spec,
    }

    result = apply_postprocessing_pipeline(
        completion,
        kw=kw,
        provider="anthropic",
        original_tool_choice="required",
        reasoning_effort=None,
        execute_retry=lambda *_args, **_kwargs: pytest.fail("unexpected retry"),
    )

    needs_retry, retry_reason = check_needs_postprocessing(
        response=result,
        provider="anthropic",
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[GREET_TOOL],
    )

    assert needs_retry is False
    assert retry_reason is None
    assert result.choices[0].message.content == json.dumps(payload)
    assert result.choices[0].message.tool_calls is None


def test_structured_wrapper_without_response_format_spec_is_removed():
    completion = _completion(
        None,
        tool_calls=[_json_tool_call({"thoughts": "only structured"})],
        finish_reason="tool_calls",
    )

    result = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=None,
        tools=[GREET_TOOL],
    )

    assert result.choices[0].message.tool_calls is None
    assert result.choices[0].message.content is None


def test_invalid_tool_retry_not_triggered_after_normalization():
    spec = canonicalize_response_format(TextResponse)
    payload = {"thoughts": "hello"}
    completion = _completion(
        None,
        tool_calls=[_json_tool_call(payload)],
        finish_reason="tool_calls",
    )

    normalized = normalize_json_tool_call_wrappers(
        completion,
        response_format_spec=spec,
        tools=[GREET_TOOL],
    )

    needs_retry, retry_reason = check_needs_postprocessing(
        response=normalized,
        provider="anthropic",
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[GREET_TOOL],
    )

    assert needs_retry is False
    assert retry_reason != RETRY_REASON_INVALID_TOOL_NAME
