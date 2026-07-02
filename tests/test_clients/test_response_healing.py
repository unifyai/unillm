import json

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel, ConfigDict

from unillm.clients.provider_postprocessing import (
    apply_postprocessing_pipeline,
    check_needs_postprocessing,
)
from unillm.clients.response_format import (
    canonicalize_response_format,
)
from unillm.clients.response_healing import (
    should_attempt_tool_call_healing,
    try_heal_embedded_tool_calls,
)

SEND_UNIFY_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_unify_message",
        "description": "Send a message to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
        },
    },
}

TOOL_A = {
    "type": "function",
    "function": {
        "name": "tool_a",
        "description": "A tool.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


class TextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thoughts: str


def _tool_call_from_message(message, index: int = 0) -> dict:
    tool_calls = message.tool_calls or []
    tool_call = tool_calls[index]
    if isinstance(tool_call, dict):
        return tool_call
    return tool_call.model_dump(warnings=False)


def _completion(content: str | None, *, tool_calls=None, finish_reason="stop"):
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
        model="deepseek-chat",
        object="chat.completion",
    )


def test_heal_staging_embedded_tool_calls():
    thoughts = (
        "The user is asking about OneDrive, but I don't have workspace access yet."
    )
    reply = "I don't have access to your workspace yet. Please connect it first."
    content = json.dumps(
        {
            "thoughts": thoughts,
            "tool_calls": [
                {
                    "name": "send_unify_message",
                    "arguments": {"content": reply},
                },
            ],
        },
    )
    response = _completion(content)
    spec = canonicalize_response_format(TextResponse)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=spec,
    )

    assert healed is not None
    msg = healed.choices[0].message
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    call = _tool_call_from_message(msg)
    assert call["function"]["name"] == "send_unify_message"
    assert json.loads(call["function"]["arguments"]) == {"content": reply}
    assert json.loads(msg.content) == {"thoughts": thoughts}
    assert healed.choices[0].finish_reason == "tool_calls"


def test_heal_rejects_unknown_tool_name():
    content = json.dumps(
        {
            "thoughts": "Try an unknown tool.",
            "tool_calls": [{"name": "missing_tool", "arguments": {}}],
        },
    )
    response = _completion(content)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is None
    assert response.choices[0].message.tool_calls is None


def test_heal_noop_when_native_tool_calls_present():
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    native_call = ChatCompletionMessageToolCall(
        id="call_native",
        type="function",
        function=Function(name="send_unify_message", arguments='{"content": "hi"}'),
    )
    content = json.dumps(
        {
            "thoughts": "ignored",
            "tool_calls": [
                {"name": "send_unify_message", "arguments": {"content": "x"}},
            ],
        },
    )
    response = _completion(content, tool_calls=[native_call])

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is None


def test_heal_noop_wrong_provider():
    content = json.dumps(
        {
            "thoughts": "Try tool.",
            "tool_calls": [
                {"name": "send_unify_message", "arguments": {"content": "x"}},
            ],
        },
    )
    response = _completion(content)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="anthropic",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is None


def test_heal_noop_when_tool_choice_auto():
    content = json.dumps(
        {
            "thoughts": "Try tool.",
            "tool_calls": [
                {"name": "send_unify_message", "arguments": {"content": "x"}},
            ],
        },
    )
    response = _completion(content)

    assert not should_attempt_tool_call_healing(
        "deepseek",
        "auto",
        response.choices[0].message,
    )

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="auto",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is None


def test_heal_validates_response_format_after_strip():
    content = json.dumps(
        {
            "thoughts": "Call the tool.",
            "tool_calls": [
                {"name": "send_unify_message", "arguments": {"content": "hi"}},
            ],
            "unexpected": "field",
        },
    )
    response = _completion(content)
    spec = canonicalize_response_format(TextResponse)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=spec,
    )

    assert healed is None


def test_heal_rejects_non_json_content():
    response = _completion("plain text response")

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is None


def test_heal_top_level_name_arguments_shape():
    content = json.dumps(
        {
            "name": "send_unify_message",
            "arguments": {"content": "hello"},
            "thoughts": "Send a greeting.",
        },
    )
    response = _completion(content)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="minimax",
        original_tool_choice="required",
        tools=[SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is not None
    msg = healed.choices[0].message
    assert msg.tool_calls is not None
    call = _tool_call_from_message(msg)
    assert call["function"]["name"] == "send_unify_message"
    assert json.loads(msg.content) == {"thoughts": "Send a greeting."}


@pytest.mark.parametrize("provider", ["deepseek", "minimax", "xiaomi-mimo"])
def test_healing_avoids_tool_choice_retry(provider):
    content = json.dumps(
        {
            "thoughts": "Reply to the user.",
            "tool_calls": [
                {
                    "name": "tool_a",
                    "arguments": {},
                },
            ],
        },
    )
    response = _completion(content)
    kw = {
        "messages": [{"role": "user", "content": "Do it."}],
        "tools": [TOOL_A],
    }
    retry_calls: list[str] = []

    def execute_retry(retry_kw, label):
        retry_calls.append(label)
        raise AssertionError("tool-choice retry should not run when healing succeeds")

    result = apply_postprocessing_pipeline(
        response,
        kw=kw,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        execute_retry=execute_retry,
    )

    assert retry_calls == []
    assert result.choices[0].message.tool_calls is not None
    assert (
        _tool_call_from_message(result.choices[0].message)["function"]["name"]
        == "tool_a"
    )
    needs_retry, retry_reason = check_needs_postprocessing(
        response=result,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[TOOL_A],
    )
    assert not needs_retry
    assert retry_reason is None


@pytest.mark.parametrize("provider", ["deepseek", "minimax", "xiaomi-mimo"])
def test_retry_response_gets_heal_attempt(provider):
    initial = _completion("I can answer directly.")
    healed_content = json.dumps(
        {
            "thoughts": "Call the tool now.",
            "tool_calls": [{"name": "tool_a", "arguments": {}}],
        },
    )
    retry_response = _completion(healed_content)
    kw = {
        "messages": [{"role": "user", "content": "Do it."}],
        "tools": [TOOL_A],
    }
    retry_calls: list[str] = []

    def execute_retry(retry_kw, label):
        retry_calls.append(label)
        assert label == "retry"
        return retry_response

    result = apply_postprocessing_pipeline(
        initial,
        kw=kw,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        execute_retry=execute_retry,
    )

    assert retry_calls == ["retry"]
    assert result.choices[0].message.tool_calls is not None
    assert (
        _tool_call_from_message(result.choices[0].message)["function"]["name"]
        == "tool_a"
    )
    needs_retry, retry_reason = check_needs_postprocessing(
        response=result,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[TOOL_A],
    )
    assert not needs_retry
    assert retry_reason is None


def test_forced_tool_name_triggers_healing():
    content = json.dumps(
        {
            "thoughts": "Use the required tool.",
            "tool_calls": [{"name": "tool_a", "arguments": {}}],
        },
    )
    response = _completion(content)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="xiaomi-mimo",
        original_tool_choice={"type": "function", "function": {"name": "tool_a"}},
        tools=[TOOL_A],
        response_format_spec=None,
    )

    assert healed is not None
    assert (
        _tool_call_from_message(healed.choices[0].message)["function"]["name"]
        == "tool_a"
    )


ASK_ABOUT_CONTACTS_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_about_contacts",
        "description": "Query contact records.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
}


def test_heal_deepseek_input_format_embedded_tool_calls():
    """DeepSeek V4 via OpenRouter uses ``input`` instead of ``arguments``."""

    query = "Find Sarah's contact preference for phone vs email."
    reply = "Let me check."
    content = json.dumps(
        {
            "thoughts": "Look up Sarah and acknowledge.",
            "tool_calls": [
                {
                    "type": "tool_call",
                    "name": "ask_about_contacts",
                    "id": "tool-1",
                    "input": {"text": query},
                },
                {
                    "type": "tool_call",
                    "name": "send_unify_message",
                    "id": "tool-2",
                    "input": {"contact_id": 1, "content": reply},
                },
            ],
        },
    )
    response = _completion(content)
    spec = canonicalize_response_format(TextResponse)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[ASK_ABOUT_CONTACTS_TOOL, SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=spec,
    )

    assert healed is not None
    msg = healed.choices[0].message
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2
    first = _tool_call_from_message(msg, 0)
    second = _tool_call_from_message(msg, 1)
    assert first["function"]["name"] == "ask_about_contacts"
    assert json.loads(first["function"]["arguments"]) == {"text": query}
    assert second["function"]["name"] == "send_unify_message"
    assert json.loads(second["function"]["arguments"]) == {
        "contact_id": 1,
        "content": reply,
    }


def test_healed_tool_calls_survive_litellm_message_model_dump():
    """Promoted tool calls must remain executable after LiteLLM Message serialization."""

    from litellm.types.utils import Choices, Message, ModelResponse

    query = "Find Sarah."
    content = json.dumps(
        {
            "thoughts": "Look up Sarah.",
            "tool_calls": [
                {
                    "type": "tool_call",
                    "name": "ask_about_contacts",
                    "id": "tool-1",
                    "input": {"text": query},
                },
                {
                    "type": "tool_call",
                    "name": "send_unify_message",
                    "id": "tool-2",
                    "input": {"contact_id": 1, "content": "One moment."},
                },
            ],
        },
    )
    message = Message(role="assistant", content=content, tool_calls=None)
    response = ModelResponse(
        id="test-id",
        choices=[Choices(finish_reason="stop", index=0, message=message)],
        created=1234567890,
        model="deepseek/deepseek-v4-pro",
        object="chat.completion",
    )

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[ASK_ABOUT_CONTACTS_TOOL, SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is not None
    dumped = healed.choices[0].message.model_dump(warnings=False)
    tool_calls = dumped["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "ask_about_contacts"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"text": query}
    assert tool_calls[1]["function"]["name"] == "send_unify_message"
    assert tool_calls[0] != {}
    assert tool_calls[1] != {}


def test_heal_deepseek_tool_name_query_format():
    query = "Find Sarah's contact record and check her preferred communication method."
    content = json.dumps(
        {
            "thoughts": "Look up Sarah.",
            "tool_calls": [
                {
                    "tool_name": "ask_about_contacts",
                    "query": query,
                },
                {
                    "tool_name": "send_unify_message",
                    "query": {"contact_id": 1, "content": "Let me check."},
                },
            ],
        },
    )
    response = _completion(content)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[ASK_ABOUT_CONTACTS_TOOL, SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is not None
    first = _tool_call_from_message(healed.choices[0].message, 0)
    second = _tool_call_from_message(healed.choices[0].message, 1)
    assert json.loads(first["function"]["arguments"]) == {"text": query}
    assert json.loads(second["function"]["arguments"]) == {
        "contact_id": 1,
        "content": "Let me check.",
    }


ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Run a general action.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "requesting_contact_id": {"type": "integer"},
            },
            "required": ["query", "requesting_contact_id"],
        },
    },
}


def test_heal_deepseek_invoke_wrapper_format():
    content = json.dumps(
        {
            "thoughts": "Search knowledge for office hours.",
            "tool_calls": [
                {
                    "invoke": {
                        "name": "act",
                        "args": {
                            "query": "What are the office hours?",
                            "requesting_contact_id": 1,
                        },
                    },
                },
                {
                    "invoke": {
                        "name": "send_unify_message",
                        "args": {"contact_id": 1, "content": "Let me check."},
                    },
                },
            ],
        },
    )
    response = _completion(content)

    healed = try_heal_embedded_tool_calls(
        response,
        provider="deepseek",
        original_tool_choice="required",
        tools=[ACT_TOOL, SEND_UNIFY_MESSAGE_TOOL],
        response_format_spec=None,
    )

    assert healed is not None
    first = _tool_call_from_message(healed.choices[0].message, 0)
    second = _tool_call_from_message(healed.choices[0].message, 1)
    assert json.loads(first["function"]["arguments"]) == {
        "query": "What are the office hours?",
        "requesting_contact_id": 1,
    }
    assert json.loads(second["function"]["arguments"]) == {
        "contact_id": 1,
        "content": "Let me check.",
    }
