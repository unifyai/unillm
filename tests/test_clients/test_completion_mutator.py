import json

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from unillm.clients.completion_mutator import (
    CompletionMutatorContext,
    inject_tool_call,
)
from unillm.clients.provider_postprocessing import apply_postprocessing_pipeline

WAIT_TOOL = {
    "type": "function",
    "function": {
        "name": "wait",
        "description": "Wait before the next turn.",
        "parameters": {
            "type": "object",
            "properties": {"delay": {"type": "number"}},
            "required": [],
        },
    },
}


def _completion(
    content: str | None,
    *,
    tool_calls=None,
    finish_reason="stop",
) -> ChatCompletion:
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
        model="minimax-v3",
        object="chat.completion",
    )


def test_inject_tool_call_shape():
    completion = _completion("Check your inbox for the verification email.")
    healed = inject_tool_call(
        completion,
        tool_name="send_unify_message",
        arguments={"content": "Check your inbox.", "contact_id": 42},
    )
    msg = healed.choices[0].message
    assert msg.content is None
    assert healed.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    tool_call = msg.tool_calls[0]
    if hasattr(tool_call, "model_dump"):
        tool_call = tool_call.model_dump(warnings=False)
    assert tool_call["function"]["name"] == "send_unify_message"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "content": "Check your inbox.",
        "contact_id": 42,
    }


def test_mutator_runs_before_retry():
    retry_calls: list[str] = []

    def mutator(completion, context):
        assert isinstance(context, CompletionMutatorContext)
        assert context.original_tool_choice == "required"
        return inject_tool_call(
            completion,
            tool_name="wait",
            arguments={"delay": 0},
        )

    completion = _completion("Plain prose with no tool call.")
    kw = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [WAIT_TOOL],
        "tool_choice": "required",
    }

    result = apply_postprocessing_pipeline(
        completion,
        kw=kw,
        provider="minimax",
        original_tool_choice="required",
        reasoning_effort=None,
        original_request_messages=kw["messages"],
        execute_retry=lambda retry_kw, label: (
            retry_calls.append(label) or _completion(None, tool_calls=[])
        ),
        completion_mutator=mutator,
    )

    assert retry_calls == []
    assert result.choices[0].message.tool_calls is not None
    tool_call = result.choices[0].message.tool_calls[0]
    if hasattr(tool_call, "model_dump"):
        tool_call = tool_call.model_dump(warnings=False)
    assert tool_call["function"]["name"] == "wait"


def test_mutator_absent_behavior_unchanged():
    retry_calls: list[str] = []

    completion = _completion("Plain prose with no tool call.")
    kw = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [WAIT_TOOL],
        "tool_choice": "required",
    }

    apply_postprocessing_pipeline(
        completion,
        kw=kw,
        provider="minimax",
        original_tool_choice="required",
        reasoning_effort=None,
        original_request_messages=kw["messages"],
        execute_retry=lambda retry_kw, label: (
            retry_calls.append(label) or _completion(None, tool_calls=[])
        ),
        completion_mutator=None,
    )

    assert len(retry_calls) == 1


def test_completion_mutator_not_in_request_kw():
    captured_kw: dict | None = None

    def mutator(completion, context):
        nonlocal captured_kw
        captured_kw = context.request_kw
        return completion

    completion = _completion("hello")
    kw = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [WAIT_TOOL],
        "tool_choice": "required",
    }

    apply_postprocessing_pipeline(
        completion,
        kw=kw,
        provider="minimax",
        original_tool_choice="required",
        reasoning_effort=None,
        original_request_messages=kw["messages"],
        execute_retry=lambda retry_kw, label: completion,
        completion_mutator=mutator,
    )

    assert captured_kw is not None
    assert "completion_mutator" not in captured_kw
