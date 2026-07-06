from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from unillm.clients.provider_postprocessing import (
    apply_postprocessing_pipeline,
    promote_reasoning_content_to_content,
)


def _completion(
    *,
    content: str | None,
    reasoning_content: str | None = None,
    tool_calls=None,
):
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    choice = Choice(index=0, message=message, finish_reason="stop")
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="minimax/MiniMax-M3",
        object="chat.completion",
    )


def test_promote_reasoning_content_when_content_is_null():
    response = _completion(
        content=None,
        reasoning_content="The result is 5.\n2 + 3 = 5",
    )

    promoted = promote_reasoning_content_to_content(response)

    assert promoted.choices[0].message.content == "The result is 5.\n2 + 3 = 5"
    assert promoted.choices[0].message.reasoning_content == (
        "The result is 5.\n2 + 3 = 5"
    )


def test_promote_noop_when_content_present():
    response = _completion(content="visible answer", reasoning_content="hidden")

    promoted = promote_reasoning_content_to_content(response)

    assert promoted.choices[0].message.content == "visible answer"


def test_promote_noop_when_tool_calls_present():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function=Function(name="add", arguments='{"x": 2, "y": 3}'),
    )
    response = _completion(
        content=None,
        reasoning_content="Calling add now.",
        tool_calls=[tool_call],
    )

    promoted = promote_reasoning_content_to_content(response)

    assert promoted.choices[0].message.content is None


def test_pipeline_promotes_reasoning_only_turn():
    response = _completion(
        content=None,
        reasoning_content="2 + 3 = 5",
    )

    result = apply_postprocessing_pipeline(
        response,
        kw={"messages": [{"role": "user", "content": "result?"}]},
        provider="minimax",
        original_tool_choice="auto",
        reasoning_effort="max",
        execute_retry=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retry should not run"),
        ),
    )

    assert result.choices[0].message.content == "2 + 3 = 5"
