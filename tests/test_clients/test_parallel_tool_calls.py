from types import SimpleNamespace

from unillm.clients.uni_llm import (
    _enforce_parallel_tool_call_response_limit,
    _normalize_assistant_message_content,
)


def _completion_with_tool_calls(*tool_calls):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=list(tool_calls)),
            ),
        ],
    )


def test_parallel_tool_calls_false_prunes_response_to_first_call():
    completion = _completion_with_tool_calls("first", "second")

    changed = _enforce_parallel_tool_call_response_limit(completion, False)

    assert changed
    assert completion.choices[0].message.tool_calls == ["first"]


def test_parallel_tool_calls_true_keeps_response_calls():
    completion = _completion_with_tool_calls("first", "second")

    changed = _enforce_parallel_tool_call_response_limit(completion, True)

    assert not changed
    assert completion.choices[0].message.tool_calls == ["first", "second"]


def test_normalize_assistant_message_content_strips_provider_wrapping_whitespace():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="\n    WORLD\n"),
            ),
        ],
    )

    changed = _normalize_assistant_message_content(completion)

    assert changed
    assert completion.choices[0].message.content == "WORLD"


def test_normalize_assistant_message_content_keeps_non_string_content():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
            ),
        ],
    )

    changed = _normalize_assistant_message_content(completion)

    assert not changed
    assert completion.choices[0].message.content is None
