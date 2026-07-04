"""
Test for tool_choice="required" compliance with thinking mode.

Anthropic's API doesn't support tool_choice="required" when extended thinking
is enabled. We work around this by:
1. Downgrading tool_choice to "auto"
2. Injecting a system prompt instructing the model to call a tool
3. If the model ignores the instruction, retrying with a stronger nudge

This test verifies that:
- The retry mechanism successfully enforces tool calls
- The intermediate messages (non-compliant response + retry nudge) are NOT
  visible to the caller - neither in the response nor in the client's history
"""

from .helpers import new_llm_client

SIMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "do_action",
        "description": "Performs an action.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def _assert_no_retry_nudge_in_history(messages: list) -> None:
    """Assert that retry nudge messages are not present in the message history."""
    # Check for the tool_choice=required nudge
    nudge_fragment = "Your previous turn FAILED"

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and nudge_fragment in content:
            raise AssertionError(
                f"Retry nudge message leaked into history: {content!r}",
            )


def test_tool_choice_required_compliance():
    """
    When tool_choice="required", the model MUST call a tool.

    With thinking mode enabled on Anthropic, we can't use the native
    tool_choice="required" API constraint. This test verifies that our
    retry-on-non-compliance mechanism works correctly AND that it's
    transparent to the caller (no intermediate messages leak through).
    """
    client = new_llm_client("claude-4.8-opus@anthropic")
    client.set_system_message(
        "You are a helpful assistant. Only use tools when absolutely necessary. "
        "For simple questions, prefer answering directly without tools.",
    )

    response = client.generate(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[SIMPLE_TOOL],
        tool_choice="required",
        return_full_completion=True,
    )

    # 1. Verify the response contains tool calls
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) > 0, (
        f"tool_choice='required' but model responded with text only: "
        f"{response.choices[0].message.content!r}"
    )

    # 2. Verify the response is the compliant one (tool call, not text-only)
    # The non-compliant response would have had content like "2 + 2 = 4"
    # The compliant response should have tool_calls and possibly no/minimal content
    assert (
        response.choices[0].finish_reason == "tool_calls"
    ), f"Expected finish_reason='tool_calls' but got {response.choices[0].finish_reason!r}"


def test_tool_choice_required_stateful_history():
    """
    When using stateful mode, verify that the retry mechanism doesn't
    leak intermediate messages into the client's conversation history.

    The client's history should only contain:
    - System message(s)
    - User message
    - Assistant message with tool_calls (the compliant response)

    It should NOT contain:
    - A text-only assistant response (the non-compliant first attempt)
    - The retry nudge user message
    """
    client = new_llm_client("claude-4.8-opus@anthropic", stateful=True)
    client.set_system_message(
        "You are a helpful assistant. Only use tools when absolutely necessary. "
        "For simple questions, prefer answering directly without tools.",
    )

    response = client.generate(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[SIMPLE_TOOL],
        tool_choice="required",
        return_full_completion=True,
    )

    # Verify tool call succeeded first
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) > 0

    # Check the client's message history
    messages = client._messages

    # Count user messages - should only be the original
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert (
        len(user_messages) == 1
    ), f"Expected 1 user message but found {len(user_messages)}: {user_messages}"
    assert (
        user_messages[0]["content"] == "What is 2+2?"
    ), f"Unexpected user message content: {user_messages[0]['content']!r}"

    # Count assistant messages - should only be the compliant response
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    assert (
        len(assistant_messages) == 1
    ), f"Expected 1 assistant message but found {len(assistant_messages)}"

    # The assistant message should have tool_calls (the compliant response)
    assistant_msg = assistant_messages[0]
    assert (
        assistant_msg.get("tool_calls") is not None
    ), f"Assistant message in history has no tool_calls: {assistant_msg}"

    # Verify no retry nudge content leaked into history
    _assert_no_retry_nudge_in_history(messages)
