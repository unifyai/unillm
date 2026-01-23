"""
Test for tool_choice="required" compliance with thinking mode.

Anthropic's API doesn't support tool_choice="required" when extended thinking
is enabled. We work around this by downgrading to "auto" and injecting a
system prompt instructing the model to call a tool.

This test demonstrates that the prompt-based nudge is insufficient - the model
may ignore the instruction and respond with text only.
"""

import pytest
from .helpers import new_llm_client


SIMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "do_action",
        "description": "Performs an action.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@pytest.mark.xfail(
    reason="Claude ignores tool_choice=required instruction when thinking mode is enabled",
    strict=True,
)
def test_tool_choice_required_compliance():
    """
    When tool_choice="required", the model MUST call a tool. With thinking
    mode enabled, we can't use the native API constraint - only a system
    prompt nudge. This test shows that nudge is insufficient when the model
    has reason to believe a tool call is unnecessary.

    We give Claude a simple question it can answer directly, with a system
    message suggesting tools are optional. A true "required" constraint would
    force a tool call anyway; our prompt-based workaround may not.
    """
    client = new_llm_client("claude-4.5-opus@anthropic")
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

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) > 0, (
        f"tool_choice='required' but model responded with text only: "
        f"{response.choices[0].message.content!r}"
    )
