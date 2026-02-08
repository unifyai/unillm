"""
Test that Anthropic constrains tool names to the schema.

FINDING: Anthropic does NOT constrain tool names to the schema.
The model can call tools mentioned in the prompt even if they're not in the
`tools` array. This happens both with and without extended thinking.

The fix is implemented via retry logic in provider_postprocessing.py.
When the model calls a tool not in the schema, we detect this and retry
with a helpful error message. The retry messages are cleaned up so they
don't appear in the client's message history.
"""

import json
from unittest.mock import MagicMock

import unillm
from unillm.clients.provider_postprocessing import (
    RETRY_REASON_INVALID_TOOL_NAME,
    build_retry_kw,
    check_needs_postprocessing,
)


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

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Semantic search over data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


def _assert_no_retry_nudge_in_history(messages: list) -> None:
    """Assert that retry nudge messages are not present in the message history."""
    # The nudge message starts with "You attempted to call"
    nudge_prefix = "You attempted to call"

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and nudge_prefix in content:
            raise AssertionError(
                f"Retry nudge message leaked into history: {content!r}",
            )


def test_anthropic_no_tool_name_constraint():
    """
    Anthropic does NOT constrain tool names to schema.

    When tool_choice="required" and the prompt mentions tool_b,
    Claude calls tool_b even though only tool_a is in the schema.

    Our retry logic detects this and retries, ultimately returning tool_a.
    The intermediate retry messages should NOT appear in the client's history.
    """
    client = unillm.Unify(
        "claude-4.5-opus@anthropic",
        cache=True,
        stateful=True,  # Enable stateful to test history cleanup
    )

    system_prompt = """Tools:
- tool_a
- tool_b

You MUST call tool_b. Do not call tool_a.
"""

    response = client.generate(
        system_message=system_prompt,
        messages=[{"role": "user", "content": "Do it."}],
        tools=[TOOL_A],  # Only tool_a in schema
        tool_choice="required",
        return_full_completion=True,
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls and len(tool_calls) > 0

    called = tool_calls[0].function.name
    assert called == "tool_a", (
        f"Called '{called}' which is NOT in schema. "
        f"Anthropic does not constrain tool names. Fix: use strict=true."
    )

    # Verify intermediate retry messages are NOT in the client's history
    messages = client._messages

    # Should only have: system message, user message, compliant assistant response
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert (
        len(user_messages) == 1
    ), f"Expected 1 user message but found {len(user_messages)}: {user_messages}"
    assert (
        user_messages[0]["content"] == "Do it."
    ), f"Unexpected user message: {user_messages[0]['content']!r}"

    # The assistant message should have the compliant tool call
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    assert (
        len(assistant_messages) == 1
    ), f"Expected 1 assistant message but found {len(assistant_messages)}"
    assert (
        assistant_messages[0].get("tool_calls") is not None
    ), f"Assistant message should have tool_calls: {assistant_messages[0]}"

    # No retry nudge content should be in the history
    _assert_no_retry_nudge_in_history(messages)


def test_anthropic_no_tool_name_constraint_with_thinking():
    """
    Anthropic does NOT constrain tool names even with extended thinking.

    Claude picks tool based on task semantics. Prompt describes both search
    and filter, but only search is in schema. For an exact-match task,
    Claude prefers filter - and calls it even though it's not available.

    Our retry logic detects this and retries, ultimately returning search.
    The intermediate retry messages should NOT appear in the client's history.
    """
    client = unillm.Unify(
        "claude-4.5-opus@anthropic",
        cache=True,
        stateful=True,  # Enable stateful to test history cleanup
    )

    tools_json = json.dumps(
        {
            "search": "(query: str) - semantic search",
            "filter": "(expr: str) - exact match filter",
        },
        indent=2,
    )

    system_prompt = f"""You are an assistant with access to tools.

Tools (name → signature):
{tools_json}

Tool selection guidance:
- Use `search` for semantic/fuzzy queries
- Use `filter` for exact matches (names, emails, IDs)

Example:
- "Find John Smith" → filter(expr="name == 'John Smith'")
- "Find someone who works in finance" → search(query="works in finance")
"""

    response = client.generate(
        system_message=system_prompt,
        messages=[{"role": "user", "content": "Find Alice Smith."}],
        tools=[TOOL_SEARCH],  # Only search in schema, not filter
        tool_choice="required",
        reasoning_effort="high",
        return_full_completion=True,
    )

    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls or len(tool_calls) == 0:
        return  # Acceptable with tool_choice="auto" (downgraded)

    called = tool_calls[0].function.name
    assert called == "search", (
        f"Called '{called}' which is NOT in schema. "
        f"Anthropic does not constrain tool names even with thinking. "
        f"Fix: use strict=true."
    )

    # Verify intermediate retry messages are NOT in the client's history
    messages = client._messages

    # Should only have: system message, user message, compliant assistant response
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert (
        len(user_messages) == 1
    ), f"Expected 1 user message but found {len(user_messages)}: {user_messages}"
    assert (
        user_messages[0]["content"] == "Find Alice Smith."
    ), f"Unexpected user message: {user_messages[0]['content']!r}"

    # The assistant message should have the compliant tool call
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    assert (
        len(assistant_messages) == 1
    ), f"Expected 1 assistant message but found {len(assistant_messages)}"
    assert (
        assistant_messages[0].get("tool_calls") is not None
    ), f"Assistant message should have tool_calls: {assistant_messages[0]}"

    # No retry nudge content should be in the history
    _assert_no_retry_nudge_in_history(messages)


def test_build_retry_kw_identifies_correct_invalid_tool():
    """
    Bug: build_retry_kw always uses tool_calls[0] in error message.

    When multiple tool calls are returned and the FIRST one is valid but a
    LATER one is invalid, the retry nudge message incorrectly reports the
    first (valid) tool as the problem.

    This test verifies the error message correctly identifies the actual
    invalid tool, not just the first tool call.
    """
    # Create a mock response with multiple tool calls:
    # - tool_calls[0] = "valid_tool" (valid)
    # - tool_calls[1] = "invalid_tool" (invalid)
    mock_tool_call_valid = MagicMock()
    mock_tool_call_valid.function.name = "valid_tool"

    mock_tool_call_invalid = MagicMock()
    mock_tool_call_invalid.function.name = "invalid_tool"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call_valid, mock_tool_call_invalid]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Original request had only "valid_tool" in schema
    kw = {
        "messages": [{"role": "user", "content": "Do it."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "valid_tool",
                    "description": "A valid tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ],
    }

    # Build the retry request
    retry_kw = build_retry_kw(
        kw=kw,
        response=mock_response,
        retry_reason=RETRY_REASON_INVALID_TOOL_NAME,
    )

    # Find the nudge message
    nudge_message = retry_kw["messages"][-1]
    assert nudge_message["role"] == "user"

    # The nudge should mention "invalid_tool", NOT "valid_tool"
    nudge_content = nudge_message["content"]
    assert (
        "invalid_tool" in nudge_content
    ), f"Retry nudge should mention 'invalid_tool' but got: {nudge_content!r}"
    assert (
        "valid_tool" not in nudge_content.split("'")[1]
    ), f"Retry nudge incorrectly reports 'valid_tool' as invalid: {nudge_content!r}"


def test_build_retry_kw_identifies_multiple_invalid_tools():
    """
    When multiple tool calls are invalid, the retry message should list all of them.
    """
    # Create a mock response with multiple tool calls:
    # - tool_calls[0] = "valid_tool" (valid)
    # - tool_calls[1] = "bad_tool_1" (invalid)
    # - tool_calls[2] = "bad_tool_2" (invalid)
    mock_tc_valid = MagicMock()
    mock_tc_valid.function.name = "valid_tool"

    mock_tc_bad1 = MagicMock()
    mock_tc_bad1.function.name = "bad_tool_1"

    mock_tc_bad2 = MagicMock()
    mock_tc_bad2.function.name = "bad_tool_2"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tc_valid, mock_tc_bad1, mock_tc_bad2]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Original request had only "valid_tool" in schema
    kw = {
        "messages": [{"role": "user", "content": "Do it."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "valid_tool",
                    "description": "A valid tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ],
    }

    # Build the retry request
    retry_kw = build_retry_kw(
        kw=kw,
        response=mock_response,
        retry_reason=RETRY_REASON_INVALID_TOOL_NAME,
    )

    # Find the nudge message
    nudge_message = retry_kw["messages"][-1]
    nudge_content = nudge_message["content"]

    # Should mention BOTH invalid tools
    assert (
        "bad_tool_1" in nudge_content
    ), f"Retry nudge should mention 'bad_tool_1' but got: {nudge_content!r}"
    assert (
        "bad_tool_2" in nudge_content
    ), f"Retry nudge should mention 'bad_tool_2' but got: {nudge_content!r}"
    # Should NOT mention valid_tool as invalid
    assert (
        "valid_tool" in nudge_content.split("tools currently available are")[1]
    ), f"Retry nudge should list 'valid_tool' as available: {nudge_content!r}"
    # Should use plural form
    assert (
        "they are not callable on this turn" in nudge_content
    ), f"Retry nudge should use plural form but got: {nudge_content!r}"


# ---------------------------------------------------------------------------
# Empty-tools edge case: tools=[] but model returns tool_calls
# ---------------------------------------------------------------------------


def test_check_needs_postprocessing_detects_tool_calls_with_empty_tools():
    """
    When tools=[] (no tools available this turn) and the model returns
    tool_calls, check_needs_postprocessing should flag a retry.

    BUG: The current guard `if msg.tool_calls and tools:` treats [] as
    falsy, so tool calls against an empty schema slip through undetected.
    """
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "filter_messages"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    needs_retry, retry_reason = check_needs_postprocessing(
        response=mock_response,
        provider="anthropic",
        original_tool_choice=None,
        reasoning_effort=None,
        tools=[],  # Explicitly empty — no tools available this turn
    )

    assert needs_retry, (
        "check_needs_postprocessing should flag a retry when tools=[] "
        "but the model returned tool_calls"
    )
    assert retry_reason == RETRY_REASON_INVALID_TOOL_NAME


def test_build_retry_kw_no_tools_nudge_says_respond_with_text():
    """
    When there are zero valid tools, the retry nudge should tell the model
    to respond with text content — not ask it to "select from the available
    tools" when there are none.

    BUG: The current nudge says "The tools currently available are: (none).
    Please select one of the available tools." which is contradictory.
    """
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search_messages"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    kw = {
        "messages": [{"role": "user", "content": "What topics were discussed?"}],
        "tools": [],  # No tools available
    }

    retry_kw = build_retry_kw(
        kw=kw,
        response=mock_response,
        retry_reason=RETRY_REASON_INVALID_TOOL_NAME,
    )

    nudge_content = retry_kw["messages"][-1]["content"]

    # The nudge must NOT ask the model to select a tool when none exist
    assert (
        "select" not in nudge_content.lower() or "no tools" in nudge_content.lower()
    ), f"Nudge asks model to select a tool when none are available: {nudge_content!r}"
    # It SHOULD instruct the model to respond with text
    assert (
        "text" in nudge_content.lower() or "content" in nudge_content.lower()
    ), f"Nudge should tell model to respond with text, but got: {nudge_content!r}"


# ---------------------------------------------------------------------------
# Whitespace-only assistant content in retry messages
# ---------------------------------------------------------------------------

_WHITESPACE_CONTENTS = ["\n\n", "  ", "\t\n", " \n "]


def _make_invalid_tool_response(content):
    """Build a mock response where the model called an invalid tool."""
    tc = MagicMock()
    tc.function.name = "json_tool_call"

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


_VALID_TOOL_KW = {
    "messages": [{"role": "user", "content": "Do it."}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "Wait.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ],
}


def _assistant_content_from_retry(retry_kw: dict) -> str | None:
    """Extract the assistant message content from the retry messages."""
    for m in retry_kw["messages"]:
        if m.get("role") == "assistant":
            return m.get("content")
    return None


def test_build_retry_kw_rejects_whitespace_only_assistant_content():
    """
    Bug: build_retry_kw includes msg.content verbatim in the retry assistant
    message. When Claude responds with tool_calls, content is often just
    "\\n\\n" (whitespace-only). Including this in the retry messages causes
    Anthropic to reject the request with:
        "messages: text content blocks must contain non-whitespace text"

    The assistant message content in the retry must either be None or
    contain non-whitespace characters.
    """
    for ws in _WHITESPACE_CONTENTS:
        resp = _make_invalid_tool_response(content=ws)
        retry_kw = build_retry_kw(
            kw=_VALID_TOOL_KW,
            response=resp,
            retry_reason=RETRY_REASON_INVALID_TOOL_NAME,
        )
        assistant_content = _assistant_content_from_retry(retry_kw)
        assert assistant_content is None or assistant_content.strip(), (
            f"Retry assistant message has whitespace-only content {assistant_content!r} "
            f"(from msg.content={ws!r}). Anthropic rejects this with: "
            f"'messages: text content blocks must contain non-whitespace text'"
        )


def test_build_retry_kw_tool_choice_required_rejects_whitespace_content():
    """Same bug for the tool_choice_required (default) retry path."""
    for ws in _WHITESPACE_CONTENTS:
        resp = _make_invalid_tool_response(content=ws)
        # Use default retry_reason (tool_choice_required path)
        retry_kw = build_retry_kw(
            kw=_VALID_TOOL_KW,
            response=resp,
            retry_reason=None,
        )
        assistant_content = _assistant_content_from_retry(retry_kw)
        assert assistant_content is None or assistant_content.strip(), (
            f"Retry assistant message has whitespace-only content {assistant_content!r} "
            f"(from msg.content={ws!r}). Anthropic rejects this with: "
            f"'messages: text content blocks must contain non-whitespace text'"
        )
