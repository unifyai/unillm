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

import pytest

import unillm
from ..settings import SETTINGS
from unillm.clients.provider_postprocessing import (
    MALFORMED_TOOL_ARGUMENTS_RETRY_NUDGE,
    RETRY_REASON_INVALID_TOOL_NAME,
    RETRY_REASON_MALFORMED_TOOL_ARGUMENTS,
    RETRY_REASON_TOOL_CHOICE_REQUIRED,
    build_retry_kw,
    build_tool_choice_required_retry_nudge,
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


def _assert_no_retry_messages_in_history(messages: list) -> None:
    """Assert that retry tool-result error messages are not in the history."""
    for msg in messages:
        content = msg.get("content", "")
        if msg.get("role") == "tool" and isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "error" in parsed:
                    raise AssertionError(
                        f"Retry tool result error leaked into history: {content!r}",
                    )
            except (json.JSONDecodeError, TypeError):
                pass


def test_anthropic_no_tool_name_constraint():
    """
    Anthropic does NOT constrain tool names to schema.

    When tool_choice="required" and the prompt mentions tool_b,
    Claude calls tool_b even though only tool_a is in the schema.

    Our retry logic detects this and retries, ultimately returning tool_a.
    The intermediate retry messages should NOT appear in the client's history.
    """
    client = unillm.Unify(
        "claude-4.8-opus@anthropic",
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
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

    # No retry messages should be in the history
    _assert_no_retry_messages_in_history(messages)


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
        "claude-4.8-opus@anthropic",
        cache=SETTINGS.UNILLM_CACHE,
        cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
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

    # No retry messages should be in the history
    _assert_no_retry_messages_in_history(messages)


def test_build_retry_kw_identifies_correct_invalid_tool():
    """
    When multiple tool calls are returned and the FIRST one is valid but a
    LATER one is invalid, the retry should produce per-tool-call error
    results that correctly identify which tool is invalid.
    """
    # Create a mock response with multiple tool calls:
    # - tool_calls[0] = "valid_tool" (valid)
    # - tool_calls[1] = "invalid_tool" (invalid)
    mock_tool_call_valid = MagicMock()
    mock_tool_call_valid.function.name = "valid_tool"
    mock_tool_call_valid.id = "call_valid"

    mock_tool_call_invalid = MagicMock()
    mock_tool_call_invalid.function.name = "invalid_tool"
    mock_tool_call_invalid.id = "call_invalid"

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

    messages = retry_kw["messages"]

    # The assistant message should preserve tool_calls for context
    assistant_msg = messages[1]  # index 0 is original user message
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"] is not None

    # There should be two tool result messages (one per tool call)
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert (
        len(tool_results) == 2
    ), f"Expected 2 tool results but got {len(tool_results)}"

    # Find the tool result for the invalid tool
    invalid_result = next(
        m for m in tool_results if m["tool_call_id"] == "call_invalid"
    )
    invalid_error = json.loads(invalid_result["content"])
    assert (
        "invalid_tool" in invalid_error["error"]["content"]
    ), f"Tool result should mention 'invalid_tool': {invalid_error}"
    assert (
        "available_tools" in invalid_error["error"]
    ), f"Tool result should include available_tools: {invalid_error}"
    assert "valid_tool" in invalid_error["error"]["available_tools"]

    # The valid tool result should say it was not executed
    valid_result = next(m for m in tool_results if m["tool_call_id"] == "call_valid")
    valid_error = json.loads(valid_result["content"])
    assert (
        "not executed" in valid_error["error"]["content"].lower()
    ), f"Valid tool result should say not executed: {valid_error}"


def test_build_retry_kw_identifies_multiple_invalid_tools():
    """
    When multiple tool calls are invalid, each gets its own tool result
    identifying that specific tool as not callable.
    """
    # Create a mock response with multiple tool calls:
    # - tool_calls[0] = "valid_tool" (valid)
    # - tool_calls[1] = "bad_tool_1" (invalid)
    # - tool_calls[2] = "bad_tool_2" (invalid)
    mock_tc_valid = MagicMock()
    mock_tc_valid.function.name = "valid_tool"
    mock_tc_valid.id = "call_valid"

    mock_tc_bad1 = MagicMock()
    mock_tc_bad1.function.name = "bad_tool_1"
    mock_tc_bad1.id = "call_bad1"

    mock_tc_bad2 = MagicMock()
    mock_tc_bad2.function.name = "bad_tool_2"
    mock_tc_bad2.id = "call_bad2"

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

    # Should have 3 tool result messages (one per tool call)
    tool_results = [m for m in retry_kw["messages"] if m.get("role") == "tool"]
    assert (
        len(tool_results) == 3
    ), f"Expected 3 tool results but got {len(tool_results)}"

    # Each invalid tool result should mention its own tool name
    bad1_result = next(m for m in tool_results if m["tool_call_id"] == "call_bad1")
    bad1_error = json.loads(bad1_result["content"])
    assert (
        "bad_tool_1" in bad1_error["error"]["content"]
    ), f"Tool result for bad_tool_1 should mention it: {bad1_error}"
    assert "not callable" in bad1_error["error"]["content"]
    assert "valid_tool" in bad1_error["error"]["available_tools"]

    bad2_result = next(m for m in tool_results if m["tool_call_id"] == "call_bad2")
    bad2_error = json.loads(bad2_result["content"])
    assert (
        "bad_tool_2" in bad2_error["error"]["content"]
    ), f"Tool result for bad_tool_2 should mention it: {bad2_error}"
    assert "not callable" in bad2_error["error"]["content"]
    assert "valid_tool" in bad2_error["error"]["available_tools"]

    # Valid tool result should say not executed
    valid_result = next(m for m in tool_results if m["tool_call_id"] == "call_valid")
    valid_error = json.loads(valid_result["content"])
    assert "not executed" in valid_error["error"]["content"].lower()


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


@pytest.mark.parametrize("provider", ["openai"])
def test_soft_forced_tool_choice_retries_text_only_response(provider):
    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = "I can answer directly."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    needs_retry, retry_reason = check_needs_postprocessing(
        response=mock_response,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[TOOL_A],
    )

    assert needs_retry
    assert retry_reason == RETRY_REASON_TOOL_CHOICE_REQUIRED


@pytest.mark.parametrize("provider", ["openai"])
def test_soft_forced_required_retries_final_answer_when_tools_remain(provider):
    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = "call 1"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    needs_retry, retry_reason = check_needs_postprocessing(
        response=mock_response,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[TOOL_A],
        request_messages=[
            {"role": "user", "content": "Call tool_a, then answer."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ],
    )

    assert needs_retry
    assert retry_reason == RETRY_REASON_TOOL_CHOICE_REQUIRED


@pytest.mark.parametrize("provider", ["openai"])
def test_soft_forced_required_allows_final_answer_after_tools_removed(provider):
    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = "2"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    needs_retry, retry_reason = check_needs_postprocessing(
        response=mock_response,
        provider=provider,
        original_tool_choice="required",
        reasoning_effort=None,
        tools=[],
        request_messages=[
            {"role": "user", "content": "Call tool_a, then answer."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ],
    )

    assert not needs_retry
    assert retry_reason is None


@pytest.mark.parametrize("provider", ["openai"])
def test_soft_forced_tool_choice_retries_wrong_tool_response(provider):
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    needs_retry, retry_reason = check_needs_postprocessing(
        response=mock_response,
        provider=provider,
        original_tool_choice={"type": "function", "function": {"name": "tool_a"}},
        reasoning_effort=None,
        tools=[TOOL_A, TOOL_SEARCH],
    )

    assert needs_retry
    assert retry_reason == RETRY_REASON_TOOL_CHOICE_REQUIRED


def test_build_retry_kw_no_tools_returns_tool_result_with_text_instruction():
    """
    Case B: When there are zero valid tools and no response_format schema,
    the tool result error should instruct the model to respond with text
    content only. It must NOT include available_tools or json_schema keys.
    """
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search_messages"
    mock_tool_call.id = "call_search"

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

    # Should have exactly one tool result
    tool_results = [m for m in retry_kw["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 1

    error_obj = json.loads(tool_results[0]["content"])
    error_inner = error_obj["error"]

    # Should instruct text-only response
    assert (
        "text content only" in error_inner["content"].lower()
    ), f"Should instruct text-only response: {error_inner}"
    # Must NOT include available_tools (no tools exist)
    assert (
        "available_tools" not in error_inner
    ), f"Should not include available_tools when no tools exist: {error_inner}"
    # Must NOT include json_schema (no schema set)
    assert (
        "json_schema" not in error_inner
    ), f"Should not include json_schema when no schema is set: {error_inner}"


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


def _make_text_only_response(content):
    """Build a mock response where the model returned text without tool calls."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_choice_retry_user_nudge(retry_kw: dict) -> str | None:
    """Return the tool-choice-required retry nudge appended as a user message."""
    original_len = len(_VALID_TOOL_KW["messages"])
    appended = retry_kw["messages"][original_len:]
    user_messages = [m for m in appended if m.get("role") == "user"]
    if not user_messages:
        return None
    assert len(user_messages) == 1, appended
    return user_messages[0].get("content")


def test_build_tool_choice_required_retry_nudge_without_rejected_content():
    nudge = build_tool_choice_required_retry_nudge(None)
    assert "FAILED" in nudge
    assert "ZERO effect" in nudge
    assert "Do NOT call `wait`" in nudge
    assert "tool_choice is set to 'required'" in nudge
    assert "for reference only" not in nudge


def test_build_tool_choice_required_retry_nudge_quotes_rejected_content():
    nudge = build_tool_choice_required_retry_nudge(
        "Check your inbox for the email I just sent.",
    )
    assert "for reference only — it was NOT sent" in nudge
    assert "> Check your inbox for the email I just sent." in nudge
    assert "call the appropriate send tool NOW" in nudge


def test_build_retry_kw_tool_choice_required_uses_user_nudge_not_assistant():
    resp = _make_text_only_response(
        "Check your inbox for the email I just sent and reply with your guess.",
    )
    retry_kw = build_retry_kw(
        kw=_VALID_TOOL_KW,
        response=resp,
        retry_reason=RETRY_REASON_TOOL_CHOICE_REQUIRED,
    )

    original_len = len(_VALID_TOOL_KW["messages"])
    appended = retry_kw["messages"][original_len:]
    assert len(appended) == 1
    assert appended[0]["role"] == "user"
    assert _assistant_content_from_retry(retry_kw) is None

    nudge = _tool_choice_retry_user_nudge(retry_kw)
    assert nudge is not None
    assert "ZERO effect" in nudge
    assert (
        "> Check your inbox for the email I just sent and reply with your guess."
        in nudge
    )


def test_build_retry_kw_tool_choice_required_rejects_whitespace_content():
    """Whitespace-only rejected content must not be quoted in the retry nudge."""
    for ws in _WHITESPACE_CONTENTS:
        resp = _make_text_only_response(content=ws)
        retry_kw = build_retry_kw(
            kw=_VALID_TOOL_KW,
            response=resp,
            retry_reason=RETRY_REASON_TOOL_CHOICE_REQUIRED,
        )
        assert _assistant_content_from_retry(retry_kw) is None
        nudge = _tool_choice_retry_user_nudge(retry_kw)
        assert nudge is not None
        assert "for reference only" not in nudge
        assert "> " not in nudge


# ---------------------------------------------------------------------------
# Case C: no tools available but response_format schema is set
# ---------------------------------------------------------------------------


def test_build_retry_kw_no_tools_with_schema_returns_json_schema():
    """
    Case C: When there are zero valid tools but a response_format Pydantic
    model is set, the tool result error should include the json_schema and
    instruct the model to return JSON only.
    """
    from pydantic import BaseModel as PydanticBaseModel

    class SummaryOutput(PydanticBaseModel):
        summary: str
        topics: list

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "analyse_data"
    mock_tool_call.id = "call_analyse"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    kw = {
        "messages": [{"role": "user", "content": "Summarise the data."}],
        "tools": [],
        "response_format": SummaryOutput,  # Pydantic model triggers Case C
    }

    retry_kw = build_retry_kw(
        kw=kw,
        response=mock_response,
        retry_reason=RETRY_REASON_INVALID_TOOL_NAME,
    )

    tool_results = [m for m in retry_kw["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 1

    error_obj = json.loads(tool_results[0]["content"])
    error_inner = error_obj["error"]

    # Should instruct JSON-only response
    assert (
        "json" in error_inner["content"].lower()
    ), f"Should instruct JSON response: {error_inner}"
    # Must include json_schema with the Pydantic schema
    assert (
        "json_schema" in error_inner
    ), f"Should include json_schema for Case C: {error_inner}"
    schema = error_inner["json_schema"]
    assert "summary" in schema.get(
        "properties",
        {},
    ), f"Schema should contain 'summary' property: {schema}"
    assert "topics" in schema.get(
        "properties",
        {},
    ), f"Schema should contain 'topics' property: {schema}"
    # Must NOT include available_tools
    assert (
        "available_tools" not in error_inner
    ), f"Should not include available_tools for Case C: {error_inner}"


# ---------------------------------------------------------------------------
# Assistant message preserves tool_calls
# ---------------------------------------------------------------------------


def test_build_retry_kw_assistant_message_preserves_tool_calls():
    """
    The retry assistant message should preserve the original tool_calls
    so the model has context about what it attempted.
    """
    mock_tc = MagicMock()
    mock_tc.function.name = "nonexistent_tool"
    mock_tc.id = "call_123"

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tc]
    mock_message.content = "Let me call the tool."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    kw = {
        "messages": [{"role": "user", "content": "Do it."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "real_tool",
                    "description": "A real tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ],
    }

    retry_kw = build_retry_kw(
        kw=kw,
        response=mock_response,
        retry_reason=RETRY_REASON_INVALID_TOOL_NAME,
    )

    # Find the assistant message
    assistant_msgs = [m for m in retry_kw["messages"] if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1

    assistant_msg = assistant_msgs[0]
    # tool_calls should be preserved (not stripped)
    assert (
        assistant_msg.get("tool_calls") is not None
    ), "Assistant message should preserve tool_calls"
    assert len(assistant_msg["tool_calls"]) == 1
    # Content should be preserved (non-whitespace)
    assert assistant_msg["content"] == "Let me call the tool."


# --------------------------------------------------------------------------- #
#  Truncated / malformed tool-call arguments                                   #
# --------------------------------------------------------------------------- #
#
# Observed in production twice: generation degenerated immediately after a key
# inside a nested free-form object and emitted whitespace until the output-token
# cap, ~8 minutes later. The OpenAI Responses bridge reports the truncation as
# `status: "incomplete"` on the raw response but still hands the transformed
# choice a `finish_reason: "tool_calls"`, so the only reliable signal left is
# that the arguments payload does not parse.

# The real shape, shortened: a valid prefix, then whitespace, never closed.
_TRUNCATED_ARGS = (
    '{"function_name":"primitives.workspace_email.list_messages",'
    '"call_kwargs":{"max_results":' + "\n  " * 400
)


def _tool_call_response(
    arguments,
    *,
    finish_reason="tool_calls",
    name="execute_function",
):
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = name
    mock_tool_call.function.arguments = arguments

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter"])
def test_truncated_tool_arguments_are_retried_for_every_provider(provider):
    """Unparseable arguments cannot be dispatched by anyone downstream."""

    needs_retry, retry_reason = check_needs_postprocessing(
        response=_tool_call_response(_TRUNCATED_ARGS),
        provider=provider,
        original_tool_choice=None,
        reasoning_effort=None,
        tools=[TOOL_A],
    )

    assert needs_retry
    assert retry_reason == RETRY_REASON_MALFORMED_TOOL_ARGUMENTS


def test_length_capped_turn_carrying_tool_calls_is_retried():
    """A turn cut off by the token cap was cut off mid-call, whatever it parses to."""

    needs_retry, retry_reason = check_needs_postprocessing(
        response=_tool_call_response('{"function_name":"x"}', finish_reason="length"),
        provider="openai",
        original_tool_choice=None,
        reasoning_effort=None,
        tools=[TOOL_A],
    )

    assert needs_retry
    assert retry_reason == RETRY_REASON_MALFORMED_TOOL_ARGUMENTS


def test_well_formed_tool_arguments_are_not_flagged_as_malformed():
    needs_retry, retry_reason = check_needs_postprocessing(
        response=_tool_call_response('{"query": "hello"}', name="search"),
        provider="openai",
        original_tool_choice=None,
        reasoning_effort=None,
        tools=[TOOL_SEARCH],
    )

    assert retry_reason != RETRY_REASON_MALFORMED_TOOL_ARGUMENTS
    assert not needs_retry


def test_malformed_retry_nudges_without_replaying_the_broken_turn():
    """An unanswered tool_call would make the retry request itself invalid."""

    kw = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "Summarise my last five emails."}],
        "tools": [TOOL_A],
    }

    retry_kw = build_retry_kw(
        kw=kw,
        response=_tool_call_response(_TRUNCATED_ARGS),
        retry_reason=RETRY_REASON_MALFORMED_TOOL_ARGUMENTS,
    )

    assert not [m for m in retry_kw["messages"] if m.get("role") == "assistant"]
    assert retry_kw["messages"][-1] == {
        "role": "user",
        "content": MALFORMED_TOOL_ARGUMENTS_RETRY_NUDGE,
    }
    # The original request is otherwise untouched.
    assert retry_kw["messages"][0] == kw["messages"][0]
    assert kw["messages"] == [
        {"role": "user", "content": "Summarise my last five emails."},
    ]
