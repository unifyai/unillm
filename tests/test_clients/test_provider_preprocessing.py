"""Tests for provider-specific preprocessing, particularly Anthropic caching."""

import json

from pydantic import BaseModel

from unillm.clients.provider_preprocessing import (
    CACHE_CONTROL_EPHEMERAL,
    THINKING_COMPLIANCE_CONTEXT_HEADER,
    THINKING_COMPLIANCE_CONTEXT_FOOTER,
    TOOL_CHOICE_REQUIRED_INSTRUCTION,
    _apply_anthropic_caching,
    _apply_thinking_compliance,
    _transform_tool_calls_to_context,
    apply_provider_preprocessing,
)


class TestApplyAnthropicCachingSystemMessage:
    """Tests for system message caching with static field handling."""

    def test_system_content_list_all_static_caches_last_item(self):
        """When all items are static, cache_control is added to the last item."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "abc0", "_static": True},
                        {"type": "text", "text": "abc1", "_static": True},
                        {"type": "text", "text": "abc2", "_static": True},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert "cache_control" not in content[1]
        assert content[2]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_system_content_list_stops_at_first_non_static(self):
        """Cache_control is added to the last static item before first static=False."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "abc0", "_static": True},
                        {"type": "text", "text": "abc1", "_static": True},
                        {"type": "text", "text": "abc2", "_static": True},
                        {"type": "text", "text": "abc3", "_static": False},
                        {"type": "text", "text": "abc4", "_static": True},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert "cache_control" not in content[1]
        assert content[2]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in content[3]
        assert "cache_control" not in content[4]

    def test_system_content_list_first_item_non_static_no_cache(self):
        """When first item is non-static, no cache_control is added."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "dynamic", "_static": False},
                        {"type": "text", "text": "abc1", "_static": True},
                        {"type": "text", "text": "abc2", "_static": True},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert "cache_control" not in content[1]
        assert "cache_control" not in content[2]

    def test_system_content_list_no_static_field_defaults_to_true(self):
        """Items without static field are treated as static=True."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "abc0"},
                        {"type": "text", "text": "abc1"},
                        {"type": "text", "text": "abc2", "_static": False},
                        {"type": "text", "text": "abc3"},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert content[1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in content[2]
        assert "cache_control" not in content[3]

    def test_system_content_list_all_non_static_no_cache(self):
        """When all items are non-static, no cache_control is added."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "dynamic1", "_static": False},
                        {"type": "text", "text": "dynamic2", "_static": False},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert "cache_control" not in content[1]

    def test_system_content_list_single_static_item(self):
        """Single static item gets cache_control."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "only one", "_static": True},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert content[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_system_content_list_single_non_static_item(self):
        """Single non-static item does not get cache_control."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "dynamic", "_static": False},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]

    def test_system_content_string_gets_cache_control_on_message(self):
        """When content is a string, cache_control is added to the message itself."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        msg = kw["messages"][0]
        assert msg["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_system_content_empty_list_no_cache(self):
        """Empty content list doesn't cause errors."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        msg = kw["messages"][0]
        # Falls through to string case since list is empty
        assert msg.get("cache_control") == CACHE_CONTROL_EPHEMERAL


class TestApplyAnthropicCachingTools:
    """Tests for tool caching."""

    def test_tools_get_cache_control_on_last_item(self):
        """Cache_control is added to the last tool."""
        kw = {
            "tools": [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": "Second tool"},
            ],
            "messages": [],
        }
        _apply_anthropic_caching(kw, ["tools"])

        assert "cache_control" not in kw["tools"][0]
        assert kw["tools"][1]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_empty_tools_no_error(self):
        """Empty tools list doesn't cause errors."""
        kw = {"tools": [], "messages": []}
        _apply_anthropic_caching(kw, ["tools"])
        # No assertion needed, just verify no error


class TestApplyAnthropicCachingUserMessages:
    """Tests for user message caching."""

    def test_user_content_list_caches_last_item(self):
        """Cache_control is added to the last item in user content list."""
        kw = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                },
            ],
        }
        _apply_anthropic_caching(kw, ["messages"])

        content = kw["messages"][0]["content"]
        assert "cache_control" not in content[0]
        assert content[1]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_user_content_string_gets_cache_control_on_message(self):
        """When content is a string, cache_control is added to the message itself."""
        kw = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                },
            ],
        }
        _apply_anthropic_caching(kw, ["messages"])

        msg = kw["messages"][0]
        assert msg["cache_control"] == CACHE_CONTROL_EPHEMERAL


class TestApplyAnthropicCachingMultipleLocations:
    """Tests for caching at multiple locations simultaneously."""

    def test_cache_system_and_messages(self):
        """Both system and user messages can be cached."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are helpful.",
                },
                {
                    "role": "user",
                    "content": "Hello!",
                },
            ],
        }
        _apply_anthropic_caching(kw, ["system", "messages"])

        assert kw["messages"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert kw["messages"][1]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_no_messages_no_error(self):
        """Missing messages doesn't cause errors."""
        kw = {}
        _apply_anthropic_caching(kw, ["system", "messages"])
        # No assertion needed, just verify no error


class TestApplyAnthropicCachingMultipleSystemMessages:
    """Cache breakpoint placement across multiple system messages.

    When multiple system messages exist (e.g. main prompt + runtime context +
    time context), the cache breakpoint must land on the last _static=True block
    across ALL system messages — not just within the last system message.

    This mirrors Unity's async tool loop where:
      - messages[0]: main prompt (list content with _static annotations)
      - messages[1]: runtime context (string, static per-loop)
      - messages[2]: time context (string, _static=False, changes every turn)
    """

    def test_breakpoint_on_static_block_not_last_dynamic_system_message(self):
        """Cache breakpoint should land on the last static content, not on a
        dynamic system message that happens to be last in the list.

        Reproduces the time-awareness KV cache bust: the time context system
        message is last, has string content, and changes every turn. The
        breakpoint must stay on the static prompt content in the first message.
        """
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant.",
                            "_static": True,
                        },
                        {
                            "type": "text",
                            "text": "Current time is 2:09 PM.",
                            "_static": False,
                        },
                    ],
                },
                {
                    "role": "system",
                    "content": "Runtime context: called by Actor.",
                },
                {
                    "role": "system",
                    "_static": False,
                    "content": "## Time Context\n- Started: 45s ago\n\n| tool | duration |\n| search | 2.1s |",
                },
                {"role": "user", "content": "hello"},
            ],
        }
        _apply_anthropic_caching(kw, ["system"])

        # The breakpoint must be on the static block in the first system message
        first_sys_content = kw["messages"][0]["content"]
        assert first_sys_content[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL

        # NOT on the dynamic time context (last system message)
        assert "cache_control" not in kw["messages"][2]


class TestInternalAnnotationStripping:
    """Underscore-prefixed keys are internal annotations (e.g. _static, _time_context).

    They must be stripped from messages and content blocks before reaching any
    provider API, regardless of provider.
    """

    def test_anthropic_strips_internal_annotations_from_content_blocks(self):
        """_static annotations on content blocks must not leak to Anthropic."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "static part", "_static": True},
                        {"type": "text", "text": "dynamic part", "_static": False},
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        }
        apply_provider_preprocessing(kw, "anthropic", prompt_caching=["system"])

        for msg in kw["messages"]:
            # No _-prefixed keys on messages
            assert not any(
                k.startswith("_") for k in msg
            ), f"Internal annotation leaked on message: {[k for k in msg if k.startswith('_')]}"
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    assert not any(k.startswith("_") for k in block), (
                        f"Internal annotation leaked on content block: "
                        f"{[k for k in block if k.startswith('_')]}"
                    )

    def test_openai_strips_internal_annotations_from_content_blocks(self):
        """_static annotations on content blocks must not leak to OpenAI."""
        kw = {
            "messages": [
                {
                    "role": "system",
                    "_custom_flag": True,
                    "content": [
                        {"type": "text", "text": "static part", "_static": True},
                        {"type": "text", "text": "dynamic part", "_static": False},
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        }
        apply_provider_preprocessing(kw, "openai")

        for msg in kw["messages"]:
            assert not any(
                k.startswith("_") for k in msg
            ), f"Internal annotation leaked on message: {[k for k in msg if k.startswith('_')]}"
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    assert not any(k.startswith("_") for k in block), (
                        f"Internal annotation leaked on content block: "
                        f"{[k for k in block if k.startswith('_')]}"
                    )


class TestAdaptiveThinking:
    def test_claude_opus_48_uses_adaptive_thinking_payload(self):
        kw = {
            "model": "anthropic/claude-opus-4-8",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hello"}],
        }

        apply_provider_preprocessing(kw, "anthropic")

        assert "reasoning_effort" not in kw
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["output_config"] == {"effort": "high"}

    def test_claude_opus_48_disables_thinking_for_response_format(self):
        kw = {
            "model": "anthropic/claude-opus-4-8",
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hello"}],
        }

        apply_provider_preprocessing(kw, "anthropic")

        assert "reasoning_effort" not in kw
        assert "thinking" not in kw
        assert "output_config" not in kw

    def test_other_anthropic_models_keep_legacy_reasoning_effort(self):
        kw = {
            "model": "anthropic/claude-opus-4-6",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hello"}],
        }

        apply_provider_preprocessing(kw, "anthropic")

        assert kw["reasoning_effort"] == "high"
        assert "extra_body" not in kw


class TestDeepSeekThinkingCompliance:
    class _Answer(BaseModel):
        answer: str

    _TOOL = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    def test_assistant_messages_get_empty_reasoning_content(self):
        kw = {
            "model": "deepseek/deepseek-v4-pro",
            "reasoning_effort": "high",
            "messages": [
                {"role": "user", "content": "call tool"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "tool", "arguments": "{}"},
                        },
                    ],
                },
            ],
        }

        apply_provider_preprocessing(kw, "deepseek")

        assert kw["messages"][1]["reasoning_content"] == ""

    def test_response_format_becomes_prompt_instruction(self):
        kw = {
            "model": "deepseek/deepseek-v4-pro",
            "response_format": self._Answer,
            "messages": [{"role": "user", "content": "hello"}],
        }

        apply_provider_preprocessing(kw, "deepseek")

        assert "response_format" not in kw
        assert kw["messages"][0]["role"] == "system"
        assert "valid JSON only" in kw["messages"][0]["content"]

    def test_required_tool_choice_is_downgraded_to_auto(self):
        kw = {
            "model": "deepseek/deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Call a tool"}],
            "tools": [self._TOOL],
            "tool_choice": "required",
        }

        apply_provider_preprocessing(kw, "deepseek")

        assert kw["tool_choice"] == "auto"
        assert kw["messages"][0]["role"] == "system"
        assert TOOL_CHOICE_REQUIRED_INSTRUCTION in kw["messages"][0]["content"]

    def test_explicit_tool_choice_is_downgraded_to_auto(self):
        kw = {
            "model": "deepseek/deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Call lookup"}],
            "tools": [self._TOOL],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }

        apply_provider_preprocessing(kw, "deepseek")

        assert kw["tool_choice"] == "auto"
        assert "lookup" in kw["messages"][0]["content"]


# A short stand-in for a real base64 screenshot (~200 chars).
# Production images are 100K+ chars; even this toy payload should never be
# serialized as text tokens.
_FAKE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" * 3

_IMAGE_BLOCK = {
    "type": "image_url",
    "image_url": {"url": f"data:image/png;base64,{_FAKE_B64}"},
}


def _make_image_tool_result_messages():
    """Build a minimal non-compliant assistant→tool sequence with an image.

    The assistant message has tool_calls but no thinking_blocks (non-compliant
    when extended thinking is enabled).  The tool result contains an image_url
    block — a screenshot returned by execute_code in CodeActActor.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Take a screenshot of the page."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps(
                            {
                                "code": "display(session.screenshot())",
                                "language": "python",
                            },
                        ),
                    },
                },
            ],
            # No thinking_blocks → non-compliant under extended thinking
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "name": "execute_code",
            "content": [
                {"type": "text", "text": "Screenshot captured:"},
                _IMAGE_BLOCK,
            ],
        },
    ]


class TestThinkingComplianceImageExtraction:
    """Images in tool results must not be JSON-serialized as text.

    When _transform_tool_calls_to_context collapses a non-compliant
    assistant+tool sequence into a user context message, any image_url
    blocks in tool results should be extracted and reattached as native
    image content blocks — the same pattern _convert_prefill_to_system_message
    already uses.  Serializing base64 data as text tokens causes massive
    context window bloat.
    """

    def test_image_blocks_not_serialized_as_text(self):
        """The base64 payload must NOT appear as text in the context message."""
        messages = _make_image_tool_result_messages()
        result = _transform_tool_calls_to_context(messages)

        # The system and user messages pass through unchanged
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

        # The third message is the collapsed context (was assistant+tool)
        ctx_msg = result[2]
        assert ctx_msg["role"] == "user"

        # Flatten all text content to check for leaked base64
        content = ctx_msg["content"]
        if isinstance(content, str):
            all_text = content
        elif isinstance(content, list):
            all_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type") != "image_url"
            )
        else:
            all_text = str(content)

        assert _FAKE_B64 not in all_text, (
            "Base64 image data was serialized as text in the context message. "
            "Image blocks should be extracted and preserved as native image_url "
            "content blocks, not dumped into a JSON text string."
        )

    def test_image_blocks_preserved_as_native_content(self):
        """Extracted images must appear as native image_url blocks."""
        messages = _make_image_tool_result_messages()
        result = _transform_tool_calls_to_context(messages)

        ctx_msg = result[2]
        content = ctx_msg["content"]

        # Content should be a list with at least one image_url block
        assert isinstance(content, list), (
            "Context message content should be a list (mixed text + image blocks), "
            f"got {type(content).__name__}"
        )

        image_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        assert (
            len(image_blocks) >= 1
        ), "Expected at least one native image_url block in the context message"

        # The image data should match the original
        assert image_blocks[0] == _IMAGE_BLOCK

    def test_no_images_unchanged_behavior(self):
        """When no images are present, the function should work as before."""
        messages = [
            {"role": "user", "content": "Do something."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_xyz",
                "name": "search",
                "content": "Found 3 results.",
            },
        ]
        result = _transform_tool_calls_to_context(messages)

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Do something."

        ctx_msg = result[1]
        assert ctx_msg["role"] == "user"
        # Plain string content (no images to extract)
        assert isinstance(ctx_msg["content"], str)
        assert THINKING_COMPLIANCE_CONTEXT_HEADER in ctx_msg["content"]
        assert THINKING_COMPLIANCE_CONTEXT_FOOTER in ctx_msg["content"]

    def test_apply_thinking_compliance_extracts_images(self):
        """End-to-end: _apply_thinking_compliance should not leak base64."""
        messages = _make_image_tool_result_messages()
        result = _apply_thinking_compliance(messages)

        # Find the context message (the collapsed one)
        ctx_msgs = [
            m
            for m in result
            if m.get("role") == "user"
            and isinstance(m.get("content"), (str, list))
            and (
                (
                    isinstance(m["content"], str)
                    and THINKING_COMPLIANCE_CONTEXT_HEADER in m["content"]
                )
                or (
                    isinstance(m["content"], list)
                    and any(
                        isinstance(b, dict)
                        and THINKING_COMPLIANCE_CONTEXT_HEADER in b.get("text", "")
                        for b in m["content"]
                    )
                )
            )
        ]
        assert (
            len(ctx_msgs) == 1
        ), f"Expected exactly one context message, found {len(ctx_msgs)}"

        content = ctx_msgs[0]["content"]
        if isinstance(content, str):
            all_text = content
        else:
            all_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type") != "image_url"
            )

        assert (
            _FAKE_B64 not in all_text
        ), "Base64 image data leaked as text through _apply_thinking_compliance"
