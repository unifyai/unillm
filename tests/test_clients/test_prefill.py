from .helpers import new_llm_client
from unillm.clients.provider_preprocessing import (
    apply_provider_preprocessing,
    _convert_prefill_to_system_message,
    _has_prefilled_assistant_before_real,
    _is_non_compliant_assistant_tool_call,
    _has_non_compliant_tool_calls,
    _transform_tool_calls_to_context,
    _apply_thinking_compliance,
    _combine_adjacent_user_messages,
    THINKING_COMPLIANCE_CONTEXT_HEADER,
    THINKING_COMPLIANCE_CONTEXT_FOOTER,
)


class TestCombineAdjacentUserMessagesImagePreservation:
    """
    Tests for preserving image blocks when combining adjacent user messages.

    Image blocks should be kept as native blocks in the combined message,
    not JSON-serialized, to maintain their semantic meaning for the model.
    """

    def test_combine_adjacent_user_messages_preserves_url_images(self):
        """
        URL image blocks should be preserved as native blocks when combining
        adjacent user messages, not JSON-serialized into text.
        """
        url_image = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"},
        }
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    url_image,
                ],
            },
        ]

        result, combined = _combine_adjacent_user_messages(messages)

        assert combined is True
        assert len(result) == 1
        assert result[0]["role"] == "user"

        # Content should be a list with text block + preserved image block
        content = result[0]["content"]
        assert isinstance(content, list)

        # Find the image block - should be preserved as native block
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert image_blocks[0] == url_image

        # Text should be JSON-dumped in a text block
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert "hello" in text_blocks[0]["text"]

    def test_combine_adjacent_user_messages_preserves_base64_images(self):
        """
        Base64 image blocks should be preserved as native blocks when combining
        adjacent user messages, not JSON-serialized into text (which would bloat context).
        """
        base64_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,SGVsbG8gV29ybGQh"},
        }
        messages = [
            {"role": "user", "content": "first message"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second message with image"},
                    base64_image,
                ],
            },
        ]

        result, combined = _combine_adjacent_user_messages(messages)

        assert combined is True
        assert len(result) == 1
        assert result[0]["role"] == "user"

        # Content should be a list with text block + preserved image block
        content = result[0]["content"]
        assert isinstance(content, list)

        # Find the image block - should be preserved as native block
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert image_blocks[0] == base64_image
        # Verify base64 data is preserved exactly, not re-encoded
        assert "SGVsbG8gV29ybGQh" in image_blocks[0]["image_url"]["url"]

        # Text should be JSON-dumped in a text block
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert "first message" in text_blocks[0]["text"]
        assert "second message with image" in text_blocks[0]["text"]

    def test_combine_adjacent_user_messages_preserves_multiple_images(self):
        """
        Multiple image blocks (both URL and base64) should all be preserved.
        """
        url_image = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }
        base64_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2VkYXRh"},
        }
        messages = [
            {"role": "user", "content": "text only"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "with images"},
                    url_image,
                    base64_image,
                ],
            },
        ]

        result, combined = _combine_adjacent_user_messages(messages)

        assert combined is True
        content = result[0]["content"]
        assert isinstance(content, list)

        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 2
        assert url_image in image_blocks
        assert base64_image in image_blocks


class TestPreprocessingPrefillRefinement:
    """
    Regression tests for the prefill-to-system-message refinement.

    These tests verify that only prefilled messages (before any real Anthropic
    response with thinking_blocks) are converted to system message context.
    Real responses should be preserved as actual messages.
    """

    def test_mixed_prefill_and_real_preserves_real_messages(self):
        """
        When messages contain both prefilled (no thinking_blocks) and real
        (with thinking_blocks) assistant messages, only the prefilled portion
        should be converted to system message. Real messages should be preserved.

        This would have failed before the refinement because ALL messages were
        being converted to system message context.
        """
        messages = [
            {"role": "user", "content": "What is 1 + 1?"},
            {"role": "assistant", "content": "The answer is 2."},  # Prefilled
            {
                "role": "assistant",
                "content": "Continuing the conversation.",
                "thinking_blocks": [{"type": "thinking", "thinking": "..."}],
            },  # Real response from Anthropic
            {"role": "user", "content": "Thanks!"},
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Should have: system (with prefilled context), real assistant, user
        assert len(result_messages) == 3

        # First message should be system with JSON context
        assert result_messages[0]["role"] == "system"
        assert "1 + 1" in result_messages[0]["content"]
        assert "The answer is 2" in result_messages[0]["content"]

        # Second message should be the REAL assistant message (preserved, not serialized)
        assert result_messages[1]["role"] == "assistant"
        assert result_messages[1]["content"] == "Continuing the conversation."
        assert "thinking_blocks" in result_messages[1]

        # Third message should be the user message (preserved)
        assert result_messages[2]["role"] == "user"
        assert result_messages[2]["content"] == "Thanks!"

    def test_no_real_responses_converts_all(self):
        """When there are no real responses, all messages should be converted."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},  # Prefilled
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Should be: system message + [continue] user message
        assert len(result_messages) == 2
        assert result_messages[0]["role"] == "system"
        assert result_messages[1]["role"] == "user"
        assert result_messages[1]["content"] == "[continue]"

    def test_all_real_responses_no_conversion(self):
        """When all assistant messages have thinking_blocks, no conversion needed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi!",
                "thinking_blocks": [{"type": "thinking", "thinking": "..."}],
            },
            {"role": "user", "content": "How are you?"},
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Should be unchanged (just deep copied)
        assert len(result_messages) == 3
        assert result_messages[0]["role"] == "user"
        assert result_messages[1]["role"] == "assistant"
        assert result_messages[2]["role"] == "user"

    def test_has_prefilled_assistant_before_real_detection(self):
        """Test the detection function for prefilled messages before real ones."""
        # Only prefilled
        assert (
            _has_prefilled_assistant_before_real(
                [
                    {"role": "assistant", "content": "prefilled"},
                ],
            )
            is True
        )

        # Only real
        assert (
            _has_prefilled_assistant_before_real(
                [
                    {"role": "assistant", "content": "real", "thinking_blocks": [{}]},
                ],
            )
            is False
        )

        # Prefilled before real
        assert (
            _has_prefilled_assistant_before_real(
                [
                    {"role": "assistant", "content": "prefilled"},
                    {"role": "assistant", "content": "real", "thinking_blocks": [{}]},
                ],
            )
            is True
        )

        # Real before prefilled (prefilled is AFTER real, so not "before real")
        assert (
            _has_prefilled_assistant_before_real(
                [
                    {"role": "assistant", "content": "real", "thinking_blocks": [{}]},
                    {"role": "assistant", "content": "prefilled"},
                ],
            )
            is False
        )


class TestPrefillToSystemMessageImageExtraction:
    """
    Tests for extracting image blocks from messages during prefill-to-system
    conversion. Without extraction, base64 image data gets JSON-serialized as
    text tokens in the system message, causing massive context window bloat.
    """

    def test_images_extracted_from_prefilled_messages(self):
        """
        Image blocks in prefilled messages should be extracted and sent as native
        image content in a user message, not JSON-serialized as text in the
        system message (Anthropic rejects non-text blocks in the system param).
        """
        base64_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,SGVsbG8gV29ybGQh"},
        }
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What do you see?"},
            {"role": "assistant", "content": "I see something."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Screenshot] User said: look"},
                    base64_image,
                ],
            },
        ]

        result = _convert_prefill_to_system_message(messages)

        # System message should be a plain string (no image blocks)
        system_msg = result[0]
        assert system_msg["role"] == "system"
        assert isinstance(system_msg["content"], str)
        assert "SGVsbG8gV29ybGQh" not in system_msg["content"]
        assert "[Screenshot]" in system_msg["content"]
        assert "[image]" in system_msg["content"]

        # User message should contain [continue] + native image blocks
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)

        image_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert image_blocks[0] == base64_image

        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
        assert any("[continue]" in b["text"] for b in text_blocks)

    def test_multiple_images_all_extracted(self):
        """Multiple images from different messages should all be extracted."""
        img1 = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aW1hZ2Ux"},
        }
        img2 = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aW1hZ2Uy"},
        }
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Webcam]"},
                    img1,
                    {"type": "text", "text": "[Screen]"},
                    img2,
                ],
            },
        ]

        result = _convert_prefill_to_system_message(messages)

        # System message should be a plain string without base64 data
        system_msg = result[0]
        assert isinstance(system_msg["content"], str)
        assert "aW1hZ2Ux" not in system_msg["content"]
        assert "aW1hZ2Uy" not in system_msg["content"]

        # User message should contain [continue] + both native images
        user_msg = result[1]
        assert isinstance(user_msg["content"], list)
        image_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 2

    def test_no_images_unchanged_behavior(self):
        """When there are no images, behavior is identical to before."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = _convert_prefill_to_system_message(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "[continue]"

    def test_images_with_real_response_after(self):
        """Images in prefilled messages are extracted when real responses follow."""
        base64_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,dGVzdA=="},
        }
        messages = [
            {"role": "user", "content": "Look at this"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Screenshot]"},
                    base64_image,
                ],
            },
            {"role": "assistant", "content": "prefilled response"},
            {
                "role": "assistant",
                "content": "Real response",
                "thinking_blocks": [{"type": "thinking", "thinking": "..."}],
            },
            {"role": "user", "content": "Continue"},
        ]

        result = _convert_prefill_to_system_message(messages)

        # System message should be a plain string without base64 data
        system_msg = result[0]
        assert system_msg["role"] == "system"
        assert isinstance(system_msg["content"], str)
        assert "dGVzdA==" not in system_msg["content"]

        # Image user message should be inserted before real messages
        image_msg = result[1]
        assert image_msg["role"] == "user"
        assert isinstance(image_msg["content"], list)
        image_blocks = [b for b in image_msg["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

        # Real messages should follow
        assert result[2]["role"] == "assistant"
        assert result[2].get("thinking_blocks") is not None
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "Continue"

    def test_integration_with_full_preprocessing(self):
        """
        End-to-end: images in prefilled conversation with reasoning_effort
        should be preserved as native image blocks, not text-serialized.
        """
        base64_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,SGVsbG8="},
        }
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hey!"},
            {"role": "user", "content": "Look at my screen"},
            {"role": "assistant", "content": "I see it."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[User Screen]"},
                    base64_image,
                ],
            },
        ]

        kw = {"messages": messages, "reasoning_effort": "low"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # System message should be a plain string without base64 data
        system_msgs = [m for m in result_messages if m["role"] == "system"]
        for msg in system_msgs:
            content = msg["content"]
            if isinstance(content, str):
                assert "SGVsbG8=" not in content

        # Images should be in a user message as native content blocks
        found_image = False
        user_msgs = [m for m in result_messages if m["role"] == "user"]
        for msg in user_msgs:
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        found_image = True
                        assert block == base64_image
        assert (
            found_image
        ), "Image should be preserved as native content block in user message"


def test_prefill_assistant_message(model):
    client = new_llm_client(model)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
            {"role": "assistant", "content": "The answer is 2."},
            {"role": "user", "content": "What was the answer?"},
        ],
    )

    assert "2" in response


def test_prefill_assistant_message_no_user(model):
    client = new_llm_client(model)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
            {"role": "assistant", "content": "The answer is 2."},
        ],
    )


def test_prefill_tool_call(model):
    call_id = "fast_tool_123"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fast_tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]

    history = [
        {
            "role": "user",
            "content": (
                "Call the tool `fast_tool` (which just returns a token) and reply with the result only"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "fast_tool", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "fast_tool",
            "content": '"fast"',
        },
    ]

    client = new_llm_client(model)
    response = client.generate(messages=history, tools=tools)
    assert "fast" in response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# Thinking Compliance Tests
# ═══════════════════════════════════════════════════════════════════════════════
#
# These tests verify that assistant messages with tool_calls but without
# thinking_blocks are automatically transformed to user context messages when
# thinking mode is enabled.


class TestThinkingComplianceDetection:
    """Tests for detecting non-compliant assistant tool call messages."""

    def test_is_non_compliant_assistant_tool_call_basic(self):
        """Assistant with tool_calls but no thinking_blocks is non-compliant."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "my_tool", "arguments": "{}"},
                },
            ],
        }
        assert _is_non_compliant_assistant_tool_call(msg) is True

    def test_is_compliant_with_thinking_blocks(self):
        """Assistant with tool_calls AND thinking_blocks is compliant."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "my_tool", "arguments": "{}"},
                },
            ],
            "thinking_blocks": [{"type": "thinking", "thinking": "Let me think..."}],
        }
        assert _is_non_compliant_assistant_tool_call(msg) is False

    def test_is_compliant_assistant_no_tool_calls(self):
        """Assistant without tool_calls doesn't need thinking_blocks (for this check)."""
        msg = {
            "role": "assistant",
            "content": "Hello!",
        }
        assert _is_non_compliant_assistant_tool_call(msg) is False

    def test_non_assistant_messages_are_compliant(self):
        """Non-assistant messages are not flagged as non-compliant."""
        assert (
            _is_non_compliant_assistant_tool_call({"role": "user", "content": "Hi"})
            is False
        )
        assert (
            _is_non_compliant_assistant_tool_call({"role": "system", "content": "..."})
            is False
        )
        assert (
            _is_non_compliant_assistant_tool_call({"role": "tool", "content": "result"})
            is False
        )

    def test_empty_tool_calls_is_compliant(self):
        """Assistant with empty tool_calls list is compliant."""
        msg = {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": [],
        }
        assert _is_non_compliant_assistant_tool_call(msg) is False

    def test_has_non_compliant_tool_calls_true(self):
        """Detects non-compliant messages in a list."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {"name": "tool", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "done", "tool_call_id": "1"},
        ]
        assert _has_non_compliant_tool_calls(messages) is True

    def test_has_non_compliant_tool_calls_false_all_have_thinking(self):
        """No non-compliant messages when all have thinking_blocks."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {"name": "tool", "arguments": "{}"}},
                ],
                "thinking_blocks": [{"type": "thinking", "thinking": "..."}],
            },
            {"role": "tool", "content": "done", "tool_call_id": "1"},
        ]
        assert _has_non_compliant_tool_calls(messages) is False

    def test_has_non_compliant_tool_calls_false_no_tool_calls(self):
        """No non-compliant messages when no tool_calls present."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert _has_non_compliant_tool_calls(messages) is False


class TestThinkingComplianceTransformation:
    """Tests for transforming non-compliant messages to user context."""

    def test_basic_transformation(self):
        """Single non-compliant tool call is transformed to user context."""
        messages = [
            {"role": "user", "content": "Call a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
            {"role": "user", "content": "Thanks"},
        ]

        result = _transform_tool_calls_to_context(messages)

        # Should have: user, user (transformed context), user
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Call a tool"

        # Transformed message should be a user message with context
        assert result[1]["role"] == "user"
        assert THINKING_COMPLIANCE_CONTEXT_HEADER in result[1]["content"]
        assert THINKING_COMPLIANCE_CONTEXT_FOOTER in result[1]["content"]
        assert "my_tool" in result[1]["content"]
        assert "result" in result[1]["content"]

        assert result[2]["role"] == "user"
        assert result[2]["content"] == "Thanks"

    def test_compliant_messages_unchanged(self):
        """Messages with thinking_blocks are not transformed."""
        messages = [
            {"role": "user", "content": "Call a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    },
                ],
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "I should call the tool"},
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
            {"role": "user", "content": "Thanks"},
        ]

        result = _transform_tool_calls_to_context(messages)

        # Should be unchanged
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1].get("thinking_blocks") is not None
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "user"

    def test_multiple_tool_results_collected(self):
        """Multiple tool results following assistant are all collected."""
        messages = [
            {"role": "user", "content": "Call tools"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result_a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result_b"},
            {"role": "user", "content": "Done"},
        ]

        result = _transform_tool_calls_to_context(messages)

        # Should have: user, user (context with both tools), user
        assert len(result) == 3
        assert result[0]["content"] == "Call tools"
        assert result[1]["role"] == "user"
        assert "tool_a" in result[1]["content"]
        assert "tool_b" in result[1]["content"]
        assert "result_a" in result[1]["content"]
        assert "result_b" in result[1]["content"]
        assert result[2]["content"] == "Done"

    def test_mixed_compliant_and_non_compliant(self):
        """Mixed transcript with both compliant and non-compliant messages."""
        messages = [
            {"role": "user", "content": "First"},
            # Compliant: has thinking_blocks
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "t1", "arguments": "{}"}},
                ],
                "thinking_blocks": [{"type": "thinking", "thinking": "..."}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "user", "content": "Second"},
            # Non-compliant: no thinking_blocks
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "function": {"name": "t2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "user", "content": "Third"},
        ]

        result = _transform_tool_calls_to_context(messages)

        # First tool call sequence should be unchanged (compliant)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First"
        assert result[1]["role"] == "assistant"
        assert result[1].get("thinking_blocks") is not None
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "Second"

        # Second tool call sequence should be transformed (non-compliant)
        assert result[4]["role"] == "user"
        assert "t2" in result[4]["content"]
        assert "r2" in result[4]["content"]

        # Final user message
        assert result[5]["role"] == "user"
        assert result[5]["content"] == "Third"

    def test_no_tool_results_following(self):
        """Non-compliant assistant without following tool results."""
        messages = [
            {"role": "user", "content": "Call tool"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "tool", "arguments": "{}"}},
                ],
            },
            {
                "role": "user",
                "content": "Never mind",
            },  # User interrupted before tool result
        ]

        result = _transform_tool_calls_to_context(messages)

        # Should still transform just the assistant message
        assert len(result) == 3
        assert result[0]["content"] == "Call tool"
        assert result[1]["role"] == "user"
        assert "tool" in result[1]["content"]
        assert result[2]["content"] == "Never mind"


class TestApplyThinkingCompliance:
    """Tests for the high-level _apply_thinking_compliance function."""

    def test_returns_unchanged_when_all_compliant(self):
        """Returns same messages when all are compliant."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {"name": "t", "arguments": "{}"}},
                ],
                "thinking_blocks": [{}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "done"},
        ]

        result = _apply_thinking_compliance(messages)
        assert result == messages

    def test_transforms_non_compliant(self):
        """Transforms non-compliant messages."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {"name": "t", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "done"},
        ]

        result = _apply_thinking_compliance(messages)
        assert len(result) == 2
        assert result[0]["content"] == "Hi"
        assert result[1]["role"] == "user"
        assert THINKING_COMPLIANCE_CONTEXT_HEADER in result[1]["content"]


class TestApplyProviderPreprocessingThinkingCompliance:
    """Integration tests for thinking compliance in apply_provider_preprocessing."""

    def test_non_compliant_tool_calls_after_real_response_transformed(self):
        """
        Non-compliant tool calls AFTER a real thinking response are transformed.

        This tests the key use case: a conversation has started with real thinking
        responses, but then synthetic tool call messages appear (from tool loops)
        that lack thinking_blocks. These should be transformed to user context.

        Note: If there are NO real thinking responses at all, the prefill conversion
        handles it differently (converts everything to system message).
        """
        messages = [
            {"role": "user", "content": "First task"},
            # Real response with thinking_blocks
            {
                "role": "assistant",
                "content": "I'll help with that.",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "User wants help"},
                ],
            },
            {"role": "user", "content": "Now do something else"},
            # Synthetic/non-compliant: no thinking_blocks (e.g., from tool loop)
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
            {"role": "user", "content": "Continue"},
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Should have:
        # 1. user (First task)
        # 2. assistant with thinking_blocks (preserved)
        # 3. user (Now do something else)
        # 4. user (transformed context from non-compliant tool call + result)
        # 5. user (Continue)
        assert len(result_messages) == 5

        assert result_messages[0]["role"] == "user"
        assert result_messages[0]["content"] == "First task"

        assert result_messages[1]["role"] == "assistant"
        assert result_messages[1].get("thinking_blocks") is not None

        assert result_messages[2]["role"] == "user"
        assert result_messages[2]["content"] == "Now do something else"

        # Transformed context message
        assert result_messages[3]["role"] == "user"
        assert THINKING_COMPLIANCE_CONTEXT_HEADER in result_messages[3]["content"]
        assert "my_tool" in result_messages[3]["content"]
        assert "result" in result_messages[3]["content"]

        assert result_messages[4]["role"] == "user"
        assert result_messages[4]["content"] == "Continue"

    def test_all_non_compliant_handled_by_prefill_conversion(self):
        """
        When there are NO real thinking responses, prefill conversion handles it.

        This is the case for seeded transcripts where all assistant messages
        are prefilled (no thinking_blocks). The prefill conversion converts
        everything to a system message + [continue] prompt.
        """
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
            {"role": "user", "content": "Continue"},
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Prefill conversion converts everything to system message + [continue]
        assert len(result_messages) == 2
        assert result_messages[0]["role"] == "system"
        assert "Do something" in result_messages[0]["content"]
        assert "my_tool" in result_messages[0]["content"]
        assert result_messages[1]["role"] == "user"
        assert result_messages[1]["content"] == "[continue]"

    def test_compliant_tool_calls_unchanged_with_reasoning_effort(self):
        """Compliant tool calls (with thinking_blocks) are not transformed."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    },
                ],
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "Let me call the tool"},
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
            {"role": "user", "content": "Continue"},
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Should be unchanged (4 messages)
        assert len(result_messages) == 4
        assert result_messages[0]["role"] == "user"
        assert result_messages[1]["role"] == "assistant"
        assert result_messages[1].get("thinking_blocks") is not None
        assert result_messages[2]["role"] == "tool"
        assert result_messages[3]["role"] == "user"

    def test_no_transformation_without_reasoning_effort(self):
        """No transformation when reasoning_effort is not set."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]

        kw = {"messages": messages}  # No reasoning_effort
        result = apply_provider_preprocessing(kw, "anthropic")

        result_messages = result["messages"]

        # Should be unchanged (no transformation without reasoning_effort)
        assert len(result_messages) == 3
        assert result_messages[0]["role"] == "user"
        assert result_messages[1]["role"] == "assistant"
        assert result_messages[2]["role"] == "tool"

    def test_no_transformation_for_non_anthropic(self):
        """No transformation for non-Anthropic providers."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {"name": "tool", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]

        kw = {"messages": messages, "reasoning_effort": "high"}
        result = apply_provider_preprocessing(kw, "openai")

        # Should be unchanged for non-Anthropic
        assert result["messages"] == messages
