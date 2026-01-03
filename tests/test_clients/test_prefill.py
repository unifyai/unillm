from .helpers import new_llm_client
from unillm.clients.provider_preprocessing import (
    apply_provider_preprocessing,
    _has_prefilled_assistant_before_real,
)


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
