"""Tests for stateful client behavior - ensuring full message is stored in history."""

import pytest
from .helpers import new_llm_client


class TestStatefulFullMessageStorage:
    """Test that stateful clients store full message in history regardless of return_full_completion."""

    def test_stateful_stores_full_message_when_return_full_completion_false(
        self,
        model,
    ):
        """Verify full message is stored in history even when return_full_completion=False."""
        client = new_llm_client(model, stateful=True, return_full_completion=False)

        # Make a request - should return string content
        response = client.generate(user_message="Say hello")
        assert isinstance(
            response,
            str,
        ), "Should return string when return_full_completion=False"

        # Check that the assistant message in history is a full message dict
        messages = client._messages
        assert len(messages) >= 2, "Should have at least user and assistant messages"

        # Find the assistant message (should be the last one)
        assistant_msg = messages[-1]
        assert assistant_msg["role"] == "assistant"

        # The stored message should be a full message dict from model_dump(),
        # not just {"role": "assistant", "content": str}
        # Full messages have additional fields like 'refusal', 'audio', 'function_call', etc.
        # At minimum, content should be a string (not the ChatCompletion repr)
        assert "content" in assistant_msg
        if assistant_msg["content"] is not None:
            # Content should be the actual text, not a repr of ChatCompletion
            assert not assistant_msg["content"].startswith("ChatCompletion(")

    def test_stateful_stores_full_message_when_return_full_completion_true(self, model):
        """Verify full message is stored in history when return_full_completion=True."""
        client = new_llm_client(model, stateful=True, return_full_completion=True)

        # Make a request - should return ChatCompletion object
        response = client.generate(user_message="Say hello")
        assert hasattr(
            response,
            "choices",
        ), "Should return ChatCompletion when return_full_completion=True"

        # Check that the assistant message in history is a full message dict
        messages = client._messages
        assert len(messages) >= 2, "Should have at least user and assistant messages"

        assistant_msg = messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert "content" in assistant_msg

    def test_stateful_message_matches_response_content(self, model):
        """Verify stored message content matches the returned content."""
        client = new_llm_client(model, stateful=True, return_full_completion=False)

        response = client.generate(
            user_message="What is 2+2? Just reply with the number.",
        )

        messages = client._messages
        assistant_msg = messages[-1]

        # The stored content should match the returned response
        stored_content = assistant_msg.get("content", "")
        if stored_content:
            stored_content = stored_content.strip()
        assert (
            stored_content == response.strip()
            if response
            else stored_content == response
        )

    @pytest.mark.asyncio
    async def test_async_stateful_stores_full_message_when_return_full_completion_false(
        self,
        model,
    ):
        """Verify async client stores full message in history even when return_full_completion=False."""
        client = new_llm_client(
            model,
            is_async=True,
            stateful=True,
            return_full_completion=False,
        )

        # Make a request - should return string content
        response = await client.generate(user_message="Say hello")
        assert isinstance(
            response,
            str,
        ), "Should return string when return_full_completion=False"

        # Check that the assistant message in history is a full message dict
        messages = client._messages
        assert len(messages) >= 2, "Should have at least user and assistant messages"

        assistant_msg = messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert "content" in assistant_msg

        # Content should be the actual text, not a repr of ChatCompletion
        if assistant_msg["content"] is not None:
            assert not assistant_msg["content"].startswith("ChatCompletion(")

    @pytest.mark.asyncio
    async def test_async_stateful_stores_full_message_when_return_full_completion_true(
        self,
        model,
    ):
        """Verify async client stores full message in history when return_full_completion=True."""
        client = new_llm_client(
            model,
            is_async=True,
            stateful=True,
            return_full_completion=True,
        )

        response = await client.generate(user_message="Say hello")
        assert hasattr(
            response,
            "choices",
        ), "Should return ChatCompletion when return_full_completion=True"

        messages = client._messages
        assert len(messages) >= 2

        assistant_msg = messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert "content" in assistant_msg
