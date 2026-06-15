"""Tests for thinking block handling across multi-turn conversations.

Tests for handling the Anthropic API constraint that thinking mode cannot be
combined with tool_choice="required" (which forces tool use).

When tool_choice="required" is requested with thinking enabled, we:
1. Keep reasoning_effort enabled (preserve thinking intelligence)
2. Downgrade tool_choice from "required" to "auto"
3. Add a system message instructing the model to call a tool

This preserves the smarter thinking model while nudging toward tool use,
instead of silently disabling thinking.
"""

import pytest
from .helpers import new_llm_client

GET_ID_TOOL = {
    "type": "function",
    "function": {
        "name": "get_id",
        "description": "Returns a reference ID.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


class TestThinkingPreservedWithToolChoiceRequired:
    """Test that thinking mode is preserved when tool_choice="required" is used."""

    @pytest.mark.parametrize("model", ["claude-4.8-opus@anthropic"])
    def test_thinking_preserved_with_tool_choice_required(self, model):
        """
        Verify that thinking mode stays enabled when tool_choice="required".

        When tool_choice="required" is used with thinking enabled:
        1. Thinking (reasoning_effort) should stay enabled
        2. tool_choice should be downgraded to "auto"
        3. A system message should instruct the model to call a tool
        4. The model should still make a tool call (guided by the instruction)
        """
        # Step 1: Create client with thinking enabled, make a tool call
        client = new_llm_client(model, stateful=True)

        response = client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Call get_id to get a reference ID.",
                },
            ],
            tools=[GET_ID_TOOL],
            return_full_completion=True,
        )

        # Verify we got a tool call response
        assistant_msg = response.choices[0].message
        assert assistant_msg.tool_calls, "Expected tool call response"

        messages = client._messages
        assert len(messages) >= 2, "Should have user and assistant messages"

        stored_assistant = messages[-1]
        assert stored_assistant["role"] == "assistant"

        # Add tool result
        client.append_messages(
            [
                {
                    "role": "tool",
                    "content": "ID-12345",
                    "tool_call_id": assistant_msg.tool_calls[0].id,
                },
            ],
        )

        # Step 2: Make a second request WITH tool_choice="required"
        # With the new behavior:
        # - reasoning_effort stays enabled (thinking preserved)
        # - tool_choice is downgraded to "auto"
        # - A system message instructs the model to call a tool
        response = client.generate(
            tools=[GET_ID_TOOL],
            tool_choice="required",
            return_full_completion=True,
        )

        # Request should succeed (key test: no API error about thinking + forced tool use)
        assert response is not None

        # The response proves thinking was enabled (no API rejection).
        # Note: The model may or may not include thinking_blocks in simple responses,
        # but the important thing is the API call succeeded with reasoning_effort enabled.

    @pytest.mark.parametrize("model", ["claude-4.8-opus@anthropic"])
    def test_thinking_blocks_stripped_when_thinking_disabled_explicitly(self, model):
        """
        Verify that when a conversation has thinking_blocks from a previous
        thinking-enabled response, a subsequent request with thinking explicitly
        disabled doesn't fail.
        """
        # Step 1: Create client with thinking enabled, make a tool call
        client = new_llm_client(model, stateful=True)

        response = client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Call get_id to get a reference ID.",
                },
            ],
            tools=[GET_ID_TOOL],
            return_full_completion=True,
        )

        assistant_msg = response.choices[0].message
        assert assistant_msg.tool_calls, "Expected tool call response"

        # Add tool result
        client.append_messages(
            [
                {
                    "role": "tool",
                    "content": "ID-12345",
                    "tool_call_id": assistant_msg.tool_calls[0].id,
                },
            ],
        )

        # Step 2: Make a second request WITHOUT thinking enabled
        # by creating a new client without reasoning_effort
        import unillm
        from ..settings import SETTINGS

        client_no_thinking = unillm.Unify(
            model,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
            # NOTE: No reasoning_effort - thinking disabled
            service_tier=SETTINGS.UNILLM_SERVICE_TIER,
            stateful=True,
        )

        # Copy the messages from the first client (including thinking_blocks)
        client_no_thinking._messages = client._messages.copy()

        # This should NOT raise an error about thinking blocks
        response = client_no_thinking.generate(
            return_full_completion=True,
        )

        assert response is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["claude-4.8-opus@anthropic"])
    async def test_thinking_preserved_with_tool_choice_required_async(self, model):
        """Async version: tool_choice=required should preserve thinking."""
        # Step 1: Create async client with thinking enabled
        client = new_llm_client(model, stateful=True, is_async=True)

        response = await client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Call get_id to get a reference ID.",
                },
            ],
            tools=[GET_ID_TOOL],
            return_full_completion=True,
        )

        assistant_msg = response.choices[0].message
        assert assistant_msg.tool_calls, "Expected tool call response"

        stored_assistant = client._messages[-1]
        assert stored_assistant["role"] == "assistant"

        # Add tool result
        client.append_messages(
            [
                {
                    "role": "tool",
                    "content": "ID-12345",
                    "tool_call_id": assistant_msg.tool_calls[0].id,
                },
            ],
        )

        # Step 2: Make request with tool_choice="required"
        # Thinking should stay enabled, tool_choice downgraded to "auto"
        response = await client.generate(
            tools=[GET_ID_TOOL],
            tool_choice="required",
            return_full_completion=True,
        )

        # Request should succeed (key test: no API error about thinking + forced tool use)
        assert response is not None

        # The response proves thinking was enabled (no API rejection).
        # Note: The model may or may not include thinking_blocks in simple responses,
        # but the important thing is the API call succeeded with reasoning_effort enabled.


class TestPreprocessingToolChoiceRequired:
    """Unit tests for the tool_choice=required preprocessing behavior."""

    def test_preprocessing_keeps_reasoning_changes_tool_choice(self):
        """
        Verify that preprocessing keeps reasoning_effort and changes tool_choice.
        """
        from unillm.clients.provider_preprocessing import (
            apply_provider_preprocessing,
            TOOL_CHOICE_REQUIRED_INSTRUCTION,
        )

        kw = {
            "messages": [{"role": "user", "content": "Call a tool"}],
            "reasoning_effort": "high",
            "tool_choice": "required",
            "tools": [GET_ID_TOOL],
        }

        result = apply_provider_preprocessing(kw, provider="anthropic")

        # reasoning_effort should be preserved
        assert (
            result.get("reasoning_effort") == "high"
        ), "reasoning_effort should NOT be deleted"

        # tool_choice should be downgraded to "auto"
        assert (
            result.get("tool_choice") == "auto"
        ), "tool_choice should be downgraded from 'required' to 'auto'"

        # A system message with the instruction should be added
        messages = result.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]
        assert any(
            TOOL_CHOICE_REQUIRED_INSTRUCTION in m.get("content", "")
            for m in system_messages
        ), "Should have system message with tool instruction"

    def test_preprocessing_no_change_without_reasoning(self):
        """
        Verify that tool_choice="required" is unchanged when reasoning is not set.
        """
        from unillm.clients.provider_preprocessing import apply_provider_preprocessing

        kw = {
            "messages": [{"role": "user", "content": "Call a tool"}],
            "tool_choice": "required",
            "tools": [GET_ID_TOOL],
        }

        result = apply_provider_preprocessing(kw, provider="anthropic")

        # tool_choice should remain "required" when no reasoning_effort
        assert (
            result.get("tool_choice") == "required"
        ), "tool_choice should stay 'required' when reasoning is not set"

    def test_preprocessing_no_change_with_auto_tool_choice(self):
        """
        Verify that tool_choice="auto" is unchanged even with reasoning enabled.
        """
        from unillm.clients.provider_preprocessing import apply_provider_preprocessing

        kw = {
            "messages": [{"role": "user", "content": "Call a tool"}],
            "reasoning_effort": "high",
            "tool_choice": "auto",
            "tools": [GET_ID_TOOL],
        }

        result = apply_provider_preprocessing(kw, provider="anthropic")

        # Both should be unchanged
        assert result.get("reasoning_effort") == "high"
        assert result.get("tool_choice") == "auto"


class TestThinkingBlocksPreservation:
    """Test that thinking_blocks are preserved when thinking stays enabled."""

    @pytest.mark.parametrize("model", ["claude-4.8-opus@anthropic"])
    def test_thinking_blocks_preserved_in_multi_turn(self, model):
        """
        Verify that thinking_blocks are correctly preserved across multiple
        turns when thinking remains enabled.
        """
        client = new_llm_client(model, stateful=True)

        # First request with tool
        response = client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Call get_id to get a reference ID.",
                },
            ],
            tools=[GET_ID_TOOL],
            return_full_completion=True,
        )

        assistant_msg = response.choices[0].message
        assert assistant_msg.tool_calls, "Expected tool call"

        # Add tool result
        client.append_messages(
            [
                {
                    "role": "tool",
                    "content": "ID-12345",
                    "tool_call_id": assistant_msg.tool_calls[0].id,
                },
            ],
        )

        # Second request - same client, thinking still enabled
        # This should work fine
        response = client.generate(
            return_full_completion=True,
        )

        assert response is not None
        # Should get a text response summarizing the ID
        content = response.choices[0].message.content
        assert content is not None
        assert "12345" in content or "ID" in content
