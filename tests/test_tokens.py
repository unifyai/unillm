"""Tests for the token tracking utility module."""

import pytest

from unillm.tokens import count_tokens, fills_context_window, get_max_input_tokens


class TestGetMaxInputTokens:
    """Tests for the get_max_input_tokens function."""

    def test_returns_positive_int_gpt(self):
        result = get_max_input_tokens("gpt-5.2@openai")
        assert isinstance(result, int)
        assert result > 0

    def test_returns_positive_int_claude(self):
        result = get_max_input_tokens("claude-4.6-opus@anthropic")
        assert isinstance(result, int)
        assert result > 0

    def test_litellm_registered_context_window_is_used(self):
        result = get_max_input_tokens("claude-4.6-opus@anthropic")
        assert result == 1_000_000

    def test_with_provider_suffix(self):
        with_suffix = get_max_input_tokens("gpt-5.2@openai")
        without_suffix = get_max_input_tokens("gpt-5.2")
        assert with_suffix == without_suffix

    def test_unknown_model_raises(self):
        with pytest.raises(
            ValueError,
            match="Model non-existent-model-xyz-12345 not found",
        ):
            get_max_input_tokens("non-existent-model-xyz-12345")

    def test_transport_alias_fallback_when_public_litellm_name_missing(
        self,
        monkeypatch,
    ):
        """OpenRouter-registered OpenAI models must resolve before LiteLLM ships them."""
        import litellm

        from unillm.endpoints.utils import get_transport_model_alias

        endpoint = "gpt-5.6-sol@openai"
        transport = get_transport_model_alias(endpoint)
        real_get_model_info = litellm.get_model_info

        def fake_get_model_info(model, *args, **kwargs):
            if model == "gpt-5.6-sol":
                raise Exception("public openai name not in pinned litellm")
            return real_get_model_info(model, *args, **kwargs)

        monkeypatch.setattr(litellm, "get_model_info", fake_get_model_info)
        assert transport.startswith("openrouter/")
        assert get_max_input_tokens(endpoint) == 1_050_000


class TestCountTokens:
    """Tests for the count_tokens function."""

    def test_simple_message(self):
        messages = [{"role": "user", "content": "Hello, world!"}]
        result = count_tokens("gpt-5.2", messages=messages)
        assert isinstance(result, int)
        assert result > 0

    def test_system_message_in_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
        without_sys = count_tokens("gpt-5.2", messages=[messages[1]])
        with_sys = count_tokens("gpt-5.2", messages=messages)
        assert with_sys > without_sys

    def test_tools_add_tokens(self):
        messages = [{"role": "user", "content": "What is the weather?"}]
        without_tools = count_tokens("gpt-5.2", messages=messages)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
        ]
        with_tools = count_tokens("gpt-5.2", messages=messages, tools=tools)
        assert with_tools > without_tools

    def test_with_provider_suffix(self):
        messages = [{"role": "user", "content": "Hello"}]
        with_suffix = count_tokens("gpt-5.2@openai", messages=messages)
        without_suffix = count_tokens("gpt-5.2", messages=messages)
        assert with_suffix == without_suffix


class TestFillsContextWindow:
    """Tests for the fills_context_window function.

    Uses monkeypatching to make get_max_input_tokens return a small value
    so we can control the threshold checks without needing huge prompts.
    """

    @pytest.fixture(autouse=True)
    def _patch_context_window(self, monkeypatch):
        """Force get_max_input_tokens to return 50 tokens for all models."""
        import unillm.tokens as _mod

        monkeypatch.setattr(_mod, "get_max_input_tokens", lambda endpoint: 50)

    def test_below_threshold(self):
        messages = [{"role": "user", "content": "Hi"}]
        assert fills_context_window(0.9, "gpt-5.2", messages=messages) is False

    def test_above_threshold(self):
        long_content = "word " * 200
        messages = [{"role": "user", "content": long_content}]
        assert fills_context_window(0.5, "gpt-5.2", messages=messages) is True

    def test_threshold_boundary(self):
        messages = [{"role": "user", "content": "Hi"}]
        token_count = count_tokens("gpt-5.2", messages=messages)
        exact_threshold = token_count / 50
        assert (
            fills_context_window(exact_threshold, "gpt-5.2", messages=messages) is True
        )
        assert (
            fills_context_window(exact_threshold + 0.01, "gpt-5.2", messages=messages)
            is False
        )
