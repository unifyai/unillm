"""Tests for provider-specific preprocessing, particularly Anthropic caching."""

from unillm.clients.provider_preprocessing import (
    CACHE_CONTROL_EPHEMERAL,
    _apply_anthropic_caching,
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
