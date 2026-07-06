"""Tests for the caching module."""

import json
import os
import tempfile
import pytest

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from unillm.caching import (
    LocalCache,
    LocalSeparateCache,
    BaseCache,
    CacheStats,
    _get_cache,
    _write_to_cache,
    is_caching_enabled,
    set_cache_backend,
    get_cache_backend,
)


class _CacheHandler:
    """Context manager for isolated cache testing."""

    def __init__(self, fname=".test_cache.ndjson"):
        self._fname = fname
        self.test_path = ""
        self._original_cache = None
        self._original_cache_dir = ""
        self._original_cache_name = ""

    def __enter__(self):
        # Store original cache state
        self._original_cache = LocalCache._cache
        self._original_cache_dir = LocalCache._cache_dir
        self._original_cache_name = LocalCache.get_cache_name()
        LocalCache._cache = None

        # Use temp directory for isolation
        self.test_path = os.path.join(tempfile.gettempdir(), self._fname)
        LocalCache._cache_dir = tempfile.gettempdir()
        LocalCache.set_cache_name(self._fname)

        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        # Restore original cache state (set_cache_name resets the in-memory
        # cache, so it must run before the _cache assignment)
        LocalCache.set_cache_name(self._original_cache_name)
        LocalCache._cache = self._original_cache
        LocalCache._cache_dir = self._original_cache_dir


class TestBaseCache:
    """Tests for BaseCache serialization."""

    def test_serialize_simple_dict(self):
        obj = {"key": "value", "number": 42}
        result = BaseCache.serialize_object(obj)
        assert json.loads(result) == obj

    def test_serialize_nested_dict(self):
        obj = {"outer": {"inner": "value"}}
        result = BaseCache.serialize_object(obj)
        assert json.loads(result) == obj

    def test_serialize_list(self):
        obj = [1, 2, 3, "four"]
        result = BaseCache.serialize_object(obj)
        assert json.loads(result) == obj

    def test_serialize_pydantic_model(self):
        """Test that Pydantic models are serialized and their types tracked."""
        message = ChatCompletionMessage(role="assistant", content="Hello")
        choice = Choice(index=0, message=message, finish_reason="stop")
        completion = ChatCompletion(
            id="test-id",
            choices=[choice],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
        )

        cached_types = {}
        result = BaseCache.serialize_object(completion, cached_types)

        # Should be JSON string at root level
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["id"] == "test-id"

        # Type should be tracked
        assert "[]" in cached_types  # Root level
        assert cached_types["[]"] == "ChatCompletion"


class TestLocalCache:
    """Tests for LocalCache backend."""

    def test_store_and_retrieve(self):
        with _CacheHandler() as handler:
            LocalCache.initialize_cache()

            LocalCache.store_entry(
                key="test_key",
                value='{"result": "test_value"}',
                res_types=None,
            )

            assert LocalCache.has_key("test_key")
            value, res_types = LocalCache.retrieve_entry("test_key")
            assert value == {"result": "test_value"}

    def test_list_keys(self):
        with _CacheHandler():
            LocalCache.initialize_cache()

            LocalCache.store_entry(key="key1", value='"value1"', res_types=None)
            LocalCache.store_entry(key="key2", value='"value2"', res_types=None)

            keys = LocalCache.list_keys()
            assert "key1" in keys
            assert "key2" in keys

    def test_remove_entry(self):
        with _CacheHandler():
            LocalCache.initialize_cache()

            LocalCache.store_entry(key="to_remove", value='"value"', res_types=None)
            assert LocalCache.has_key("to_remove")

            LocalCache.remove_entry("to_remove")
            assert not LocalCache.has_key("to_remove")

    def test_persistence(self):
        """Test that cache persists across reloads."""
        with _CacheHandler() as handler:
            LocalCache.initialize_cache()
            LocalCache.store_entry(key="persistent", value='"data"', res_types=None)

            # Force cache reload
            LocalCache._cache = None
            LocalCache.initialize_cache()

            assert LocalCache.has_key("persistent")
            value, _ = LocalCache.retrieve_entry("persistent")
            assert value == "data"


class TestCacheHighLevel:
    """Tests for high-level caching functions."""

    def test_get_cache_returns_none_on_miss(self):
        with _CacheHandler():
            result = _get_cache(
                fn_name="test_fn",
                kw={"arg": "value"},
            )
            assert result is None

    def test_write_and_get_cache(self):
        with _CacheHandler():
            kw = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}

            _write_to_cache(
                fn_name="completion",
                kw=kw,
                response={"result": "Hello!"},
            )

            result = _get_cache(
                fn_name="completion",
                kw=kw,
            )
            assert result == {"result": "Hello!"}

    def test_cache_with_chat_completion(self):
        """Test caching actual ChatCompletion objects."""
        with _CacheHandler():
            message = ChatCompletionMessage(role="assistant", content="Hello")
            choice = Choice(index=0, message=message, finish_reason="stop")
            completion = ChatCompletion(
                id="test-id",
                choices=[choice],
                created=1234567890,
                model="gpt-4",
                object="chat.completion",
            )

            kw = {"model": "gpt-4", "messages": []}

            _write_to_cache(
                fn_name="completion",
                kw=kw,
                response=completion,
            )

            result = _get_cache(
                fn_name="completion",
                kw=kw,
            )

            # Result should be reconstructed as ChatCompletion
            assert isinstance(result, ChatCompletion)
            assert result.id == "test-id"
            assert result.choices[0].message.content == "Hello"


class TestCacheStats:
    """Tests for cache statistics."""

    def test_cache_stats_initial_state(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.reads == 0
        assert stats.writes == 0

    def test_cache_stats_percentage(self):
        stats = CacheStats(hits=3, misses=1, reads=4, writes=2)
        assert stats.get_percentage_of_cache_hits() == 75.0
        assert stats.get_percentage_of_cache_misses() == 25.0

    def test_cache_stats_percentage_zero_reads(self):
        stats = CacheStats()
        assert stats.get_percentage_of_cache_hits() == 0.0
        assert stats.get_percentage_of_cache_misses() == 0.0

    def test_cache_stats_add(self):
        stats1 = CacheStats(hits=1, misses=1, reads=2, writes=1)
        stats2 = CacheStats(hits=2, misses=0, reads=2, writes=1)
        combined = stats1 + stats2
        assert combined.hits == 3
        assert combined.misses == 1
        assert combined.reads == 4
        assert combined.writes == 2

    def test_cache_stats_repr(self):
        stats = CacheStats(hits=2, misses=2, reads=4, writes=1)
        repr_str = repr(stats)
        assert "hits=2" in repr_str
        assert "50.0%" in repr_str


class TestCacheBackend:
    """Tests for cache backend selection."""

    def test_default_backend_is_local(self):
        backend = get_cache_backend()
        assert backend == LocalCache

    def test_set_backend_local_separate(self):
        original = get_cache_backend()
        try:
            set_cache_backend("local_separate")
            assert get_cache_backend() == LocalSeparateCache
        finally:
            set_cache_backend("local")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            set_cache_backend("nonexistent")

    def test_get_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            get_cache_backend("nonexistent")


class TestIsCachingEnabled:
    """Tests for is_caching_enabled function."""

    def test_caching_disabled_by_default(self):
        # The global CACHING_ENABLED is False by default
        assert is_caching_enabled() is False


class TestMalformedCacheEntries:
    """Tests for handling malformed cache entries gracefully."""

    def test_missing_key_field_is_skipped(self):
        """Cache entry with missing 'key' field should be skipped, not crash."""
        with _CacheHandler() as handler:
            # Manually write a malformed entry (valid JSON but missing "key")
            with open(handler.test_path, "w") as f:
                f.write('{"value": "test", "res_types": null}\n')

            # This should NOT raise - malformed entries should be skipped
            LocalCache.initialize_cache()

            # Cache should be empty (malformed entry skipped)
            assert LocalCache.list_keys() == []

    def test_missing_value_field_is_skipped(self):
        """Cache entry with missing 'value' field should be skipped."""
        with _CacheHandler() as handler:
            with open(handler.test_path, "w") as f:
                f.write('{"key": "test_key", "res_types": null}\n')

            LocalCache.initialize_cache()
            assert not LocalCache.has_key("test_key")

    def test_missing_res_types_field_is_skipped(self):
        """Cache entry with missing 'res_types' field should be skipped."""
        with _CacheHandler() as handler:
            with open(handler.test_path, "w") as f:
                f.write('{"key": "test_key", "value": "test"}\n')

            LocalCache.initialize_cache()
            assert not LocalCache.has_key("test_key")

    def test_valid_entries_after_malformed_are_loaded(self):
        """Valid entries after a malformed one should still be loaded."""
        with _CacheHandler() as handler:
            with open(handler.test_path, "w") as f:
                # Malformed entry (missing key)
                f.write('{"value": "bad", "res_types": null}\n')
                # Valid entry
                f.write(
                    '{"key": "good_key", "value": "\\"good_value\\"", "res_types": null}\n',
                )

            LocalCache.initialize_cache()

            # Malformed entry skipped, valid entry loaded
            assert LocalCache.has_key("good_key")
            value, _ = LocalCache.retrieve_entry("good_key")
            assert value == "good_value"

    def test_get_cache_handles_malformed_gracefully(self):
        """_get_cache should not crash when cache has malformed entries."""
        with _CacheHandler() as handler:
            with open(handler.test_path, "w") as f:
                f.write('{"value": "bad"}\n')  # Missing key and res_types

            # Should return None (cache miss), not raise an exception
            result = _get_cache(
                fn_name="test_fn",
                kw={"arg": "value"},
            )
            assert result is None
