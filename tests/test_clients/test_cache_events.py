"""Tests for cache event capture using context managers."""

import os
import pytest
from unittest.mock import patch, MagicMock

import unillm
from unillm import capture_cache_events, acapture_cache_events
from unillm.cache_events import _emit_cache_event


class TestCaptureContextManager:
    """Tests for the capture_cache_events context manager."""

    def test_capture_returns_empty_list_initially(self):
        with capture_cache_events() as events:
            pass
        assert events == []

    def test_capture_receives_emitted_events(self):
        with capture_cache_events() as events:
            _emit_cache_event(
                {
                    "cache_status": "hit",
                    "endpoint": "test@provider",
                    "request_kw": {"model": "test"},
                },
            )
        assert len(events) == 1
        assert events[0]["cache_status"] == "hit"

    def test_capture_receives_multiple_events(self):
        with capture_cache_events() as events:
            _emit_cache_event(
                {
                    "cache_status": "miss",
                    "endpoint": "test@provider",
                    "request_kw": {},
                },
            )
            _emit_cache_event(
                {
                    "cache_status": "hit",
                    "endpoint": "test@provider",
                    "request_kw": {},
                },
            )
        assert len(events) == 2
        assert events[0]["cache_status"] == "miss"
        assert events[1]["cache_status"] == "hit"

    def test_events_outside_context_not_captured(self):
        # Emit before context
        _emit_cache_event(
            {
                "cache_status": "miss",
                "endpoint": "before@provider",
                "request_kw": {},
            },
        )

        with capture_cache_events() as events:
            _emit_cache_event(
                {
                    "cache_status": "hit",
                    "endpoint": "inside@provider",
                    "request_kw": {},
                },
            )

        # Emit after context
        _emit_cache_event(
            {
                "cache_status": "miss",
                "endpoint": "after@provider",
                "request_kw": {},
            },
        )

        # Only the inside event should be captured
        assert len(events) == 1
        assert events[0]["endpoint"] == "inside@provider"

    def test_nested_contexts_are_independent(self):
        with capture_cache_events() as outer_events:
            _emit_cache_event(
                {
                    "cache_status": "miss",
                    "endpoint": "outer@provider",
                    "request_kw": {},
                },
            )

            with capture_cache_events() as inner_events:
                _emit_cache_event(
                    {
                        "cache_status": "hit",
                        "endpoint": "inner@provider",
                        "request_kw": {},
                    },
                )

            # After inner context, outer should receive events again
            _emit_cache_event(
                {
                    "cache_status": "miss",
                    "endpoint": "outer-again@provider",
                    "request_kw": {},
                },
            )

        # Inner only captured inner event
        assert len(inner_events) == 1
        assert inner_events[0]["endpoint"] == "inner@provider"

        # Outer captured outer events (but not inner, due to context override)
        assert len(outer_events) == 2
        assert outer_events[0]["endpoint"] == "outer@provider"
        assert outer_events[1]["endpoint"] == "outer-again@provider"


class TestAsyncCaptureContextManager:
    """Tests for the acapture_cache_events async context manager."""

    @pytest.mark.asyncio
    async def test_async_capture_returns_empty_list_initially(self):
        async with acapture_cache_events() as events:
            pass
        assert events == []

    @pytest.mark.asyncio
    async def test_async_capture_receives_emitted_events(self):
        async with acapture_cache_events() as events:
            _emit_cache_event(
                {
                    "cache_status": "miss",
                    "endpoint": "test@provider",
                    "request_kw": {"model": "test"},
                },
            )
        assert len(events) == 1
        assert events[0]["cache_status"] == "miss"

    @pytest.mark.asyncio
    async def test_async_events_outside_context_not_captured(self):
        _emit_cache_event(
            {
                "cache_status": "miss",
                "endpoint": "before@provider",
                "request_kw": {},
            },
        )

        async with acapture_cache_events() as events:
            _emit_cache_event(
                {
                    "cache_status": "hit",
                    "endpoint": "inside@provider",
                    "request_kw": {},
                },
            )

        _emit_cache_event(
            {
                "cache_status": "miss",
                "endpoint": "after@provider",
                "request_kw": {},
            },
        )

        assert len(events) == 1
        assert events[0]["endpoint"] == "inside@provider"


class TestCacheEventEmissionMocked:
    """Tests for cache event emission during LLM requests using mocked LLM calls.

    These tests mock all external dependencies including logging to avoid
    creating log files that would pollute CI cache statistics.
    """

    @pytest.fixture(autouse=True)
    def mock_logging(self):
        """Mock logging functions to prevent log file creation in mocked tests."""
        with patch("unillm.clients.uni_llm.write_request_pending", return_value=None):
            with patch("unillm.clients.uni_llm.append_response_and_finalize"):
                yield

    def test_sync_client_emits_cache_miss_on_llm_call(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        with patch(
            "unillm.clients.uni_llm.litellm.completion",
            return_value=mock_response,
        ):
            with patch("unillm.clients.uni_llm._get_cache", return_value=None):
                with patch("unillm.clients.uni_llm._write_to_cache"):
                    with patch(
                        "unillm.clients.uni_llm.compute_cost_from_response",
                        return_value=0.001,
                    ):
                        with patch("unillm.clients.uni_llm.unify.deduct_credits"):
                            client = unillm.Unify("gpt-4@openai", cache=True)
                            with capture_cache_events() as events:
                                client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        assert len(events) == 1
        assert events[0]["cache_status"] == "miss"
        assert events[0]["endpoint"] == "gpt-4@openai"

    def test_sync_client_emits_cache_hit_when_cached(self):
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"

        with patch(
            "unillm.clients.uni_llm._get_cache",
            return_value=mock_cached_response,
        ):
            with patch("unillm.clients.uni_llm._write_to_cache"):
                client = unillm.Unify("gpt-4@openai", cache=True)
                with capture_cache_events() as events:
                    client.generate(messages=[{"role": "user", "content": "Hi"}])

        assert len(events) == 1
        assert events[0]["cache_status"] == "hit"

    @pytest.mark.asyncio
    async def test_async_client_emits_cache_miss_on_llm_call(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        async def mock_acompletion(*args, **kwargs):
            return mock_response

        with patch(
            "unillm.clients.uni_llm.litellm.acompletion",
            side_effect=mock_acompletion,
        ):
            with patch("unillm.clients.uni_llm._get_cache", return_value=None):
                with patch("unillm.clients.uni_llm._write_to_cache"):
                    with patch(
                        "unillm.clients.uni_llm.compute_cost_from_response",
                        return_value=0.001,
                    ):
                        client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                        async with acapture_cache_events() as events:
                            await client.generate(
                                messages=[{"role": "user", "content": "Hi"}],
                            )

        assert len(events) == 1
        assert events[0]["cache_status"] == "miss"
        assert events[0]["endpoint"] == "gpt-4@openai"

    @pytest.mark.asyncio
    async def test_async_client_emits_cache_hit_when_cached(self):
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"

        with patch(
            "unillm.clients.uni_llm._get_cache",
            return_value=mock_cached_response,
        ):
            with patch("unillm.clients.uni_llm._write_to_cache"):
                client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                async with acapture_cache_events() as events:
                    await client.generate(messages=[{"role": "user", "content": "Hi"}])

        assert len(events) == 1
        assert events[0]["cache_status"] == "hit"

    def test_event_contains_request_kw(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        with patch(
            "unillm.clients.uni_llm.litellm.completion",
            return_value=mock_response,
        ):
            with patch("unillm.clients.uni_llm._get_cache", return_value=None):
                with patch("unillm.clients.uni_llm._write_to_cache"):
                    with patch(
                        "unillm.clients.uni_llm.compute_cost_from_response",
                        return_value=0.001,
                    ):
                        with patch("unillm.clients.uni_llm.unify.deduct_credits"):
                            client = unillm.Unify(
                                "gpt-4@openai",
                                cache=True,
                                temperature=0.5,
                            )
                            with capture_cache_events() as events:
                                client.generate(
                                    messages=[{"role": "user", "content": "Test"}],
                                )

        event = events[0]
        assert "model" in event["request_kw"]
        assert "messages" in event["request_kw"]
        assert event["request_kw"].get("temperature") == 0.5

    def test_no_capture_context_means_events_dropped(self):
        """Events emitted without a capture context should be silently dropped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        with patch(
            "unillm.clients.uni_llm.litellm.completion",
            return_value=mock_response,
        ):
            with patch("unillm.clients.uni_llm._get_cache", return_value=None):
                with patch("unillm.clients.uni_llm._write_to_cache"):
                    with patch(
                        "unillm.clients.uni_llm.compute_cost_from_response",
                        return_value=0.001,
                    ):
                        with patch("unillm.clients.uni_llm.unify.deduct_credits"):
                            client = unillm.Unify("gpt-4@openai", cache=True)
                            # No capture context - should not error
                            response = client.generate(
                                messages=[{"role": "user", "content": "Hi"}],
                            )

        # Just verify it completed without error
        assert response is not None


# Integration tests - only run when API keys are available
_HAS_API_KEYS = bool(
    os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
)


@pytest.mark.skipif(not _HAS_API_KEYS, reason="No API keys available")
class TestCacheEventEmissionIntegration:
    """Integration tests for cache events with real LLM calls."""

    def test_real_sync_client_emits_cache_event(self):
        from ..settings import SETTINGS

        client = unillm.Unify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        with capture_cache_events() as events:
            client.generate(
                messages=[{"role": "user", "content": "Say 'hello' [cache_events]"}],
            )

        assert len(events) == 1
        assert events[0]["cache_status"] in ("hit", "miss")
        assert events[0]["endpoint"] == SETTINGS.UNILLM_DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_real_async_client_emits_cache_event(self):
        from ..settings import SETTINGS

        client = unillm.AsyncUnify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        async with acapture_cache_events() as events:
            await client.generate(
                messages=[{"role": "user", "content": "Say 'hello' [cache_events]"}],
            )

        assert len(events) == 1
        assert events[0]["cache_status"] in ("hit", "miss")
        assert events[0]["endpoint"] == SETTINGS.UNILLM_DEFAULT_MODEL

    def test_cache_miss_then_hit_sequence(self):
        """Second identical request should be a cache hit."""
        from ..settings import SETTINGS

        client = unillm.Unify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        messages = [
            {
                "role": "user",
                "content": "What is 7+7? Reply with just the number. [cache_events]",
            },
        ]

        # First request
        with capture_cache_events() as events1:
            client.generate(messages=messages)
        first_status = events1[0]["cache_status"]

        # Second identical request
        with capture_cache_events() as events2:
            client.generate(messages=messages)
        second_status = events2[0]["cache_status"]

        # If first was a miss, second must be a hit
        if first_status == "miss":
            assert (
                second_status == "hit"
            ), "Second identical request should be a cache hit"

    @pytest.mark.asyncio
    async def test_async_cache_miss_then_hit_sequence(self):
        """Async: Second identical request should be a cache hit."""
        from ..settings import SETTINGS

        client = unillm.AsyncUnify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        messages = [
            {
                "role": "user",
                "content": "What is 8+8? Reply with just the number. [cache_events]",
            },
        ]

        # First request
        async with acapture_cache_events() as events1:
            await client.generate(messages=messages)
        first_status = events1[0]["cache_status"]

        # Second identical request
        async with acapture_cache_events() as events2:
            await client.generate(messages=messages)
        second_status = events2[0]["cache_status"]

        # If first was a miss, second must be a hit
        if first_status == "miss":
            assert (
                second_status == "hit"
            ), "Second identical request should be a cache hit"
