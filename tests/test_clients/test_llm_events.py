"""Tests for LLM event hooks."""

import os
import pytest
from unittest.mock import patch, MagicMock

import unillm
from unillm import (
    LLMEvent,
    set_llm_event_hook,
    get_llm_event_hook,
    llm_event_hook_scope,
    allm_event_hook_scope,
)
from unillm.llm_events import _emit_llm_event


class TestLLMEventDataclass:
    """Tests for the LLMEvent dataclass."""

    def test_create_event(self):
        event = LLMEvent(
            endpoint="gpt-4@openai",
            model="gpt-4",
            provider="openai",
            request_kw={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert event.endpoint == "gpt-4@openai"
        assert event.model == "gpt-4"
        assert event.provider == "openai"
        assert event.response is None
        assert event.cache_status is None
        assert event.error is None
        assert event.stream is False

    def test_create_event_with_response(self):
        mock_response = MagicMock()
        event = LLMEvent(
            endpoint="claude-4@anthropic",
            model="claude-4",
            provider="anthropic",
            request_kw={"messages": []},
            response=mock_response,
            cache_status="miss",
            stream=False,
        )
        assert event.response is mock_response
        assert event.cache_status == "miss"

    def test_create_error_event(self):
        error = Exception("API error")
        event = LLMEvent(
            endpoint="gpt-4@openai",
            model="gpt-4",
            provider="openai",
            request_kw={},
            error=error,
            cache_status="error",
        )
        assert event.error is error
        assert event.cache_status == "error"

    def test_streaming_event(self):
        event = LLMEvent(
            endpoint="gpt-4@openai",
            model="gpt-4",
            provider="openai",
            request_kw={},
            stream=True,
        )
        assert event.stream is True


class TestSetLLMEventHook:
    """Tests for set_llm_event_hook and get_llm_event_hook."""

    def test_initially_no_hook(self):
        # Clear any existing hook first
        set_llm_event_hook(None)
        assert get_llm_event_hook() is None

    def test_set_and_get_hook(self):
        def my_hook(event: LLMEvent) -> None:
            pass

        set_llm_event_hook(my_hook)
        assert get_llm_event_hook() is my_hook

        # Clean up
        set_llm_event_hook(None)

    def test_clear_hook_with_none(self):
        def my_hook(event: LLMEvent) -> None:
            pass

        set_llm_event_hook(my_hook)
        set_llm_event_hook(None)
        assert get_llm_event_hook() is None


class TestEmitLLMEvent:
    """Tests for _emit_llm_event."""

    def test_emit_to_hook(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        set_llm_event_hook(capture_hook)
        try:
            event = LLMEvent(
                endpoint="test@provider",
                model="test",
                provider="provider",
                request_kw={},
            )
            _emit_llm_event(event)

            assert len(captured) == 1
            assert captured[0] is event
        finally:
            set_llm_event_hook(None)

    def test_emit_without_hook_is_silent(self):
        set_llm_event_hook(None)
        # Should not raise
        event = LLMEvent(
            endpoint="test@provider",
            model="test",
            provider="provider",
            request_kw={},
        )
        _emit_llm_event(event)

    def test_hook_exception_is_swallowed(self):
        def bad_hook(event: LLMEvent) -> None:
            raise RuntimeError("Hook failed!")

        set_llm_event_hook(bad_hook)
        try:
            event = LLMEvent(
                endpoint="test@provider",
                model="test",
                provider="provider",
                request_kw={},
            )
            # Should not raise even though hook raises
            _emit_llm_event(event)
        finally:
            set_llm_event_hook(None)


class TestLLMEventHookScope:
    """Tests for llm_event_hook_scope context manager."""

    def test_scope_captures_events(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with llm_event_hook_scope(capture_hook):
            _emit_llm_event(
                LLMEvent(
                    endpoint="test@provider",
                    model="test",
                    provider="provider",
                    request_kw={},
                ),
            )

        assert len(captured) == 1

    def test_scope_restores_previous_hook(self):
        original_captured = []
        scoped_captured = []

        def original_hook(event: LLMEvent) -> None:
            original_captured.append(event)

        def scoped_hook(event: LLMEvent) -> None:
            scoped_captured.append(event)

        set_llm_event_hook(original_hook)
        try:
            # Emit to original
            _emit_llm_event(
                LLMEvent(
                    endpoint="before@provider",
                    model="test",
                    provider="provider",
                    request_kw={},
                ),
            )

            with llm_event_hook_scope(scoped_hook):
                _emit_llm_event(
                    LLMEvent(
                        endpoint="inside@provider",
                        model="test",
                        provider="provider",
                        request_kw={},
                    ),
                )

            # After scope, emit should go back to original
            _emit_llm_event(
                LLMEvent(
                    endpoint="after@provider",
                    model="test",
                    provider="provider",
                    request_kw={},
                ),
            )

            assert len(original_captured) == 2
            assert original_captured[0].endpoint == "before@provider"
            assert original_captured[1].endpoint == "after@provider"

            assert len(scoped_captured) == 1
            assert scoped_captured[0].endpoint == "inside@provider"
        finally:
            set_llm_event_hook(None)

    def test_scope_restores_none_hook(self):
        set_llm_event_hook(None)
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with llm_event_hook_scope(capture_hook):
            _emit_llm_event(
                LLMEvent(
                    endpoint="test@provider",
                    model="test",
                    provider="provider",
                    request_kw={},
                ),
            )

        assert get_llm_event_hook() is None


class TestAsyncLLMEventHookScope:
    """Tests for allm_event_hook_scope async context manager."""

    @pytest.mark.asyncio
    async def test_async_scope_captures_events(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        async with allm_event_hook_scope(capture_hook):
            _emit_llm_event(
                LLMEvent(
                    endpoint="test@provider",
                    model="test",
                    provider="provider",
                    request_kw={},
                ),
            )

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_async_scope_restores_previous_hook(self):
        set_llm_event_hook(None)
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        async with allm_event_hook_scope(capture_hook):
            _emit_llm_event(
                LLMEvent(
                    endpoint="inside@provider",
                    model="test",
                    provider="provider",
                    request_kw={},
                ),
            )

        assert get_llm_event_hook() is None
        assert len(captured) == 1


class TestLLMEventEmissionMocked:
    """Tests for LLM event emission during LLM requests using mocked LLM calls."""

    @pytest.fixture(autouse=True)
    def mock_logging(self):
        """Mock logging functions to prevent log file creation in mocked tests."""
        with patch("unillm.clients.uni_llm.write_request_pending", return_value=None):
            with patch("unillm.clients.uni_llm.append_response_and_finalize"):
                yield

    @pytest.fixture(autouse=True)
    def clear_hook(self):
        """Clear hook before and after each test."""
        set_llm_event_hook(None)
        yield
        set_llm_event_hook(None)

    def test_sync_client_emits_event(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

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
                            with llm_event_hook_scope(capture_hook):
                                client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        # Should have one event per LLM call
        assert len(captured) == 1

        event = captured[0]
        assert event.endpoint == "gpt-4@openai"
        assert event.model == "gpt-4"
        assert event.provider == "openai"
        assert event.stream is False
        assert "messages" in event.request_kw
        assert event.cache_status == "miss"
        assert event.error is None
        assert event.response is mock_response

    def test_sync_client_emits_cache_hit_status(self):
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with patch(
            "unillm.clients.uni_llm._get_cache",
            return_value=mock_cached_response,
        ):
            with patch("unillm.clients.uni_llm._write_to_cache"):
                client = unillm.Unify("gpt-4@openai", cache=True)
                with llm_event_hook_scope(capture_hook):
                    client.generate(messages=[{"role": "user", "content": "Hi"}])

        assert len(captured) == 1
        assert captured[0].cache_status == "hit"

    def test_sync_client_captures_error(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with patch("unillm.clients.uni_llm._get_cache", return_value=None):
            with patch(
                "unillm.clients.uni_llm.litellm.completion",
                side_effect=Exception("API Error"),
            ):
                client = unillm.Unify("gpt-4@openai", cache=True)
                with llm_event_hook_scope(capture_hook):
                    with pytest.raises(Exception, match="API Error"):
                        client.generate(
                            messages=[{"role": "user", "content": "Hi"}],
                        )

        # Should still emit event even on error
        assert len(captured) == 1

        event = captured[0]
        assert event.error is not None
        assert "API Error" in str(event.error)
        assert event.cache_status == "error"

    @pytest.mark.asyncio
    async def test_async_client_emits_event(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        async def mock_acompletion(*args, **kwargs):
            return mock_response

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

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
                        with patch("unillm.clients.uni_llm.asyncio.create_task"):
                            client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                            async with allm_event_hook_scope(capture_hook):
                                await client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        # Should have one event per LLM call
        assert len(captured) == 1

        event = captured[0]
        assert event.endpoint == "gpt-4@openai"
        assert event.cache_status == "miss"
        assert event.error is None

    @pytest.mark.asyncio
    async def test_async_client_captures_error(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        async def mock_acompletion_error(*args, **kwargs):
            raise Exception("Async API Error")

        with patch("unillm.clients.uni_llm._get_cache", return_value=None):
            with patch(
                "unillm.clients.uni_llm.litellm.acompletion",
                side_effect=mock_acompletion_error,
            ):
                client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                async with allm_event_hook_scope(capture_hook):
                    with pytest.raises(Exception, match="Async API Error"):
                        await client.generate(
                            messages=[{"role": "user", "content": "Hi"}],
                        )

        assert len(captured) == 1
        assert captured[0].error is not None
        assert captured[0].cache_status == "error"

    def test_request_kw_contains_messages_and_model(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

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
                                temperature=0.7,
                            )
                            with llm_event_hook_scope(capture_hook):
                                client.generate(
                                    messages=[{"role": "user", "content": "Test"}],
                                )

        event = captured[0]
        assert "model" in event.request_kw
        assert "messages" in event.request_kw
        assert event.request_kw.get("temperature") == 0.7


class TestStreamingLLMEvents:
    """Tests for LLM events during streaming requests."""

    @pytest.fixture(autouse=True)
    def mock_logging(self):
        """Mock logging functions to prevent log file creation."""
        with patch("unillm.clients.uni_llm.write_request_pending", return_value=None):
            with patch("unillm.clients.uni_llm.append_response_and_finalize"):
                yield

    @pytest.fixture(autouse=True)
    def clear_hook(self):
        """Clear hook before and after each test."""
        set_llm_event_hook(None)
        yield
        set_llm_event_hook(None)

    def test_sync_streaming_emits_event(self):
        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk2.usage = None

        def mock_completion(*args, **kwargs):
            return iter([mock_chunk1, mock_chunk2])

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with patch(
            "unillm.clients.uni_llm.litellm.completion",
            side_effect=mock_completion,
        ):
            client = unillm.Unify("gpt-4@openai", stream=True)
            with llm_event_hook_scope(capture_hook):
                # Consume the generator
                list(client.generate(messages=[{"role": "user", "content": "Hi"}]))

        # Should have one event per LLM call (after streaming completes)
        assert len(captured) == 1

        event = captured[0]
        assert event.stream is True
        assert event.cache_status is None  # Streaming doesn't use cache
        assert event.response is None  # No single response for streams

    @pytest.mark.asyncio
    async def test_async_streaming_emits_event(self):
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk2.usage = None

        async def mock_acompletion(*args, **kwargs):
            async def async_gen():
                yield mock_chunk1
                yield mock_chunk2

            return async_gen()

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with patch(
            "unillm.clients.uni_llm.litellm.acompletion",
            side_effect=mock_acompletion,
        ):
            client = unillm.AsyncUnify("gpt-4@openai", stream=True)
            async with allm_event_hook_scope(capture_hook):
                # Consume the async generator
                result = []
                gen = await client.generate(
                    messages=[{"role": "user", "content": "Hi"}],
                )
                async for chunk in gen:
                    result.append(chunk)

        assert len(captured) == 1

        event = captured[0]
        assert event.stream is True


# Integration tests - only run when API keys are available
_HAS_API_KEYS = bool(
    os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
)


@pytest.mark.skipif(not _HAS_API_KEYS, reason="No API keys available")
class TestLLMEventEmissionIntegration:
    """Integration tests for LLM events with real LLM calls."""

    @pytest.fixture(autouse=True)
    def clear_hook(self):
        """Clear hook before and after each test."""
        set_llm_event_hook(None)
        yield
        set_llm_event_hook(None)

    def test_real_sync_client_emits_event(self):
        from ..settings import SETTINGS

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        client = unillm.Unify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        with llm_event_hook_scope(capture_hook):
            client.generate(
                messages=[{"role": "user", "content": "Say 'hello' [llm_events]"}],
            )

        assert len(captured) == 1
        assert captured[0].endpoint == SETTINGS.UNILLM_DEFAULT_MODEL
        assert captured[0].cache_status in ("hit", "miss")

    @pytest.mark.asyncio
    async def test_real_async_client_emits_event(self):
        from ..settings import SETTINGS

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        client = unillm.AsyncUnify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        async with allm_event_hook_scope(capture_hook):
            await client.generate(
                messages=[{"role": "user", "content": "Say 'hello' [llm_events]"}],
            )

        assert len(captured) == 1
        assert captured[0].cache_status in ("hit", "miss")
