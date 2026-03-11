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

    def test_create_event_minimal(self):
        event = LLMEvent(
            request={
                "model": "gpt-4@openai",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert event.request["model"] == "gpt-4@openai"
        assert event.response is None
        assert event.provider_cost is None
        assert event.billed_cost is None

    def test_create_event_with_response(self):
        event = LLMEvent(
            request={"model": "gpt-4@openai", "messages": []},
            response={"id": "chatcmpl-123", "model": "gpt-4", "choices": []},
        )
        assert event.response is not None
        assert event.response["id"] == "chatcmpl-123"

    def test_create_event_with_costs(self):
        event = LLMEvent(
            request={"model": "gpt-4@openai"},
            provider_cost=0.001,
            billed_cost=0.005,
        )
        assert event.provider_cost == 0.001
        assert event.billed_cost == 0.005

    def test_event_costs_default_to_none(self):
        event = LLMEvent(request={"model": "gpt-4@openai"})
        assert event.provider_cost is None
        assert event.billed_cost is None


class TestCostMargin:
    """Tests for the cost margin configuration."""

    def test_default_margin(self):
        from unillm.costs import get_cost_margin

        # Clear env var if set
        with patch.dict(os.environ, {}, clear=True):
            # Remove UNILLM_COST_MARGIN if it exists
            os.environ.pop("UNILLM_COST_MARGIN", None)
            assert get_cost_margin() == 1.2

    def test_margin_from_env_var(self):
        from unillm.costs import get_cost_margin

        with patch.dict(os.environ, {"UNILLM_COST_MARGIN": "3.5"}):
            assert get_cost_margin() == 3.5

    def test_invalid_margin_falls_back_to_default(self):
        from unillm.costs import get_cost_margin

        with patch.dict(os.environ, {"UNILLM_COST_MARGIN": "not_a_number"}):
            assert get_cost_margin() == 1.2


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
            event = LLMEvent(request={"model": "test@provider"})
            _emit_llm_event(event)

            assert len(captured) == 1
            assert captured[0] is event
        finally:
            set_llm_event_hook(None)

    def test_emit_without_hook_is_silent(self):
        set_llm_event_hook(None)
        # Should not raise
        event = LLMEvent(request={"model": "test@provider"})
        _emit_llm_event(event)

    def test_hook_exception_is_swallowed(self):
        def bad_hook(event: LLMEvent) -> None:
            raise RuntimeError("Hook failed!")

        set_llm_event_hook(bad_hook)
        try:
            event = LLMEvent(request={"model": "test@provider"})
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
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

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
            _emit_llm_event(LLMEvent(request={"model": "before@provider"}))

            with llm_event_hook_scope(scoped_hook):
                _emit_llm_event(LLMEvent(request={"model": "inside@provider"}))

            # After scope, emit should go back to original
            _emit_llm_event(LLMEvent(request={"model": "after@provider"}))

            assert len(original_captured) == 2
            assert original_captured[0].request["model"] == "before@provider"
            assert original_captured[1].request["model"] == "after@provider"

            assert len(scoped_captured) == 1
            assert scoped_captured[0].request["model"] == "inside@provider"
        finally:
            set_llm_event_hook(None)

    def test_scope_restores_none_hook(self):
        set_llm_event_hook(None)
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with llm_event_hook_scope(capture_hook):
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert get_llm_event_hook() is None


class TestAsyncLLMEventHookScope:
    """Tests for allm_event_hook_scope async context manager."""

    @pytest.mark.asyncio
    async def test_async_scope_captures_events(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        async with allm_event_hook_scope(capture_hook):
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_async_scope_restores_previous_hook(self):
        set_llm_event_hook(None)
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        async with allm_event_hook_scope(capture_hook):
            _emit_llm_event(LLMEvent(request={"model": "inside@provider"}))

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
        mock_response.model_dump.return_value = {"id": "test", "choices": []}

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
        # Request should contain the full request data
        assert "model" in event.request
        assert "messages" in event.request
        # Response should be the serialized dict
        assert event.response is not None
        assert event.response["id"] == "test"

    def test_sync_client_emits_cache_hit_with_no_costs(self):
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"
        mock_cached_response.model_dump.return_value = {"id": "cached", "choices": []}

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
        # Cache hits don't compute costs
        assert captured[0].provider_cost is None
        assert captured[0].billed_cost is None

    def test_sync_client_captures_error_with_no_response(self):
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
        # Error means no valid response
        assert event.response is None
        # No costs on error
        assert event.provider_cost is None
        assert event.billed_cost is None

    @pytest.mark.asyncio
    async def test_async_client_emits_event(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {"id": "async-test", "choices": []}

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
                        client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                        async with allm_event_hook_scope(capture_hook):
                            await client.generate(
                                messages=[{"role": "user", "content": "Hi"}],
                            )

        # Should have one event per LLM call
        assert len(captured) == 1

        event = captured[0]
        assert "model" in event.request
        assert event.response is not None
        assert event.response["id"] == "async-test"

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
        assert captured[0].response is None
        assert captured[0].provider_cost is None

    def test_request_contains_full_kwargs(self):
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
        # Request should contain full kwargs
        assert "model" in event.request
        assert "messages" in event.request
        assert event.request.get("temperature") == 0.7


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

    def test_sync_streaming_emits_event_with_no_response(self):
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
        # Streaming has no single response
        assert event.response is None
        # Request still captured
        assert "messages" in event.request

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
        assert captured[0].response is None


class TestLLMEventCosts:
    """Tests for cost information in LLM events."""

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

    def test_sync_client_includes_costs_in_event(self):
        """Non-streaming events should include provider_cost and billed_cost."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

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

        assert len(captured) == 1
        event = captured[0]

        # Provider cost should be set
        assert event.provider_cost == 0.001

        # Billed cost should be provider_cost * margin (default 1.2)
        assert event.billed_cost == pytest.approx(0.0012)

    def test_costs_with_custom_margin(self):
        """Billed cost should respect UNILLM_COST_MARGIN env var."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with patch.dict(os.environ, {"UNILLM_COST_MARGIN": "3"}):
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

        assert len(captured) == 1
        event = captured[0]

        # Provider cost should be set
        assert event.provider_cost == 0.001

        # Billed cost should be provider_cost * 3 (custom margin)
        assert event.billed_cost == 0.003

    def test_cache_hit_has_no_costs(self):
        """Cache hits should not incur costs."""
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"
        mock_cached_response.model_dump.return_value = {}

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
        event = captured[0]
        # Cache hits are free - no costs
        assert event.provider_cost is None
        assert event.billed_cost is None

    def test_error_has_no_costs(self):
        """Errors should not have costs (nothing was computed)."""
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

        assert len(captured) == 1
        event = captured[0]
        # No costs on error
        assert event.provider_cost is None
        assert event.billed_cost is None

    @pytest.mark.asyncio
    async def test_async_client_includes_costs_in_event(self):
        """Async non-streaming events should include costs."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

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
                        return_value=0.002,
                    ):
                        client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                        async with allm_event_hook_scope(capture_hook):
                            await client.generate(
                                messages=[{"role": "user", "content": "Hi"}],
                            )

        assert len(captured) == 1
        event = captured[0]

        assert event.provider_cost == 0.002
        assert event.billed_cost == 0.01  # 0.002 * 5


# ---------------------------------------------------------------------------
#  Cross-thread global hook tests
# ---------------------------------------------------------------------------


class TestGlobalLLMEventHook:
    """Tests for set_global_llm_event_hook - process-wide hook that works across threads."""

    @pytest.fixture(autouse=True)
    def clear_hooks(self):
        """Clear both context and global hooks before and after each test."""
        set_llm_event_hook(None)
        # Clear global hook (will exist after implementation)
        try:
            from unillm import set_global_llm_event_hook

            set_global_llm_event_hook(None)
        except ImportError:
            pass
        yield
        set_llm_event_hook(None)
        try:
            from unillm import set_global_llm_event_hook

            set_global_llm_event_hook(None)
        except ImportError:
            pass

    def test_global_hook_called_from_different_thread(self):
        """Global hook should be called even when LLM call happens in a different thread.

        This is the key test for the production use case: hook is installed at startup
        in one thread, but LLM calls may happen from worker threads.
        """
        import concurrent.futures

        from unillm import set_global_llm_event_hook

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        # Set global hook in main thread
        set_global_llm_event_hook(capture_hook)

        # Emit event from a different thread
        def emit_in_thread():
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(emit_in_thread)
            future.result()  # Wait for completion

        # Global hook should have caught the event
        assert len(captured) == 1
        assert captured[0].request["model"] == "test@provider"

    def test_global_hook_called_from_thread_where_hook_was_not_set(self):
        """Global hook should work when hook is set in thread A but event emitted in thread B.

        This mimics the production scenario where:
        - unity.init() sets the hook (in a worker thread via asyncio.to_thread)
        - LLM calls happen from the main async context (different thread)
        """
        import concurrent.futures

        from unillm import set_global_llm_event_hook

        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        # Set hook from a worker thread (mimicking asyncio.to_thread behavior)
        def set_hook_in_thread():
            set_global_llm_event_hook(capture_hook)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(set_hook_in_thread).result()

        # Now emit event from main thread (hook was set in different thread)
        _emit_llm_event(LLMEvent(request={"model": "main-thread@provider"}))

        assert len(captured) == 1
        assert captured[0].request["model"] == "main-thread@provider"

    def test_context_hook_takes_precedence_over_global_hook(self):
        """Context-specific hook should take precedence over global hook.

        This preserves the existing scoped capture behavior for tests.
        """
        from unillm import set_global_llm_event_hook

        global_captured = []
        context_captured = []

        def global_hook(event: LLMEvent) -> None:
            global_captured.append(event)

        def context_hook(event: LLMEvent) -> None:
            context_captured.append(event)

        set_global_llm_event_hook(global_hook)

        # Without context hook, global should catch it
        _emit_llm_event(LLMEvent(request={"model": "global-only@provider"}))
        assert len(global_captured) == 1
        assert len(context_captured) == 0

        # With context hook, context should catch it (not global)
        with llm_event_hook_scope(context_hook):
            _emit_llm_event(LLMEvent(request={"model": "context-scoped@provider"}))

        assert len(global_captured) == 1  # Still just the first event
        assert len(context_captured) == 1
        assert context_captured[0].request["model"] == "context-scoped@provider"

        # After context exits, global should catch again
        _emit_llm_event(LLMEvent(request={"model": "back-to-global@provider"}))
        assert len(global_captured) == 2
        assert global_captured[1].request["model"] == "back-to-global@provider"


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
        # Request should have the model (note: model name only, not endpoint)
        model_name = SETTINGS.UNILLM_DEFAULT_MODEL.split("@")[0]
        assert captured[0].request.get("model") == model_name
        # Response should be a dict (serialized)
        assert isinstance(captured[0].response, dict)

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
        assert isinstance(captured[0].response, dict)
