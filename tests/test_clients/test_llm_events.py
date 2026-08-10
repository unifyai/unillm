"""Tests for LLM event hooks."""

import pytest
from unittest.mock import patch, MagicMock

import unillm
from unillm import (
    LLMEvent,
    add_llm_event_listener,
    clear_llm_event_listeners,
    llm_event_listeners,
    set_llm_event_hook,
    get_llm_event_hook,
    llm_event_hook_scope,
    allm_event_hook_scope,
)
from unillm.endpoints.utils import get_model_alias
from unillm.llm_events import _emit_llm_event


class TestLLMEventDataclass:
    """Tests for the LLMEvent dataclass."""

    def test_create_event_minimal(self):
        event = LLMEvent(
            request={
                "model": "openai/gpt-4o@openrouter",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert event.request["model"] == "openai/gpt-4o@openrouter"
        assert event.response is None
        assert event.provider_cost is None

    def test_create_event_with_response(self):
        event = LLMEvent(
            request={"model": "openai/gpt-4o@openrouter", "messages": []},
            response={"id": "chatcmpl-123", "model": "gpt-4", "choices": []},
        )
        assert event.response is not None
        assert event.response["id"] == "chatcmpl-123"

    def test_create_event_with_costs(self):
        event = LLMEvent(
            request={"model": "openai/gpt-4o@openrouter"},
            provider_cost=0.001,
        )
        assert event.provider_cost == 0.001

    def test_event_costs_default_to_none(self):
        event = LLMEvent(request={"model": "openai/gpt-4o@openrouter"})
        assert event.provider_cost is None


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
                        with patch("unillm.clients.uni_llm.unisdk.deduct_credits"):
                            client = unillm.Unify(
                                "openai/gpt-4o@openrouter",
                                cache=True,
                            )
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

    def test_structured_output_event_request_is_json_serializable(self):
        """A Pydantic ``response_format`` class must not leak into the event.

        Downstream sinks (EventBus -> Orchestra, file logs) persist the request
        with ``json.dumps``; a raw model class raised ``TypeError: Object of
        type ModelMetaclass is not JSON serializable`` and silently dropped the
        Events/LLM row for every structured-output call.
        """
        import json

        from pydantic import BaseModel

        class _Decision(BaseModel):
            classification: str
            content: str = ""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"classification": "defer"}'
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
                        with patch("unillm.clients.uni_llm.unisdk.deduct_credits"):
                            client = unillm.Unify(
                                "openai/gpt-4o@openrouter",
                                cache=True,
                            )
                            client.set_response_format(_Decision)
                            with llm_event_hook_scope(capture_hook):
                                client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        assert len(captured) == 1
        event = captured[0]
        # The class is replaced by its JSON schema, keeping the contract visible.
        serialized = json.dumps(event.request)
        assert "ModelMetaclass" not in serialized
        assert event.request["response_format"]["json_schema"]["name"] == "_Decision"
        assert (
            "classification"
            in event.request["response_format"]["json_schema"]["schema"]["properties"]
        )

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
                client = unillm.Unify("openai/gpt-4o@openrouter", cache=True)
                with llm_event_hook_scope(capture_hook):
                    client.generate(messages=[{"role": "user", "content": "Hi"}])

        assert len(captured) == 1
        # Cache hits don't compute costs
        assert captured[0].provider_cost is None

    def test_sync_client_captures_error_with_no_response(self):
        captured = []

        def capture_hook(event: LLMEvent) -> None:
            captured.append(event)

        with patch("unillm.clients.uni_llm._get_cache", return_value=None):
            with patch(
                "unillm.clients.uni_llm.litellm.completion",
                side_effect=Exception("API Error"),
            ):
                client = unillm.Unify("openai/gpt-4o@openrouter", cache=True)
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
                        client = unillm.AsyncUnify(
                            "openai/gpt-4o@openrouter",
                            cache=True,
                        )
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
                client = unillm.AsyncUnify("openai/gpt-4o@openrouter", cache=True)
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
                        with patch("unillm.clients.uni_llm.unisdk.deduct_credits"):
                            client = unillm.Unify(
                                "openai/gpt-4o@openrouter",
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
            client = unillm.Unify("openai/gpt-4o@openrouter", stream=True)
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
            client = unillm.AsyncUnify("openai/gpt-4o@openrouter", stream=True)
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
        """Non-streaming events should include the call's cost."""
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
                        with patch("unillm.clients.uni_llm.unisdk.deduct_credits"):
                            client = unillm.Unify(
                                "openai/gpt-4o@openrouter",
                                cache=True,
                            )
                            with llm_event_hook_scope(capture_hook):
                                client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        assert len(captured) == 1
        event = captured[0]

        # Provider cost should be set
        assert event.provider_cost == 0.001

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
                client = unillm.Unify("openai/gpt-4o@openrouter", cache=True)
                with llm_event_hook_scope(capture_hook):
                    client.generate(messages=[{"role": "user", "content": "Hi"}])

        assert len(captured) == 1
        event = captured[0]
        # Cache hits are free - no costs
        assert event.provider_cost is None

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
                client = unillm.Unify("openai/gpt-4o@openrouter", cache=True)
                with llm_event_hook_scope(capture_hook):
                    with pytest.raises(Exception, match="API Error"):
                        client.generate(
                            messages=[{"role": "user", "content": "Hi"}],
                        )

        assert len(captured) == 1
        event = captured[0]
        # No costs on error
        assert event.provider_cost is None

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
                        client = unillm.AsyncUnify(
                            "openai/gpt-4o@openrouter",
                            cache=True,
                        )
                        async with allm_event_hook_scope(capture_hook):
                            await client.generate(
                                messages=[{"role": "user", "content": "Hi"}],
                            )

        assert len(captured) == 1
        event = captured[0]

        assert event.provider_cost == 0.002


# ---------------------------------------------------------------------------
#  Process-global listener tests
# ---------------------------------------------------------------------------


class TestLLMEventListeners:
    """Tests for add_llm_event_listener - additive, process-wide metering."""

    @pytest.fixture(autouse=True)
    def clear_hooks(self):
        """Clear the scoped hook and every listener around each test."""
        set_llm_event_hook(None)
        clear_llm_event_listeners()
        yield
        set_llm_event_hook(None)
        clear_llm_event_listeners()

    def test_listener_receives_events(self):
        captured = []
        listener = add_llm_event_listener(captured.append)

        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert len(captured) == 1
        assert listener.delivered == 1
        assert listener.healthy

    def test_second_listener_does_not_displace_the_first(self):
        """Registration is additive: no last-write-wins race.

        A metering consumer that installs after the runtime's own wiring used to
        have to read the incumbent hook and chain to it, and whichever consumer
        wrote last owned the slot. Both must now receive every event regardless
        of registration order.
        """
        first, second = [], []
        add_llm_event_listener(first.append)
        add_llm_event_listener(second.append)

        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert len(first) == 1
        assert len(second) == 1

    def test_removing_one_listener_leaves_the_other(self):
        first, second = [], []
        handle = add_llm_event_listener(first.append)
        add_llm_event_listener(second.append)

        handle.remove()
        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert first == []
        assert len(second) == 1

    def test_remove_is_idempotent(self):
        handle = add_llm_event_listener(lambda event: None)
        handle.remove()
        handle.remove()
        assert llm_event_listeners() == ()

    def test_listener_called_from_different_thread(self):
        """A listener must see calls made from a worker thread.

        The production case: the listener is registered at startup in one
        thread, but LLM calls may happen from worker threads.
        """
        import concurrent.futures

        captured = []
        add_llm_event_listener(captured.append)

        def emit_in_thread():
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(emit_in_thread).result()

        assert len(captured) == 1
        assert captured[0].request["model"] == "test@provider"

    def test_listener_registered_in_thread_where_it_was_not_emitted(self):
        """Registering in thread A must cover events emitted in thread B.

        This mimics the production scenario where:
        - unify.init() registers the listener (in a worker thread via
          asyncio.to_thread)
        - LLM calls happen from the main async context (different thread)
        """
        import concurrent.futures

        captured = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(add_llm_event_listener, captured.append).result()

        _emit_llm_event(LLMEvent(request={"model": "main-thread@provider"}))

        assert len(captured) == 1
        assert captured[0].request["model"] == "main-thread@provider"

    def test_listener_survives_a_fresh_event_loop(self):
        """Registration is not per-loop state.

        Work dispatched onto a loop created after registration — a nested
        asyncio.run, a runtime that builds its own loop per turn — must still
        be metered.
        """
        import asyncio

        captured = []
        add_llm_event_listener(captured.append)

        async def emit():
            _emit_llm_event(LLMEvent(request={"model": "fresh-loop@provider"}))

        asyncio.run(emit())
        asyncio.run(emit())

        assert len(captured) == 2

    def test_listener_registered_inside_a_loop_outlives_it(self):
        """A listener registered inside one loop still fires under the next.

        ContextVar-based state would be discarded with the context that set it;
        listener registration is a plain module-level registry, so it is not.
        """
        import asyncio

        captured = []

        async def register():
            add_llm_event_listener(captured.append)

        asyncio.run(register())
        _emit_llm_event(LLMEvent(request={"model": "after-loop@provider"}))

        assert len(captured) == 1

    def test_scoped_hook_does_not_suppress_listeners(self):
        """A scoped hook is additive, not an override.

        A scoped capture anywhere in the call path used to displace the
        process-wide hook entirely, so whole stretches of a run went unmetered
        with nothing to show it. Both recipients must see the event.
        """
        listener_captured, scoped_captured = [], []
        add_llm_event_listener(listener_captured.append)

        _emit_llm_event(LLMEvent(request={"model": "before@provider"}))

        with llm_event_hook_scope(scoped_captured.append):
            _emit_llm_event(LLMEvent(request={"model": "inside@provider"}))

        _emit_llm_event(LLMEvent(request={"model": "after@provider"}))

        assert [e.request["model"] for e in listener_captured] == [
            "before@provider",
            "inside@provider",
            "after@provider",
        ]
        assert [e.request["model"] for e in scoped_captured] == ["inside@provider"]

    def test_raising_listener_does_not_break_the_llm_call(self):
        def bad_listener(event: LLMEvent) -> None:
            raise RuntimeError("listener failed!")

        add_llm_event_listener(bad_listener)
        # Must not raise into the caller
        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

    def test_raising_listener_does_not_starve_the_others(self):
        def bad_listener(event: LLMEvent) -> None:
            raise RuntimeError("listener failed!")

        before, after = [], []
        add_llm_event_listener(before.append)
        add_llm_event_listener(bad_listener)
        add_llm_event_listener(after.append)

        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert len(before) == 1
        assert len(after) == 1

    def test_raising_listener_reports_its_own_failure(self):
        """A listener that records nothing must not look like a quiet run.

        Isolation keeps a broken listener from failing LLM calls, which also
        means it silently records nothing — and a consumer reporting totals
        cannot tell "$0 was spent" from "spending was never observed". The
        handle carries that distinction.
        """

        def bad_listener(event: LLMEvent) -> None:
            raise RuntimeError("listener failed!")

        listener = add_llm_event_listener(bad_listener)

        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))
        _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert listener.delivered == 0
        assert listener.failed == 2
        assert not listener.healthy
        assert isinstance(listener.last_error, RuntimeError)

    def test_listener_reading_an_absent_event_field_is_reported(self):
        """The exact drift that lost three benchmark runs.

        A consumer reading a field LLMEvent no longer carries raises on every
        event, records nothing, and reports zero cost. The failure has to be
        visible on the handle, because the emit path deliberately swallows it.
        """
        recorded = []

        def stale_listener(event: LLMEvent) -> None:
            recorded.append(event.billed_cost)  # removed from LLMEvent

        listener = add_llm_event_listener(stale_listener)

        _emit_llm_event(
            LLMEvent(request={"model": "test@provider"}, provider_cost=1.25),
        )

        assert recorded == []
        assert listener.failed == 1
        assert not listener.healthy
        assert isinstance(listener.last_error, AttributeError)

    def test_failure_is_logged(self, caplog):
        """A silent drop is the failure mode; the log is the other half of the
        signal for a consumer that never checks the handle."""
        import logging

        def bad_listener(event: LLMEvent) -> None:
            raise RuntimeError("listener failed!")

        add_llm_event_listener(bad_listener)

        with caplog.at_level(logging.ERROR, logger="unillm"):
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert any(
            "listener" in record.message.lower() for record in caplog.records
        ), caplog.text

    def test_raising_scoped_hook_does_not_break_the_call(self):
        def bad_hook(event: LLMEvent) -> None:
            raise RuntimeError("hook failed!")

        captured = []
        add_llm_event_listener(captured.append)

        with llm_event_hook_scope(bad_hook):
            _emit_llm_event(LLMEvent(request={"model": "test@provider"}))

        assert len(captured) == 1


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
        # Request carries the public accounting alias, not the endpoint string.
        assert captured[0].request.get("model") == get_model_alias(
            SETTINGS.UNILLM_DEFAULT_MODEL,
        )
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
