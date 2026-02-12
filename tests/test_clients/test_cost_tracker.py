"""Tests for cost event capture using context managers."""

import os
import pytest
from unittest.mock import patch, MagicMock

import unillm
from unillm import (
    capture_costs,
    acapture_costs,
    summarize_costs,
    CostEvent,
    CostSummary,
)
from unillm.cost_tracker import _emit_cost_event

# ---------------------------------------------------------------------------
#  Unit tests for CostEvent / CostSummary dataclasses
# ---------------------------------------------------------------------------


class TestCostEventDataclass:
    """Tests for the CostEvent dataclass."""

    def test_create_event_minimal(self):
        event = CostEvent(model="gpt-4")
        assert event.model == "gpt-4"
        assert event.provider_cost == 0.0
        assert event.billed_cost == 0.0
        assert event.prompt_tokens == 0
        assert event.completion_tokens == 0
        assert event.cache_status == "disabled"

    def test_create_event_with_costs(self):
        event = CostEvent(
            model="gpt-4",
            provider_cost=0.001,
            billed_cost=0.005,
            prompt_tokens=100,
            completion_tokens=50,
            cache_status="miss",
        )
        assert event.provider_cost == 0.001
        assert event.billed_cost == 0.005
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50
        assert event.cache_status == "miss"

    def test_cache_hit_event_has_zero_cost(self):
        event = CostEvent(
            model="gpt-4",
            provider_cost=0.0,
            billed_cost=0.0,
            cache_status="hit",
        )
        assert event.provider_cost == 0.0
        assert event.billed_cost == 0.0
        assert event.cache_status == "hit"


class TestCostSummaryDataclass:
    """Tests for the CostSummary dataclass."""

    def test_create_empty_summary(self):
        summary = CostSummary()
        assert summary.total_provider_cost == 0.0
        assert summary.total_billed_cost == 0.0
        assert summary.total_prompt_tokens == 0
        assert summary.total_completion_tokens == 0
        assert summary.total_requests == 0
        assert summary.cache_hits == 0
        assert summary.cache_misses == 0
        assert summary.events == []


# ---------------------------------------------------------------------------
#  Unit tests for capture_costs / acapture_costs context managers
# ---------------------------------------------------------------------------


class TestCaptureCostsContextManager:
    """Tests for the capture_costs context manager."""

    def test_capture_returns_empty_list_initially(self):
        with capture_costs() as events:
            pass
        assert events == []

    def test_capture_receives_emitted_events(self):
        with capture_costs() as events:
            _emit_cost_event(
                CostEvent(
                    model="gpt-4",
                    provider_cost=0.001,
                    billed_cost=0.005,
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_status="miss",
                ),
            )
        assert len(events) == 1
        assert events[0].model == "gpt-4"
        assert events[0].provider_cost == 0.001

    def test_capture_receives_multiple_events(self):
        with capture_costs() as events:
            _emit_cost_event(
                CostEvent(model="gpt-4", provider_cost=0.001, cache_status="miss"),
            )
            _emit_cost_event(
                CostEvent(model="gpt-4", provider_cost=0.0, cache_status="hit"),
            )
        assert len(events) == 2
        assert events[0].cache_status == "miss"
        assert events[1].cache_status == "hit"

    def test_events_outside_context_not_captured(self):
        # Emit before context
        _emit_cost_event(CostEvent(model="before"))

        with capture_costs() as events:
            _emit_cost_event(CostEvent(model="inside"))

        # Emit after context
        _emit_cost_event(CostEvent(model="after"))

        # Only the inside event should be captured
        assert len(events) == 1
        assert events[0].model == "inside"

    def test_nested_contexts_are_independent(self):
        with capture_costs() as outer_events:
            _emit_cost_event(CostEvent(model="outer"))

            with capture_costs() as inner_events:
                _emit_cost_event(CostEvent(model="inner"))

            # After inner context, outer should receive events again
            _emit_cost_event(CostEvent(model="outer-again"))

        # Inner only captured inner event
        assert len(inner_events) == 1
        assert inner_events[0].model == "inner"

        # Outer captured outer events (but not inner, due to context override)
        assert len(outer_events) == 2
        assert outer_events[0].model == "outer"
        assert outer_events[1].model == "outer-again"


class TestAsyncCaptureCostsContextManager:
    """Tests for the acapture_costs async context manager."""

    @pytest.mark.asyncio
    async def test_async_capture_returns_empty_list_initially(self):
        async with acapture_costs() as events:
            pass
        assert events == []

    @pytest.mark.asyncio
    async def test_async_capture_receives_emitted_events(self):
        async with acapture_costs() as events:
            _emit_cost_event(
                CostEvent(
                    model="gpt-4",
                    provider_cost=0.002,
                    billed_cost=0.01,
                    cache_status="miss",
                ),
            )
        assert len(events) == 1
        assert events[0].provider_cost == 0.002

    @pytest.mark.asyncio
    async def test_async_events_outside_context_not_captured(self):
        _emit_cost_event(CostEvent(model="before"))

        async with acapture_costs() as events:
            _emit_cost_event(CostEvent(model="inside"))

        _emit_cost_event(CostEvent(model="after"))

        assert len(events) == 1
        assert events[0].model == "inside"


# ---------------------------------------------------------------------------
#  Unit tests for summarize_costs
# ---------------------------------------------------------------------------


class TestSummarizeCosts:
    """Tests for the summarize_costs aggregation function."""

    def test_empty_list(self):
        summary = summarize_costs([])
        assert summary.total_requests == 0
        assert summary.total_provider_cost == 0.0
        assert summary.total_billed_cost == 0.0

    def test_single_event(self):
        events = [
            CostEvent(
                model="gpt-4",
                provider_cost=0.001,
                billed_cost=0.005,
                prompt_tokens=100,
                completion_tokens=50,
                cache_status="miss",
            ),
        ]
        summary = summarize_costs(events)
        assert summary.total_requests == 1
        assert summary.total_provider_cost == 0.001
        assert summary.total_billed_cost == 0.005
        assert summary.total_prompt_tokens == 100
        assert summary.total_completion_tokens == 50
        assert summary.cache_hits == 0
        assert summary.cache_misses == 1
        assert len(summary.events) == 1

    def test_multiple_events(self):
        events = [
            CostEvent(
                model="gpt-4",
                provider_cost=0.001,
                billed_cost=0.005,
                prompt_tokens=100,
                completion_tokens=50,
                cache_status="miss",
            ),
            CostEvent(
                model="gpt-4",
                provider_cost=0.0,
                billed_cost=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                cache_status="hit",
            ),
            CostEvent(
                model="gpt-4",
                provider_cost=0.002,
                billed_cost=0.01,
                prompt_tokens=200,
                completion_tokens=100,
                cache_status="miss",
            ),
        ]
        summary = summarize_costs(events)
        assert summary.total_requests == 3
        assert summary.total_provider_cost == pytest.approx(0.003)
        assert summary.total_billed_cost == pytest.approx(0.015)
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 150
        assert summary.cache_hits == 1
        assert summary.cache_misses == 2
        assert len(summary.events) == 3

    def test_disabled_cache_status_not_counted_as_hit_or_miss(self):
        events = [
            CostEvent(model="gpt-4", provider_cost=0.001, cache_status="disabled"),
        ]
        summary = summarize_costs(events)
        assert summary.cache_hits == 0
        assert summary.cache_misses == 0
        assert summary.total_requests == 1


# ---------------------------------------------------------------------------
#  Mocked integration tests: verify cost events are emitted during generate()
# ---------------------------------------------------------------------------


class TestCostEventEmissionMocked:
    """Tests for cost event emission during LLM requests using mocked LLM calls."""

    @pytest.fixture(autouse=True)
    def mock_logging(self):
        """Mock logging functions to prevent log file creation in mocked tests."""
        with patch("unillm.clients.uni_llm.write_request_pending", return_value=None):
            with patch("unillm.clients.uni_llm.append_response_and_finalize"):
                yield

    def test_sync_client_emits_cost_event_on_cache_miss(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

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
                            with capture_costs() as events:
                                client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        assert len(events) == 1
        event = events[0]
        assert event.cache_status == "miss"
        assert event.provider_cost == 0.001
        assert event.billed_cost == 0.005  # 0.001 * 5 (default margin)
        assert event.prompt_tokens == 10
        assert event.completion_tokens == 5

    def test_sync_client_emits_cost_event_on_cache_hit(self):
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"
        mock_cached_response.usage = MagicMock()
        mock_cached_response.usage.prompt_tokens = 10
        mock_cached_response.usage.completion_tokens = 5

        with patch(
            "unillm.clients.uni_llm._get_cache",
            return_value=mock_cached_response,
        ):
            with patch("unillm.clients.uni_llm._write_to_cache"):
                client = unillm.Unify("gpt-4@openai", cache=True)
                with capture_costs() as events:
                    client.generate(messages=[{"role": "user", "content": "Hi"}])

        assert len(events) == 1
        event = events[0]
        assert event.cache_status == "hit"
        # Cache hits are free
        assert event.provider_cost == 0.0
        assert event.billed_cost == 0.0

    @pytest.mark.asyncio
    async def test_async_client_emits_cost_event_on_cache_miss(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10

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
                        return_value=0.002,
                    ):
                        client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                        async with acapture_costs() as events:
                            await client.generate(
                                messages=[{"role": "user", "content": "Hi"}],
                            )

        assert len(events) == 1
        event = events[0]
        assert event.cache_status == "miss"
        assert event.provider_cost == 0.002
        assert event.billed_cost == 0.01  # 0.002 * 5
        assert event.prompt_tokens == 20
        assert event.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_async_client_emits_cost_event_on_cache_hit(self):
        mock_cached_response = MagicMock()
        mock_cached_response.choices = [MagicMock()]
        mock_cached_response.choices[0].message.content = "Cached response"
        mock_cached_response.usage = MagicMock()
        mock_cached_response.usage.prompt_tokens = 20
        mock_cached_response.usage.completion_tokens = 10

        with patch(
            "unillm.clients.uni_llm._get_cache",
            return_value=mock_cached_response,
        ):
            with patch("unillm.clients.uni_llm._write_to_cache"):
                client = unillm.AsyncUnify("gpt-4@openai", cache=True)
                async with acapture_costs() as events:
                    await client.generate(
                        messages=[{"role": "user", "content": "Hi"}],
                    )

        assert len(events) == 1
        event = events[0]
        assert event.cache_status == "hit"
        assert event.provider_cost == 0.0
        assert event.billed_cost == 0.0

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

    def test_summarize_after_capture(self):
        """End-to-end: capture + summarize."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model_dump.return_value = {}
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

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
                            with capture_costs() as events:
                                client.generate(
                                    messages=[{"role": "user", "content": "Hi"}],
                                )

        summary = summarize_costs(events)
        assert summary.total_requests == 1
        assert summary.total_provider_cost == 0.001
        assert summary.total_billed_cost == 0.005
        assert summary.cache_misses == 1
        assert summary.cache_hits == 0


# ---------------------------------------------------------------------------
#  Integration tests - only run when API keys are available
# ---------------------------------------------------------------------------

_HAS_API_KEYS = bool(
    os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
)


@pytest.mark.skipif(not _HAS_API_KEYS, reason="No API keys available")
class TestCostEventEmissionIntegration:
    """Integration tests for cost events with real LLM calls."""

    def test_real_sync_client_emits_cost_event(self):
        from ..settings import SETTINGS

        client = unillm.Unify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        with capture_costs() as events:
            client.generate(
                messages=[{"role": "user", "content": "Say 'hello' [cost_tracker]"}],
            )

        assert len(events) == 1
        event = events[0]
        assert event.cache_status in ("hit", "miss", "disabled")
        # If it was a cache miss, there should be a cost
        if event.cache_status == "miss":
            assert event.provider_cost > 0
            assert event.billed_cost > 0

    @pytest.mark.asyncio
    async def test_real_async_client_emits_cost_event(self):
        from ..settings import SETTINGS

        client = unillm.AsyncUnify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        async with acapture_costs() as events:
            await client.generate(
                messages=[{"role": "user", "content": "Say 'hello' [cost_tracker]"}],
            )

        assert len(events) == 1
        event = events[0]
        assert event.cache_status in ("hit", "miss", "disabled")

    def test_real_summarize_costs(self):
        from ..settings import SETTINGS

        client = unillm.Unify(
            SETTINGS.UNILLM_DEFAULT_MODEL,
            cache=SETTINGS.UNILLM_CACHE,
            cache_backend=SETTINGS.UNILLM_CACHE_BACKEND,
        )
        with capture_costs() as events:
            client.generate(
                messages=[
                    {
                        "role": "user",
                        "content": "What is 2+2? Reply with just the number. [cost_tracker]",
                    },
                ],
            )

        summary = summarize_costs(events)
        assert summary.total_requests == 1
        assert summary.total_requests == summary.cache_hits + summary.cache_misses + (
            1 if events[0].cache_status == "disabled" else 0
        )
