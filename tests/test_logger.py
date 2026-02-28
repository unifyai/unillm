"""
Tests for LLM logging and OpenTelemetry tracing functionality.

These tests verify that:
1. The logger module correctly writes request/response payloads to log files
2. OpenTelemetry spans are created when UNILLM_OTEL is enabled
3. Trace context propagates correctly through unillm → unify hierarchy
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from unillm.logger import (
    _expand_string_newlines,
    _normalize_body,
    _sanitize_origin,
    _serialize_kw,
    write_request_pending,
    append_response_and_finalize,
    llm_span,
    set_span_response,
    get_tracer,
    is_otel_enabled,
)

# --------------------------------------------------------------------------- #
#  Fixtures for OTel testing
# --------------------------------------------------------------------------- #


@pytest.fixture
def reset_otel():
    """Reset OTel state before and after test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Reset OTel global state
    # pylint: disable=protected-access
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = None

    trace.set_tracer_provider(provider)

    yield {"provider": provider, "exporter": exporter}

    exporter.clear()


# --------------------------------------------------------------------------- #
#  _serialize_kw tests
# --------------------------------------------------------------------------- #


def test_serialize_kw_simple_dict():
    """Simple dicts pass through unchanged."""
    data = {"model": "gpt-4", "temperature": 0.7}
    result = _serialize_kw(data)
    assert result == data


def test_serialize_kw_messages_list():
    """Messages list is preserved."""
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    result = _serialize_kw(data)
    assert result["messages"] == data["messages"]


def test_serialize_kw_pydantic_instance():
    """Pydantic model instances are serialized via model_dump."""

    class TestModel(BaseModel):
        name: str
        value: int

    data = {"nested": TestModel(name="test", value=42)}
    result = _serialize_kw(data)
    assert result["nested"] == {"name": "test", "value": 42}


def test_serialize_kw_pydantic_class():
    """Pydantic model classes are serialized via model_json_schema."""

    class ResponseFormat(BaseModel):
        answer: str
        confidence: float

    data = {"response_format": ResponseFormat}
    result = _serialize_kw(data)
    assert "__pydantic_schema__" in result["response_format"]
    schema = result["response_format"]["__pydantic_schema__"]
    assert "properties" in schema
    assert "answer" in schema["properties"]


def test_serialize_kw_nested_structures():
    """Nested dicts and lists are handled recursively."""

    class Inner(BaseModel):
        x: int

    data = {
        "outer": {
            "list": [Inner(x=1), Inner(x=2)],
            "dict": {"a": Inner(x=3)},
        },
    }
    result = _serialize_kw(data)
    assert result["outer"]["list"] == [{"x": 1}, {"x": 2}]
    assert result["outer"]["dict"]["a"] == {"x": 3}


def test_serialize_kw_none_values():
    """None values pass through."""
    data = {"model": "gpt-4", "tools": None}
    result = _serialize_kw(data)
    assert result["tools"] is None


def test_serialize_kw_non_json_serializable():
    """Non-JSON-serializable objects are converted to strings."""

    class Custom:
        def __str__(self):
            return "custom-object"

    data = {"custom": Custom()}
    result = _serialize_kw(data)
    assert result["custom"] == "custom-object"


# --------------------------------------------------------------------------- #
#  File writing tests
# --------------------------------------------------------------------------- #


def test_write_request_pending_creates_file(tmp_path, monkeypatch):
    """Writing a pending request creates a timestamped file."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    path = write_request_pending({"model": "gpt-4"}, label="test")

    assert path is not None
    assert path.exists()
    assert ".cache_pending." in path.name

    content = path.read_text()
    assert "🔄 [test] LLM request ➡️" in content
    assert '"model": "gpt-4"' in content


def test_append_response_and_finalize(tmp_path, monkeypatch):
    """Appending response and finalizing renames the file."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    # Write pending request
    pending_path = write_request_pending({"model": "gpt-4"}, label="test")
    assert pending_path is not None

    # Append response and finalize
    append_response_and_finalize(
        pending_path,
        {"choices": [{"message": {"content": "Hello"}}]},
        "hit",
        label="test",
    )

    # Pending file should be gone
    assert not pending_path.exists()

    hit_files = list(tmp_path.glob("*.cache_hit.txt"))
    assert len(hit_files) == 1

    content = hit_files[0].read_text()
    assert "LLM request ➡️" in content
    assert "LLM response ⬅️" in content
    assert "[cache: hit]" in content


def test_append_response_and_finalize_with_none_response(tmp_path, monkeypatch):
    """Finalization works when response is None (exception during LLM call)."""
    from unillm import settings
    from unillm import logger

    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    # Write pending request
    pending_path = write_request_pending({"model": "gpt-4"}, label="test")
    assert pending_path is not None

    # Finalize with None response (simulates exception during LLM call)
    append_response_and_finalize(
        pending_path,
        None,
        "error",
        label="test",
    )

    # Pending file should be gone
    assert not pending_path.exists()

    error_files = list(tmp_path.glob("*.cache_error.txt"))
    assert len(error_files) == 1

    content = error_files[0].read_text()
    assert "LLM request ➡️" in content
    assert "LLM response ⬅️" in content
    assert "[cache: error]" in content


def test_append_response_disabled_drops_cache_extension(tmp_path, monkeypatch):
    """When cache is disabled the final file has no .cache_* extension."""
    from unillm import settings
    from unillm import logger

    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    pending_path = write_request_pending({"model": "gpt-4"}, label="test")
    assert pending_path is not None

    final_path = append_response_and_finalize(
        pending_path,
        {"choices": [{"message": {"content": "Hello"}}]},
        "disabled",
        label="test",
    )

    assert not pending_path.exists()
    assert final_path is not None
    assert ".cache_" not in final_path.name
    assert final_path.name.endswith(".txt")


def test_write_request_without_label(tmp_path, monkeypatch):
    """Writing without a label omits the label prefix."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    path = write_request_pending({"data": 1})

    content = path.read_text()
    assert "🔄 LLM request ➡️" in content
    # No label brackets in the header line
    assert "[" not in content.split("\n")[0]


def test_multiple_writes_unique_filenames(tmp_path, monkeypatch):
    """Multiple writes create unique files."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    paths = []
    for i in range(3):
        path = write_request_pending({"i": i})
        paths.append(path)

    # All paths should be unique
    assert len(set(paths)) == 3
    files = list(tmp_path.glob("*_pending.txt"))
    assert len(files) == 3


def test_no_dir_returns_none_but_console_logs(monkeypatch, caplog):
    """When no log directory is set, returns None for file but still logs to console."""
    import logging
    from unillm import settings
    from unillm import logger

    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", "")
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    caplog.set_level(logging.DEBUG, logger="unillm")

    path = write_request_pending({"model": "gpt-4"}, label="test")

    assert path is None
    assert "LLM request" in caplog.text
    assert "gpt-4" in caplog.text


def test_terminal_off_still_writes_files(tmp_path, monkeypatch, caplog):
    """With terminal logging off but log dir set, files are still written."""
    import logging
    from unillm import settings
    from unillm import logger

    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", False)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    caplog.set_level(logging.DEBUG, logger="unillm")

    path = write_request_pending({"model": "gpt-4"}, label="test")

    assert path is not None
    assert path.exists()
    # No console output when terminal logging is off
    assert "LLM request" not in caplog.text


# --------------------------------------------------------------------------- #
#  OpenTelemetry tests
# --------------------------------------------------------------------------- #


class TestOtelEnabled:
    """Tests for UNILLM_OTEL master switch."""

    def test_otel_disabled_by_default(self, monkeypatch):
        """UNILLM_OTEL defaults to false."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        assert is_otel_enabled() is False

    def test_otel_enabled_via_setting(self, monkeypatch):
        """UNILLM_OTEL=true enables OTel."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        assert is_otel_enabled() is True


class TestLlmSpan:
    """Tests for llm_span context manager."""

    def test_span_created_when_otel_enabled(self, reset_otel, monkeypatch):
        """llm_span creates a span when OTel is enabled."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        with llm_span("gpt-4@openai", "gpt-4", provider="openai") as span:
            assert span is not None
            span.set_attribute("test", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "LLM gpt-4@openai"
        assert spans[0].attributes.get("llm.endpoint") == "gpt-4@openai"
        assert spans[0].attributes.get("llm.model") == "gpt-4"
        assert spans[0].attributes.get("llm.provider") == "openai"

    def test_span_none_when_otel_disabled(self, monkeypatch):
        """llm_span yields None when OTel is disabled."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        with llm_span("gpt-4@openai", "gpt-4") as span:
            assert span is None

    def test_span_records_error_on_exception(self, reset_otel, monkeypatch):
        """llm_span records errors when exceptions occur."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        with pytest.raises(ValueError, match="test error"):
            with llm_span("gpt-4@openai", "gpt-4") as span:
                raise ValueError("test error")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("error.type") == "ValueError"
        assert "test error" in spans[0].attributes.get("error.message", "")


class TestSetSpanResponse:
    """Tests for set_span_response helper."""

    def test_sets_cache_status(self, reset_otel, monkeypatch):
        """set_span_response sets cache_status attribute."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        with llm_span("gpt-4@openai", "gpt-4") as span:
            set_span_response(span, "hit")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("llm.cache_status") == "hit"

    def test_sets_usage_attributes(self, reset_otel, monkeypatch):
        """set_span_response sets usage attributes from response."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        # Mock response with usage
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-4-turbo"

        with llm_span("gpt-4@openai", "gpt-4") as span:
            set_span_response(span, "miss", mock_response)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get("llm.usage.prompt_tokens") == 10
        assert spans[0].attributes.get("llm.usage.completion_tokens") == 20
        assert spans[0].attributes.get("llm.usage.total_tokens") == 30
        assert spans[0].attributes.get("llm.response_model") == "gpt-4-turbo"

    def test_handles_none_span(self):
        """set_span_response handles None span gracefully."""
        # Should not raise
        set_span_response(None, "hit")


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_returns_none_when_disabled(self, monkeypatch):
        """get_tracer returns None when OTel is disabled."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        assert get_tracer() is None

    def test_returns_tracer_when_enabled(self, reset_otel, monkeypatch):
        """get_tracer returns a tracer when OTel is enabled."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        tracer = get_tracer()
        assert tracer is not None

    def test_uses_existing_provider(self, reset_otel, monkeypatch):
        """get_tracer uses existing TracerProvider if available."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        existing_provider = reset_otel["provider"]

        tracer = get_tracer()
        assert tracer is not None
        # Should use existing provider
        assert trace.get_tracer_provider() is existing_provider


class TestTraceHierarchy:
    """Tests for trace hierarchy with unify under the hood."""

    def test_unillm_span_becomes_child_of_parent(self, reset_otel, monkeypatch):
        """Unillm span becomes child of parent span."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]
        parent_tracer = trace.get_tracer("unity")

        # Simulate Unity creating a parent span
        with parent_tracer.start_as_current_span("unity.conductor.ask") as parent:
            parent_ctx = parent.get_span_context()

            # Now unillm creates a child span
            with llm_span("gpt-4@openai", "gpt-4") as child:
                child_ctx = child.get_span_context()
                # Same trace ID
                assert child_ctx.trace_id == parent_ctx.trace_id
                # Different span ID
                assert child_ctx.span_id != parent_ctx.span_id

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Find the spans
        unity_span = next(s for s in spans if "unity" in s.name)
        llm_span_obj = next(s for s in spans if "LLM" in s.name)

        # Verify parent-child relationship
        assert llm_span_obj.parent.span_id == unity_span.context.span_id

    def test_nested_unillm_unify_hierarchy(self, reset_otel, monkeypatch):
        """Tests full hierarchy: parent -> unillm -> unify (simulated)."""
        from unillm import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]
        parent_tracer = trace.get_tracer("unity")
        unify_tracer = trace.get_tracer("unify")

        # Simulate: Unity -> Unillm -> Unify HTTP call
        with parent_tracer.start_as_current_span("unity.conductor.ask") as unity_span:
            with llm_span("gpt-4@openai", "gpt-4") as unillm_span:
                # Simulate unify HTTP span (which would be created by unify/utils/http.py)
                with unify_tracer.start_as_current_span("GET projects") as unify_span:
                    pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        # Find spans by name
        unity_s = next(s for s in spans if "unity" in s.name)
        unillm_s = next(s for s in spans if "LLM" in s.name)
        unify_s = next(s for s in spans if "GET" in s.name)

        # Verify hierarchy
        # All same trace
        assert (
            unity_s.context.trace_id
            == unillm_s.context.trace_id
            == unify_s.context.trace_id
        )

        # unillm is child of unity
        assert unillm_s.parent.span_id == unity_s.context.span_id

        # unify is child of unillm
        assert unify_s.parent.span_id == unillm_s.context.span_id


class TestLogUsage:
    """Tests for log_usage() — external session usage logging (e.g. Realtime API)."""

    def test_writes_log_file_with_usage_and_transcript(self, tmp_path, monkeypatch):
        """log_usage writes a complete log file with request context and usage."""
        from unittest.mock import patch
        from unillm import settings
        from unillm import logger
        from unillm.logger import log_usage

        # Configure file logging
        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
        monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(logger, "_LOG_DIR", None)

        # Realistic Realtime API usage from a single response.done event
        usage = {
            "input_tokens": 150,
            "output_tokens": 80,
            "total_tokens": 230,
            "input_token_details": {
                "audio_tokens": 130,
                "text_tokens": 20,
                "cached_tokens": 0,
            },
            "output_token_details": {
                "audio_tokens": 70,
                "text_tokens": 10,
            },
        }

        transcript = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "Let me check on that for you."},
        ]

        with patch("unillm.logger.unify.deduct_credits"):
            billed_cost = log_usage(
                "gpt-4o-realtime-preview",
                usage,
                transcript=transcript,
                label="gpt-4o-realtime-preview",
            )

        # Should return a positive billed cost
        assert billed_cost > 0

        # Should have created exactly one log file with _usage suffix
        log_files = list(tmp_path.glob("*_usage.txt"))
        assert len(log_files) == 1

        content = log_files[0].read_text()

        # Log file should contain the request section with model and transcript
        assert "LLM request" in content
        assert "gpt-4o-realtime-preview" in content
        assert "What's the weather like?" in content
        assert "Let me check on that for you." in content

        # Log file should contain the response section with usage stats
        assert "LLM response" in content
        assert "[usage]" in content
        assert "audio_tokens" in content
        assert "provider_cost" in content
        assert "billed_cost" in content

    def test_deducts_credits(self, tmp_path, monkeypatch):
        """log_usage deducts the billed cost via unify.deduct_credits."""
        from unittest.mock import patch, MagicMock
        from unillm import settings
        from unillm import logger
        from unillm.logger import log_usage

        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
        monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(logger, "_LOG_DIR", None)

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {
                "audio_tokens": 80,
                "text_tokens": 20,
            },
            "output_token_details": {
                "audio_tokens": 40,
                "text_tokens": 10,
            },
        }

        mock_deduct = MagicMock()
        with patch("unillm.logger.unify.deduct_credits", mock_deduct):
            billed_cost = log_usage("gpt-4o-realtime-preview", usage)

        # Should have called deduct_credits with the billed amount
        mock_deduct.assert_called_once()
        deducted_amount = mock_deduct.call_args[0][0]
        assert deducted_amount == billed_cost
        assert deducted_amount > 0

    def test_works_without_transcript(self, tmp_path, monkeypatch):
        """log_usage works when no transcript is provided."""
        from unittest.mock import patch
        from unillm import settings
        from unillm import logger
        from unillm.logger import log_usage

        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
        monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(logger, "_LOG_DIR", None)

        usage = {"input_tokens": 50, "output_tokens": 30}

        with patch("unillm.logger.unify.deduct_credits"):
            billed_cost = log_usage("gpt-4o-realtime-preview", usage)

        assert billed_cost > 0

        log_files = list(tmp_path.glob("*_usage.txt"))
        assert len(log_files) == 1

        content = log_files[0].read_text()
        # Should NOT contain "messages" key when no transcript
        assert "messages" not in content
        assert "gpt-4o-realtime-preview" in content

    def test_resilient_to_deduct_failure(self, tmp_path, monkeypatch):
        """log_usage still writes the log file even if credit deduction fails."""
        from unittest.mock import patch
        from unillm import settings
        from unillm import logger
        from unillm.logger import log_usage

        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
        monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(logger, "_LOG_DIR", None)

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"audio_tokens": 80, "text_tokens": 20},
            "output_token_details": {"audio_tokens": 40, "text_tokens": 10},
        }

        with patch(
            "unillm.logger.unify.deduct_credits",
            side_effect=ConnectionError("no connection"),
        ):
            # Should not raise
            billed_cost = log_usage("gpt-4o-realtime-preview", usage)

        # Cost should still be computed
        assert billed_cost > 0

        # Log file should still exist
        log_files = list(tmp_path.glob("*_usage.txt"))
        assert len(log_files) == 1

    def test_emits_llm_event(self, tmp_path, monkeypatch):
        """log_usage emits an LLMEvent so downstream hooks (e.g. cumulative
        spend tracking) fire."""
        from unittest.mock import patch
        from unillm import settings
        from unillm import logger
        from unillm.logger import log_usage
        from unillm.llm_events import LLMEvent

        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
        monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(logger, "_LOG_DIR", None)

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"audio_tokens": 80, "text_tokens": 20},
            "output_token_details": {"audio_tokens": 40, "text_tokens": 10},
        }

        captured_events = []

        def capture_hook(event: LLMEvent) -> None:
            captured_events.append(event)

        with (
            patch("unillm.logger.unify.deduct_credits"),
            patch(
                "unillm.llm_events._emit_llm_event",
                side_effect=lambda e: captured_events.append(e),
            ),
        ):
            log_usage("gpt-4o-realtime-preview", usage)

        assert len(captured_events) == 1
        event = captured_events[0]
        assert isinstance(event, LLMEvent)
        assert event.request["model"] == "gpt-4o-realtime-preview"
        assert event.provider_cost > 0
        assert event.billed_cost > 0
        assert event.response["usage"] == usage


class TestStreamingFileLogging:
    """Tests that streaming LLM calls write file-based I/O logs.

    The non-streaming path (_generate_non_stream) calls write_request_pending
    and append_response_and_finalize. The streaming path (_generate_stream)
    must do the same so that CI artifacts capture the full request/response
    for every LLM call regardless of modality.
    """

    @pytest.mark.asyncio
    async def test_async_streaming_writes_log_file(self, tmp_path, monkeypatch):
        """AsyncUnify with stream=True should produce a log file."""
        from unittest.mock import patch
        from unillm import settings
        from unillm import logger
        import unillm

        # Configure file logging to tmp_path
        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_TERMINAL_LOG", True)
        monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(logger, "_TERMINAL_LOG_ENABLED", True)
        monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(logger, "_LOG_DIR", None)

        # Build mock streaming chunks
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

        with patch(
            "unillm.clients.uni_llm.litellm.acompletion",
            side_effect=mock_acompletion,
        ):
            client = unillm.AsyncUnify("gpt-4@openai", stream=True)
            gen = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # Verify the LLM call itself worked
        assert "".join(chunks) == "Hello world"

        # A log file must exist — streaming should log just like non-streaming
        log_files = list(tmp_path.glob("*.txt"))
        assert len(log_files) >= 1, (
            f"Streaming call produced no log files in {tmp_path}. "
            f"Non-streaming calls write a .cache_pending → .cache_hit/.cache_miss file, "
            f"but the streaming path skips write_request_pending entirely."
        )


class TestSanitizeOrigin:
    """Tests for _sanitize_origin filename safety."""

    def test_dots_replaced_with_underscores(self):
        """Dots in origin must become underscores so they don't look like file extensions."""
        assert "." not in _sanitize_origin("ConversationManager.decide")
        assert (
            _sanitize_origin("ConversationManager.decide")
            == "ConversationManager_decide"
        )


class TestExpandStringNewlines:
    """Tests for newline expansion in JSON string values."""

    def test_normalize_body_expands_newlines(self):
        """_normalize_body expands \\n in string values into real newlines."""
        body = {
            "role": "system",
            "content": "Line one\nLine two\nLine three",
        }
        result = _normalize_body(body)
        lines = result.split("\n")

        content_line = next(l for l in lines if "Line one" in l)
        assert '"content"' in content_line

        idx = lines.index(content_line)
        assert "Line two" in lines[idx + 1]
        assert "Line three" in lines[idx + 2]

    def test_continuation_lines_aligned_to_content_start(self):
        """Expanded newlines produce continuation lines aligned to content start."""
        body = {"key": "aaa\nbbb"}
        result = _normalize_body(body)
        lines = result.split("\n")

        key_line = next(l for l in lines if "aaa" in l)
        idx = lines.index(key_line)
        cont_line = lines[idx + 1]

        aaa_col = key_line.index("aaa")
        bbb_col = cont_line.index("bbb")
        assert aaa_col == bbb_col

    def test_escaped_backslash_n_not_expanded(self):
        r"""Literal backslash+n in values (\\n in JSON) is not expanded."""
        import json

        body = {"text": "before\\nafter"}
        json_text = json.dumps(body, indent=4)
        result = _expand_string_newlines(json_text)

        for line in result.split("\n"):
            if "before" in line:
                assert "after" in line
                break

    def test_preserves_json_structure(self):
        """Expansion only affects string interiors, not JSON structure."""
        body = {
            "messages": [
                {"role": "system", "content": "Hello\nWorld"},
                {"role": "user", "content": "No newlines here"},
            ],
        }
        result = _normalize_body(body)

        assert '"messages"' in result
        assert '"role"' in result
        assert '"system"' in result
        assert '"user"' in result
        assert '"No newlines here"' in result

    def test_empty_strings_unchanged(self):
        """Empty string values pass through without issue."""
        body = {"key": ""}
        result = _normalize_body(body)
        assert '"key": ""' in result

    def test_deeply_nested_string_indent(self):
        """Strings at deeper nesting levels get correct continuation indent."""
        body = {"outer": {"inner": {"deep": "first\nsecond"}}}
        result = _normalize_body(body)
        lines = result.split("\n")

        first_line = next(l for l in lines if "first" in l)
        idx = lines.index(first_line)
        cont_line = lines[idx + 1]

        first_col = first_line.index("first")
        second_col = cont_line.index("second")
        assert first_col == second_col

    def test_multiple_strings_with_newlines(self):
        """Multiple string values with newlines are each expanded independently."""
        body = {
            "a": "x\ny",
            "b": "p\nq",
        }
        result = _normalize_body(body)
        lines = result.split("\n")

        x_line = next(l for l in lines if '"a"' in l)
        x_idx = lines.index(x_line)
        assert "y" in lines[x_idx + 1]

        p_line = next(l for l in lines if '"b"' in l)
        p_idx = lines.index(p_line)
        assert "q" in lines[p_idx + 1]


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_otel_setting_parses_true(self):
        """UNILLM_OTEL parses 'true' string correctly."""
        from unillm.settings import _parse_bool

        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True

    def test_otel_setting_parses_false(self):
        """UNILLM_OTEL parses 'false' string correctly."""
        from unillm.settings import _parse_bool

        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("") is False
