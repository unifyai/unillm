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
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    path = write_request_pending({"model": "gpt-4"}, label="test")

    assert path is not None
    assert path.exists()
    assert "_pending" in path.name

    content = path.read_text()
    assert "🔄 [test] LLM request ➡️" in content
    assert '"model": "gpt-4"' in content


def test_append_response_and_finalize(tmp_path, monkeypatch):
    """Appending response and finalizing renames the file."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_LOG_ENABLED", True)
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

    # Should have a _hit file now
    hit_files = list(tmp_path.glob("*_hit.txt"))
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
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_LOG_ENABLED", True)
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

    # Should have an _error file now
    error_files = list(tmp_path.glob("*_error.txt"))
    assert len(error_files) == 1

    content = error_files[0].read_text()
    assert "LLM request ➡️" in content
    assert "LLM response ⬅️" in content
    assert "[cache: error]" in content


def test_write_request_without_label(tmp_path, monkeypatch):
    """Writing without a label omits the label prefix."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_LOG_ENABLED", True)
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
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_LOG_ENABLED", True)
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


def test_logging_disabled_returns_none(tmp_path, monkeypatch):
    """When logging is disabled, write_request_pending returns None (no file)."""
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logger, "_LOG_ENABLED", False)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    path = write_request_pending({"model": "gpt-4"}, label="test")

    assert path is None
    # No files should be created
    assert list(tmp_path.glob("*.txt")) == []


def test_logging_enabled_no_dir_returns_none_but_console_logs(monkeypatch, caplog):
    """When logging is enabled but no directory set, returns None but still logs to console."""
    import logging
    from unillm import settings
    from unillm import logger

    # Clear env var so monkeypatch takes effect (env var takes precedence in _get_log_dir)
    monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", "")
    monkeypatch.setattr(logger, "_LOG_ENABLED", True)
    monkeypatch.setattr(logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(logger, "_LOG_DIR", None)

    caplog.set_level(logging.DEBUG, logger="unillm")

    path = write_request_pending({"model": "gpt-4"}, label="test")

    # File path is None (no directory)
    assert path is None

    # But console log should still happen
    assert "LLM request" in caplog.text
    assert "gpt-4" in caplog.text


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
