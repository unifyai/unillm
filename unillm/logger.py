"""
LLM I/O Logging and OpenTelemetry Tracing
=========================================

Provides console and file-based logging for LLM request/response payloads,
plus OpenTelemetry tracing for distributed observability.

Logging is controlled by two environment variables:
- UNILLM_LOG: Enable/disable all logging (default: true)
- UNILLM_LOG_DIR: Directory for log files (default: console only)

When UNILLM_LOG_DIR is set, structured files are written:
- During the call: ``{timestamp}_pending.txt`` (contains request only)
- After completion: ``{timestamp}_hit.txt`` or ``{timestamp}_miss.txt``
  (contains both request and response, with cache status in filename)

If an LLM call hangs or crashes, the ``_pending.txt`` file remains as evidence
of the incomplete request.

OpenTelemetry tracing is controlled by:
- UNILLM_OTEL: Enable/disable OTel tracing (default: false)
- UNILLM_OTEL_ENDPOINT: OTLP endpoint for trace export (optional)
- UNILLM_OTEL_LOG_DIR: Directory for file-based span export (optional)

When UNILLM_OTEL is enabled, LLM calls create OTel spans that can be
correlated with parent spans (from Unity) and child spans (in Unify).

File-based span export:
When UNILLM_OTEL_LOG_DIR is set, spans are written to JSONL files keyed
by trace_id: {UNILLM_OTEL_LOG_DIR}/{trace_id}.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .settings import SETTINGS

# ---------------------------------------------------------------------------
# Console logging setup
# ---------------------------------------------------------------------------

_LOGGER = logging.getLogger("unillm")
_LOG_ENABLED = SETTINGS.UNILLM_LOG
_LOGGER.setLevel(logging.DEBUG if _LOG_ENABLED else logging.WARNING)

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------

_OTEL_ENABLED = SETTINGS.UNILLM_OTEL
_OTEL_ENDPOINT = SETTINGS.UNILLM_OTEL_ENDPOINT
_OTEL_LOG_DIR = SETTINGS.UNILLM_OTEL_LOG_DIR
_OTEL_INITIALIZED = False
_TRACER = None


# ---------------------------------------------------------------------------
# File-based Span Exporter
# ---------------------------------------------------------------------------


class FileSpanExporter:
    """Exports spans to JSONL files, one file per trace_id.

    This enables standalone trace logging without a parent TracerProvider
    or external collector (Tempo/Jaeger).
    """

    def __init__(self, log_dir: Path, service_name: str = "unillm"):
        self.log_dir = log_dir
        self.service_name = service_name
        self._lock = threading.Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def export(self, spans) -> int:
        """Export a batch of spans to files."""
        try:
            for span in spans:
                self._write_span(span)
            return 0  # SUCCESS
        except Exception as e:
            _LOGGER.warning(f"FileSpanExporter failed to export spans: {e}")
            return 1  # FAILURE

    def _write_span(self, span) -> None:
        """Write a single span to its trace file."""
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return

        trace_id = f"{ctx.trace_id:032x}"
        span_id = f"{ctx.span_id:016x}"

        parent_span_id = None
        if span.parent is not None:
            parent_span_id = f"{span.parent.span_id:016x}"

        span_data: dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": span.name,
            "service": self.service_name,
            "start_time": (
                datetime.fromtimestamp(
                    span.start_time / 1e9,
                    tz=timezone.utc,
                ).isoformat()
                if span.start_time
                else None
            ),
            "end_time": (
                datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc).isoformat()
                if span.end_time
                else None
            ),
            "duration_ms": (
                (span.end_time - span.start_time) / 1e6
                if span.end_time and span.start_time
                else None
            ),
            "status": span.status.status_code.name if span.status else None,
            "attributes": dict(span.attributes) if span.attributes else {},
        }

        trace_file = self.log_dir / f"{trace_id}.jsonl"
        with self._lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(span_data, default=str) + "\n")

    def shutdown(self) -> None:
        """Shutdown the exporter (no-op for file exporter)."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans (no-op for file exporter)."""
        return True


def _setup_otel() -> None:
    """Initialize OpenTelemetry if enabled and not already configured.

    When UNILLM_OTEL_LOG_DIR is set, spans are written to JSONL files
    keyed by trace_id for standalone trace logging.
    """
    global _OTEL_INITIALIZED, _TRACER

    if _OTEL_INITIALIZED or not _OTEL_ENABLED:
        return

    _OTEL_INITIALIZED = True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Check if a TracerProvider already exists (parent set it up)
        existing = trace.get_tracer_provider()
        if existing and not isinstance(
            existing,
            (trace.NoOpTracerProvider, trace.ProxyTracerProvider),
        ):
            # Parent (e.g., Unity) already configured OTel - use theirs
            _TRACER = trace.get_tracer("unillm")
            _LOGGER.debug("Using existing OTel TracerProvider from parent")
            return

        # We're the outermost layer - set up our own provider
        resource = Resource.create({SERVICE_NAME: "unillm"})
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint configured
        if _OTEL_ENDPOINT:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=_OTEL_ENDPOINT, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                _LOGGER.debug(f"Configured OTLP exporter at {_OTEL_ENDPOINT}")
            except ImportError:
                _LOGGER.warning(
                    "OTLP exporter not available - install opentelemetry-exporter-otlp",
                )
            except Exception as e:
                _LOGGER.warning(f"Failed to configure OTLP exporter: {e}")

        # Add file exporter if log directory configured
        if _OTEL_LOG_DIR:
            try:
                log_dir = Path(_OTEL_LOG_DIR)
                file_exporter = FileSpanExporter(log_dir, service_name="unillm")
                provider.add_span_processor(SimpleSpanProcessor(file_exporter))
                _LOGGER.debug(f"Configured file span exporter at {_OTEL_LOG_DIR}")
            except Exception as e:
                _LOGGER.warning(f"Failed to configure file span exporter: {e}")

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("unillm")
        _LOGGER.debug("Initialized OTel TracerProvider for unillm")

    except ImportError:
        _LOGGER.debug("OpenTelemetry not available - tracing disabled")
    except Exception as e:
        _LOGGER.warning(f"Failed to initialize OpenTelemetry: {e}")


def get_tracer():
    """Get the OpenTelemetry tracer, initializing if needed."""
    global _TRACER
    if _TRACER is None and _OTEL_ENABLED:
        _setup_otel()
    return _TRACER


def is_otel_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled."""
    return _OTEL_ENABLED


def get_otel_log_dir() -> Optional[Path]:
    """Get the OTel span log directory, if configured."""
    if not _OTEL_LOG_DIR:
        return None
    return Path(_OTEL_LOG_DIR)


@contextmanager
def llm_span(endpoint: str, model: str, **attributes):
    """Create an OTel span for an LLM call.

    Args:
        endpoint: The endpoint being called (e.g., "gpt-4@openai")
        model: The model name
        **attributes: Additional span attributes

    Yields:
        The span (or None if OTel disabled)
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    with tracer.start_as_current_span(
        f"LLM {endpoint}",
        kind=SpanKind.CLIENT,
    ) as span:
        span.set_attribute("llm.endpoint", endpoint)
        span.set_attribute("llm.model", model)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(
                    f"llm.{key}",
                    str(value) if not isinstance(value, (int, float, bool)) else value,
                )

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise


def set_span_response(span, cache_status: str, response: Any = None) -> None:
    """Set response attributes on a span.

    Args:
        span: The OTel span (or None)
        cache_status: "hit" or "miss"
        response: The LLM response object (optional)
    """
    if span is None:
        return

    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_attribute("llm.cache_status", cache_status)

        if response is not None:
            # Try to extract usage info
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute("llm.usage.prompt_tokens", usage.prompt_tokens)
                if hasattr(usage, "completion_tokens"):
                    span.set_attribute(
                        "llm.usage.completion_tokens",
                        usage.completion_tokens,
                    )
                if hasattr(usage, "total_tokens"):
                    span.set_attribute("llm.usage.total_tokens", usage.total_tokens)

            # Try to extract model from response
            if hasattr(response, "model"):
                span.set_attribute("llm.response_model", response.model)

        span.set_status(Status(StatusCode.OK))
    except Exception:
        pass  # Silent best-effort


# ---------------------------------------------------------------------------
# File-based trace logging
# ---------------------------------------------------------------------------

_LOG_DIR: Optional[Path] = None
_LOG_DIR_CHECKED = False


def configure_log_dir(log_dir: Optional[str] = None) -> Optional[Path]:
    """Configure or reconfigure the log directory for file-based logging.

    Call this after setting UNILLM_LOG_DIR if the env var was set
    after this module was imported.

    Args:
        log_dir: Explicit log directory path. If None, reads from
                 UNILLM_LOG_DIR env var / settings.

    Returns:
        The configured log directory Path, or None if disabled.
    """
    global _LOG_DIR, _LOG_DIR_CHECKED

    _LOG_DIR_CHECKED = False
    _LOG_DIR = None

    if log_dir is not None:
        os.environ["UNILLM_LOG_DIR"] = log_dir

    return _get_log_dir()


def _get_log_dir() -> Path | None:
    """Get the log directory path, or None if logging is disabled.

    Returns None if:
    - UNILLM_LOG is False (master switch off)
    - UNILLM_LOG_DIR is not set
    """
    global _LOG_DIR, _LOG_DIR_CHECKED

    if _LOG_DIR_CHECKED:
        return _LOG_DIR

    _LOG_DIR_CHECKED = True

    if not _LOG_ENABLED:
        return None

    # Check env var first (allows runtime override), then settings
    log_dir_str = os.getenv("UNILLM_LOG_DIR", "").strip() or SETTINGS.UNILLM_LOG_DIR
    if not log_dir_str:
        return None

    try:
        log_dir = Path(log_dir_str)
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_DIR = log_dir
        _LOGGER.debug(f"LLM I/O file logging enabled: {log_dir}")
    except Exception as e:
        _LOGGER.warning(f"Failed to create LLM log directory {log_dir_str}: {e}")
        _LOG_DIR = None

    return _LOG_DIR


def _serialize_kw(kw: dict) -> dict:
    """Serialize the kw dict for logging, handling Pydantic models and special types."""
    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = None  # type: ignore

    def _convert(obj: Any) -> Any:
        if obj is None:
            return None
        if BaseModel is not None and isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "model_json_schema"):
            # Pydantic model class (not instance)
            return {"__pydantic_schema__": obj.model_json_schema()}
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    return _convert(kw)


def _normalize_body(body: Any) -> str:
    """Normalize a body payload to a string for writing."""
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, indent=4, default=str)
    except Exception:
        return str(body)


def _truncate_for_console(body_str: str, max_len: int = 500) -> str:
    """Truncate a body string for console output."""
    if len(body_str) <= max_len:
        return body_str
    return body_str[:max_len] + "...(truncated)"


def write_request_pending(
    request_kw: dict,
    *,
    label: str | None = None,
) -> Path | None:
    """Write the request payload immediately with a _pending suffix.

    Logs to console (always when enabled) and to file (if directory set).
    Returns the file path so we can append the response and rename later.
    If the LLM call hangs/crashes, the _pending file remains as evidence.
    """
    if not _LOG_ENABLED:
        return None

    label_prefix = f"[{label}] " if label else ""
    serialized = _serialize_kw(request_kw)
    body_str = _normalize_body(serialized)

    # Console log (always when enabled)
    _LOGGER.debug(
        f"🔄 {label_prefix}LLM request ➡️\n{_truncate_for_console(body_str)}",
    )

    # File log (only if directory set)
    log_dir = _get_log_dir()
    if log_dir is None:
        return None

    try:
        now = datetime.now(timezone.utc)
        hhmmss = now.strftime("%H%M%S")
        ns = time.time_ns() % 1_000_000_000
        base = f"{hhmmss}_{ns:09d}_pending"
        path = log_dir / f"{base}.txt"

        # Handle filename collision
        i = 1
        while path.exists():
            path = log_dir / f"{base}_{i}.txt"
            i += 1

        with path.open("w", encoding="utf-8") as f:
            f.write(f"🔄 {label_prefix}LLM request ➡️\n")
            f.write(body_str.rstrip())
            f.write("\n")

        return path
    except Exception:
        return None


def append_response_and_finalize(
    pending_path: Path | None,
    response_body: Any,
    cache_status: str,
    *,
    label: str | None = None,
) -> None:
    """Append the response to the pending file and rename to reflect cache status.

    Logs to console (always when enabled) and finalizes file (if path provided).
    The final filename will be: {timestamp}_hit.txt or {timestamp}_miss.txt
    """
    if not _LOG_ENABLED:
        return

    label_prefix = f"[{label}] " if label else ""
    body_str = _normalize_body(response_body)

    # Console log (always when enabled)
    _LOGGER.debug(
        f"🔄 {label_prefix}LLM response ⬅️ [cache: {cache_status}]\n"
        f"{_truncate_for_console(body_str)}",
    )

    # File log (only if we have a pending path)
    if pending_path is None or not pending_path.exists():
        return

    try:
        # Append response to the file
        with pending_path.open("a", encoding="utf-8") as f:
            f.write(f"\n🔄 {label_prefix}LLM response ⬅️ [cache: {cache_status}]\n")
            f.write(body_str.rstrip())
            f.write("\n")

        # Rename from _pending to _hit or _miss
        new_name = pending_path.name.replace("_pending", f"_{cache_status}")
        new_path = pending_path.parent / new_name
        pending_path.rename(new_path)
    except Exception:
        # Silent best-effort
        pass
