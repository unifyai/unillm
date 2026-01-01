"""
Tests for LLM logging functionality.

These tests verify that the log module correctly writes request/response
payloads to log files when UNILLM_LOG is enabled.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from unillm.log import (
    _serialize_kw,
    write_request_pending,
    append_response_and_finalize,
)


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
    from unillm import log

    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(log, "_LOG_ENABLED", True)
    monkeypatch.setattr(log, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(log, "_LOG_DIR", None)

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
    from unillm import log

    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(log, "_LOG_ENABLED", True)
    monkeypatch.setattr(log, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(log, "_LOG_DIR", None)

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


def test_write_request_without_label(tmp_path, monkeypatch):
    """Writing without a label omits the label prefix."""
    from unillm import settings
    from unillm import log

    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(log, "_LOG_ENABLED", True)
    monkeypatch.setattr(log, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(log, "_LOG_DIR", None)

    path = write_request_pending({"data": 1})

    content = path.read_text()
    assert "🔄 LLM request ➡️" in content
    # No label brackets in the header line
    assert "[" not in content.split("\n")[0]


def test_multiple_writes_unique_filenames(tmp_path, monkeypatch):
    """Multiple writes create unique files."""
    from unillm import settings
    from unillm import log

    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(log, "_LOG_ENABLED", True)
    monkeypatch.setattr(log, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(log, "_LOG_DIR", None)

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
    from unillm import log

    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", False)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(log, "_LOG_ENABLED", False)
    monkeypatch.setattr(log, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(log, "_LOG_DIR", None)

    path = write_request_pending({"model": "gpt-4"}, label="test")

    assert path is None
    # No files should be created
    assert list(tmp_path.glob("*.txt")) == []


def test_logging_enabled_no_dir_returns_none_but_console_logs(monkeypatch, caplog):
    """When logging is enabled but no directory set, returns None but still logs to console."""
    import logging
    from unillm import settings
    from unillm import log

    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG", True)
    monkeypatch.setattr(settings.SETTINGS, "UNILLM_LOG_DIR", "")
    monkeypatch.setattr(log, "_LOG_ENABLED", True)
    monkeypatch.setattr(log, "_LOG_DIR_CHECKED", False)
    monkeypatch.setattr(log, "_LOG_DIR", None)

    caplog.set_level(logging.DEBUG, logger="unillm")

    path = write_request_pending({"model": "gpt-4"}, label="test")

    # File path is None (no directory)
    assert path is None

    # But console log should still happen
    assert "LLM request" in caplog.text
    assert "gpt-4" in caplog.text
