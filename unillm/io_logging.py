"""
LLM I/O Logging
===============

Writes LLM request/response payloads to log files for debugging.

Enabled via the ``UNILLM_IO_LOG`` environment variable. Logs are written to
``{UNILLM_LOG_DIR}/`` when enabled.

Log file format:
- During the call: ``{timestamp}_pending.txt`` (contains request only)
- After completion: ``{timestamp}_hit.txt`` or ``{timestamp}_miss.txt``
  (contains both request and response, with cache status in filename)

If an LLM call hangs or crashes, the ``_pending.txt`` file remains as evidence
of the incomplete request.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import SETTINGS


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


def _get_log_dir() -> Path | None:
    """Get the log directory path, or None if logging is disabled."""
    if not SETTINGS.UNILLM_IO_LOG:
        return None
    if not SETTINGS.UNILLM_LOG_DIR:
        return None
    log_dir = Path(SETTINGS.UNILLM_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_request_pending(
    request_kw: dict,
    *,
    label: str | None = None,
) -> Path | None:
    """Write the request payload immediately with a _pending suffix.

    Returns the file path so we can append the response and rename later.
    If the LLM call hangs/crashes, the _pending file remains as evidence.
    """
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

        body_str = _normalize_body(_serialize_kw(request_kw))
        label_prefix = f"[{label}] " if label else ""

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

    The final filename will be: {timestamp}_hit.txt or {timestamp}_miss.txt
    """
    if pending_path is None or not pending_path.exists():
        return

    try:
        body_str = _normalize_body(response_body)
        label_prefix = f"[{label}] " if label else ""

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
