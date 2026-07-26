"""OpenRouter model catalog snapshot for listing, pricing fallback, and Orchestra sync.

Any ``<openrouter-id>@openrouter`` endpoint is callable without being in this
catalog. The snapshot is used for:
- ``list_endpoints`` / ``list_models("openrouter")`` enrichment
- cost fallback when a response omits ``usage.cost``
- downstream product catalogs (Orchestra) that need modalities / pricing
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("unillm.openrouter_catalog")

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours

_lock = threading.RLock()
_snapshot: dict[str, Any] | None = None
_snapshot_fetched_at: float = 0.0


def _ttl_seconds() -> float:
    raw = os.environ.get("UNILLM_OPENROUTER_CATALOG_TTL_SECONDS")
    if raw is None or raw == "":
        return float(_DEFAULT_TTL_SECONDS)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(_DEFAULT_TTL_SECONDS)


def _cache_path() -> Path | None:
    raw = os.environ.get("UNILLM_OPENROUTER_CATALOG_PATH")
    if raw:
        return Path(raw)
    log_dir = os.environ.get("UNILLM_LOG_DIR") or os.environ.get("UNILLM_CACHE_DIR")
    if log_dir:
        return Path(log_dir) / "openrouter_models.json"
    return Path.home() / ".cache" / "unillm" / "openrouter_models.json"


def _normalize_model_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None

    pricing = raw.get("pricing") or {}
    architecture = raw.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or raw.get(
        "input_modalities",
    )
    if not isinstance(input_modalities, list):
        input_modalities = []

    def _per_token(key: str) -> float | None:
        value = pricing.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    supported_params = raw.get("supported_parameters") or []
    if not isinstance(supported_params, list):
        supported_params = []
    supported_l = {str(p).lower() for p in supported_params}

    return {
        "id": model_id,
        "name": raw.get("name") or model_id,
        "context_length": raw.get("context_length"),
        "input_cost_per_token": _per_token("prompt"),
        "output_cost_per_token": _per_token("completion"),
        "cache_read_input_token_cost": _per_token("input_cache_read"),
        "cache_creation_input_token_cost": _per_token("input_cache_write"),
        "input_modalities": [str(m).lower() for m in input_modalities],
        "supports_image_input": "image" in {str(m).lower() for m in input_modalities},
        "supports_tools": bool(
            {"tools", "tool_choice", "functions", "function_call"} & supported_l,
        ),
        "supports_reasoning": bool(
            {"reasoning", "include_reasoning", "reasoning_effort"} & supported_l,
        ),
        "supported_parameters": [str(p) for p in supported_params],
    }


def _load_disk_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "models" not in data:
            return None
        return data
    except Exception:
        _LOGGER.debug("Failed to load OpenRouter catalog from %s", path, exc_info=True)
        return None


def _write_disk_snapshot(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        _LOGGER.debug("Failed to persist OpenRouter catalog to %s", path, exc_info=True)


def _fetch_remote_snapshot() -> dict[str, Any] | None:
    try:
        import urllib.request

        headers = {"Accept": "application/json", "User-Agent": "unillm"}
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(_OPENROUTER_MODELS_URL, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        _LOGGER.warning("Failed to fetch OpenRouter model catalog", exc_info=True)
        return None

    raw_models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return None

    models: dict[str, dict[str, Any]] = {}
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_model_entry(entry)
        if normalized is None:
            continue
        models[normalized["id"]] = normalized

    return {
        "fetched_at": time.time(),
        "source": _OPENROUTER_MODELS_URL,
        "models": models,
    }


def get_openrouter_catalog(
    *,
    refresh: bool = False,
    allow_fetch: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return ``{openrouter_id: metadata}`` from memory, disk, or remote fetch."""

    global _snapshot, _snapshot_fetched_at

    with _lock:
        now = time.time()
        ttl = _ttl_seconds()
        if (
            not refresh
            and _snapshot is not None
            and (ttl <= 0 or now - _snapshot_fetched_at < ttl)
        ):
            return dict(_snapshot.get("models") or {})

        path = _cache_path()
        disk = _load_disk_snapshot(path) if path is not None else None
        if (
            not refresh
            and disk is not None
            and (ttl <= 0 or now - float(disk.get("fetched_at") or 0) < ttl)
        ):
            _snapshot = disk
            _snapshot_fetched_at = float(disk.get("fetched_at") or now)
            return dict(_snapshot.get("models") or {})

        if allow_fetch:
            remote = _fetch_remote_snapshot()
            if remote is not None:
                _snapshot = remote
                _snapshot_fetched_at = float(remote["fetched_at"])
                if path is not None:
                    _write_disk_snapshot(path, remote)
                return dict(_snapshot.get("models") or {})

        if disk is not None:
            _snapshot = disk
            _snapshot_fetched_at = float(disk.get("fetched_at") or now)
            return dict(_snapshot.get("models") or {})

        if _snapshot is not None:
            return dict(_snapshot.get("models") or {})
        return {}


def get_openrouter_model_info(model_id: str) -> dict[str, Any] | None:
    """Return catalog metadata for an OpenRouter model id, if known."""

    catalog = get_openrouter_catalog(allow_fetch=False)
    info = catalog.get(model_id)
    if info is not None:
        return info
    # Opportunistic refresh when missing (best-effort; may no-op offline).
    catalog = get_openrouter_catalog(refresh=True, allow_fetch=True)
    return catalog.get(model_id)


def list_openrouter_model_ids(*, allow_fetch: bool = False) -> list[str]:
    """Return sorted OpenRouter model ids from the catalog snapshot."""

    return sorted(get_openrouter_catalog(allow_fetch=allow_fetch))


def openrouter_endpoint(model_id: str) -> str:
    """Return the UniLLM public endpoint for an OpenRouter catalog id."""

    return f"{model_id}@openrouter"


def catalog_pricing_as_litellm_info(model_id: str) -> dict[str, Any] | None:
    """Map catalog pricing into LiteLLM-style cost keys for fallback billing."""

    info = get_openrouter_model_info(model_id)
    if info is None:
        return None
    result: dict[str, Any] = {
        "litellm_provider": "openrouter",
        "mode": "chat",
    }
    if info.get("context_length") is not None:
        result["max_input_tokens"] = info["context_length"]
    for src, dest in (
        ("input_cost_per_token", "input_cost_per_token"),
        ("output_cost_per_token", "output_cost_per_token"),
        ("cache_read_input_token_cost", "cache_read_input_token_cost"),
        ("cache_creation_input_token_cost", "cache_creation_input_token_cost"),
    ):
        if info.get(src) is not None:
            result[dest] = info[src]
    if info.get("supports_tools"):
        result["supports_function_calling"] = True
        result["supports_parallel_function_calling"] = True
    if info.get("supports_reasoning"):
        result["supports_reasoning"] = True
    if info.get("supports_image_input"):
        result["supports_vision"] = True
    return result
