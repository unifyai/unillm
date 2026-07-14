"""
Hash→offset index over NDJSON LLM cache files.

Keeps the on-disk line format unchanged while avoiding an eager in-memory
dict of full (often multi-MB) cache keys. Lookups use sha256(key) → byte
offset, then seek + verify the full key on the stored line.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

_INDEX_VERSION = 1
_DEFAULT_LRU_MAX = 128


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _idx_path_for(cache_path: str) -> str:
    return f"{cache_path}.idx"


def delete_index_sidecar(cache_path: str) -> None:
    """Remove the sidecar index for a cache file if it exists."""
    idx_path = _idx_path_for(cache_path)
    try:
        os.remove(idx_path)
    except FileNotFoundError:
        pass
    # Historical shared temp path plus per-process mkstemp leftovers.
    parent = os.path.dirname(idx_path) or "."
    base = os.path.basename(idx_path)
    for name in os.listdir(parent):
        if name == f"{base}.tmp" or (
            name.startswith(f"{base}.") and name.endswith(".tmp")
        ):
            try:
                os.remove(os.path.join(parent, name))
            except FileNotFoundError:
                pass


class NdjsonIndexedStore:
    """Indexed NDJSON store with optional value LRU."""

    def __init__(self, path: str, *, lru_max: int = _DEFAULT_LRU_MAX) -> None:
        self.path = path
        self.idx_path = _idx_path_for(path)
        self._offsets: Dict[str, int] = {}
        self._fingerprint: Optional[Tuple[int, int]] = None
        self._keys_cache: Optional[List[str]] = None
        self._lru_max = lru_max
        self._lru: OrderedDict[str, Tuple[Any, Any]] = OrderedDict()

    def open_or_create(self) -> None:
        """Ensure the NDJSON file exists and load or rebuild the index."""
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "wb"):
                pass
        self._load_or_rebuild_index()

    def __len__(self) -> int:
        self._ensure_fresh()
        return len(self._offsets)

    def has_key(self, key: str) -> bool:
        self._ensure_fresh()
        digest = _key_hash(key)
        if digest not in self._offsets:
            return False
        # Confirm the line still matches (collision / stale offset).
        return self._read_entry_at(self._offsets[digest], expected_key=key) is not None

    def get(self, key: str) -> Optional[Tuple[Any, Any]]:
        """
        Return (deserialized_value, res_types) or None on miss.

        deserialized_value is the JSON-decoded `value` field (same as the
        eager backends after json.loads on the stored value string).
        """
        self._ensure_fresh()
        digest = _key_hash(key)
        cached = self._lru_get(digest)
        if cached is not None:
            return cached

        offset = self._offsets.get(digest)
        if offset is None:
            return None
        entry = self._read_entry_at(offset, expected_key=key)
        if entry is None:
            # Stale index entry — drop and treat as miss.
            self._offsets.pop(digest, None)
            return None
        value_str, res_types = entry
        deserialized = json.loads(value_str)
        self._lru_put(digest, (deserialized, res_types))
        return deserialized, res_types

    def append(
        self,
        key: str,
        value: Any,
        res_types: Optional[Any] = None,
    ) -> None:
        """Append one NDJSON line and update the in-memory + sidecar index."""
        self._ensure_fresh()
        line = (
            json.dumps(
                {
                    "key": key,
                    "value": value,
                    "res_types": res_types,
                },
            )
            + "\n"
        )
        payload = line.encode("utf-8")
        with open(self.path, "ab") as f:
            offset = f.tell()
            f.write(payload)
        digest = _key_hash(key)
        self._offsets[digest] = offset
        self._keys_cache = None
        # store_entry passes value as a JSON string from serialize_object
        if isinstance(value, str):
            try:
                deserialized = json.loads(value)
            except json.JSONDecodeError:
                deserialized = value
        else:
            deserialized = value
        self._lru_put(digest, (deserialized, res_types))
        self._fingerprint = self._file_fingerprint()
        self._write_sidecar()

    def remove(self, key: str) -> bool:
        """
        Rewrite the NDJSON file without `key`, then rebuild the index.

        Returns True if an entry was removed.
        """
        self._ensure_fresh()
        digest = _key_hash(key)
        if digest not in self._offsets and (
            self._keys_cache is None or key not in self._keys_cache
        ):
            # Still scan in case index is incomplete.
            if not self.has_key(key):
                return False

        kept: List[bytes] = []
        removed = False
        with open(self.path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    item = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    kept.append(line if line.endswith(b"\n") else line + b"\n")
                    continue
                if not isinstance(item, dict) or "key" not in item:
                    kept.append(line if line.endswith(b"\n") else line + b"\n")
                    continue
                if item["key"] == key:
                    removed = True
                    continue
                kept.append(line if line.endswith(b"\n") else line + b"\n")

        if not removed:
            return False

        with open(self.path, "wb") as f:
            f.writelines(kept)

        self._keys_cache = None
        self._lru.pop(digest, None)
        self._rebuild_index(write_sidecar=True)
        return True

    def list_keys(self) -> List[str]:
        """Scan the file for full keys (introspection / tests)."""
        self._ensure_fresh()
        if self._keys_cache is not None:
            return list(self._keys_cache)

        keys: "OrderedDict[str, None]" = OrderedDict()
        with open(self.path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    continue
                if not isinstance(item, dict):
                    continue
                if not all(k in item for k in ("key", "value", "res_types")):
                    continue
                key = item["key"]
                if not isinstance(key, str):
                    continue
                keys[key] = None

        self._keys_cache = list(keys.keys())
        return list(self._keys_cache)

    def _ensure_fresh(self) -> None:
        fp = self._file_fingerprint()
        if fp is None:
            self._offsets = {}
            self._fingerprint = None
            self._keys_cache = None
            return
        if self._fingerprint != fp:
            self._load_or_rebuild_index()

    def _file_fingerprint(self) -> Optional[Tuple[int, int]]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        return (mtime_ns, st.st_size)

    def _load_or_rebuild_index(self) -> None:
        if self._try_load_sidecar():
            return
        self._rebuild_index(write_sidecar=True)

    def _try_load_sidecar(self) -> bool:
        fp = self._file_fingerprint()
        if fp is None:
            self._offsets = {}
            self._fingerprint = None
            return True
        if not os.path.exists(self.idx_path):
            return False
        try:
            with open(self.idx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
        if not isinstance(data, dict) or data.get("version") != _INDEX_VERSION:
            return False
        if data.get("mtime_ns") != fp[0] or data.get("size") != fp[1]:
            return False
        offsets = data.get("offsets")
        if not isinstance(offsets, dict):
            return False
        # Coerce JSON keys/values
        parsed: Dict[str, int] = {}
        for k, v in offsets.items():
            if not isinstance(k, str) or not isinstance(v, int):
                return False
            parsed[k] = v
        self._offsets = parsed
        self._fingerprint = fp
        self._keys_cache = None
        return True

    def _rebuild_index(self, *, write_sidecar: bool) -> None:
        offsets: Dict[str, int] = {}
        if not os.path.exists(self.path):
            with open(self.path, "wb"):
                pass
        with open(self.path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
                    warnings.warn(
                        f"Cache file {self.path} contains invalid cache entry, "
                        f"skipping at offset {offset}: {raw[:40]!r}... "
                        f"({type(e).__name__}: {e})",
                    )
                    continue
                if not isinstance(item, dict):
                    warnings.warn(
                        f"Cache file {self.path} contains non-object entry at "
                        f"offset {offset}, skipping",
                    )
                    continue
                if not all(k in item for k in ("key", "value", "res_types")):
                    warnings.warn(
                        f"Cache file {self.path} contains incomplete cache entry "
                        f"at offset {offset}, skipping",
                    )
                    continue
                key = item["key"]
                if not isinstance(key, str):
                    continue
                offsets[_key_hash(key)] = offset
        self._offsets = offsets
        self._fingerprint = self._file_fingerprint()
        self._keys_cache = None
        self._lru.clear()
        if write_sidecar:
            self._write_sidecar()

    def _write_sidecar(self) -> None:
        fp = self._fingerprint or self._file_fingerprint()
        if fp is None:
            return
        payload = {
            "version": _INDEX_VERSION,
            "mtime_ns": fp[0],
            "size": fp[1],
            "offsets": self._offsets,
        }
        # Unique temp path per writer: parallel pytest workers previously
        # shared `{idx}.tmp`, so one process's os.replace could delete the
        # other's temp before the second replace ran (FileNotFoundError).
        parent = os.path.dirname(self.idx_path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{os.path.basename(self.idx_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(tmp_path, self.idx_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _read_entry_at(
        self,
        offset: int,
        *,
        expected_key: str,
    ) -> Optional[Tuple[str, Any]]:
        try:
            with open(self.path, "rb") as f:
                f.seek(offset)
                line = f.readline()
        except OSError:
            return None
        if not line.strip():
            return None
        try:
            item = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return None
        if not isinstance(item, dict):
            return None
        if not all(k in item for k in ("key", "value", "res_types")):
            return None
        if item["key"] != expected_key:
            return None
        value = item["value"]
        if not isinstance(value, str):
            # Defensive: normalize non-string values to JSON string form.
            value = json.dumps(value)
        return value, item["res_types"]

    def _lru_get(self, digest: str) -> Optional[Tuple[Any, Any]]:
        if digest not in self._lru:
            return None
        self._lru.move_to_end(digest)
        return self._lru[digest]

    def _lru_put(self, digest: str, entry: Tuple[Any, Any]) -> None:
        if self._lru_max <= 0:
            return
        self._lru[digest] = entry
        self._lru.move_to_end(digest)
        while len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)
