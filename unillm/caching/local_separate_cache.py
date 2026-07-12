"""
Local cache with separate read and write storage.

The shared read file (`.cache.ndjson`) is indexed lazily. The write file
(`.cache_write.ndjson`) stays a small in-memory dict plus append-only
file, matching CI shard behavior.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base_cache import BaseCache
from .ndjson_index import NdjsonIndexedStore
from .ndsjson_cache_utils import _write_to_ndjson_cache


class LocalSeparateCache(BaseCache):
    """Local cache with separate read and write storage for better performance."""

    _read_store: Optional[NdjsonIndexedStore] = None
    _cache_write: Optional[Dict[str, Any]] = None
    _cache_dir: str = os.environ.get("UNILLM_CACHE_DIR", os.getcwd())
    _cache_name_read: str = ".cache.ndjson"
    _cache_name_write: str = ".cache_write.ndjson"
    _enabled: bool = False

    @classmethod
    def set_cache_name(cls, name: str) -> None:
        """Set the cache names and reset both caches."""
        cls._cache_name_read = f"{name}_read"
        cls._cache_name_write = f"{name}_write"
        cls._read_store = None
        cls._cache_write = None

    @classmethod
    def get_cache_name(cls) -> str:
        """Get the current read cache name."""
        return cls._cache_name_read

    @classmethod
    def get_cache_filepath(cls, name: str = None) -> str:
        """Get the full filepath for the cache file."""
        if name is None:
            name = cls.get_cache_name()
        return os.path.join(cls._cache_dir, name)

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if the cache is enabled."""
        return cls._enabled

    @classmethod
    def store_entry(
        cls,
        *,
        key: str,
        value: Any,
        res_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a key-value pair in the write cache."""
        cls._cache_write[key] = {"value": value, "res_types": res_types}
        with open(cls.get_cache_filepath(cls._cache_name_write), "a") as f:
            _write_to_ndjson_cache(
                f,
                key,
                value,
                res_types,
            )

    @classmethod
    def initialize_cache(cls, name: str = None) -> None:
        """Initialize both read and write caches."""
        if cls._cache_write is None:
            cls._cache_write = {}

        if cls._read_store is None:
            read_path = cls.get_cache_filepath(cls._cache_name_read)
            store = NdjsonIndexedStore(read_path)
            store.open_or_create()
            cls._read_store = store

    @classmethod
    def list_keys(cls) -> List[str]:
        write_keys = list(cls._cache_write.keys()) if cls._cache_write else []
        read_keys = cls._read_store.list_keys() if cls._read_store else []
        return read_keys + write_keys

    @classmethod
    def retrieve_entry(cls, key: str) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Retrieve a value from the cache, checking write cache first.

        Returns:
            Tuple of (value, res_types) or (None, None) if not found
        """
        if cls._cache_write and key in cls._cache_write:
            value = cls._cache_write[key]
            deserialized_value = json.loads(value["value"])
            return deserialized_value, value["res_types"]

        if cls._read_store is not None:
            result = cls._read_store.get(key)
            if result is not None:
                deserialized_value, res_types = result
                # Promote to write cache for faster future access
                cls.store_entry(
                    key=key,
                    value=cls.serialize_object(deserialized_value),
                    res_types=res_types,
                )
                return deserialized_value, res_types

        return None, None

    @classmethod
    def has_key(cls, key: str) -> bool:
        """Check if a key exists in either cache."""
        if cls._cache_write is not None and key in cls._cache_write:
            return True
        return cls._read_store is not None and cls._read_store.has_key(key)

    @classmethod
    def remove_entry(cls, key: str) -> None:
        """Remove an entry from both caches."""
        if cls._cache_write:
            item = cls._cache_write.pop(key, None)
            if item is not None:
                with open(cls.get_cache_filepath(cls._cache_name_write), "w") as f:
                    for write_key, value in cls._cache_write.items():
                        _write_to_ndjson_cache(
                            f,
                            write_key,
                            value["value"],
                            value["res_types"],
                        )

        if cls._read_store is not None:
            cls._read_store.remove(key)
