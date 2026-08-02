"""
Local file-based cache implementation.

Uses a hash→offset index over the NDJSON file so full cache keys are not
held in memory after startup.
"""

import os
from typing import Any, Dict, List, Optional

from .base_cache import BaseCache
from .ndjson_index import NdjsonIndexedStore


class LocalCache(BaseCache):
    """Local file-based cache implementation."""

    _store: Optional[NdjsonIndexedStore] = None
    _cache_dir: str = os.environ.get("UNILLM_CACHE_DIR", os.getcwd())
    _cache_filename: str = ".cache.ndjson"
    _enabled: bool = False

    @classmethod
    def set_cache_name(cls, name: str) -> None:
        """Set the cache filename and reset the indexed store."""
        cls._cache_filename = name
        cls._store = None

    @classmethod
    def get_cache_name(cls) -> str:
        """Get the current cache filename."""
        return cls._cache_filename

    @classmethod
    def set_cache_dir(cls, path: str) -> None:
        """Point the cache at *path* and drop the store opened under the old one."""
        cls._cache_dir = path
        cls._store = None

    @classmethod
    def get_cache_dir(cls) -> str:
        """Get the directory the cache file lives in."""
        return cls._cache_dir

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
        """Store a key-value pair in the cache."""
        cls._store.append(key, value, res_types)

    @classmethod
    def initialize_cache(cls, name: str = None) -> None:
        """Initialize or load the indexed cache from disk."""
        if cls._store is None:
            cache_filepath = cls.get_cache_filepath(name)
            store = NdjsonIndexedStore(cache_filepath)
            store.open_or_create()
            cls._store = store

    @classmethod
    def list_keys(cls) -> List[str]:
        if cls._store is None:
            return []
        return cls._store.list_keys()

    @classmethod
    def retrieve_entry(cls, key: str) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Retrieve a value from the cache.

        Returns:
            Tuple of (value, type_registry) or (None, None) if not found
        """
        if cls._store is None:
            return None, None
        result = cls._store.get(key)
        if result is None:
            return None, None
        return result

    @classmethod
    def has_key(cls, key: str) -> bool:
        """Check if a key exists in the cache."""
        return cls._store is not None and cls._store.has_key(key)

    @classmethod
    def remove_entry(cls, key: str) -> None:
        """Remove an entry and rebuild the on-disk cache + index."""
        if cls._store is not None:
            cls._store.remove(key)
