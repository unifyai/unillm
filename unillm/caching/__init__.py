"""
Caching utilities for the unillm framework.

This module provides a flexible caching system with multiple backends:
- LocalCache: Indexed local file-based caching
- LocalSeparateCache: Separate read/write caches for CI (indexed read file)
"""

from .base_cache import BaseCache
from .canonical import (
    CANON_VERSION,
    canonical_digest,
    canonical_digest_of_raw_key,
    canonical_kw,
    parse_raw_key,
)
from .local_cache import LocalCache
from .local_separate_cache import LocalSeparateCache
from .cache_benchmark import CacheStats, get_cache_stats
from ._caching import (
    _get_cache,
    _write_to_cache,
    is_caching_enabled,
    set_cache_backend,
    get_cache_backend,
    set_cache_dir,
)

__all__ = [
    "BaseCache",
    "CANON_VERSION",
    "canonical_digest",
    "canonical_digest_of_raw_key",
    "canonical_kw",
    "parse_raw_key",
    "LocalCache",
    "LocalSeparateCache",
    "CacheStats",
    "get_cache_stats",
    "is_caching_enabled",
    "set_cache_backend",
    "get_cache_backend",
    "set_cache_dir",
]
