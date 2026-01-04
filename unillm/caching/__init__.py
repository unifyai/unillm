"""
Caching utilities for the unillm framework.

This module provides a flexible caching system with multiple backends:
- LocalCache: Simple local file-based caching
- LocalSeparateCache: Separate read/write caches for better performance
"""

from .base_cache import BaseCache
from .local_cache import LocalCache
from .local_separate_cache import LocalSeparateCache
from .cache_benchmark import CacheStats, get_cache_stats
from ._caching import (
    _get_cache,
    _write_to_cache,
    is_caching_enabled,
    set_cache_backend,
    get_cache_backend,
)

__all__ = [
    "BaseCache",
    "LocalCache",
    "LocalSeparateCache",
    "CacheStats",
    "get_cache_stats",
    "_get_cache",
    "_write_to_cache",
    "is_caching_enabled",
    "set_cache_backend",
    "get_cache_backend",
]
