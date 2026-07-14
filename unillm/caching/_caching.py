"""
Main caching module providing high-level caching functionality.

This module provides decorators and utilities for caching function results
with multiple backend options and flexible caching modes.
"""

import json
import threading
from typing import Any, Dict, Optional, Type

from litellm.types.utils import ModelResponse
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

from .base_cache import BaseCache
from .local_cache import LocalCache
from .local_separate_cache import LocalSeparateCache
from .cache_benchmark import record_get_cache, record_write_to_cache
from unillm.clients.response_format import RESPONSE_FORMAT_SPEC_KEY

# Internal transport metadata must not affect cache keys.
_CACHE_INTERNAL_KW_KEYS = frozenset(
    {
        RESPONSE_FORMAT_SPEC_KEY,
        "_unillm_response_format",
    },
)


def _cache_key_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in kw.items()
        if v is not None and k not in _CACHE_INTERNAL_KW_KEYS
    }


# Global state
CACHE_LOCK = threading.Lock()
CACHING_ENABLED = False
CURRENT_CACHE_BACKEND = "local"

# Available cache backends
CACHE_BACKENDS = {
    "local": LocalCache,
    "local_separate": LocalSeparateCache,
}


def set_cache_backend(backend: str) -> None:
    """Set the current cache backend."""
    global CURRENT_CACHE_BACKEND
    if backend not in CACHE_BACKENDS:
        raise ValueError(
            f"Invalid backend: {backend}. Available: {list(CACHE_BACKENDS.keys())}",
        )
    CURRENT_CACHE_BACKEND = backend


def get_cache_backend(backend: Optional[str] = None) -> Type[BaseCache]:
    """Get the cache backend class."""
    if backend is None:
        backend = CURRENT_CACHE_BACKEND
    if backend not in CACHE_BACKENDS:
        raise ValueError(
            f"Invalid backend: {backend}. Available: {list(CACHE_BACKENDS.keys())}",
        )
    return CACHE_BACKENDS[backend]


def is_caching_enabled() -> bool:
    """Check if caching is globally enabled."""
    return CACHING_ENABLED


@record_get_cache
def _get_cache(
    fn_name: str,
    kw: Dict[str, Any],
    filename: str = None,
    raise_on_empty: bool = False,
    backend: Optional[str] = None,
) -> Optional[Any]:
    global CACHE_LOCK

    type_mapping = {
        "ChatCompletion": ChatCompletion,
        "ModelResponse": ModelResponse,
        "ParsedChatCompletion": ParsedChatCompletion,
    }
    CACHE_LOCK.acquire()
    try:
        current_backend = get_cache_backend(backend)
        current_backend.initialize_cache(filename)
        kw = _cache_key_kwargs(kw)
        kw_str = BaseCache.serialize_object(kw)
        cache_str = f"{fn_name}_{kw_str}"
        if not current_backend.has_key(cache_str):
            if raise_on_empty:
                CACHE_LOCK.release()
                raise Exception(
                    f"Failed to get cache for function {fn_name} with kwargs "
                    f"{BaseCache.serialize_object(kw, indent=4)} "
                    f"from cache at {filename}. Key was not found in the cache.",
                )
            CACHE_LOCK.release()
            return
        ret, res_types = current_backend.retrieve_entry(cache_str)
        if res_types is None:
            CACHE_LOCK.release()
            return ret
        for idx_str, type_str in res_types.items():
            type_str = type_str.split("[")[0]
            idx_list = json.loads(idx_str)
            if len(idx_list) == 0:
                CACHE_LOCK.release()
                typ = type_mapping[type_str]
                if issubclass(typ, BaseModel):
                    return typ(**ret)
                raise Exception(f"Cache indexing found for unsupported type: {typ}")
            item = ret
            for i, idx in enumerate(idx_list):
                if i == len(idx_list) - 1:
                    typ = type_mapping[type_str]
                    if issubclass(typ, BaseModel):
                        item[idx] = typ.from_json(item[idx])
                    else:
                        raise Exception(
                            f"Cache indexing found for unsupported type: {typ}",
                        )
                    break
                item = item[idx]
        CACHE_LOCK.release()
        return ret
    except Exception as e:
        if CACHE_LOCK.locked():
            CACHE_LOCK.release()
        raise Exception(
            f"Failed to get cache for function {fn_name} with kwargs {kw} "
            f"from cache at {filename}",
        ) from e


@record_write_to_cache
def _write_to_cache(
    fn_name: str,
    kw: Dict[str, Any],
    response: Any,
    backend: Optional[str] = None,
    filename: str = None,
):

    global CACHE_LOCK
    CACHE_LOCK.acquire()
    try:
        current_backend = get_cache_backend(backend)
        current_backend.initialize_cache(filename)
        kw = _cache_key_kwargs(kw)
        kw_str = BaseCache.serialize_object(kw)
        cache_str = f"{fn_name}_{kw_str}"
        res_types = {}
        response_str = BaseCache.serialize_object(response, res_types)
        current_backend.store_entry(
            key=cache_str,
            value=response_str,
            res_types=res_types if len(res_types) > 0 else None,
        )
        CACHE_LOCK.release()
    except Exception as e:
        CACHE_LOCK.release()
        raise Exception(
            f"Failed to write function {fn_name} with kwargs {kw} and "
            f"response {response} to cache at {filename}",
        ) from e
