from typing import Literal, Union

CACHE_MODES = (
    "both",
    "write",
    "read",
    "read-only",
)

CacheMode = Literal[
    "both",
    "write",
    "read",
    "read-only",
]

CacheParam = Union[bool, CacheMode]

# All values accepted by the cache parameter (for runtime validation)
VALID_CACHE_VALUES = (*CACHE_MODES, True, False, None)
