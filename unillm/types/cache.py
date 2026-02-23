from typing import Literal, Union

CACHE_MODES = (
    "both",
    "write",
    "read",
    "read-only",
    "read-closest",
)

CacheMode = Literal[
    "both",
    "write",
    "read",
    "read-only",
    "read-closest",
]

CacheParam = Union[bool, CacheMode]

# All values accepted by the cache parameter (for runtime validation)
VALID_CACHE_VALUES = (*CACHE_MODES, True, False, None)
