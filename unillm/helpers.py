from typing import Any, Optional


class _UnsetSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        # Stable textual form to avoid process-specific object addresses
        return "<UNSET>"


UNSET = _UnsetSentinel()


def _default(value: Any, default_value: Any) -> Any:
    return value if value is not None else default_value


# Global seed for reproducibility
_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: Optional[int]) -> None:
    """Set the global seed for reproducibility."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed


def get_seed() -> Optional[int]:
    """Get the global seed, or None if not set."""
    return _GLOBAL_SEED
