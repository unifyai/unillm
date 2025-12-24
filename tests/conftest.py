import warnings
import atexit

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

# TODO: Won't be needed once LiteLLM handles their type annotations correctly...

# Suppress Pydantic serialization warnings from litellm's logging system.
# Litellm registers an atexit handler that flushes logs after pytest finishes,
# which triggers warnings due to type annotation mismatches (Choices vs StreamingChoices).
# The warnings.filterwarnings approach doesn't work during interpreter shutdown,
# so we need to unregister the atexit handler and flush manually with warnings suppressed.

_flush_on_exit = GLOBAL_LOGGING_WORKER._flush_on_exit

# Unregister litellm's atexit handler
atexit.unregister(_flush_on_exit)


def _flush_with_suppressed_warnings():
    """Flush litellm logs with Pydantic serialization warnings suppressed."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings:",
            category=UserWarning,
        )
        _flush_on_exit()


# Register our wrapper instead
atexit.register(_flush_with_suppressed_warnings)
