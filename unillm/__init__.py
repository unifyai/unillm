# Suppress LiteLLM's LoggingWorker "Task exception was never retrieved" errors.
# These occur when pytest-asyncio creates new event loops between tests, while
# LiteLLM's global singleton worker holds a queue bound to the old loop.
# This is a known LiteLLM issue; suppressing asyncio ERROR logs is the cleanest fix.
import logging

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .clients.uni_llm import Unify, AsyncUnify
from .cache_events import (
    CacheEvent,
    capture_cache_events,
    acapture_cache_events,
)
from .caching import (
    get_cache_stats,
    set_cache_backend,
    CacheStats,
)
from .costs import (
    compute_cost,
    compute_cost_from_response,
    compute_full_cost_from_usage,
    deduct_credits_for_usage,
)
from .helpers import get_seed, set_seed
from .llm_events import (
    LLMEvent,
    set_llm_event_hook,
    get_llm_event_hook,
    llm_event_hook_scope,
    allm_event_hook_scope,
)
from .logger import (
    configure_log_dir,
    write_request_pending,
    append_response_and_finalize,
)
from .settings import SETTINGS
