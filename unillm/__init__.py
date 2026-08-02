# Suppress LiteLLM's LoggingWorker "Task exception was never retrieved" errors.
# These occur when pytest-asyncio creates new event loops between tests, while
# LiteLLM's global singleton worker holds a queue bound to the old loop.
# This is a known LiteLLM issue; suppressing asyncio ERROR logs is the cleanest fix.
import logging

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .clients.uni_llm import Unify, AsyncUnify
from .clients.provider_postprocessing import ModelRefusalError
from .clients.completion_mutator import (
    CompletionMutator,
    CompletionMutatorContext,
    inject_tool_call,
)
from .cache_events import (
    CacheEvent,
    capture_cache_events,
    acapture_cache_events,
)
from .caching import (
    get_cache_stats,
    set_cache_backend,
    set_cache_dir,
    CacheStats,
)
from .cost_tracker import (
    CostEvent,
    capture_costs,
    acapture_costs,
)
from .costs import (
    compute_cost,
    compute_cost_from_response,
    compute_full_cost_from_usage,
    extract_openrouter_usage_cost,
)
from .helpers import get_seed, set_seed
from .tokens import (
    count_tokens,
    fills_context_window,
    get_max_input_tokens,
)
from .llm_events import (
    LLMEvent,
    set_llm_event_hook,
    get_llm_event_hook,
    set_global_llm_event_hook,
    get_global_llm_event_hook,
    llm_event_hook_scope,
    allm_event_hook_scope,
)
from .logger import (
    configure_log_dir,
    log_usage,
    write_request_pending,
    append_response_and_finalize,
)
from .settings import SETTINGS
from .limit_hooks import (
    LimitCheckRequest,
    LimitCheckResponse,
    LimitType,
    SpendingLimitExceededError,
    set_limit_check_hook,
    get_limit_check_hook,
    clear_limit_check_hook,
    is_limit_check_enabled,
    check_limits,
    check_limits_sync,
)
from .billing_context import set_billing_context, get_billing_context
