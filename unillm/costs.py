"""Cost computation using LiteLLM's pricing data."""

import os
import re
from typing import Optional, Union

import litellm

# Cost margin multiplier for billing users.
# Configurable via UNILLM_COST_MARGIN environment variable.
_DEFAULT_COST_MARGIN = 1.2


def get_cost_margin() -> float:
    """Get the cost margin multiplier from environment or use default.

    The margin is applied to provider costs to determine what users are billed.
    Configurable via the UNILLM_COST_MARGIN environment variable.

    Returns:
        The cost margin multiplier.
    """
    margin_str = os.environ.get("UNILLM_COST_MARGIN")
    if margin_str is not None:
        try:
            return float(margin_str)
        except ValueError:
            pass
    return _DEFAULT_COST_MARGIN


def _normalize_model_name(model: str) -> str:
    """
    Normalize model name for LiteLLM lookup.

    Strips the @provider suffix used by unify/unillm (e.g., 'gpt-5.2@openai' -> 'gpt-5.2').

    Args:
        model: The model identifier, possibly with @provider suffix.

    Returns:
        The model name without provider suffix.
    """
    if "@" in model:
        return model.rsplit("@", 1)[0]
    return model


def _is_openrouter_model(model: str) -> bool:
    """Return whether *model* is an OpenRouter transport or ``*@openrouter`` endpoint."""

    if model.endswith("@openrouter"):
        return True
    return model.startswith("openrouter/")


def extract_openrouter_usage_cost(usage: Union[dict, object, None]) -> Optional[float]:
    """Return OpenRouter's authoritative ``usage.cost`` (USD) when present."""

    if usage is None:
        return None
    cost = _get_nested_attr(usage, "cost", default=None)
    if cost is None and isinstance(usage, dict):
        cost = usage.get("cost")
    if cost is None:
        return None
    try:
        value = float(cost)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _get_model_info(model: str) -> dict:
    """
    Get model pricing info from LiteLLM.

    Args:
        model: The model identifier (may include @provider suffix which will be stripped).

    Returns:
        The model info dict from LiteLLM.

    Raises:
        ValueError: If the model is not found in LiteLLM's pricing data.
    """
    from .endpoints.utils import (
        ensure_endpoints_imported,
        get_model_alias,
        get_transport_model_alias,
        openrouter_model,
    )

    ensure_endpoints_imported()
    candidates: list[str] = []
    if "@" in model:
        try:
            candidates.append(get_model_alias(model))
            candidates.append(get_transport_model_alias(model))
        except ValueError:
            candidates.append(_normalize_model_name(model))
            if model.endswith("@openrouter"):
                candidates.append(openrouter_model(_normalize_model_name(model)))
    else:
        candidates.append(model)
        if model.startswith("openrouter/"):
            pass
        elif "/" in model:
            candidates.append(openrouter_model(model))

    last_error: Exception | None = None
    for candidate in dict.fromkeys(candidates):
        try:
            return litellm.get_model_info(candidate)
        except Exception as e:
            last_error = e

    # Fall back to OpenRouter catalog snapshot pricing when LiteLLM is missing
    # the model (common for the long tail of *@openrouter endpoints).
    if _is_openrouter_model(model):
        from .openrouter_catalog import catalog_pricing_as_litellm_info

        model_id = _normalize_model_name(model)
        if model_id.startswith("openrouter/"):
            model_id = model_id[len("openrouter/") :]
        catalog_info = catalog_pricing_as_litellm_info(model_id)
        if catalog_info is not None:
            return catalog_info

    raise ValueError(f"Could not find pricing info for model '{model}': {last_error}")


_TIER_RE = re.compile(r"^(.+)_above_(\d+)k?_tokens$")


def _get_tiered_rate(
    model_info: dict,
    base_key: str,
    prompt_tokens: int,
) -> tuple[float, int | None]:
    """Return the effective per-token rate, accounting for tier thresholds.

    LiteLLM encodes tiered pricing as ``{base_key}_above_{N}k_tokens``
    (e.g. ``input_cost_per_token_above_200k_tokens``).  If the prompt
    exceeds a tier threshold *and* a higher rate is defined for that tier,
    the higher rate applies.  When multiple tiers exist, the highest
    applicable threshold wins.
    """
    base_rate = model_info.get(base_key, 0)
    best_threshold = None
    best_rate = None
    for key, value in model_info.items():
        if value is None:
            continue
        m = _TIER_RE.match(key)
        if m and m.group(1) == base_key:
            threshold = int(m.group(2)) * 1_000
            if prompt_tokens > threshold:
                if best_threshold is None or threshold > best_threshold:
                    best_threshold = threshold
                    best_rate = value
    if best_rate is not None:
        return best_rate, best_threshold
    return base_rate, None


def _get_cache_read_rate(
    model_info: dict,
    prompt_tokens: int,
) -> tuple[float, int | None]:
    """Return the cache-read prompt token rate when the provider exposes one."""

    for base_key in (
        "cache_read_input_token_cost",
        "input_cost_per_token_cache_hit",
    ):
        if model_info.get(base_key) is not None:
            return _get_tiered_rate(model_info, base_key, prompt_tokens)
    return 0, None


def _compute_text_token_cost(
    model_info: dict,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Compute text token cost with tiered long-context pricing.

    Some providers charge higher rates when the prompt exceeds a threshold
    (e.g. Anthropic at 200k tokens).  This function discovers tier
    boundaries from the model_info keys automatically.
    """
    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    billable_input_tokens = prompt_tokens - cached_tokens

    input_rate = model_info.get("input_cost_per_token", 0)
    input_rate_tier, input_threshold = _get_tiered_rate(
        model_info,
        "input_cost_per_token",
        prompt_tokens,
    )
    cache_read_rate, _ = _get_cache_read_rate(model_info, prompt_tokens)
    output_rate_tier, _ = _get_tiered_rate(
        model_info,
        "output_cost_per_token",
        prompt_tokens,
    )

    if input_threshold is not None and billable_input_tokens:
        threshold_tokens = min(input_threshold, billable_input_tokens)
        input_cost = (
            threshold_tokens * input_rate
            + max(0, billable_input_tokens - input_threshold) * input_rate_tier
        )
    else:
        input_cost = billable_input_tokens * input_rate

    cache_read_cost = cached_tokens * cache_read_rate

    output_cost = completion_tokens * output_rate_tier

    return input_cost + cache_read_cost + output_cost


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Compute the cost of an LLM request using LiteLLM's pricing data.

    Args:
        model: The model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        The cost in USD.

    Raises:
        ValueError: If the model is not found in LiteLLM's pricing data.
    """
    model_info = _get_model_info(model)
    return _compute_text_token_cost(
        model_info,
        prompt_tokens,
        completion_tokens,
        cached_tokens=cached_tokens,
    )


def compute_cost_from_response(
    model: str,
    response: Union[dict, object],
) -> Optional[float]:
    """
    Compute cost from a ChatCompletion response object.

    For OpenRouter transports, prefers the authoritative ``usage.cost`` field
    when present. Otherwise extracts token usage and prices via LiteLLM /
    catalog fallback.

    Args:
        model: The model identifier.
        response: The ChatCompletion response (either a dict or object with usage attr).

    Returns:
        The cost in USD, or None if usage information is unavailable.
    """
    # Extract usage from response
    if hasattr(response, "usage"):
        usage = response.usage
        if usage is None:
            return None
    elif isinstance(response, dict):
        usage = response.get("usage", {})
        if not usage:
            return None
    else:
        return None

    # OpenRouter reports the true charged USD amount on usage.cost.
    if _is_openrouter_model(model):
        reported = extract_openrouter_usage_cost(usage)
        if reported is not None:
            return reported

    if hasattr(usage, "prompt_tokens"):
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
    elif isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
    else:
        return None

    cached_tokens = _get_nested_attr(
        usage,
        "prompt_tokens_details",
        "cached_tokens",
    ) or _get_nested_attr(usage, "cache_read_input_tokens")

    if prompt_tokens == 0 and completion_tokens == 0:
        # Still allow OpenRouter zero-token responses with an explicit cost.
        if _is_openrouter_model(model):
            reported = extract_openrouter_usage_cost(usage)
            if reported is not None:
                return reported
        return None

    try:
        return compute_cost(
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens=cached_tokens,
        )
    except ValueError:
        # Model not in LiteLLM's pricing database - skip cost tracking
        return None


def _extract_usage_from_response(response: Union[dict, object]) -> Optional[dict]:
    """
    Extract usage dict from a response object.

    Handles both object responses (with .usage attribute) and dict responses.

    Args:
        response: The API response.

    Returns:
        The usage dict, or None if not available.
    """
    if hasattr(response, "usage"):
        usage = response.usage
        if usage is None:
            return None
        # Convert object to dict if needed
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        elif hasattr(usage, "__dict__"):
            return dict(usage.__dict__)
        elif isinstance(usage, dict):
            return usage
        return None
    elif isinstance(response, dict):
        return response.get("usage")
    return None


def _get_nested_attr(obj: Union[dict, object], *keys, default=0):
    """
    Get a nested attribute from a dict or object.

    Args:
        obj: The object or dict to traverse.
        *keys: The keys/attributes to traverse.
        default: Default value if not found.

    Returns:
        The value at the nested path, or default.
    """
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key, None)
        else:
            return default
    return current if current is not None else default


def compute_full_cost_from_usage(model: str, usage: Union[dict, object]) -> float:
    """
    Compute the full cost from a usage object, including audio tokens.

    Supports both standard chat completion usage (prompt_tokens, completion_tokens)
    and Realtime API usage (input_token_details.audio_tokens, etc.).

    Args:
        model: The model identifier.
        usage: The usage object/dict from an API response.

    Returns:
        The cost in USD.

    Raises:
        ValueError: If the model is not found in LiteLLM's pricing data.
    """
    model_info = _get_model_info(model)

    # Get pricing info
    input_cost_per_token = model_info.get("input_cost_per_token", 0)
    output_cost_per_token = model_info.get("output_cost_per_token", 0)
    input_cost_per_audio_token = model_info.get("input_cost_per_audio_token", 0)
    output_cost_per_audio_token = model_info.get("output_cost_per_audio_token", 0)

    # Try to get detailed token breakdown (Realtime API format)
    input_text_tokens = _get_nested_attr(usage, "input_token_details", "text_tokens")
    input_audio_tokens = _get_nested_attr(usage, "input_token_details", "audio_tokens")
    output_text_tokens = _get_nested_attr(usage, "output_token_details", "text_tokens")
    output_audio_tokens = _get_nested_attr(
        usage,
        "output_token_details",
        "audio_tokens",
    )

    has_detailed_breakdown = (
        input_text_tokens > 0
        or input_audio_tokens > 0
        or output_text_tokens > 0
        or output_audio_tokens > 0
    )

    if has_detailed_breakdown:
        # Use detailed breakdown for accurate audio billing
        text_cost = (input_text_tokens * input_cost_per_token) + (
            output_text_tokens * output_cost_per_token
        )
        audio_cost = (input_audio_tokens * input_cost_per_audio_token) + (
            output_audio_tokens * output_cost_per_audio_token
        )
        return text_cost + audio_cost

    # Fall back to standard chat completion format (prompt_tokens/completion_tokens)
    prompt_tokens = _get_nested_attr(usage, "prompt_tokens")
    completion_tokens = _get_nested_attr(usage, "completion_tokens")

    # Also try input_tokens/output_tokens (alternative format)
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _get_nested_attr(usage, "input_tokens")
        completion_tokens = _get_nested_attr(usage, "output_tokens")

    cached_tokens = _get_nested_attr(
        usage,
        "prompt_tokens_details",
        "cached_tokens",
    ) or _get_nested_attr(usage, "cache_read_input_tokens")

    return _compute_text_token_cost(
        model_info,
        prompt_tokens,
        completion_tokens,
        cached_tokens=cached_tokens,
    )
