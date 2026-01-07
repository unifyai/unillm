"""Cost computation using LiteLLM's pricing data."""

from typing import Optional, Union

import litellm
import unify


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
        return model.split("@")[0]
    return model


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
    normalized = _normalize_model_name(model)
    try:
        return litellm.get_model_info(normalized)
    except Exception as e:
        raise ValueError(f"Could not find pricing info for model '{model}': {e}")


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
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

    input_cost_per_token = model_info.get("input_cost_per_token", 0)
    output_cost_per_token = model_info.get("output_cost_per_token", 0)

    cost = (prompt_tokens * input_cost_per_token) + (
        completion_tokens * output_cost_per_token
    )
    return cost


def compute_cost_from_response(
    model: str,
    response: Union[dict, object],
) -> Optional[float]:
    """
    Compute cost from a ChatCompletion response object.

    Extracts token usage from the response and computes the cost.

    Args:
        model: The model identifier.
        response: The ChatCompletion response (either a dict or object with usage attr).

    Returns:
        The cost in USD, or None if usage information is unavailable.
    """
    # Extract usage from response
    if hasattr(response, "usage"):
        usage = response.usage
        if hasattr(usage, "prompt_tokens"):
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
        else:
            return None
    elif isinstance(response, dict):
        usage = response.get("usage", {})
        if not usage:
            return None
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    else:
        return None

    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    try:
        return compute_cost(model, prompt_tokens, completion_tokens)
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

    return (prompt_tokens * input_cost_per_token) + (
        completion_tokens * output_cost_per_token
    )


def deduct_credits_for_usage(
    model: str,
    response: Union[dict, object],
) -> float:
    """
    Compute the cost of an LLM response and deduct credits from the user's account.

    This function extracts usage information from an API response, computes the
    cost using LiteLLM's pricing data (including audio tokens for Realtime API),
    and deducts the appropriate credits.

    Supports:
    - Standard chat completions (prompt_tokens, completion_tokens)
    - Realtime API responses (input_token_details.audio_tokens, etc.)

    Args:
        model: The model identifier (e.g., 'gpt-4o', 'gpt-4o-realtime-preview').
        response: The full API response object or dict containing usage information.

    Returns:
        The cost that was deducted in USD.

    Raises:
        ValueError: If the model is not found in LiteLLM's pricing data.
        ValueError: If no usage information is found in the response.
    """
    # Extract usage from response
    usage = _extract_usage_from_response(response)
    if usage is None:
        raise ValueError("No usage information found in response")

    # Compute cost (this will raise ValueError if model not found)
    cost = compute_full_cost_from_usage(model, usage)

    # Deduct credits if cost is positive
    if cost > 0:
        unify.deduct_credits(cost)

    return cost
