"""Cost computation using LiteLLM's pricing data."""

from typing import Optional, Union

import litellm


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
    try:
        model_info = litellm.get_model_info(model)
    except Exception as e:
        raise ValueError(f"Could not find pricing info for model '{model}': {e}")

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

    return compute_cost(model, prompt_tokens, completion_tokens)
