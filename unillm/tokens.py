"""Token tracking utilities wrapping LiteLLM's token counting and model info."""

from typing import Iterable, List, Optional

import litellm
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


def _normalize_model_name(endpoint: str) -> str:
    """Strip the ``@provider`` suffix (e.g. ``"gpt-4o@openai"`` -> ``"gpt-4o"``)."""
    if "@" in endpoint:
        return endpoint.split("@")[0]
    return endpoint


def get_max_input_tokens(endpoint: str) -> int:
    """Return the context window (max input tokens) for the given model.

    Args:
        endpoint: Model identifier, optionally with @provider suffix
                  (e.g. ``"gpt-4o"`` or ``"gpt-4o@openai"``).

    Returns:
        Maximum number of input tokens the model accepts.

    Raises:
        ValueError: If the model is not found in LiteLLM's model data.
    """
    model = _normalize_model_name(endpoint)
    try:
        info = litellm.get_model_info(model)
    except Exception as e:
        raise ValueError(f"Could not find model info for '{endpoint}': {e}")
    return info["max_input_tokens"]


def count_tokens(
    endpoint: str,
    messages: List[ChatCompletionMessageParam],
    tools: Optional[Iterable[ChatCompletionToolParam]] = None,
) -> int:
    """Count the number of tokens in a request payload.

    Args:
        endpoint: Model identifier, optionally with @provider suffix.
        messages: Conversation messages.
        tools: Tool/function definitions attached to the request.

    Returns:
        Token count for the combined request payload.
    """
    model = _normalize_model_name(endpoint)
    tools_list = list(tools) if tools is not None else None
    return litellm.token_counter(
        model=model,
        messages=messages or [],
        tools=tools_list,
    )


def fills_context_window(
    threshold: float,
    endpoint: str,
    messages: List[ChatCompletionMessageParam],
    tools: Optional[Iterable[ChatCompletionToolParam]] = None,
) -> bool:
    """Check whether a request's token usage meets or exceeds a fraction of the context window.

    Args:
        threshold: Fraction of the context window (0.0–1.0). The check
                   returns ``True`` when ``tokens / context_window >= threshold``.
        endpoint: Model identifier, optionally with @provider suffix.
        messages: Conversation messages.
        tools: Tool/function definitions attached to the request.

    Returns:
        ``True`` if the ratio of request tokens to the context window is
        at or above *threshold*.
    """
    tokens = count_tokens(endpoint, messages, tools)
    window = get_max_input_tokens(endpoint)
    return tokens / window >= threshold
