"""Native OpenAI endpoint registrations.

``gpt-*@openai`` maps to the OpenAI API (via LiteLLM), not OpenRouter.
OpenRouter-hosted OpenAI catalog models use ``openai/<id>@openrouter``.
"""

from .utils import register_litellm_model_info, register_model_alias_map

# Bare LiteLLM OpenAI ids. Public accounting alias remains the dict key
# (see ``_public_model_alias`` for provider ``openai``).
models = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4o": "gpt-4o",
    "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "gpt-4o-mini": "gpt-4o-mini",
    "chatgpt-4o-latest": "chatgpt-4o-latest",
    "o1": "o1",
    "o3-mini": "o3-mini",
    "gpt-4o-search-preview": "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview": "gpt-4o-mini-search-preview",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5-chat-latest": "gpt-5-chat-latest",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-chat-latest": "gpt-5.1-chat-latest",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.4-nano": "gpt-5.4-nano",
    "gpt-5.5": "gpt-5.5",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-luna": "gpt-5.6-luna",
}

register_model_alias_map("openai", models)


def _native_openai_info(
    context: int,
    input_cost: float,
    output_cost: float,
    *,
    cache_read: float | None = None,
    cache_write: float | None = None,
    supports_reasoning: bool = False,
) -> dict:
    """Metadata for OpenAI models missing from the pinned LiteLLM release."""

    info: dict = {
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": context,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
        "supports_function_calling": True,
        "supports_parallel_function_calling": True,
        "supports_reasoning": supports_reasoning,
    }
    if cache_read is not None:
        info["cache_read_input_token_cost"] = cache_read
    if cache_write is not None:
        info["cache_creation_input_token_cost"] = cache_write
    return info


# Register newer GPT catalog entries that LiteLLM may not yet price.
register_litellm_model_info(
    {
        "gpt-5.5": _native_openai_info(
            1_050_000,
            5.00 / 1_000_000,
            30.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
            supports_reasoning=True,
        ),
        "gpt-5.6-sol": _native_openai_info(
            1_050_000,
            5.00 / 1_000_000,
            30.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
            cache_write=6.25 / 1_000_000,
            supports_reasoning=True,
        ),
        "gpt-5.6-terra": _native_openai_info(
            1_050_000,
            2.50 / 1_000_000,
            15.00 / 1_000_000,
            cache_read=0.25 / 1_000_000,
            cache_write=3.125 / 1_000_000,
            supports_reasoning=True,
        ),
        "gpt-5.6-luna": _native_openai_info(
            1_050_000,
            1.00 / 1_000_000,
            6.00 / 1_000_000,
            cache_read=0.10 / 1_000_000,
            cache_write=1.25 / 1_000_000,
            supports_reasoning=True,
        ),
    },
)
