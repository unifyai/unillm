"""Corrections to OpenRouter model metadata.

Endpoint resolution registers pricing and capability flags straight from the
OpenRouter catalog for any id the pinned LiteLLM release does not know, so a
newly released model needs no entry here. Add one only to correct the catalog
or to cover a model the catalog omits.
"""

from .utils import register_openrouter_model_info


def _pricing(
    context: int,
    input_cost: float,
    output_cost: float,
    *,
    cache_read: float | None = None,
    cache_write: float | None = None,
    supports_reasoning: bool | None = None,
    supports_function_calling: bool | None = None,
    supports_parallel_function_calling: bool | None = None,
) -> dict:
    info = {
        "max_input_tokens": context,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
    }
    if cache_read is not None:
        info["cache_read_input_token_cost"] = cache_read
    if cache_write is not None:
        info["cache_creation_input_token_cost"] = cache_write
    if supports_reasoning is not None:
        info["supports_reasoning"] = supports_reasoning
    if supports_function_calling is not None:
        info["supports_function_calling"] = supports_function_calling
    if supports_parallel_function_calling is not None:
        info["supports_parallel_function_calling"] = supports_parallel_function_calling
    return info


def _openai_chat(
    context: int,
    input_cost: float,
    output_cost: float,
    *,
    cache_read: float | None = None,
    cache_write: float | None = None,
    supports_reasoning: bool = False,
) -> dict:
    """OpenAI catalog models (shared metadata for OpenRouter transport ids).

    LiteLLM's ``drop_params=True`` strips ``reasoning_effort`` / tool controls
    unless these capability flags are present on the OpenRouter model entry.
    """

    return _pricing(
        context,
        input_cost,
        output_cost,
        cache_read=cache_read,
        cache_write=cache_write,
        supports_reasoning=supports_reasoning,
        supports_function_calling=True,
        supports_parallel_function_calling=True,
    )


register_openrouter_model_info(
    {
        "google/gemini-2.5-flash-lite": _pricing(
            1_048_576,
            0.10 / 1_000_000,
            0.40 / 1_000_000,
            cache_read=0.01 / 1_000_000,
            cache_write=0.08333333333333334 / 1_000_000,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
        ),
        "meta-llama/llama-3.1-8b-instruct": _pricing(
            131_072,
            0.02 / 1_000_000,
            0.03 / 1_000_000,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
        ),
        "meta-llama/llama-3.3-70b-instruct": _pricing(
            131_072,
            0.10 / 1_000_000,
            0.32 / 1_000_000,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
        ),
        "meta-llama/llama-4-maverick": _pricing(
            1_048_576,
            0.15 / 1_000_000,
            0.60 / 1_000_000,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
        ),
        "mistralai/mistral-medium-3": _pricing(
            131_072,
            0.40 / 1_000_000,
            2.00 / 1_000_000,
            cache_read=0.04 / 1_000_000,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
        ),
        "mistralai/mistral-medium-3.1": _pricing(
            131_072,
            0.40 / 1_000_000,
            2.00 / 1_000_000,
            cache_read=0.04 / 1_000_000,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
        ),
        "openai/gpt-4-turbo": _openai_chat(
            128_000,
            10.00 / 1_000_000,
            30.00 / 1_000_000,
        ),
        "openai/gpt-4o-mini": _openai_chat(
            128_000,
            0.15 / 1_000_000,
            0.60 / 1_000_000,
            cache_read=0.075 / 1_000_000,
        ),
        "openai/gpt-4o-mini-search-preview": _openai_chat(
            128_000,
            0.15 / 1_000_000,
            0.60 / 1_000_000,
        ),
        "openai/gpt-4o-search-preview": _openai_chat(
            128_000,
            2.50 / 1_000_000,
            10.00 / 1_000_000,
        ),
        "openai/gpt-5.1": _openai_chat(
            400_000,
            1.25 / 1_000_000,
            10.00 / 1_000_000,
            cache_read=0.13 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.1-chat": _openai_chat(
            128_000,
            1.25 / 1_000_000,
            10.00 / 1_000_000,
            cache_read=0.13 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.4": _openai_chat(
            1_050_000,
            2.50 / 1_000_000,
            15.00 / 1_000_000,
            cache_read=0.25 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.4-mini": _openai_chat(
            400_000,
            0.75 / 1_000_000,
            4.50 / 1_000_000,
            cache_read=0.075 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.4-nano": _openai_chat(
            400_000,
            0.20 / 1_000_000,
            1.25 / 1_000_000,
            cache_read=0.02 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.5": _openai_chat(
            1_050_000,
            5.00 / 1_000_000,
            30.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.6-sol": _openai_chat(
            1_050_000,
            5.00 / 1_000_000,
            30.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
            cache_write=6.25 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.6-terra": _openai_chat(
            1_050_000,
            2.50 / 1_000_000,
            15.00 / 1_000_000,
            cache_read=0.25 / 1_000_000,
            cache_write=3.125 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/gpt-5.6-luna": _openai_chat(
            1_050_000,
            1.00 / 1_000_000,
            6.00 / 1_000_000,
            cache_read=0.10 / 1_000_000,
            cache_write=1.25 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/o3": _openai_chat(
            200_000,
            2.00 / 1_000_000,
            8.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
            supports_reasoning=True,
        ),
        "openai/o4-mini": _openai_chat(
            200_000,
            1.10 / 1_000_000,
            4.40 / 1_000_000,
            cache_read=0.275 / 1_000_000,
            supports_reasoning=True,
        ),
    },
)
