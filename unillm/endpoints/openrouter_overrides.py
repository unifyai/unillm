from .utils import register_openrouter_model_info


def _pricing(
    context: int,
    input_cost: float,
    output_cost: float,
    *,
    cache_read: float | None = None,
    cache_write: float | None = None,
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
    return info


register_openrouter_model_info(
    {
        "google/gemini-2.5-flash-lite": _pricing(
            1_048_576,
            0.10 / 1_000_000,
            0.40 / 1_000_000,
            cache_read=0.01 / 1_000_000,
            cache_write=0.08333333333333334 / 1_000_000,
        ),
        "meta-llama/llama-3.1-8b-instruct": _pricing(
            131_072,
            0.02 / 1_000_000,
            0.03 / 1_000_000,
        ),
        "meta-llama/llama-3.3-70b-instruct": _pricing(
            131_072,
            0.10 / 1_000_000,
            0.32 / 1_000_000,
        ),
        "meta-llama/llama-4-maverick": _pricing(
            1_048_576,
            0.15 / 1_000_000,
            0.60 / 1_000_000,
        ),
        "mistralai/mistral-medium-3": _pricing(
            131_072,
            0.40 / 1_000_000,
            2.00 / 1_000_000,
            cache_read=0.04 / 1_000_000,
        ),
        "mistralai/mistral-medium-3.1": _pricing(
            131_072,
            0.40 / 1_000_000,
            2.00 / 1_000_000,
            cache_read=0.04 / 1_000_000,
        ),
        "openai/gpt-4-turbo": _pricing(
            128_000,
            10.00 / 1_000_000,
            30.00 / 1_000_000,
        ),
        "openai/gpt-4o-mini": _pricing(
            128_000,
            0.15 / 1_000_000,
            0.60 / 1_000_000,
            cache_read=0.075 / 1_000_000,
        ),
        "openai/gpt-4o-mini-search-preview": _pricing(
            128_000,
            0.15 / 1_000_000,
            0.60 / 1_000_000,
        ),
        "openai/gpt-4o-search-preview": _pricing(
            128_000,
            2.50 / 1_000_000,
            10.00 / 1_000_000,
        ),
        "openai/gpt-5.1": _pricing(
            400_000,
            1.25 / 1_000_000,
            10.00 / 1_000_000,
            cache_read=0.13 / 1_000_000,
        ),
        "openai/gpt-5.1-chat": _pricing(
            128_000,
            1.25 / 1_000_000,
            10.00 / 1_000_000,
            cache_read=0.13 / 1_000_000,
        ),
        "openai/gpt-5.4": _pricing(
            1_050_000,
            2.50 / 1_000_000,
            15.00 / 1_000_000,
            cache_read=0.25 / 1_000_000,
        ),
        "openai/gpt-5.4-mini": _pricing(
            400_000,
            0.75 / 1_000_000,
            4.50 / 1_000_000,
            cache_read=0.075 / 1_000_000,
        ),
        "openai/gpt-5.4-nano": _pricing(
            400_000,
            0.20 / 1_000_000,
            1.25 / 1_000_000,
            cache_read=0.02 / 1_000_000,
        ),
        "openai/gpt-5.5": _pricing(
            1_050_000,
            5.00 / 1_000_000,
            30.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
        ),
        "openai/o3": _pricing(
            200_000,
            2.00 / 1_000_000,
            8.00 / 1_000_000,
            cache_read=0.50 / 1_000_000,
        ),
        "openai/o4-mini": _pricing(
            200_000,
            1.10 / 1_000_000,
            4.40 / 1_000_000,
            cache_read=0.275 / 1_000_000,
        ),
    },
)
