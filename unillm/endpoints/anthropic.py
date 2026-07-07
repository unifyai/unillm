from .utils import (
    register_litellm_model_info,
    register_model_alias_map,
)

provider = "anthropic"
models = {
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-3.5-haiku": "anthropic/claude-3-5-haiku-20241022",
    "claude-4-sonnet": "anthropic/claude-sonnet-4-20250514",
    "claude-4-opus": "anthropic/claude-opus-4-20250514",
    "claude-4.1-opus": "anthropic/claude-opus-4-1-20250805",
    "claude-4.5-sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "claude-4.5-haiku": "anthropic/claude-haiku-4-5-20251001",
    "claude-4.5-opus": "anthropic/claude-opus-4-5-20251101",
    "claude-4.6-opus": "anthropic/claude-opus-4-6",
    "claude-4.6-sonnet": "anthropic/claude-sonnet-4-6",
    "claude-4.8-opus": "anthropic/claude-opus-4-8",
    "claude-fable-5": "anthropic/claude-fable-5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
}

CONTEXT_1M_BETA = "context-1m-2025-08-07"
CONTEXT_1M_MODELS = {
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-8",
    models["claude-4-sonnet"],
    models["claude-4.5-sonnet"],
    models["claude-4.6-opus"],
    models["claude-4.6-sonnet"],
    models["claude-4.8-opus"],
}

ADAPTIVE_THINKING_MODELS = {
    "anthropic/claude-opus-4-8",
    models["claude-4.8-opus"],
    "anthropic/claude-fable-5",
    models["claude-fable-5"],
    "anthropic/claude-sonnet-5",
    models["claude-sonnet-5"],
}

# Models (Opus 4.7 generation onwards) that return a 400 error when
# temperature/top_p/top_k are set to any non-default value. Requests to these
# models must omit sampling parameters entirely.
SAMPLING_PARAMS_REJECTED_MODELS = {
    "anthropic/claude-opus-4-8",
    models["claude-4.8-opus"],
    "anthropic/claude-fable-5",
    models["claude-fable-5"],
    "anthropic/claude-sonnet-5",
    models["claude-sonnet-5"],
}

# Models whose safety classifiers can decline a request. A refusal arrives as
# a successful response with stop_reason "refusal" (surfaced by litellm as
# finish_reason "content_filter"), not an HTTP error. Anthropic's guidance is
# that a refused request can usually be served by another Claude model, so
# each classifier-gated model maps to the fallback used for a one-shot retry.
REFUSAL_FALLBACK_MODELS = {
    "anthropic/claude-fable-5": "anthropic/claude-opus-4-8",
}

register_model_alias_map(provider, models)
register_litellm_model_info(
    {
        "anthropic/claude-3-5-haiku-20241022": {
            "litellm_provider": provider,
            "mode": "chat",
            "max_input_tokens": 200_000,
            "input_cost_per_token": 0.80 / 1_000_000,
            "cache_creation_input_token_cost": 1.00 / 1_000_000,
            "cache_read_input_token_cost": 0.08 / 1_000_000,
            "output_cost_per_token": 4.00 / 1_000_000,
        },
        "anthropic/claude-opus-4-6": {
            "litellm_provider": provider,
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 5.00 / 1_000_000,
            "output_cost_per_token": 25.00 / 1_000_000,
        },
        "anthropic/claude-sonnet-4-6": {
            "litellm_provider": provider,
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 3.00 / 1_000_000,
            "output_cost_per_token": 15.00 / 1_000_000,
        },
        "anthropic/claude-opus-4-8": {
            "litellm_provider": provider,
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 5.00 / 1_000_000,
            "output_cost_per_token": 25.00 / 1_000_000,
        },
        # Claude Fable 5: 1M context by default (no beta header), adaptive
        # thinking is the only thinking mode.
        "anthropic/claude-fable-5": {
            "litellm_provider": provider,
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 128_000,
            "input_cost_per_token": 10.00 / 1_000_000,
            "output_cost_per_token": 50.00 / 1_000_000,
        },
        # Claude Sonnet 5: 1M context by default, adaptive thinking on by
        # default. Registered at the standard $3/$15 rate that applies from
        # September 1, 2026 (an intro $2/$10 rate runs until August 31).
        "anthropic/claude-sonnet-5": {
            "litellm_provider": provider,
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 128_000,
            "input_cost_per_token": 3.00 / 1_000_000,
            "output_cost_per_token": 15.00 / 1_000_000,
        },
    },
)
