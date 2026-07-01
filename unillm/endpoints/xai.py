from .utils import (
    openrouter_model,
    register_model_alias_map,
    register_openrouter_model_info,
)

models = {
    "grok-4.1-fast-reasoning": openrouter_model("x-ai/grok-4.3"),
    "grok-4.1-fast-non-reasoning": openrouter_model("x-ai/grok-4.3"),
    "grok-code-fast": openrouter_model("x-ai/grok-4.3"),
    "grok-4-fast-reasoning": openrouter_model("x-ai/grok-4.20"),
    "grok-4-fast-non-reasoning": openrouter_model("x-ai/grok-4.20"),
    "grok-4": openrouter_model("x-ai/grok-4.20"),
    "grok-3": openrouter_model("x-ai/grok-4.20"),
    "grok-3-mini": openrouter_model("x-ai/grok-4.20"),
}

register_model_alias_map("xai", models)
register_openrouter_model_info(
    {
        "x-ai/grok-4.3": {
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 1.25 / 1_000_000,
            "output_cost_per_token": 2.50 / 1_000_000,
        },
        "x-ai/grok-4.20": {
            "max_input_tokens": 2_000_000,
            "input_cost_per_token": 1.25 / 1_000_000,
            "output_cost_per_token": 2.50 / 1_000_000,
        },
    },
)
