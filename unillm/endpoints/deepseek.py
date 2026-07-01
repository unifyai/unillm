from .utils import (
    openrouter_model,
    register_litellm_model_info,
    register_model_alias_map,
    register_openrouter_model_info,
)

models = {
    "deepseek-v4-max": openrouter_model("deepseek/deepseek-v4-pro"),
    "deepseek-v4": openrouter_model("deepseek/deepseek-chat"),
    "deepseek-v3": openrouter_model("deepseek/deepseek-chat"),
    "deepseek-r1": openrouter_model("deepseek/deepseek-r1"),
}

register_model_alias_map("deepseek", models)
register_litellm_model_info(
    {
        "deepseek-v4-pro": {
            "litellm_provider": "deepseek",
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "max_tokens": 393_216,
            "input_cost_per_token": 0.435 / 1_000_000,
            "cache_read_input_token_cost": 0.003625 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
        "deepseek/deepseek-v4-pro": {
            "litellm_provider": "deepseek",
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "max_tokens": 393_216,
            "input_cost_per_token": 0.435 / 1_000_000,
            "cache_read_input_token_cost": 0.003625 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
    },
)
register_openrouter_model_info(
    {
        "deepseek/deepseek-v4-pro": {
            "max_input_tokens": 1_048_576,
            "max_tokens": 393_216,
            "input_cost_per_token": 0.435 / 1_000_000,
            "cache_read_input_token_cost": 0.003625 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
    },
)
