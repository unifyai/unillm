from .utils import register_litellm_model_info, register_model_alias_map

models = {
    "deepseek-v4-max": "deepseek/deepseek-v4-pro",
    "deepseek-v4": "deepseek/deepseek-chat",
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
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
