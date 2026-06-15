from .utils import register_litellm_model_info, register_model_alias_map

provider = "xiaomi-mimo"

models = {
    "mimo-v2.5": "xiaomi_mimo/mimo-v2.5",
    "mimo-v2.5-pro": "xiaomi_mimo/mimo-v2.5-pro",
}

register_model_alias_map(provider, models)
register_litellm_model_info(
    {
        "mimo-v2.5": {
            "litellm_provider": "xiaomi_mimo",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 0.14 / 1_000_000,
            "cache_read_input_token_cost": 0.0028 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
        },
        models["mimo-v2.5"]: {
            "litellm_provider": "xiaomi_mimo",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 0.14 / 1_000_000,
            "cache_read_input_token_cost": 0.0028 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
        },
        "mimo-v2.5-pro": {
            "litellm_provider": "xiaomi_mimo",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 0.435 / 1_000_000,
            "cache_read_input_token_cost": 0.0036 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
        models["mimo-v2.5-pro"]: {
            "litellm_provider": "xiaomi_mimo",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 0.435 / 1_000_000,
            "cache_read_input_token_cost": 0.0036 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
    },
)
