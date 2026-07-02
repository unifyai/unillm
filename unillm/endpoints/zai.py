from .utils import (
    openrouter_model,
    register_litellm_model_info,
    register_model_alias_map,
    register_openrouter_model_info,
)

provider = "zai"

models = {
    "glm-5.2": openrouter_model("z-ai/glm-5.2"),
}

register_model_alias_map(provider, models)
register_litellm_model_info(
    {
        "glm-5.2": {
            "litellm_provider": "openrouter",
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "input_cost_per_token": 0.93 / 1_000_000,
            "cache_read_input_token_cost": 0.18 / 1_000_000,
            "output_cost_per_token": 3.0 / 1_000_000,
        },
        models["glm-5.2"]: {
            "litellm_provider": "openrouter",
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "input_cost_per_token": 0.93 / 1_000_000,
            "cache_read_input_token_cost": 0.18 / 1_000_000,
            "output_cost_per_token": 3.0 / 1_000_000,
        },
        "z-ai/glm-5.2": {
            "litellm_provider": "openrouter",
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "input_cost_per_token": 0.93 / 1_000_000,
            "cache_read_input_token_cost": 0.18 / 1_000_000,
            "output_cost_per_token": 3.0 / 1_000_000,
        },
    },
)
register_openrouter_model_info(
    {
        "z-ai/glm-5.2": {
            "max_input_tokens": 1_048_576,
            "input_cost_per_token": 0.93 / 1_000_000,
            "cache_read_input_token_cost": 0.18 / 1_000_000,
            "output_cost_per_token": 3.0 / 1_000_000,
        },
    },
)
