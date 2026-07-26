from .utils import (
    openrouter_model,
    register_model_alias_map,
    register_openrouter_model_info,
)

provider = "moonshotai"

models = {
    "kimi-k3": openrouter_model("moonshotai/kimi-k3"),
}

register_model_alias_map(provider, models)
register_openrouter_model_info(
    {
        "moonshotai/kimi-k3": {
            "max_input_tokens": 1_048_576,
            "input_cost_per_token": 3.00 / 1_000_000,
            "cache_read_input_token_cost": 0.30 / 1_000_000,
            "output_cost_per_token": 15.00 / 1_000_000,
            "supports_reasoning": True,
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
        },
    },
)
