from .utils import (
    openrouter_model,
    register_model_alias_map,
    register_openrouter_model_info,
)

provider = "minimax"

models = {
    "minimax-v3": openrouter_model("minimax/minimax-m3"),
}

register_model_alias_map(provider, models)
register_openrouter_model_info(
    {
        "minimax/minimax-m3": {
            "max_input_tokens": 1_000_000,
            "input_cost_per_token": 0.30 / 1_000_000,
            "output_cost_per_token": 1.20 / 1_000_000,
        },
    },
)
