from .utils import (
    openrouter_model,
    register_litellm_model_info,
    register_model_alias_map,
    register_openrouter_model_info,
)

provider = "qwen"

models = {
    "qwen3.7-plus": openrouter_model("qwen/qwen3.7-plus"),
}

_QWEN_37_PLUS_INFO = {
    "max_input_tokens": 1_000_000,
    "input_cost_per_token": 0.32 / 1_000_000,
    "cache_read_input_token_cost": 0.064 / 1_000_000,
    "output_cost_per_token": 1.28 / 1_000_000,
}

register_model_alias_map(provider, models)
register_litellm_model_info(
    {
        models["qwen3.7-plus"]: {
            "litellm_provider": "openrouter",
            "mode": "chat",
            **_QWEN_37_PLUS_INFO,
        },
        "qwen/qwen3.7-plus": {
            "litellm_provider": "openrouter",
            "mode": "chat",
            **_QWEN_37_PLUS_INFO,
        },
    },
)
register_openrouter_model_info(
    {
        "qwen/qwen3.7-plus": _QWEN_37_PLUS_INFO,
    },
)
