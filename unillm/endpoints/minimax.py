from .utils import register_litellm_model_info, register_model_alias_map

provider = "minimax"

models = {
    "minimax-v3": "minimax/MiniMax-M3",
}

register_model_alias_map(provider, models)
register_litellm_model_info(
    {
        models["minimax-v3"]: {
            "max_input_tokens": 1_000_000,
        },
    },
)
