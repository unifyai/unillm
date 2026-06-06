from .utils import register_model_alias_map, register_model_info

provider = "minimax"

models = {
    "minimax-v3": "minimax/MiniMax-M3",
}

model_info = {
    "minimax-v3": {"max_input_tokens": 1_000_000},
}

register_model_alias_map(provider, models)
register_model_info(provider, model_info)
