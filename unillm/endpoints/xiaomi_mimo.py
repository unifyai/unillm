from .utils import register_model_alias_map, register_model_info

provider = "xiaomi-mimo"

models = {
    "mimo-v2.5-pro": "xiaomi_mimo/mimo-v2.5-pro",
}

model_info = {
    "mimo-v2.5-pro": {"max_input_tokens": 1_000_000},
}

register_model_alias_map(provider, models)
register_model_info(provider, model_info)
