from .utils import register_model_alias_map, register_model_info

models = {
    "deepseek-v4-max": "deepseek/deepseek-v4-pro",
    "deepseek-v4": "deepseek/deepseek-chat",
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
}

model_info = {
    "deepseek-v4-max": {"max_input_tokens": 128_000},
}

register_model_alias_map("deepseek", models)
register_model_info("deepseek", model_info)
