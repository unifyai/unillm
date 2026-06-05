from .utils import register_model_alias_map

models = {
    "deepseek-v4-max": "deepseek/deepseek-v4-pro",
    "deepseek-v4": "deepseek/deepseek-chat",
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
}

register_model_alias_map("deepseek", models)
