from .utils import openrouter_model, register_model_alias_map

models = {
    "mistral-large": openrouter_model("mistralai/mistral-large"),
    "mistral-medium": openrouter_model("mistralai/mistral-medium-3.1"),
}

register_model_alias_map("mistral", models)
