from .utils import openrouter_model, register_model_alias_map

models = {
    "llama-3.1-8b-chat": openrouter_model("meta-llama/llama-3.1-8b-instruct"),
    "llama-3.3-70b-chat": openrouter_model("meta-llama/llama-3.3-70b-instruct"),
    "llama-4-maverick-instruct": openrouter_model("meta-llama/llama-4-maverick"),
}

register_model_alias_map("groq", models)
