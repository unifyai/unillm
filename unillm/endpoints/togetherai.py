from .utils import openrouter_model, register_model_alias_map

models = {
    "gpt-oss-120b": openrouter_model("openai/gpt-oss-120b"),
    "deepseek-v3.1": openrouter_model("deepseek/deepseek-chat-v3.1"),
    "llama-4-maverick-instruct": openrouter_model("meta-llama/llama-4-maverick"),
    "llama-3.3-70b-chat": openrouter_model("meta-llama/llama-3.3-70b-instruct"),
    "qwen-3-235b-a22b-instruct": openrouter_model("qwen/qwen3-235b-a22b-2507"),
}
register_model_alias_map("togetherai", models)
