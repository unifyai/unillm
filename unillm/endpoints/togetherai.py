from .utils import openrouter_model, register_model_alias_map

models = {
    "gpt-oss-20b": openrouter_model("openai/gpt-oss-20b"),
    "gpt-oss-120b": openrouter_model("openai/gpt-oss-120b"),
    "deepseek-v3.1": openrouter_model("deepseek/deepseek-chat-v3.1"),
    "deepseek-r1": openrouter_model("deepseek/deepseek-r1"),
    "deepseek-v3": openrouter_model("deepseek/deepseek-chat"),
    "llama-4-maverick-instruct": openrouter_model("meta-llama/llama-4-maverick"),
    "llama-3.3-70b-chat": openrouter_model("meta-llama/llama-3.3-70b-instruct"),
    "llama-3.2-3b-chat": openrouter_model("meta-llama/llama-3.2-3b-instruct"),
    "llama-3.1-70b-chat": openrouter_model("meta-llama/llama-3.1-70b-instruct"),
    "llama-3.1-405b-chat": openrouter_model("nousresearch/hermes-3-llama-3.1-405b"),
    "mistral-small": openrouter_model("mistralai/mistral-small-24b-instruct-2501"),
    "qwen-3-235b-a22b-instruct": openrouter_model("qwen/qwen3-235b-a22b-2507"),
    "qwen-2.5-7b-instruct": openrouter_model("qwen/qwen-2.5-7b-instruct"),
    "qwen-2.5-72b-instruct": openrouter_model("qwen/qwen-2.5-72b-instruct"),
}
register_model_alias_map("togetherai", models)
