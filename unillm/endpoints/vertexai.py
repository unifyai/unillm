from .utils import openrouter_model, register_model_alias_map

models = {
    "gemini-3-pro": openrouter_model("google/gemini-3.1-pro-preview"),
    "gemini-3-flash": openrouter_model("google/gemini-3-flash-preview"),
    "gemini-2.5-flash-lite": openrouter_model("google/gemini-2.5-flash-lite"),
    "gemini-2.5-flash": openrouter_model("google/gemini-2.5-flash"),
    "gemini-2.5-pro": openrouter_model("google/gemini-2.5-pro"),
    "gemini-2.0-flash-lite": openrouter_model("google/gemini-2.5-flash-lite"),
    "gemini-2.0-flash": openrouter_model("google/gemini-2.5-flash"),
    "claude-3-haiku": openrouter_model("anthropic/claude-3-haiku"),
    "claude-3.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "claude-4-sonnet": openrouter_model("anthropic/claude-sonnet-4"),
    "claude-4-opus": openrouter_model("anthropic/claude-opus-4"),
    "claude-4.1-opus": openrouter_model("anthropic/claude-opus-4.1"),
    "claude-4.5-sonnet": openrouter_model("anthropic/claude-sonnet-4.5"),
    "claude-4.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "claude-4.5-opus": openrouter_model("anthropic/claude-opus-4.5"),
    "llama-3.3-70b-chat": openrouter_model("meta-llama/llama-3.3-70b-instruct"),
    "llama-4-maverick-instruct": openrouter_model("meta-llama/llama-4-maverick"),
    "mistral-medium": openrouter_model("mistralai/mistral-medium-3"),
    "qwen-3-235b-a22b-instruct": openrouter_model("qwen/qwen3-235b-a22b-2507"),
    "deepseek-v3.1": openrouter_model("deepseek/deepseek-chat-v3.1"),
    "deepseek-r1": openrouter_model("deepseek/deepseek-r1-0528"),
    "gpt-oss-120b": openrouter_model("openai/gpt-oss-120b"),
}

register_model_alias_map("vertex-ai", models)
