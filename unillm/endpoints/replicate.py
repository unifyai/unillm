from .utils import openrouter_model, register_model_alias_map

models = {
    "llama-4-maverick-instruct": openrouter_model("meta-llama/llama-4-maverick"),
    "deepseek-v3.1": openrouter_model("deepseek/deepseek-chat-v3.1"),
    "o4-mini": openrouter_model("openai/o4-mini"),
    "gpt-4.1": openrouter_model("openai/gpt-4.1"),
    "gpt-4.1-mini": openrouter_model("openai/gpt-4.1-mini"),
    "gpt-4.1-nano": openrouter_model("openai/gpt-4.1-nano"),
    "gpt-5": openrouter_model("openai/gpt-5"),
    "gpt-5-mini": openrouter_model("openai/gpt-5-mini"),
    "gpt-5-nano": openrouter_model("openai/gpt-5-nano"),
    "gpt-oss-120b": openrouter_model("openai/gpt-oss-120b"),
    "gpt-5.1": openrouter_model("openai/gpt-5.1"),
    "claude-4.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "claude-4.5-sonnet": openrouter_model("anthropic/claude-sonnet-4.5"),
    "claude-4-sonnet": openrouter_model("anthropic/claude-sonnet-4"),
    "claude-3.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "gemini-3-pro": openrouter_model("google/gemini-3.1-pro-preview"),
}

register_model_alias_map("replicate", models)
