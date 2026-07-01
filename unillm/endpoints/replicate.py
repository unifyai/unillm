from .utils import openrouter_model, register_model_alias_map

models = {
    "llama-4-maverick-instruct": openrouter_model("meta-llama/llama-4-maverick"),
    "llama-4-scout-instruct": openrouter_model("meta-llama/llama-4-scout"),
    "llama-3-8b-chat": openrouter_model("meta-llama/llama-3-8b-instruct"),
    "llama-3-70b-chat": openrouter_model("meta-llama/llama-3.1-70b-instruct"),
    "llama-3.1-405b-chat": openrouter_model("nousresearch/hermes-3-llama-3.1-405b"),
    "deepseek-v3.1": openrouter_model("deepseek/deepseek-chat-v3.1"),
    "deepseek-v3": openrouter_model("deepseek/deepseek-chat"),
    "deepseek-r1": openrouter_model("deepseek/deepseek-r1"),
    "o4-mini": openrouter_model("openai/o4-mini"),
    "gpt-4.1": openrouter_model("openai/gpt-4.1"),
    "gpt-4.1-mini": openrouter_model("openai/gpt-4.1-mini"),
    "gpt-4.1-nano": openrouter_model("openai/gpt-4.1-nano"),
    "gpt-5": openrouter_model("openai/gpt-5"),
    "gpt-5-mini": openrouter_model("openai/gpt-5-mini"),
    "gpt-5-nano": openrouter_model("openai/gpt-5-nano"),
    "gpt-oss-20b": openrouter_model("openai/gpt-oss-20b"),
    "gpt-oss-120b": openrouter_model("openai/gpt-oss-120b"),
    "gpt-5.1": openrouter_model("openai/gpt-5.1"),
    "claude-4.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "claude-4.5-sonnet": openrouter_model("anthropic/claude-sonnet-4.5"),
    "claude-4-sonnet": openrouter_model("anthropic/claude-sonnet-4"),
    "claude-3.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "gemini-3-pro": openrouter_model("google/gemini-3.1-pro-preview"),
}

register_model_alias_map("replicate", models)
