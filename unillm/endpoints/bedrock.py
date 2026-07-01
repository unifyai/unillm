from .utils import openrouter_model, register_model_alias_map

models = {
    "gpt-oss-20b": openrouter_model("openai/gpt-oss-20b"),
    "gpt-oss-120b": openrouter_model("openai/gpt-oss-120b"),
    "deepseek-r1": openrouter_model("deepseek/deepseek-r1"),
    "llama-3.3-70b-chat": openrouter_model("meta-llama/llama-3.3-70b-instruct"),
    "llama-3.2-1b-chat": openrouter_model("meta-llama/llama-3.2-1b-instruct"),
    "llama-3.2-3b-chat": openrouter_model("meta-llama/llama-3.2-3b-instruct"),
    "llama-3.1-8b-chat": openrouter_model("meta-llama/llama-3.1-8b-instruct"),
    "llama-3.1-70b-chat": openrouter_model("meta-llama/llama-3.1-70b-instruct"),
    "llama-3.1-405b-chat": openrouter_model("nousresearch/hermes-3-llama-3.1-405b"),
    "llama-3-8b-chat": openrouter_model("meta-llama/llama-3-8b-instruct"),
    "llama-3-70b-chat": openrouter_model("meta-llama/llama-3.1-70b-instruct"),
    "claude-3-haiku": openrouter_model("anthropic/claude-3-haiku"),
    "claude-3.5-haiku": openrouter_model("anthropic/claude-haiku-4.5"),
    "claude-4-sonnet": openrouter_model("anthropic/claude-sonnet-4"),
    "claude-4-opus": openrouter_model("anthropic/claude-opus-4"),
    "claude-4.1-opus": openrouter_model("anthropic/claude-opus-4.1"),
    "claude-4.5-sonnet": openrouter_model("anthropic/claude-sonnet-4.5"),
    "claude-4.5-opus": openrouter_model("anthropic/claude-opus-4.5"),
}

register_model_alias_map("bedrock", models)
