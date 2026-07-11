from .utils import openrouter_model, register_model_alias_map

models = {
    "gpt-3.5-turbo": openrouter_model("openai/gpt-3.5-turbo"),
    "gpt-4": openrouter_model("openai/gpt-4"),
    "gpt-4-turbo": openrouter_model("openai/gpt-4-turbo"),
    "gpt-4o": openrouter_model("openai/gpt-4o"),
    "gpt-4o-2024-05-13": openrouter_model("openai/gpt-4o-2024-05-13"),
    "gpt-4o-mini": openrouter_model("openai/gpt-4o-mini"),
    "chatgpt-4o-latest": openrouter_model("openai/gpt-4o"),
    "o1": openrouter_model("openai/o1"),
    "o3-mini": openrouter_model("openai/o3-mini"),
    "gpt-4o-search-preview": openrouter_model("openai/gpt-4o-search-preview"),
    "gpt-4o-mini-search-preview": openrouter_model(
        "openai/gpt-4o-mini-search-preview",
    ),
    "gpt-4.1": openrouter_model("openai/gpt-4.1"),
    "gpt-4.1-mini": openrouter_model("openai/gpt-4.1-mini"),
    "gpt-4.1-nano": openrouter_model("openai/gpt-4.1-nano"),
    "o3": openrouter_model("openai/o3"),
    "o4-mini": openrouter_model("openai/o4-mini"),
    "gpt-5": openrouter_model("openai/gpt-5"),
    "gpt-5-mini": openrouter_model("openai/gpt-5-mini"),
    "gpt-5-nano": openrouter_model("openai/gpt-5-nano"),
    "gpt-5-chat-latest": openrouter_model("openai/gpt-5-chat"),
    "gpt-5.1": openrouter_model("openai/gpt-5.1"),
    "gpt-5.1-chat-latest": openrouter_model("openai/gpt-5.1-chat"),
    "gpt-5.2": openrouter_model("openai/gpt-5.2"),
    "gpt-5.4": openrouter_model("openai/gpt-5.4"),
    "gpt-5.4-mini": openrouter_model("openai/gpt-5.4-mini"),
    "gpt-5.4-nano": openrouter_model("openai/gpt-5.4-nano"),
    "gpt-5.5": openrouter_model("openai/gpt-5.5"),
    "gpt-5.6-sol": openrouter_model("openai/gpt-5.6-sol"),
    "gpt-5.6-terra": openrouter_model("openai/gpt-5.6-terra"),
    "gpt-5.6-luna": openrouter_model("openai/gpt-5.6-luna"),
}

register_model_alias_map("openai", models)
