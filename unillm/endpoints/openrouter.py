"""First-class ``@openrouter`` provider.

Static registrations are optional; any ``<openrouter-id>@openrouter`` endpoint is
accepted dynamically via :mod:`unillm.endpoints.utils`. This module ensures
``openrouter`` appears in :func:`list_providers` even before a catalog fetch.
"""

from .utils import openrouter_model, register_model_alias_map

# Seed a few commonly used OpenRouter ids so list_providers() includes
# ``openrouter`` without requiring a network catalog fetch. Dynamic resolution
# still accepts the rest of the OpenRouter catalog.
models = {
    "openai/gpt-5.6-sol": openrouter_model("openai/gpt-5.6-sol"),
    "openai/gpt-5.6-terra": openrouter_model("openai/gpt-5.6-terra"),
    "openai/gpt-5.6-luna": openrouter_model("openai/gpt-5.6-luna"),
    "openai/gpt-5.5": openrouter_model("openai/gpt-5.5"),
    "openai/gpt-5.4": openrouter_model("openai/gpt-5.4"),
    "openai/gpt-5.4-mini": openrouter_model("openai/gpt-5.4-mini"),
    "openai/gpt-4o": openrouter_model("openai/gpt-4o"),
    "openai/gpt-4o-mini": openrouter_model("openai/gpt-4o-mini"),
    "anthropic/claude-sonnet-4.6": openrouter_model("anthropic/claude-sonnet-4.6"),
    "anthropic/claude-opus-4.6": openrouter_model("anthropic/claude-opus-4.6"),
}

register_model_alias_map("openrouter", models)
