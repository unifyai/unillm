from .utils import register_model_alias_map, register_model_info

provider = "anthropic"
models = {
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-3.5-haiku": "anthropic/claude-3-5-haiku-20241022",
    "claude-4-sonnet": "anthropic/claude-sonnet-4-20250514",
    "claude-4-opus": "anthropic/claude-opus-4-20250514",
    "claude-4.1-opus": "anthropic/claude-opus-4-1-20250805",
    "claude-4.5-sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "claude-4.5-haiku": "anthropic/claude-haiku-4-5-20251001",
    "claude-4.5-opus": "anthropic/claude-opus-4-5-20251101",
    "claude-4.6-opus": "anthropic/claude-opus-4-6",
    "claude-4.6-sonnet": "anthropic/claude-sonnet-4-6",
    "claude-4.8-opus": "anthropic/claude-opus-4-8",
}

model_info = {
    "claude-4.6-opus": {"max_input_tokens": 1_000_000},
    "claude-4.6-sonnet": {"max_input_tokens": 1_000_000},
    "claude-4.8-opus": {"max_input_tokens": 1_000_000},
}

CONTEXT_1M_BETA = "context-1m-2025-08-07"
CONTEXT_1M_MODELS = {
    models["claude-4-sonnet"],
    models["claude-4.5-sonnet"],
    models["claude-4.6-opus"],
    models["claude-4.6-sonnet"],
    models["claude-4.8-opus"],
}

ADAPTIVE_THINKING_MODELS = {
    models["claude-4.8-opus"],
}

register_model_alias_map(provider, models)
register_model_info(provider, model_info)
