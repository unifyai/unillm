from importlib import import_module
from typing import Any, Dict

import litellm

_MODEL_ALIAS_MAP: Dict[str, str] = {}
_MODEL_TRANSPORT_ALIAS_MAP: Dict[str, str] = {}
_MODEL_INFO_MAP: Dict[str, Dict[str, Any]] = {}
_ENDPOINTS_IMPORTED = False
_OPENROUTER_PREFIX = "openrouter/"

_ANTHROPIC_PUBLIC_ALIASES = {
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
    "claude-opus-5": "anthropic/claude-opus-5",
    "claude-fable-5": "anthropic/claude-fable-5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
}

_DEEPSEEK_PUBLIC_ALIASES = {
    "deepseek-v4-max": "deepseek/deepseek-v4-pro",
}


def ensure_endpoints_imported() -> None:
    """Import provider endpoint modules so the model registry is populated."""

    global _ENDPOINTS_IMPORTED
    if _ENDPOINTS_IMPORTED:
        return
    import_module("unillm.endpoints")
    _ENDPOINTS_IMPORTED = True


def register_model_alias_map(
    provider: str,
    model_map: Dict[str, Any],
) -> None:
    _MODEL_TRANSPORT_ALIAS_MAP.update(
        {f"{model}@{provider}": alias for model, alias in model_map.items()},
    )
    _MODEL_ALIAS_MAP.update(
        {
            f"{model}@{provider}": _public_model_alias(provider, model, alias)
            for model, alias in model_map.items()
        },
    )


def register_model_info(
    provider: str,
    model_info: Dict[str, Dict[str, Any]],
) -> None:
    _MODEL_INFO_MAP.update(
        {f"{model}@{provider}": info for model, info in model_info.items()},
    )


def register_litellm_model_info(model_info: Dict[str, Dict[str, Any]]) -> None:
    """Register missing or corrected model metadata in LiteLLM's registry."""

    litellm.register_model(model_info)


def openrouter_model(model_id: str) -> str:
    """Return the LiteLLM model id for an OpenRouter catalog model."""

    return (
        model_id
        if model_id.startswith(_OPENROUTER_PREFIX)
        else f"{_OPENROUTER_PREFIX}{model_id}"
    )


def _public_model_alias(provider: str, model: str, alias: str) -> str:
    if provider == "openai":
        return model
    if provider == "anthropic":
        return _ANTHROPIC_PUBLIC_ALIASES.get(model, alias)
    if provider == "deepseek":
        return _DEEPSEEK_PUBLIC_ALIASES.get(model, alias)
    if provider == "minimax" and model == "minimax-v3":
        return "minimax/MiniMax-M3"
    if provider == "moonshotai":
        return f"moonshotai/{model}"
    if provider == "xiaomi-mimo":
        return f"xiaomi_mimo/{model}"
    if provider == "zai":
        return f"z-ai/{model}"
    return alias


def register_openrouter_model_info(model_info: Dict[str, Dict[str, Any]]) -> None:
    """Register OpenRouter metadata not yet present in the pinned LiteLLM release."""

    register_litellm_model_info(
        {
            openrouter_model(model): {
                "litellm_provider": "openrouter",
                "mode": "chat",
                **info,
            }
            for model, info in model_info.items()
        },
    )


def get_model_alias(endpoint: str) -> str:
    """
    Get the alias for a model. If the model is not found, throws an exception.

    Args:
        endpoint: The endpoint of the model.
    Returns:
        LiteLLM model name for the model.
    """
    ensure_endpoints_imported()
    alias = _MODEL_ALIAS_MAP.get(endpoint)
    if alias is None:
        raise ValueError(f"Model {endpoint} not found")
    return alias


def get_transport_model_alias(endpoint: str) -> str:
    """
    Return the LiteLLM transport model for *endpoint*.

    Public aliases remain stable for accounting, logs, and cache keys; transport
    aliases select the configured backend for inference.
    """

    ensure_endpoints_imported()
    alias = _MODEL_TRANSPORT_ALIAS_MAP.get(endpoint)
    if alias is None:
        return get_model_alias(endpoint)
    return alias


def list_models(provider: str) -> list[str]:
    ensure_endpoints_imported()
    suffix = f"@{provider}"
    return sorted(
        endpoint[: -len(suffix)]
        for endpoint in _MODEL_ALIAS_MAP
        if endpoint.endswith(suffix)
    )


def list_providers() -> list[str]:
    """Return provider names with at least one registered model endpoint."""

    ensure_endpoints_imported()
    return sorted(
        {
            endpoint.rsplit("@", 1)[1]
            for endpoint in _MODEL_ALIAS_MAP
            if "@" in endpoint
        },
    )


def list_endpoints(provider: str | None = None) -> list[str]:
    """Return supported model endpoints in ``model@provider`` form.

    Args:
        provider: Optional provider filter, e.g. ``"openai"``.
    """

    ensure_endpoints_imported()
    if provider is None:
        return sorted(_MODEL_ALIAS_MAP)
    suffix = f"@{provider}"
    return sorted(
        endpoint for endpoint in _MODEL_ALIAS_MAP if endpoint.endswith(suffix)
    )


def get_model_info(endpoint: str) -> Dict[str, Any]:
    """Return model info for *endpoint*, with local overrides taking priority.

    Resolves the endpoint alias, fetches LiteLLM's model info as a base,
    then overlays any entries registered via :func:`register_model_info`.

    OpenAI-public aliases (e.g. ``gpt-5.6-sol``) may be unknown to the pinned
    LiteLLM release while still registered under the OpenRouter transport id
    (``openrouter/openai/gpt-5.6-sol``). Try the transport alias when the
    public name has no LiteLLM metadata yet.

    Args:
        endpoint: Model identifier with ``@provider`` suffix.

    Returns:
        Merged model-info dict.

    Raises:
        ValueError: If the model is not found in either source.
    """
    model = get_model_alias(endpoint)
    transport = get_transport_model_alias(endpoint)
    info: Dict[str, Any] = {}
    for candidate in dict.fromkeys((model, transport)):
        try:
            candidate_info = dict(litellm.get_model_info(candidate))
        except Exception:
            continue
        if candidate_info:
            info = candidate_info
            break
    overrides = _MODEL_INFO_MAP.get(endpoint, {})
    info.update(overrides)
    if not info:
        raise ValueError(f"Could not find model info for '{endpoint}'")
    return info
