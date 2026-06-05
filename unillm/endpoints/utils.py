from importlib import import_module
from typing import Any, Dict

import litellm

_MODEL_ALIAS_MAP: Dict[str, str] = {}
_MODEL_INFO_MAP: Dict[str, Dict[str, Any]] = {}
_ENDPOINTS_IMPORTED = False


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
    _MODEL_ALIAS_MAP.update(
        {f"{model}@{provider}": alias for model, alias in model_map.items()},
    )


def register_model_info(
    provider: str,
    model_info: Dict[str, Dict[str, Any]],
) -> None:
    _MODEL_INFO_MAP.update(
        {f"{model}@{provider}": info for model, info in model_info.items()},
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

    Args:
        endpoint: Model identifier with ``@provider`` suffix.

    Returns:
        Merged model-info dict.

    Raises:
        ValueError: If the model is not found in either source.
    """
    model = get_model_alias(endpoint)
    try:
        info = dict(litellm.get_model_info(model))
    except Exception:
        info = {}
    overrides = _MODEL_INFO_MAP.get(endpoint, {})
    info.update(overrides)
    if not info:
        raise ValueError(f"Could not find model info for '{endpoint}'")
    return info
