from typing import Any, Dict

import litellm

_MODEL_ALIAS_MAP: Dict[str, str] = {}
_MODEL_INFO_MAP: Dict[str, Dict[str, Any]] = {}


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
    alias = _MODEL_ALIAS_MAP.get(endpoint)
    if alias is None:
        raise ValueError(f"Model {endpoint} not found")
    return alias


def list_models(provider: str) -> list[str]:
    suffix = f"@{provider}"
    return sorted(
        endpoint[: -len(suffix)]
        for endpoint in _MODEL_ALIAS_MAP
        if endpoint.endswith(suffix)
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
