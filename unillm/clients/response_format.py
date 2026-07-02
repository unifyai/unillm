"""Internal response_format normalization and provider transport."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type, Union

from pydantic import BaseModel

try:
    import jsonschema
except (
    ImportError
):  # pragma: no cover - optional at import, required for dict validation
    jsonschema = None  # type: ignore[assignment]

RESPONSE_FORMAT_SPEC_KEY = "_unillm_response_format_spec"

ResponseFormatMode = Literal["native", "hybrid_prompt"]
OpenAIResponseFormatType = Literal["json_schema", "json_object"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ResponseFormatSpec:
    """Canonical internal representation of a caller's response_format."""

    source: Union[Type[BaseModel], dict]
    pydantic_model: Optional[Type[BaseModel]]
    json_schema: dict
    openai_type: OpenAIResponseFormatType


def canonicalize_response_format(
    response_format: Any,
) -> Optional[ResponseFormatSpec]:
    """Normalize a Pydantic class or OpenAI response_format dict into a spec."""
    if response_format is None:
        return None

    if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
        return ResponseFormatSpec(
            source=response_format,
            pydantic_model=response_format,
            json_schema=response_format.model_json_schema(),
            openai_type="json_schema",
        )

    if isinstance(response_format, dict):
        if "__pydantic_schema__" in response_format:
            schema = response_format["__pydantic_schema__"]
            return ResponseFormatSpec(
                source=response_format,
                pydantic_model=None,
                json_schema=schema,
                openai_type="json_schema",
            )

        rf_type = response_format.get("type")
        if rf_type == "json_object":
            return ResponseFormatSpec(
                source=response_format,
                pydantic_model=None,
                json_schema={"type": "object"},
                openai_type="json_object",
            )
        if rf_type == "json_schema":
            inner = response_format.get("json_schema") or {}
            schema = inner.get("schema") if isinstance(inner, dict) else None
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            return ResponseFormatSpec(
                source=response_format,
                pydantic_model=None,
                json_schema=schema,
                openai_type="json_schema",
            )

        return ResponseFormatSpec(
            source=response_format,
            pydantic_model=None,
            json_schema=response_format,
            openai_type="json_schema",
        )

    return None


def ensure_response_format_spec(kw: dict) -> None:
    """Build and stash a ResponseFormatSpec from kw['response_format'] when present."""
    if kw.get(RESPONSE_FORMAT_SPEC_KEY) is not None:
        return

    response_format = kw.get("response_format")
    if response_format is None:
        response_format = kw.get("_unillm_response_format")

    spec = canonicalize_response_format(response_format)
    if spec is not None:
        kw[RESPONSE_FORMAT_SPEC_KEY] = spec


def get_response_format_spec(kw: dict) -> Optional[ResponseFormatSpec]:
    """Return the stashed spec, falling back to canonicalizing legacy kw keys."""
    spec = kw.get(RESPONSE_FORMAT_SPEC_KEY)
    if isinstance(spec, ResponseFormatSpec):
        return spec

    response_format = kw.get("response_format") or kw.get("_unillm_response_format")
    return canonicalize_response_format(response_format)


def provider_response_format_mode(
    provider: Optional[str],
) -> ResponseFormatMode:
    if provider == "deepseek":
        return "hybrid_prompt"
    return "native"


def build_schema_instruction(json_schema: dict) -> str:
    return (
        "Respond with valid JSON only, with no markdown or commentary. "
        "The JSON must conform to this schema:\n"
        f"{json.dumps(json_schema, indent=2)}"
    )


def apply_response_format_transport(
    spec: ResponseFormatSpec,
    provider: Optional[str],
    kw: dict,
) -> None:
    """Mutate kw for LiteLLM transport based on provider capabilities."""
    if provider_response_format_mode(provider) != "hybrid_prompt":
        return

    kw.pop("response_format", None)
    kw["response_format"] = {"type": "json_object"}
    instruction = build_schema_instruction(spec.json_schema)
    messages = list(kw.get("messages", []))
    messages = [
        message
        for message in messages
        if not (
            message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("Respond with valid JSON only")
        )
    ]
    messages.insert(0, {"role": "system", "content": instruction})
    kw["messages"] = messages
    kw[RESPONSE_FORMAT_SPEC_KEY] = spec


def parse_structured_content(content: str) -> tuple[Any, Optional[str]]:
    """Parse JSON from model text, tolerating fences and surrounding prose."""
    text = content.strip()
    if not text:
        return None, "Response is empty"

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError):
        pass

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        return json.loads(candidate), None
                    except (json.JSONDecodeError, TypeError) as exc:
                        return None, f"Response is not valid JSON: {exc}"

    return None, "Response is not valid JSON: no JSON object or array found"


def validate_against_spec(parsed: Any, spec: ResponseFormatSpec) -> Optional[str]:
    """Validate parsed JSON against the spec. Returns an error message or None."""
    if spec.pydantic_model is not None:
        try:
            spec.pydantic_model.model_validate(parsed)
        except Exception as exc:
            return str(exc)
        return None

    if jsonschema is None:
        return None

    try:
        jsonschema.validate(instance=parsed, schema=spec.json_schema)
    except Exception as exc:
        return str(exc)
    return None


def serialize_response_format_spec(spec: ResponseFormatSpec) -> dict:
    """JSON-serializable representation for logs and cache keys."""
    if spec.pydantic_model is not None:
        source = {
            "__pydantic_schema__": spec.pydantic_model.model_json_schema(),
            "__pydantic_name__": spec.pydantic_model.__name__,
        }
    else:
        source = spec.source if isinstance(spec.source, dict) else str(spec.source)
    return {
        "source": source,
        "json_schema": spec.json_schema,
        "openai_type": spec.openai_type,
    }
