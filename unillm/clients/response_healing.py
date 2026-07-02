"""Recover native tool_calls from JSON content for soft-forced providers."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from .provider_preprocessing import SOFT_FORCED_TOOL_CHOICE_PROVIDERS
from .response_format import (
    ResponseFormatSpec,
    parse_structured_content,
    validate_against_spec,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage

logger = logging.getLogger(__name__)


def _forced_tool_name(tool_choice: Any) -> Optional[str]:
    if not isinstance(tool_choice, dict):
        return None
    function_choice = tool_choice.get("function")
    if not isinstance(function_choice, dict):
        return None
    name = function_choice.get("name")
    return name if isinstance(name, str) and name else None


def _tool_choice_is_forced(original_tool_choice: Any) -> bool:
    if original_tool_choice == "required":
        return True
    return _forced_tool_name(original_tool_choice) is not None


def _valid_tool_names(tools: Optional[List[dict]]) -> set[str]:
    if not tools:
        return set()
    names: set[str] = set()
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            name = tool["function"].get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def should_attempt_tool_call_healing(
    provider: str,
    original_tool_choice: Any,
    msg: "ChatCompletionMessage",
) -> bool:
    if provider not in SOFT_FORCED_TOOL_CHOICE_PROVIDERS:
        return False
    if not _tool_choice_is_forced(original_tool_choice):
        return False
    return not msg.tool_calls


def _serialize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _embedded_call_name(raw_call: dict) -> Optional[str]:
    if "function" in raw_call and isinstance(raw_call["function"], dict):
        name = raw_call["function"].get("name")
        return name if isinstance(name, str) and name else None

    for key in ("name", "tool_name", "tool"):
        name = raw_call.get(key)
        if isinstance(name, str) and name:
            return name
    return None


def _tool_parameter_properties(
    tool_name: str,
    tools: Optional[List[dict]],
) -> set[str]:
    if not tools:
        return set()
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        if function.get("name") != tool_name:
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        if isinstance(properties, dict):
            return set(properties)
    return set()


def _unwrap_embedded_call_shape(raw_call: dict) -> dict:
    invoke = raw_call.get("invoke")
    if isinstance(invoke, dict):
        name = invoke.get("name")
        if isinstance(name, str) and name:
            for args_key in ("args", "arguments", "input", "parameters"):
                if args_key in invoke:
                    return {"name": name, "arguments": invoke[args_key]}
            return {"name": name, "arguments": {}}
    return raw_call


def _embedded_call_arguments(
    raw_call: dict,
    *,
    tool_name: str,
    tools: Optional[List[dict]],
) -> Any:
    raw_call = _unwrap_embedded_call_shape(raw_call)

    if "function" in raw_call and isinstance(raw_call["function"], dict):
        return raw_call["function"].get("arguments", {})

    for key in ("arguments", "tool_args", "input", "parameters", "args"):
        if key in raw_call:
            return raw_call[key]

    query = raw_call.get("query")
    if isinstance(query, dict):
        return query
    if isinstance(query, str):
        props = _tool_parameter_properties(tool_name, tools)
        if "text" in props:
            return {"text": query}
        if "query" in props:
            return {"query": query}
    return {}


def _normalize_embedded_call(
    raw_call: Any,
    *,
    tools: Optional[List[dict]],
) -> Optional[tuple[str, str]]:
    if not isinstance(raw_call, dict):
        return None

    raw_call = _unwrap_embedded_call_shape(raw_call)
    name = _embedded_call_name(raw_call)
    if name is None:
        return None
    arguments = _embedded_call_arguments(raw_call, tool_name=name, tools=tools)
    return name, _serialize_arguments(arguments)


def _promoted_tool_call_dicts(
    embedded_calls: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    return [
        ChatCompletionMessageToolCall(
            id=f"call_healed_{index}",
            type="function",
            function=Function(name=name, arguments=arguments),
        ).model_dump(warnings=False)
        for index, (name, arguments) in enumerate(embedded_calls)
    ]


def _extract_embedded_calls(
    parsed: dict,
    *,
    tools: Optional[List[dict]],
) -> tuple[list[tuple[str, str]], set[str]]:
    promoted_keys: set[str] = set()

    if "tool_calls" in parsed:
        inner = parsed["tool_calls"]
        if isinstance(inner, str):
            inner = json.loads(inner) if inner else []
        if not isinstance(inner, list) or not inner:
            return [], set()
        calls = [_normalize_embedded_call(item, tools=tools) for item in inner]
        if any(call is None for call in calls):
            return [], set()
        promoted_keys.add("tool_calls")
        return calls, promoted_keys  # type: ignore[return-value]

    if _embedded_call_name(parsed) is not None and any(
        key in parsed
        for key in (
            "arguments",
            "tool_args",
            "input",
            "parameters",
            "args",
            "query",
            "function",
            "invoke",
        )
    ):
        call = _normalize_embedded_call(parsed, tools=tools)
        if call is None:
            return [], set()
        promoted_keys.update(
            key
            for key in (
                "name",
                "tool_name",
                "tool",
                "arguments",
                "tool_args",
                "input",
                "parameters",
                "args",
                "query",
                "function",
                "invoke",
                "type",
                "id",
                "response_format",
            )
            if key in parsed
        )
        return [call], promoted_keys

    return [], set()


def _clean_content_payload(
    parsed: dict,
    promoted_keys: set[str],
    response_format_spec: Optional[ResponseFormatSpec],
) -> Optional[str]:
    remaining = {
        key: value for key, value in parsed.items() if key not in promoted_keys
    }
    if not remaining:
        return None
    return json.dumps(remaining, ensure_ascii=False)


def _content_passes_response_format(
    content: Optional[str],
    response_format_spec: Optional[ResponseFormatSpec],
) -> bool:
    if response_format_spec is None:
        return True
    if content is None:
        return False

    parsed, parse_error = parse_structured_content(content)
    if parse_error is not None:
        return False
    return validate_against_spec(parsed, response_format_spec) is None


def try_heal_embedded_tool_calls(
    response: "ChatCompletion",
    *,
    provider: str,
    original_tool_choice: Any,
    tools: Optional[List[dict]],
    response_format_spec: Optional[ResponseFormatSpec],
) -> Optional["ChatCompletion"]:
    msg = response.choices[0].message
    if not should_attempt_tool_call_healing(provider, original_tool_choice, msg):
        return None

    content = msg.content
    if not isinstance(content, str) or not content.strip():
        return None

    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    embedded_calls, promoted_keys = _extract_embedded_calls(
        parsed,
        tools=tools,
    )
    if not embedded_calls:
        return None

    allowed_names = _valid_tool_names(tools)
    if not allowed_names:
        return None

    for name, _arguments in embedded_calls:
        if name not in allowed_names:
            return None

    cleaned_content = _clean_content_payload(
        parsed,
        promoted_keys,
        response_format_spec,
    )
    if not _content_passes_response_format(cleaned_content, response_format_spec):
        return None

    promoted_tool_calls = _promoted_tool_call_dicts(embedded_calls)

    msg.content = cleaned_content
    msg.tool_calls = promoted_tool_calls
    response.choices[0].finish_reason = "tool_calls"

    logger.info(
        "Healed embedded tool_calls from JSON content for provider=%s tools=%s",
        provider,
        [
            call["function"]["name"]
            for call in promoted_tool_calls
            if isinstance(call.get("function"), dict)
        ],
    )
    return response


def maybe_heal_tool_calls_in_completion(
    chat_completion: "ChatCompletion",
    *,
    provider: str,
    original_tool_choice: Any,
    tools: Optional[List[dict]],
    response_format_spec: Optional[ResponseFormatSpec],
) -> "ChatCompletion":
    healed = try_heal_embedded_tool_calls(
        chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        tools=tools,
        response_format_spec=response_format_spec,
    )
    return healed if healed is not None else chat_completion
