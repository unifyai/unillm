"""Normalize ``json_tool_call`` wrapper tool calls to OpenAI-standard shape."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from .response_format import ResponseFormatSpec, validate_against_spec

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

JSON_TOOL_CALL_NAME = "json_tool_call"


def _parse_json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    if isinstance(raw, dict):
        return raw
    return {}


def _tool_call_to_dict(call: Any) -> dict[str, Any]:
    if isinstance(call, dict):
        return call
    return call.model_dump(warnings=False)


def _serialize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _make_tool_call_dict(
    *,
    name: str,
    arguments: Any,
    call_id: str,
    index: int,
) -> dict[str, Any]:
    return ChatCompletionMessageToolCall(
        id=call_id or f"call_unwrapped_{index}",
        type="function",
        function=Function(name=name, arguments=_serialize_arguments(arguments)),
    ).model_dump(warnings=False)


def _unwrap_json_tool_call_args(
    args: dict[str, Any],
    *,
    wrapper_id: str,
    start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """Expand ``json_tool_call`` arguments into OpenAI tool call dicts."""
    if "tool_calls" in args:
        inner = args.get("tool_calls")
        if isinstance(inner, str):
            inner = json.loads(inner) if inner else []
        if not isinstance(inner, list):
            raise ValueError(
                f"json_tool_call.tool_calls must be a list, got {type(inner)}",
            )

        expanded: list[dict[str, Any]] = []
        index = start_index
        for item in inner:
            if not isinstance(item, dict):
                continue
            if "function" in item:
                expanded.append(item)
                continue
            name = item.get("name") or item.get("tool_name")
            item_args = item.get("arguments", item.get("tool_args", {}))
            if name:
                expanded.append(
                    _make_tool_call_dict(
                        name=name,
                        arguments=item_args,
                        call_id=wrapper_id,
                        index=index,
                    ),
                )
                index += 1
        return expanded, index

    name = args.get("name") or args.get("tool_name")
    inner_args = args.get("arguments", args.get("tool_args", {}))
    if name:
        return [
            _make_tool_call_dict(
                name=name,
                arguments=inner_args,
                call_id=wrapper_id,
                index=start_index,
            ),
        ], start_index + 1

    return [], start_index


def normalize_json_tool_call_wrappers(
    chat_completion: "ChatCompletion",
    *,
    response_format_spec: Optional[ResponseFormatSpec],
    tools: Optional[List[dict]],
) -> "ChatCompletion":
    """Promote structured output to content and unwrap inner tool calls."""
    _ = tools
    msg = chat_completion.choices[0].message
    tool_calls = msg.tool_calls
    if not tool_calls:
        return chat_completion

    normalized_calls: list[dict[str, Any]] = []
    had_wrapper = False
    next_index = 0

    for call in tool_calls:
        call_dict = _tool_call_to_dict(call)
        fn_info = call_dict.get("function") or {}
        fn_name = fn_info.get("name") if isinstance(fn_info, dict) else None

        if fn_name != JSON_TOOL_CALL_NAME:
            normalized_calls.append(call_dict)
            continue

        had_wrapper = True
        args = _parse_json_args(fn_info.get("arguments", "{}"))

        if response_format_spec is not None and isinstance(args, dict):
            if validate_against_spec(args, response_format_spec) is None:
                content = msg.content
                if content is None or not str(content).strip():
                    msg.content = json.dumps(args, ensure_ascii=False)

        if isinstance(args, dict):
            unwrapped, next_index = _unwrap_json_tool_call_args(
                args,
                wrapper_id=str(call_dict.get("id", "")),
                start_index=next_index,
            )
            normalized_calls.extend(unwrapped)

    if not had_wrapper:
        return chat_completion

    msg.tool_calls = normalized_calls or None
    choice = chat_completion.choices[0]
    choice.finish_reason = "tool_calls" if normalized_calls else "stop"

    logger.info(
        "Normalized json_tool_call wrappers to %s tool call(s)",
        len(normalized_calls),
    )
    return chat_completion
