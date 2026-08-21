"""Normalize ``json_tool_call`` wrapper tool calls to OpenAI-standard shape.

Also promotes a lone schema-shaped tool call to structured content: some
endpoints answer a ``json_schema`` response_format with the payload inside a
tool call of their own naming (``content`` null) instead of the
``json_tool_call`` wrapper. When the request declared no tools, such a call
cannot be a real invocation — its arguments are the structured output.
"""

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
        return _promote_schema_shaped_tool_call(
            chat_completion,
            response_format_spec=response_format_spec,
            tools=tools,
        )

    msg.tool_calls = normalized_calls or None
    choice = chat_completion.choices[0]
    choice.finish_reason = "tool_calls" if normalized_calls else "stop"

    logger.info(
        "Normalized json_tool_call wrappers to %s tool call(s)",
        len(normalized_calls),
    )
    return chat_completion


def _names_a_declared_tool(name: Any, tools: Optional[List[dict]]) -> bool:
    """Whether *name* is one of the tools the request actually offered.

    A call naming one is a real invocation whatever else it looks like. A call
    naming anything else had nothing to invoke, which is what makes it
    readable as misplaced structured output.
    """

    if not tools or not isinstance(name, str) or not name:
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        declared = fn.get("name") if isinstance(fn, dict) else tool.get("name")
        if declared == name:
            return True
    return False


def _promote_schema_shaped_tool_call(
    chat_completion: "ChatCompletion",
    *,
    response_format_spec: Optional[ResponseFormatSpec],
    tools: Optional[List[dict]],
) -> "ChatCompletion":
    """Promote a lone tool call that carries the requested structured output.

    Applies when the caller asked for a ``json_schema`` response and the
    answer is a single call the caller never offered: that call cannot be a
    real invocation, so arguments validating against the requested schema are
    the structured output wearing a tool call's shape.

    The test is per-call, not per-request. Declaring tools does not stop a
    provider putting structured output in a call of its own naming, and a
    caller that both offers tools and asks for a schema had no way to read
    that answer -- the payload sits in ``tool_calls`` while it reads
    ``content``. Every agentic loop that wants typed output is in exactly
    that position, so scoping this to tool-less requests left the case
    unreachable for them.

    A call naming a declared tool is still left alone: that is a real
    invocation, and promoting it would swallow the request. Several calls,
    non-empty content, or arguments that fail the schema also pass through.
    """
    if response_format_spec is None:
        return chat_completion
    msg = chat_completion.choices[0].message
    if msg.content is not None and str(msg.content).strip():
        return chat_completion
    calls = msg.tool_calls or []
    if len(calls) != 1:
        return chat_completion
    fn_info = _tool_call_to_dict(calls[0]).get("function") or {}
    if not isinstance(fn_info, dict):
        return chat_completion
    if _names_a_declared_tool(fn_info.get("name"), tools):
        return chat_completion
    try:
        args = _parse_json_args(fn_info.get("arguments", "{}"))
    except json.JSONDecodeError:
        return chat_completion
    if not isinstance(args, dict):
        return chat_completion
    if validate_against_spec(args, response_format_spec) is not None:
        return chat_completion
    msg.content = json.dumps(args, ensure_ascii=False)
    msg.tool_calls = None
    chat_completion.choices[0].finish_reason = "stop"
    logger.info(
        "Promoted schema-shaped tool call %r to structured content",
        fn_info.get("name"),
    )
    return chat_completion
