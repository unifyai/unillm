"""Recover native tool_calls from structured/XML/parenthesized model text."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from typing import Any, List, Optional, TYPE_CHECKING

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from .response_format import (
    ResponseFormatSpec,
    parse_structured_content,
    validate_against_spec,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage

logger = logging.getLogger(__name__)

# Provider JSON often uses these keys for the primary string payload field.
_ARGUMENT_ALIASES = (
    "text",
    "query",
    "input",
    "message",
    "content",
    "prompt",
    "request",
    "q",
)


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
    *,
    tools: Optional[List[dict]] = None,
) -> bool:
    del provider
    if msg.tool_calls:
        return False
    if _tool_choice_is_forced(original_tool_choice):
        return True
    if original_tool_choice in (None, "auto") and _valid_tool_names(tools):
        return True
    return False


def _serialize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _embedded_call_name(raw_call: dict) -> Optional[str]:
    if "function" in raw_call and isinstance(raw_call["function"], dict):
        name = raw_call["function"].get("name")
        return name if isinstance(name, str) and name else None

    for key in ("name", "tool_name", "tool_call_name", "tool"):
        name = raw_call.get(key)
        if isinstance(name, str) and name:
            return name
    return None


def _tool_is_argumentless(tool_name: str, tools: Optional[List[dict]]) -> bool:
    if not tools:
        return False
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        if function.get("name") != tool_name:
            continue
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            return True
        required = parameters.get("required", [])
        return not required
    return False


def _candidate_tool_names(
    tools: Optional[List[dict]],
    *,
    only_name: Optional[str] = None,
) -> list[str]:
    names = sorted(_valid_tool_names(tools), key=len, reverse=True)
    if only_name is None:
        return names
    return [only_name] if only_name in names else []


def _tool_required_parameters(
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
        if not isinstance(parameters, dict):
            return set()
        required = parameters.get("required", [])
        if isinstance(required, list):
            return {name for name in required if isinstance(name, str)}
    return set()


def _parse_arguments_dict(arguments: Any) -> Optional[dict[str, Any]]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_tool_arguments(
    tool_name: str,
    arguments: Any,
    tools: Optional[List[dict]],
) -> Optional[dict[str, Any]]:
    """Map provider-specific argument shapes onto the tool schema."""

    parsed = _parse_arguments_dict(arguments)
    if parsed is None:
        return None

    properties = _tool_parameter_properties(tool_name, tools)
    required = _tool_required_parameters(tool_name, tools)
    if not properties and not required:
        return {} if not parsed else None

    coerced = dict(parsed)

    for key in required:
        if key in coerced and coerced[key] is not None:
            continue
        for alias in _ARGUMENT_ALIASES:
            if alias == key or alias not in coerced:
                continue
            if alias not in properties:
                coerced[key] = coerced.pop(alias)
                break

    coerced = {key: value for key, value in coerced.items() if key in properties}

    if not _validate_tool_arguments(tool_name, coerced, tools):
        return None
    return coerced


def _validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tools: Optional[List[dict]],
) -> bool:
    properties = _tool_parameter_properties(tool_name, tools)
    required = _tool_required_parameters(tool_name, tools)
    if not properties and not required:
        return not arguments
    for key in arguments:
        if key not in properties:
            return False
    for key in required:
        if key not in arguments:
            return False
    return True


def _parse_python_call_arguments(args_str: str) -> Optional[dict[str, Any]]:
    args_str = args_str.strip()
    if not args_str:
        return {}

    try:
        tree = ast.parse(f"_placeholder({args_str})", mode="eval")
    except SyntaxError:
        return None

    call = tree.body
    if not isinstance(call, ast.Call):
        return None
    if call.args:
        return None
    if any(keyword.arg is None for keyword in call.keywords):
        return None

    arguments: dict[str, Any] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return arguments


def _issued_call_keys(
    request_messages: Optional[List[dict]],
) -> frozenset[tuple[str, str]]:
    """Collect ``(name, canonical_arguments_json)`` for every assistant tool call
    already issued in the conversation."""
    keys: set[tuple[str, str]] = set()
    for message in request_messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = _parse_arguments_dict(function.get("arguments"))
            if isinstance(name, str) and name and arguments is not None:
                keys.add((name, json.dumps(arguments, sort_keys=True)))
    return frozenset(keys)


def _find_python_tool_call_in_text(
    text: str,
    *,
    tools: Optional[List[dict]],
    candidates: list[str],
    issued_call_keys: frozenset[tuple[str, str]] = frozenset(),
) -> Optional[tuple[str, dict[str, Any]]]:
    """Promote ``tool(arg=...)`` substrings only when arguments are non-empty.

    Bare name mentions and argumentless ``tool()`` calls are ignored: they produce
    too many false positives when models narrate completed tools in prose.

    ``issued_call_keys`` carries the ``(name, canonical_arguments_json)`` of tool
    calls already issued earlier in the conversation. A text call that exactly
    replays one of them is the model narrating that call (e.g. a final report
    quoting ``store_skills(request="...")`` next to its result), not a new
    attempt — promoting it re-executes the tool on every report and never lets
    the loop finish.
    """
    best: Optional[tuple[int, int, str, dict[str, Any]]] = None

    for name in candidates:
        pattern = re.compile(rf"\b{re.escape(name)}\s*\(([^)]*)\)")
        for match in pattern.finditer(text):
            arguments = _parse_python_call_arguments(match.group(1))
            if arguments is None or not arguments:
                continue
            if not _validate_tool_arguments(name, arguments, tools):
                continue
            if (name, json.dumps(arguments, sort_keys=True)) in issued_call_keys:
                continue
            position = match.start()
            candidate = (position, len(name), name, arguments)
            if best is None:
                best = candidate
                continue
            if candidate[0] < best[0]:
                best = candidate
                continue
            if candidate[0] == best[0] and candidate[1] > best[1]:
                best = candidate

    if best is None:
        return None
    return best[2], best[3]


def _content_search_text(content: str) -> str:
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        return content

    if not isinstance(parsed, dict):
        return content

    parts = [content]
    for value in parsed.values():
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


_STRUCTURED_TOOL_INDICATOR_KEYS = (
    "action",
    "tool",
    "tool_name",
    "next_tool",
    "planned_tool",
    "planned_action",
)

_STRUCTURED_ARGUMENT_KEYS = (
    "arguments",
    "parameters",
    "input",
    "query",
    "text",
    "args",
    "tool_args",
    "tool_call_parameters",
)


def _parse_structured_content_dict(content: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _infer_from_structured_tool_fields(
    parsed: dict[str, Any],
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
) -> Optional[tuple[str, dict[str, Any], set[str]]]:
    forced_name = _forced_tool_name(original_tool_choice)
    candidates = set(_candidate_tool_names(tools, only_name=forced_name))
    if not candidates:
        return None

    for indicator_key in _STRUCTURED_TOOL_INDICATOR_KEYS:
        tool_name = parsed.get(indicator_key)
        if not isinstance(tool_name, str) or tool_name not in candidates:
            continue

        raw_arguments: Any = {}
        promoted_keys = {indicator_key}
        for arg_key in _STRUCTURED_ARGUMENT_KEYS:
            if arg_key in parsed:
                raw_arguments = parsed[arg_key]
                promoted_keys.add(arg_key)
                break

        coerced = _coerce_tool_arguments(tool_name, raw_arguments, tools)
        if coerced is None and _tool_is_argumentless(tool_name, tools):
            coerced = {}
        if coerced is None:
            continue

        return tool_name, coerced, promoted_keys

    return None


def _clean_structured_content(
    content: str,
    promoted_keys: set[str],
) -> Optional[str]:
    parsed = _parse_structured_content_dict(content)
    if parsed is None:
        return None if promoted_keys else content
    remaining = {
        key: value for key, value in parsed.items() if key not in promoted_keys
    }
    if not remaining:
        return None
    return json.dumps(remaining, ensure_ascii=False)


def _strip_provider_markup_delimiters(text: str) -> str:
    return re.sub(r"\]<\][^[]+\[>\[?", "", text)


_INVOKE_BLOCK_RE = re.compile(
    r"<\s*invoke\s+name=(['\"])([^'\"]+)\1\s*>(.*?)</\s*invoke\s*>",
    re.IGNORECASE | re.DOTALL,
)

_XML_CHILD_TAG_RE = re.compile(
    r"<\s*(\w+)\s*>(.*?)</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _extract_xml_invoke_child_arguments(body: str) -> dict[str, str]:
    arguments: dict[str, str] = {}
    for match in _XML_CHILD_TAG_RE.finditer(body):
        key = match.group(1)
        value = match.group(2).strip()
        if value:
            arguments[key] = value
    return arguments


def _find_xml_invoke_tool_call(
    content: str,
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
) -> Optional[tuple[str, dict[str, Any]]]:
    cleaned = _strip_provider_markup_delimiters(content)
    if "<invoke" not in cleaned.lower():
        return None

    forced_name = _forced_tool_name(original_tool_choice)
    candidates = _candidate_tool_names(tools, only_name=forced_name)
    if not candidates:
        return None

    allowed = set(candidates)
    for match in _INVOKE_BLOCK_RE.finditer(cleaned):
        name = match.group(2)
        if name not in allowed:
            continue
        if _tool_is_argumentless(name, tools):
            return name, {}
        body = match.group(3)
        raw_arguments = _extract_xml_invoke_child_arguments(body)
        if not raw_arguments:
            continue
        coerced = _coerce_tool_arguments(name, raw_arguments, tools)
        if coerced is None:
            continue
        return name, coerced
    return None


def _infer_explicit_tool_call_from_content(
    content: str,
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
    request_messages: Optional[List[dict]] = None,
) -> Optional[tuple[str, dict[str, Any], set[str]]]:
    """Promote only explicit tool-call markup from assistant content.

    Used for both forced and auto tool_choice. Soft/prose inference is handled
    separately and must not run under ``tool_choice="auto"``.

    When ``request_messages`` is provided, python-style text calls that exactly
    replay a tool call already issued in the conversation are treated as
    narration and not promoted. Callers pass it only under auto/None
    tool_choice: a forced choice leaves the model no final-answer alternative,
    so there a repeated call is a genuine retry rather than a report.
    """
    forced_name = _forced_tool_name(original_tool_choice)
    candidates = _candidate_tool_names(tools, only_name=forced_name)
    if not candidates:
        return None

    xml_call = _find_xml_invoke_tool_call(
        content,
        tools=tools,
        original_tool_choice=original_tool_choice,
    )
    if xml_call is not None:
        return xml_call[0], xml_call[1], {"content"}

    search_text = _content_search_text(content)
    python_call = _find_python_tool_call_in_text(
        search_text,
        tools=tools,
        candidates=candidates,
        issued_call_keys=_issued_call_keys(request_messages),
    )
    if python_call is not None:
        return python_call[0], python_call[1], set()

    parsed = _parse_structured_content_dict(content)
    if parsed is not None:
        structured = _infer_from_structured_tool_fields(
            parsed,
            tools=tools,
            original_tool_choice=original_tool_choice,
        )
        if structured is not None:
            return structured
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

    for key in (
        "arguments",
        "tool_args",
        "tool_call_parameters",
        "input",
        "parameters",
        "args",
    ):
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
    if not name:
        return None
    arguments = _embedded_call_arguments(raw_call, tool_name=name, tools=tools)
    coerced = _coerce_tool_arguments(name, arguments, tools)
    if coerced is None:
        return None
    return name, _serialize_arguments(coerced)


def _nested_embedded_tool_calls(raw_call: dict) -> Optional[list[Any]]:
    nested = raw_call.get("tool_calls")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested) if nested else []
        except json.JSONDecodeError:
            return None
    if isinstance(nested, list):
        return nested
    return None


def _collect_embedded_calls(
    items: list[Any],
    *,
    tools: Optional[List[dict]],
) -> list[tuple[str, str]]:
    """Extract valid tool calls, flattening nested provider ``tool_calls`` trees."""

    calls: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        raw_call = _unwrap_embedded_call_shape(item)
        normalized = _normalize_embedded_call(raw_call, tools=tools)
        if normalized is not None:
            calls.append(normalized)
            continue

        nested = _nested_embedded_tool_calls(raw_call)
        if nested:
            calls.extend(_collect_embedded_calls(nested, tools=tools))

    return calls


def _promoted_tool_call_dicts(
    embedded_calls: list[tuple[str, str]],
    *,
    response_id: str | None,
) -> list[dict[str, Any]]:
    response_token = hashlib.sha1((response_id or "").encode()).hexdigest()[:10]
    return [
        ChatCompletionMessageToolCall(
            id=f"call_healed_{response_token}_{index}",
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
        calls = _collect_embedded_calls(inner, tools=tools)
        if not calls:
            return [], set()
        promoted_keys.add("tool_calls")
        return calls, promoted_keys

    if _embedded_call_name(parsed) is not None and any(
        key in parsed
        for key in (
            "arguments",
            "tool_args",
            "tool_call_parameters",
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
                "tool_call_name",
                "tool",
                "arguments",
                "tool_args",
                "tool_call_parameters",
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
    if not should_attempt_tool_call_healing(
        provider,
        original_tool_choice,
        msg,
        tools=tools,
    ):
        return None

    content = msg.content
    if not isinstance(content, str) or not content.strip():
        return None

    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        parsed = None

    # Providers sometimes emit a bare JSON array of tool-call objects as
    # message content (e.g. MiniMax: [{"name": "...", "parameters": {...}}])
    # instead of a dict with a ``tool_calls`` key or a native tool_calls field.
    if isinstance(parsed, list):
        embedded_calls = _collect_embedded_calls(parsed, tools=tools)
        cleaned_content = None
    elif isinstance(parsed, dict):
        embedded_calls, promoted_keys = _extract_embedded_calls(
            parsed,
            tools=tools,
        )
        if not embedded_calls:
            return None
        cleaned_content = _clean_content_payload(
            parsed,
            promoted_keys,
            response_format_spec,
        )
    else:
        return None

    if not embedded_calls:
        return None

    allowed_names = _valid_tool_names(tools)
    if not allowed_names:
        return None

    for name, _arguments in embedded_calls:
        if name not in allowed_names:
            return None

    if not _content_passes_response_format(cleaned_content, response_format_spec):
        return None

    promoted_tool_calls = _promoted_tool_call_dicts(
        embedded_calls,
        response_id=response.id,
    )

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


def try_infer_tool_call_from_content(
    response: "ChatCompletion",
    *,
    provider: str,
    original_tool_choice: Any,
    tools: Optional[List[dict]],
    response_format_spec: Optional[ResponseFormatSpec],
    request_messages: Optional[List[dict]] = None,
) -> Optional["ChatCompletion"]:
    msg = response.choices[0].message
    if not should_attempt_tool_call_healing(
        provider,
        original_tool_choice,
        msg,
        tools=tools,
    ):
        return None

    content = msg.content
    if not isinstance(content, str) or not content.strip():
        return None

    # On auto/None, only promote explicit markup (XML invoke, python-style
    # calls, structured tool fields), and never a python-style call that
    # replays one already issued in the conversation — that is the model
    # narrating a completed call in its answer, and promoting it re-runs the
    # tool on every report. Under forced tool_choice the model cannot answer
    # in text at all, so repeats are genuine retries and stay promotable.
    inferred = _infer_explicit_tool_call_from_content(
        content,
        tools=tools,
        original_tool_choice=original_tool_choice,
        request_messages=(
            None if _tool_choice_is_forced(original_tool_choice) else request_messages
        ),
    )
    if inferred is None:
        return None

    tool_name, arguments, promoted_keys = inferred
    cleaned_content = _clean_structured_content(content, promoted_keys)
    if not _content_passes_response_format(cleaned_content, response_format_spec):
        return None

    promoted_tool_calls = _promoted_tool_call_dicts(
        [(tool_name, _serialize_arguments(arguments))],
        response_id=response.id,
    )

    msg.content = cleaned_content
    msg.tool_calls = promoted_tool_calls
    response.choices[0].finish_reason = "tool_calls"

    logger.info(
        "Inferred tool call from text for provider=%s tool=%s",
        provider,
        tool_name,
    )
    return response


try_infer_argumentless_tool_from_content = try_infer_tool_call_from_content


def maybe_heal_tool_calls_in_completion(
    chat_completion: "ChatCompletion",
    *,
    provider: str,
    original_tool_choice: Any,
    tools: Optional[List[dict]],
    response_format_spec: Optional[ResponseFormatSpec],
    request_messages: Optional[List[dict]] = None,
) -> "ChatCompletion":
    healed = try_heal_embedded_tool_calls(
        chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        tools=tools,
        response_format_spec=response_format_spec,
    )
    if healed is not None:
        return healed
    inferred = try_infer_tool_call_from_content(
        chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        tools=tools,
        response_format_spec=response_format_spec,
        request_messages=request_messages,
    )
    return inferred if inferred is not None else chat_completion
