"""Recover native tool_calls from model text when tools are available."""

from __future__ import annotations

import ast
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


def _find_python_tool_call_in_text(
    text: str,
    *,
    tools: Optional[List[dict]],
    candidates: list[str],
) -> Optional[tuple[str, dict[str, Any]]]:
    best: Optional[tuple[int, int, str, dict[str, Any]]] = None

    for name in candidates:
        pattern = re.compile(rf"\b{re.escape(name)}\s*\(([^)]*)\)")
        for match in pattern.finditer(text):
            arguments = _parse_python_call_arguments(match.group(1))
            if arguments is None:
                continue
            if not _validate_tool_arguments(name, arguments, tools):
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


def _argumentless_tool_names(
    tools: Optional[List[dict]],
    *,
    only_name: Optional[str] = None,
) -> list[str]:
    return [
        name
        for name in _candidate_tool_names(tools, only_name=only_name)
        if _tool_is_argumentless(name, tools)
    ]


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


def _tool_name_in_text(tool_name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(tool_name)}\b", text) is not None


def _exact_tool_name_values(parsed: dict, allowed: set[str]) -> set[str]:
    found: set[str] = set()
    for value in parsed.values():
        if isinstance(value, str) and value in allowed:
            found.add(value)
    return found


def _infer_argumentless_tool_name(
    content: str,
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
) -> Optional[str]:
    forced_name = _forced_tool_name(original_tool_choice)
    candidates = _argumentless_tool_names(
        tools,
        only_name=forced_name,
    )
    if not candidates:
        return None

    allowed = set(candidates)
    exact_matches: set[str] = set()
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict):
            exact_matches = _exact_tool_name_values(parsed, allowed)
    except json.JSONDecodeError:
        pass

    search_text = _content_search_text(content)
    matched = [
        name
        for name in candidates
        if name in exact_matches or _tool_name_in_text(name, search_text)
    ]
    if not matched:
        return None
    return matched[0]


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


def _clean_prose_payload(payload: str) -> str:
    cleaned = payload.strip().strip("`'\"")
    cleaned = re.sub(
        r"\s+for\s+contact_id\s*=\s*\d+\.?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:and then|then|before|after|while)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _has_tool_intent(text: str, tool_name: str) -> bool:
    escaped = re.escape(tool_name)
    patterns = (
        rf"(?i)(?:use|call|invoke|try)\s+(?:the\s+)?(?:`|')?{escaped}(?:`|')?(?:\s+tool)?\b",
        rf"(?i)\bI'll\s+use\s+{escaped}\b",
        rf"(?i)\b{escaped}\s+(?:to|for)\b",
        rf"(?i)\b{escaped}\s+directly\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _intent_match_position(text: str, tool_name: str) -> Optional[int]:
    escaped = re.escape(tool_name)
    patterns = (
        rf"(?i)(?:use|call|invoke|try)\s+(?:the\s+)?(?:`|')?{escaped}(?:`|')?(?:\s+tool)?\b",
        rf"(?i)\bI'll\s+use\s+{escaped}\b",
        rf"(?i)\b{escaped}\s+(?:to|for)\b",
        rf"(?i)\b{escaped}\s+directly\b",
    )
    positions = [
        match.start()
        for pattern in patterns
        for match in [re.search(pattern, text)]
        if match is not None
    ]
    if positions:
        return min(positions)
    if _tool_name_in_text(tool_name, text):
        match = re.search(rf"\b{escaped}\b", text)
        return match.start() if match else None
    return None


def _extract_prose_int_argument(text: str, field_name: str) -> Optional[int]:
    patterns = (
        rf"(?i)\b{re.escape(field_name)}\s*[=:]\s*(\d+)\b",
        r"(?i)\bcontact_id\s*[=:]\s*(\d+)\b",
        r"(?i)\bcontact\s+(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_task_description(
    text: str,
    tool_name: str,
    field_name: str,
) -> Optional[str]:
    if not _has_tool_intent(text, tool_name):
        return None

    if tool_name == "ask_about_contacts" or field_name == "text":
        patterns = (
            r"(?i)(?:search(?:ing)?|check(?:ing)?|look(?:ing)?\s+up|find(?:ing)?)"
            r"(?:\s+\w+){0,5}\s+(?:contacts?(?:\s+records?)?\s+)?(?:for\s+)?(.+?)(?:[.?!]|$)",
            r"(?i)(?:need to|should|I'll)\s+(?:search|check|look up|find)\s+(.+?)(?:[.?!]|$)",
            r"(?i)(?:query|question)\s+(?:about\s+)?(.+?)(?:[.?!]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return _clean_prose_payload(match.group(1))

    if tool_name == "act" or field_name == "query":
        patterns = (
            r"(?i)(?:search(?:ing)?|find(?:ing)?|look(?:ing)?\s+up)"
            r"(?:\s+\w+){0,4}\s+(?:for\s+)?(.+?)(?:[.?!]|$)",
            r"(?i)(?:view|analyze|inspect)\s+(?:their\s+)?(.+?)(?:[.?!]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return _clean_prose_payload(match.group(1))

    return None


def _extract_prose_string_argument(
    text: str,
    tool_name: str,
    field_name: str,
) -> Optional[str]:
    escaped = re.escape(tool_name)
    for preposition in ("to", "for"):
        pattern = rf"(?i)\b{escaped}\s+{preposition}\s+(.+?)(?:[.?!]|$)"
        match = re.search(pattern, text)
        if match:
            return _clean_prose_payload(match.group(1))

    pattern = (
        rf"(?i)(?:use|call|invoke|try)\s+(?:the\s+)?(?:`|')?"
        rf"{escaped}(?:`|')?(?:\s+tool)?\s+(?:to|for)\s+(.+?)(?:[.?!]|$)"
    )
    match = re.search(pattern, text)
    if match:
        return _clean_prose_payload(match.group(1))

    extracted = _extract_task_description(text, tool_name, field_name)
    if extracted is not None:
        return extracted

    return _fallback_contextual_argument(text, tool_name, field_name)


def _fallback_contextual_argument(
    text: str,
    tool_name: str,
    field_name: str,
) -> Optional[str]:
    if not _has_tool_intent(text, tool_name):
        return None

    sentences = [part.strip() for part in re.split(r"[.!?]\s+", text) if part.strip()]
    if not sentences:
        return None

    if tool_name == "ask_about_contacts" or field_name == "text":
        for sentence in sentences:
            if re.search(
                r"(?i)(contact|preference|phone|email|look up|look her up|find|search)",
                sentence,
            ) and not re.fullmatch(
                rf"(?i).*\b{re.escape(tool_name)}\b.*",
                sentence,
            ):
                return sentence

    for sentence in sentences:
        if _tool_name_in_text(tool_name, sentence):
            continue
        if len(sentence) >= 20:
            return sentence

    return sentences[0]


def _extract_prose_arguments(
    text: str,
    tool_name: str,
    tools: Optional[List[dict]],
) -> Optional[dict[str, Any]]:
    if not _has_tool_intent(text, tool_name):
        return None

    required = _tool_required_parameters(tool_name, tools)
    properties = _tool_parameter_properties(tool_name, tools)
    if not required:
        return None

    arguments: dict[str, Any] = {}
    for field_name in sorted(required):
        if field_name not in properties:
            return None
        if field_name in {"requesting_contact_id", "contact_id"} or field_name.endswith(
            "_id",
        ):
            extracted_int = _extract_prose_int_argument(text, field_name)
            if extracted_int is None:
                return None
            arguments[field_name] = extracted_int
            continue

        extracted = _extract_prose_string_argument(text, tool_name, field_name)
        if extracted is None:
            return None
        arguments[field_name] = extracted

    return arguments


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
            thoughts = parsed.get("thoughts")
            if isinstance(thoughts, str):
                prose_args = _extract_prose_arguments(thoughts, tool_name, tools)
                if prose_args is not None:
                    coerced = _coerce_tool_arguments(tool_name, prose_args, tools)
        if coerced is None:
            continue

        return tool_name, coerced, promoted_keys

    return None


def _integers_from_messages(messages: Optional[List[dict]]) -> list[int]:
    if not messages:
        return []
    blob = json.dumps(messages, ensure_ascii=False)
    patterns = (
        r'"(?:contact_id|requesting_contact_id)"\s*:\s*(\d+)',
        r"contact_id=(\d+)",
        r'contact_id\\"=(\\d+)',
        r'contact_id=\\"(\d+)\\"',
        r'<contact contact_id=\\"(\d+)\\"',
        r'requesting_contact_id\\":\s*(\d+)',
    )
    ids: list[int] = []
    for pattern in patterns:
        ids.extend(int(match.group(1)) for match in re.finditer(pattern, blob))
    return ids


def _boss_contact_id_from_messages(messages: Optional[List[dict]]) -> Optional[int]:
    if not messages:
        return None
    blob = json.dumps(messages, ensure_ascii=False)
    match = re.search(
        r'<contact contact_id=\\"(\d+)\\"[^>]*is_boss=\\"True\\"',
        blob,
    )
    if match:
        return int(match.group(1))
    match = re.search(
        r'is_boss=\\"True\\"[^>]*contact_id=\\"(\d+)\\"',
        blob,
    )
    if match:
        return int(match.group(1))
    return None


def _requesting_contact_id_from_context(
    context_text: str,
    request_messages: Optional[List[dict]],
) -> Optional[int]:
    extracted_int = _extract_prose_int_argument(context_text, "requesting_contact_id")
    if extracted_int is not None:
        return extracted_int
    boss_id = _boss_contact_id_from_messages(request_messages)
    if boss_id is not None:
        return boss_id
    message_ids = _integers_from_messages(request_messages)
    if len(message_ids) == 1:
        return message_ids[0]
    return None


def _fill_missing_arguments_from_context(
    arguments: dict[str, Any],
    *,
    text: str,
    tool_name: str,
    tools: Optional[List[dict]],
    request_messages: Optional[List[dict]],
) -> Optional[dict[str, Any]]:
    required = _tool_required_parameters(tool_name, tools)
    filled = dict(arguments)
    context_text = text
    if request_messages:
        context_text += "\n" + json.dumps(request_messages, ensure_ascii=False)

    for field_name in required:
        if field_name in filled:
            continue
        if field_name.endswith("_id") or field_name == "contact_id":
            extracted_int = _extract_prose_int_argument(context_text, field_name)
            if extracted_int is None and field_name == "requesting_contact_id":
                extracted_int = _requesting_contact_id_from_context(
                    context_text,
                    request_messages,
                )
            if extracted_int is None:
                message_ids = _integers_from_messages(request_messages)
                extracted_int = message_ids[0] if len(message_ids) == 1 else None
            if extracted_int is None:
                return None
            filled[field_name] = extracted_int
            continue
        extracted = _extract_prose_string_argument(context_text, tool_name, field_name)
        if extracted is None:
            return None
        filled[field_name] = extracted

    coerced = _coerce_tool_arguments(tool_name, filled, tools)
    return coerced


_CONTACT_LOOKUP_INTENT_PATTERNS = (
    r"(?i)\bsearch(?:ing)?\s+(?:\w+\s+){0,3}contacts?\b",
    r"(?i)\bcheck(?:ing)?\s+(?:\w+\s+){0,3}contacts?\b",
    r"(?i)\blook(?:\w*\s+){0,2}(?:up|for)\s+(?:\w+\s+){0,3}(?:in\s+)?contacts?\b",
    r"(?i)\bfind(?:ing)?\s+(?:\w+\s+){0,2}(?:contact|preference|phone|email)\b",
    r"(?i)\bcontact[- ]related query\b",
)

_ACT_TASK_INTENT_PATTERNS = (
    r"(?i)\bsummarize\b",
    r"(?i)\bsearch(?:ing)?\s+(?:the\s+)?(?:knowledge base|records?)\b",
    r"(?i)\blook(?:\w*\s+){0,2}up\b.*(?:office hours|shipment|memphis)\b",
    r"(?i)\bfind(?:ing)?\s+(?:any\s+)?(?:relevant|shipment|office)\b",
)


def _matches_contact_lookup_intent(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _CONTACT_LOOKUP_INTENT_PATTERNS)


def _matches_act_task_intent(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _ACT_TASK_INTENT_PATTERNS)


def _extract_contact_lookup_text(text: str) -> Optional[str]:
    sentences = [
        part.strip() for part in re.split(r"[.!?]\n?\s+", text) if part.strip()
    ]
    for sentence in sentences:
        if (
            _matches_contact_lookup_intent(sentence)
            or re.search(r"(?i)\bsarah\b|preference|phone or email", sentence)
        ) and len(sentence) >= 15:
            return sentence
    if _matches_contact_lookup_intent(text):
        return text.strip()[:500]
    return None


def _extract_act_task_query(text: str) -> Optional[str]:
    patterns = (
        r"(?i)(?:summarize|summary of)\s+(.+?)(?:[.?!]|$)",
        r"(?i)(?:search(?:ing)?|find(?:ing)?|look(?:ing)?\s+up)\s+(?:\w+\s+){0,4}(?:for\s+)?(.+?)(?:[.?!]|$)",
        r"(?i)(?:respond|reply)\s+(?:to\s+)?(?:my\s+)?boss\s+(?:with\s+)?(.+?)(?:[.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_prose_payload(match.group(1))
    return _extract_task_description(text, "act", "query")


def _infer_implicit_tool_call(
    content: str,
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
    request_messages: Optional[List[dict]],
) -> Optional[tuple[str, dict[str, Any], set[str]]]:
    forced_name = _forced_tool_name(original_tool_choice)
    candidates = set(_candidate_tool_names(tools, only_name=forced_name))
    if not candidates:
        return None

    search_text = _content_search_text(content)

    if "ask_about_contacts" in candidates and _matches_contact_lookup_intent(
        search_text,
    ):
        lookup_text = _extract_contact_lookup_text(search_text)
        if lookup_text is not None:
            coerced = _coerce_tool_arguments(
                "ask_about_contacts",
                {"text": lookup_text},
                tools,
            )
            if coerced is not None:
                return "ask_about_contacts", coerced, set()

    if "act" in candidates and (
        _has_tool_intent(search_text, "act") or _matches_act_task_intent(search_text)
    ):
        query = _extract_prose_string_argument(search_text, "act", "query")
        if query is None:
            query = _extract_act_task_query(search_text)
        if query is not None:
            coerced = _fill_missing_arguments_from_context(
                {"query": query},
                text=search_text,
                tool_name="act",
                tools=tools,
                request_messages=request_messages,
            )
            if coerced is not None:
                return "act", coerced, set()

    return None


def _infer_prose_tool_call(
    content: str,
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
    request_messages: Optional[List[dict]] = None,
) -> Optional[tuple[str, dict[str, Any], set[str]]]:
    forced_name = _forced_tool_name(original_tool_choice)
    candidates = _candidate_tool_names(tools, only_name=forced_name)
    if not candidates:
        return None

    search_text = _content_search_text(content)
    best: Optional[tuple[int, int, str, dict[str, Any]]] = None

    for tool_name in candidates:
        if _tool_is_argumentless(tool_name, tools):
            continue
        arguments = _extract_prose_arguments(search_text, tool_name, tools)
        if arguments is None:
            continue
        coerced = _fill_missing_arguments_from_context(
            arguments,
            text=search_text,
            tool_name=tool_name,
            tools=tools,
            request_messages=request_messages,
        )
        if coerced is None:
            continue
        position = _intent_match_position(search_text, tool_name)
        if position is None:
            continue
        candidate = (position, len(tool_name), tool_name, coerced)
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
    return best[2], best[3], set()


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


def _infer_tool_call_from_content(
    content: str,
    *,
    tools: Optional[List[dict]],
    original_tool_choice: Any,
    request_messages: Optional[List[dict]] = None,
) -> Optional[tuple[str, dict[str, Any], set[str]]]:
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

    prose_call = _infer_prose_tool_call(
        content,
        tools=tools,
        original_tool_choice=original_tool_choice,
        request_messages=request_messages,
    )
    if prose_call is not None:
        return prose_call

    implicit_call = _infer_implicit_tool_call(
        content,
        tools=tools,
        original_tool_choice=original_tool_choice,
        request_messages=request_messages,
    )
    if implicit_call is not None:
        return implicit_call

    argumentless_name = _infer_argumentless_tool_name(
        content,
        tools=tools,
        original_tool_choice=original_tool_choice,
    )
    if argumentless_name is None:
        return None
    return argumentless_name, {}, set()


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

    inferred = _infer_tool_call_from_content(
        content,
        tools=tools,
        original_tool_choice=original_tool_choice,
        request_messages=request_messages,
    )
    if inferred is None:
        return None

    tool_name, arguments, promoted_keys = inferred
    cleaned_content = _clean_structured_content(content, promoted_keys)
    if not _content_passes_response_format(cleaned_content, response_format_spec):
        return None

    promoted_tool_calls = _promoted_tool_call_dicts(
        [(tool_name, _serialize_arguments(arguments))],
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
