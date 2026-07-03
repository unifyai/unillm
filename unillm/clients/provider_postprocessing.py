"""
Provider-specific post-processing for LLM responses.

This module handles response transformations and fixes that need to happen
after the LLM call returns. It follows the same pattern as provider_preprocessing.py
but operates on responses rather than requests.

Currently handles:
- Anthropic: tool_choice="required" compliance with thinking mode
- Anthropic/DeepSeek: invalid tool name detection (tool called not in schema)
- DeepSeek/MiniMax/Xiaomi MiMo: soft forced tool choice compliance
- DeepSeek/MiniMax/Xiaomi MiMo: embedded tool_calls recovery from JSON content
- json_tool_call wrapper normalization (structured output + inner tool unwrapping)
- response_format schema validation with retry (all providers)
"""

import json
import logging
from typing import Any, List, Optional, Set, Tuple, TYPE_CHECKING

from .response_format import (
    RESPONSE_FORMAT_SPEC_KEY,
    ResponseFormatSpec,
    get_response_format_spec,
    parse_structured_content,
    validate_against_spec,
)
from .json_tool_call_normalization import normalize_json_tool_call_wrappers
from .response_healing import maybe_heal_tool_calls_in_completion

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage


def _tool_call_name(tool_call: Any) -> Optional[str]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            return name if isinstance(name, str) and name else None
        return None
    function = getattr(tool_call, "function", None)
    if function is None:
        return None
    return function.name


def _tool_call_arguments(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            return str(function.get("arguments", ""))
        return ""
    return str(tool_call.function.arguments)


def _tool_call_id(tool_call: Any) -> Optional[str]:
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id")
        return call_id if isinstance(call_id, str) else None
    call_id = getattr(tool_call, "id", None)
    return call_id if isinstance(call_id, str) else None


logger = logging.getLogger(__name__)

# Retry reason constants
RETRY_REASON_TOOL_CHOICE_REQUIRED = "tool_choice_required"
RETRY_REASON_INVALID_TOOL_NAME = "invalid_tool_name"
RETRY_REASON_REPEATED_COMPLETED_TOOL = "repeated_completed_tool"

SOFT_FORCED_TOOL_CHOICE_PROVIDERS = {"deepseek", "minimax", "xiaomi-mimo"}

# Nudge message for retrying when model ignores tool_choice="required" instruction
TOOL_CHOICE_REQUIRED_RETRY_NUDGE = (
    "I understand you may not think a tool call is necessary on this step, but "
    "tool_choice is set to 'required' which means you MUST select the most "
    "appropriate tool with the most appropriate arguments. Please call a tool now."
)

# Error message for valid tool calls that were not executed because
# sibling tool calls in the same batch were invalid.
_VALID_TOOL_NOT_EXECUTED_MSG = (
    "Not executed because other tool calls in this batch "
    "called tools not in the schema."
)


def check_needs_postprocessing(
    *,
    response: "ChatCompletion",
    provider: str,
    original_tool_choice: Optional[Any],
    reasoning_effort: Optional[str],
    tools: Optional[List[dict]] = None,
    request_messages: Optional[List[dict]] = None,
    original_request_messages: Optional[List[dict]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a response needs post-processing (retry).

    Returns a tuple of (needs_retry, retry_reason).
    If needs_retry is True, retry_reason is one of:
        - RETRY_REASON_TOOL_CHOICE_REQUIRED
        - RETRY_REASON_INVALID_TOOL_NAME
    If needs_retry is False, retry_reason is None.

    This design allows the caller to handle the retry (sync or async) themselves.
    """
    if provider == "anthropic":
        return _check_anthropic_postprocessing(
            response=response,
            original_tool_choice=original_tool_choice,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
    return _check_soft_forced_tool_choice_postprocessing(
        response=response,
        original_tool_choice=original_tool_choice,
        tools=tools,
        request_messages=request_messages,
        original_request_messages=original_request_messages,
    )


def _get_valid_tool_names(tools: Optional[List[dict]]) -> List[str]:
    """Extract tool names from the tools array."""
    if not tools:
        return []
    names = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            name = tool["function"].get("name")
            if name:
                names.append(name)
    return names


def _forced_tool_name(tool_choice: Any) -> Optional[str]:
    if not isinstance(tool_choice, dict):
        return None
    function_choice = tool_choice.get("function")
    if not isinstance(function_choice, dict):
        return None
    name = function_choice.get("name")
    return name if isinstance(name, str) and name else None


def _called_tool_names(msg: "ChatCompletionMessage") -> List[str]:
    names = [_tool_call_name(tc) for tc in (msg.tool_calls or [])]
    return [name for name in names if name]


def _has_tool_result_history(messages: Optional[List[dict]]) -> bool:
    if not messages:
        return False
    return any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in messages
    )


def _completed_tool_signatures(messages: Optional[List[dict]]) -> set[tuple[str, str]]:
    if not messages:
        return set()

    call_id_to_signature: dict[str, tuple[str, str]] = {}
    completed_call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                call_id = tool_call.get("id")
                name = function.get("name")
                arguments = function.get("arguments", "")
                if isinstance(call_id, str) and isinstance(name, str):
                    call_id_to_signature[call_id] = (name, str(arguments))
        elif message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str):
                completed_call_ids.add(tool_call_id)

    return {
        signature
        for call_id, signature in call_id_to_signature.items()
        if call_id in completed_call_ids
    }


def _repeats_completed_tool_call(
    msg: "ChatCompletionMessage",
    messages: Optional[List[dict]],
) -> bool:
    completed = _completed_tool_signatures(messages)
    if not completed or not msg.tool_calls:
        return False

    return all(
        (_tool_call_name(tool_call), _tool_call_arguments(tool_call)) in completed
        for tool_call in msg.tool_calls
    )


def _make_tool_error(
    content: str,
    *,
    available_tools: Optional[Set[str]] = None,
    json_schema: Optional[dict] = None,
) -> dict:
    """Build a structured error object for an invalid-tool-call tool result.

    Returns a dict of the form ``{"error": {"content": ..., ...}}``.
    Optional keys are omitted when ``None``.
    """
    inner: dict = {"content": content}
    if available_tools is not None:
        inner["available_tools"] = list(available_tools)
    if json_schema is not None:
        inner["json_schema"] = json_schema
    return {"error": inner}


def _build_invalid_tool_name_retry_messages(
    *,
    kw: dict,
    msg: "ChatCompletionMessage",
    assistant_content: Optional[str],
) -> list:
    """Build retry messages for the invalid-tool-name case.

    Instead of a sanitised user-text nudge, this keeps the original tool_calls
    in the assistant message (for context) and replies with structured
    ``role: "tool"`` result messages containing a JSON error object.

    Three cases are handled:

    * **Case A** – at least one valid tool is available: each invalid tool
      result includes ``available_tools``; valid tool results explain they
      were not executed.
    * **Case B** – no tools available, no ``response_format`` schema:
      instruct the model to respond with text content only.
    * **Case C** – no tools available, ``response_format`` schema present:
      instruct the model to return JSON matching the schema and include the
      schema in the error object.

    Returns the full retry message list
    (original messages + assistant w/ tool_calls + tool results).
    """
    retry_messages = list(kw.get("messages", []))
    tool_calls = msg.tool_calls or []
    valid_tool_names = set(_get_valid_tool_names(kw.get("tools", [])))

    # --- assistant message: preserve tool_calls for model context ----------
    retry_messages.append(
        {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        },
    )

    # --- tool result messages for every tool call --------------------------
    if valid_tool_names:
        # Case A: at least one valid tool is available
        for tc in tool_calls:
            called_name = _tool_call_name(tc)
            if called_name not in valid_tool_names:
                error_obj = _make_tool_error(
                    f"'{called_name}' is not callable on this turn. "
                    "It may be mentioned in the system message but is "
                    "not in the current tool schema. "
                    "Please select from the available tools only.",
                    available_tools=valid_tool_names,
                )
            else:
                error_obj = _make_tool_error(
                    _VALID_TOOL_NOT_EXECUTED_MSG,
                    available_tools=valid_tool_names,
                )
            retry_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _tool_call_id(tc),
                    "content": json.dumps(error_obj),
                },
            )
    else:
        # No tools available — determine Case B vs Case C
        rf_spec = get_response_format_spec(kw)
        schema = rf_spec.json_schema if rf_spec is not None else None
        for tc in tool_calls:
            called_name = _tool_call_name(tc)
            if schema is not None:
                # Case C: response_format schema is set
                error_obj = _make_tool_error(
                    f"'{called_name}' is not callable. "
                    "No tools are available on this turn. "
                    "Do not call any tools. "
                    "Respond with valid JSON only, nothing else, "
                    "conforming to the required schema.",
                    json_schema=schema,
                )
            else:
                # Case B: no schema, no tools
                error_obj = _make_tool_error(
                    f"'{_tool_call_name(tc)}' is not callable. "
                    "No tools are available on this turn. "
                    "Do not call any tools. "
                    "Respond with text content only.",
                )
            retry_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(error_obj),
                },
            )

    return retry_messages


def build_retry_kw(
    *,
    kw: dict,
    response: "ChatCompletion",
    retry_reason: Optional[str] = None,
) -> dict:
    """
    Build the retry request kwargs with nudge messages appended.

    Call this only when check_needs_postprocessing returns True.

    Args:
        kw: The original request kwargs
        response: The non-compliant response
        retry_reason: One of RETRY_REASON_* constants, or None for default behavior
    """
    msg = response.choices[0].message

    # Anthropic rejects assistant messages whose text content is whitespace-only
    # ("messages: text content blocks must contain non-whitespace text").
    # Claude commonly returns content="\n\n" alongside tool_calls; sanitize
    # it to None so the retry request stays valid.
    assistant_content = msg.content if msg.content and msg.content.strip() else None

    if retry_reason == RETRY_REASON_INVALID_TOOL_NAME:
        retry_messages = _build_invalid_tool_name_retry_messages(
            kw=kw,
            msg=msg,
            assistant_content=assistant_content,
        )
    elif retry_reason == RETRY_REASON_REPEATED_COMPLETED_TOOL:
        retry_messages = list(kw.get("messages", []))
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    "The tool call you just requested has already completed and "
                    "its result is present in the conversation history. Do not "
                    "call it again. Answer using the completed tool result."
                ),
            },
        )
    else:
        # Default: tool_choice_required case
        retry_messages = list(kw.get("messages", []))
        retry_messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            },
        )
        retry_messages.append(
            {
                "role": "user",
                "content": TOOL_CHOICE_REQUIRED_RETRY_NUDGE,
            },
        )

    # Create retry request
    retry_kw = dict(kw)
    retry_kw["messages"] = retry_messages
    if retry_reason == RETRY_REASON_REPEATED_COMPLETED_TOOL:
        retry_kw.pop("tools", None)
        retry_kw.pop("tool_choice", None)
    return retry_kw


def _check_anthropic_postprocessing(
    *,
    response: "ChatCompletion",
    original_tool_choice: Optional[Any],
    reasoning_effort: Optional[str],
    tools: Optional[List[dict]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if Anthropic response needs post-processing.

    Handles two cases:
    1. When thinking mode is enabled (reasoning_effort is set), Anthropic's API
       doesn't support tool_choice="required". We work around this by:
       - Preprocessing: downgrade to "auto" + add system instruction
       - Postprocessing (here): if model ignored instruction, retry with nudge

    2. Anthropic doesn't constrain tool names to the schema - the model can call
       tools mentioned in the prompt even if they're not in the `tools` array.
       We detect this and retry with a helpful error message.
    """
    msg = response.choices[0].message

    # Check for invalid tool names first (applies regardless of thinking mode).
    # This must run even when tools is [] or None — Anthropic can call tools
    # mentioned in the system prompt even if they're not in the tools array.
    if msg.tool_calls:
        valid_names = set(_get_valid_tool_names(tools))
        for tool_call in msg.tool_calls:
            called_name = _tool_call_name(tool_call)
            if called_name not in valid_names:
                # Model called a tool not in the schema
                return True, RETRY_REASON_INVALID_TOOL_NAME

    # Check for tool_choice="required" non-compliance with thinking mode
    if reasoning_effort is not None and original_tool_choice == "required":
        if not msg.tool_calls:
            # Non-compliant: model responded with text only despite instruction
            return True, RETRY_REASON_TOOL_CHOICE_REQUIRED

    return False, None


def _check_soft_forced_tool_choice_postprocessing(
    *,
    response: "ChatCompletion",
    original_tool_choice: Optional[Any],
    tools: Optional[List[dict]] = None,
    request_messages: Optional[List[dict]] = None,
    original_request_messages: Optional[List[dict]] = None,
) -> Tuple[bool, Optional[str]]:
    msg = response.choices[0].message

    if msg.tool_calls:
        valid_names = set(_get_valid_tool_names(tools))
        for tool_call in msg.tool_calls:
            called_name = _tool_call_name(tool_call)
            if called_name not in valid_names:
                return True, RETRY_REASON_INVALID_TOOL_NAME

        if original_tool_choice in (None, "auto") and _repeats_completed_tool_call(
            msg,
            original_request_messages,
        ):
            return True, RETRY_REASON_REPEATED_COMPLETED_TOOL

    if original_tool_choice == "required" and not msg.tool_calls:
        if (
            not _get_valid_tool_names(tools)
            and msg.content
            and msg.content.strip()
            and _has_tool_result_history(request_messages)
        ):
            return False, None
        return True, RETRY_REASON_TOOL_CHOICE_REQUIRED

    required_tool_name = _forced_tool_name(original_tool_choice)
    if required_tool_name is not None:
        if required_tool_name not in _called_tool_names(msg):
            return True, RETRY_REASON_TOOL_CHOICE_REQUIRED

    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# response_format schema validation retry
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_FORMAT_RETRY_NUDGE = (
    "Your previous response did not conform to the required output schema.\n\n"
    "Validation error:\n{error}\n\n"
    "Your response was:\n{response}\n\n"
    "You MUST respond with valid JSON that exactly matches this schema:\n{schema}\n\n"
    "Return ONLY the JSON object — no markdown, no commentary, no code fences."
)


def check_response_format_compliance(
    *,
    response: "ChatCompletion",
    kw: dict,
) -> Tuple[bool, Optional[str], Optional[ResponseFormatSpec]]:
    """Check whether the response satisfies the response_format schema.

    Returns (needs_retry, validation_error_message, response_format_spec).
    If needs_retry is False the other values are None.
    """
    spec = get_response_format_spec(kw)
    if spec is None:
        return False, None, None

    msg = response.choices[0].message
    content = msg.content
    if content is None:
        return False, None, None

    parsed, parse_error = parse_structured_content(content)
    if parse_error is not None:
        return True, parse_error, spec

    validation_error = validate_against_spec(parsed, spec)
    if validation_error is not None:
        return True, validation_error, spec

    return False, None, None


def apply_postprocessing_pipeline(
    chat_completion: "ChatCompletion",
    *,
    kw: dict,
    provider: str,
    original_tool_choice: Optional[Any],
    reasoning_effort: Optional[str],
    original_request_messages: Optional[List[dict]] = None,
    execute_retry,
) -> "ChatCompletion":
    """Run healing, tool-choice retries, and response_format validation."""
    raw_tools = kw.get("tools")
    tools = list(raw_tools) if raw_tools is not None else None
    response_format_spec = get_response_format_spec(kw)

    chat_completion = maybe_heal_tool_calls_in_completion(
        chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        tools=tools,
        response_format_spec=response_format_spec,
        request_messages=kw.get("messages"),
    )
    chat_completion = normalize_json_tool_call_wrappers(
        chat_completion,
        response_format_spec=response_format_spec,
        tools=tools,
    )

    needs_retry, retry_reason = check_needs_postprocessing(
        response=chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        reasoning_effort=reasoning_effort,
        tools=tools,
        request_messages=kw.get("messages"),
        original_request_messages=original_request_messages,
    )
    if needs_retry:
        retry_kw = build_retry_kw(
            kw=kw,
            response=chat_completion,
            retry_reason=retry_reason,
        )
        chat_completion = execute_retry(retry_kw, "retry")
        chat_completion = maybe_heal_tool_calls_in_completion(
            chat_completion,
            provider=provider,
            original_tool_choice=original_tool_choice,
            tools=tools,
            response_format_spec=response_format_spec,
            request_messages=retry_kw.get("messages"),
        )
        chat_completion = normalize_json_tool_call_wrappers(
            chat_completion,
            response_format_spec=response_format_spec,
            tools=tools,
        )
        check_needs_postprocessing(
            response=chat_completion,
            provider=provider,
            original_tool_choice=original_tool_choice,
            reasoning_effort=reasoning_effort,
            tools=tools,
            request_messages=kw.get("messages"),
            original_request_messages=original_request_messages,
        )

    rf_needs_retry, rf_error, rf_spec = check_response_format_compliance(
        response=chat_completion,
        kw=kw,
    )
    if rf_needs_retry and rf_spec is not None:
        rf_retry_kw = build_response_format_retry_kw(
            kw=kw,
            response=chat_completion,
            validation_error=rf_error,
            response_format_spec=rf_spec,
        )
        chat_completion = execute_retry(rf_retry_kw, "rf-retry")

    return chat_completion


async def apply_postprocessing_pipeline_async(
    chat_completion: "ChatCompletion",
    *,
    kw: dict,
    provider: str,
    original_tool_choice: Optional[Any],
    reasoning_effort: Optional[str],
    original_request_messages: Optional[List[dict]] = None,
    execute_retry,
) -> "ChatCompletion":
    """Async variant of apply_postprocessing_pipeline."""
    raw_tools = kw.get("tools")
    tools = list(raw_tools) if raw_tools is not None else None
    response_format_spec = get_response_format_spec(kw)

    chat_completion = maybe_heal_tool_calls_in_completion(
        chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        tools=tools,
        response_format_spec=response_format_spec,
        request_messages=kw.get("messages"),
    )
    chat_completion = normalize_json_tool_call_wrappers(
        chat_completion,
        response_format_spec=response_format_spec,
        tools=tools,
    )

    needs_retry, retry_reason = check_needs_postprocessing(
        response=chat_completion,
        provider=provider,
        original_tool_choice=original_tool_choice,
        reasoning_effort=reasoning_effort,
        tools=tools,
        request_messages=kw.get("messages"),
        original_request_messages=original_request_messages,
    )
    if needs_retry:
        retry_kw = build_retry_kw(
            kw=kw,
            response=chat_completion,
            retry_reason=retry_reason,
        )
        chat_completion = await execute_retry(retry_kw, "retry")
        chat_completion = maybe_heal_tool_calls_in_completion(
            chat_completion,
            provider=provider,
            original_tool_choice=original_tool_choice,
            tools=tools,
            response_format_spec=response_format_spec,
            request_messages=retry_kw.get("messages"),
        )
        chat_completion = normalize_json_tool_call_wrappers(
            chat_completion,
            response_format_spec=response_format_spec,
            tools=tools,
        )
        check_needs_postprocessing(
            response=chat_completion,
            provider=provider,
            original_tool_choice=original_tool_choice,
            reasoning_effort=reasoning_effort,
            tools=tools,
            request_messages=kw.get("messages"),
            original_request_messages=original_request_messages,
        )

    rf_needs_retry, rf_error, rf_spec = check_response_format_compliance(
        response=chat_completion,
        kw=kw,
    )
    if rf_needs_retry and rf_spec is not None:
        rf_retry_kw = build_response_format_retry_kw(
            kw=kw,
            response=chat_completion,
            validation_error=rf_error,
            response_format_spec=rf_spec,
        )
        chat_completion = await execute_retry(rf_retry_kw, "rf-retry")

    return chat_completion


def build_response_format_retry_kw(
    *,
    kw: dict,
    response: "ChatCompletion",
    validation_error: str,
    response_format_spec: ResponseFormatSpec,
) -> dict:
    """Build retry kwargs with a nudge explaining the schema violation."""
    msg = response.choices[0].message

    retry_messages = list(kw.get("messages", []))
    retry_messages.append(
        {
            "role": "assistant",
            "content": msg.content,
        },
    )

    schema_str = json.dumps(response_format_spec.json_schema, indent=2)
    nudge = RESPONSE_FORMAT_RETRY_NUDGE.format(
        error=validation_error,
        response=msg.content[:500] if msg.content else "(empty)",
        schema=schema_str,
    )
    retry_messages.append(
        {
            "role": "user",
            "content": nudge,
        },
    )

    retry_kw = dict(kw)
    retry_kw["messages"] = retry_messages
    retry_kw.pop("tools", None)
    retry_kw.pop("tool_choice", None)
    retry_kw["response_format"] = response_format_spec.source
    retry_kw[RESPONSE_FORMAT_SPEC_KEY] = response_format_spec
    return retry_kw
