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
- reasoning_content promotion when content is empty (thinking models)
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
from .completion_mutator import CompletionMutator, apply_completion_mutator
from .response_healing import maybe_heal_tool_calls_in_completion

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage

logger = logging.getLogger(__name__)


def _message_content_is_empty(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    return False


def _message_reasoning_content(msg: "ChatCompletionMessage") -> Optional[str]:
    reasoning = getattr(msg, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    provider_fields = getattr(msg, "provider_specific_fields", None)
    if isinstance(provider_fields, dict):
        nested = provider_fields.get("reasoning_content")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def promote_reasoning_content_to_content(
    chat_completion: "ChatCompletion",
) -> "ChatCompletion":
    """Copy ``reasoning_content`` into ``content`` when the visible payload is empty.

    Some thinking models (MiniMax, DeepSeek, etc.) place the user-visible answer
    only in ``reasoning_content`` with ``content: null``. Downstream consumers
    that read ``content`` (tool loops, tests, UIs) otherwise see a blank turn.
    """
    msg = chat_completion.choices[0].message
    if msg.tool_calls:
        return chat_completion
    if not _message_content_is_empty(msg.content):
        return chat_completion

    reasoning = _message_reasoning_content(msg)
    if reasoning is None:
        return chat_completion

    msg.content = reasoning
    logger.info("Promoted reasoning_content to content")
    return chat_completion


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


def _tool_call_id(tool_call: Any) -> Optional[str]:
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id")
        return call_id if isinstance(call_id, str) else None
    call_id = getattr(tool_call, "id", None)
    return call_id if isinstance(call_id, str) else None


class ModelRefusalError(Exception):
    """The model's safety classifiers declined the request.

    Raised only when the refusal-fallback model also declined (or no fallback
    is configured for the refusing model). Anthropic returns refusals as
    successful responses with ``stop_reason: "refusal"`` (surfaced by litellm
    as ``finish_reason: "content_filter"``), so without this check a refusal
    would silently propagate as an empty assistant turn.
    """


# Retry reason constants
RETRY_REASON_TOOL_CHOICE_REQUIRED = "tool_choice_required"
RETRY_REASON_INVALID_TOOL_NAME = "invalid_tool_name"
RETRY_REASON_MALFORMED_TOOL_ARGUMENTS = "malformed_tool_arguments"

MALFORMED_TOOL_ARGUMENTS_RETRY_NUDGE = (
    "Your previous turn FAILED: the arguments you emitted for a tool call were "
    "not valid JSON, so the call was discarded and had ZERO effect — nothing "
    "was executed. This usually means the arguments were cut off part-way "
    "through. Re-issue the call now, emitting the complete arguments object in "
    "one go. Keep every value in the type the tool declares — numbers and "
    "booleans unquoted — and close every brace and bracket."
)

# Base nudge for retrying when model ignores tool_choice="required" instruction.
# The rejected plain-text attempt is never appended as an assistant turn — that
# format makes models treat undelivered prose as already sent (e.g. calling wait).
TOOL_CHOICE_REQUIRED_RETRY_NUDGE_BASE = (
    "Your previous turn FAILED: you returned plain text instead of calling a tool. "
    "That text was NOT delivered to the user and had ZERO effect — nothing was "
    "sent, logged, or committed. Do NOT call `wait` or treat that text as already "
    "delivered. tool_choice is set to 'required', which means you MUST call the "
    "most appropriate tool with the most appropriate arguments on this turn."
)


def build_tool_choice_required_retry_nudge(
    rejected_content: str | None,
) -> str:
    """Build the user nudge for a rejected text-only tool-required response."""
    if rejected_content:
        return (
            f"{TOOL_CHOICE_REQUIRED_RETRY_NUDGE_BASE}\n\n"
            "If you intended to communicate the following, call the appropriate "
            "send tool NOW with this content (for reference only — it was NOT sent):\n"
            f"> {rejected_content}"
        )
    return (
        f"{TOOL_CHOICE_REQUIRED_RETRY_NUDGE_BASE}\n\n"
        "Please call the appropriate tool now."
    )


# Error message for valid tool calls that were not executed because
# sibling tool calls in the same batch were invalid.
_VALID_TOOL_NOT_EXECUTED_MSG = (
    "Not executed because other tool calls in this batch "
    "called tools not in the schema."
)


def _finish_reason(response: "ChatCompletion") -> Optional[str]:
    return response.choices[0].finish_reason if response.choices else None


def check_malformed_tool_arguments(response: "ChatCompletion") -> bool:
    """Return True when a returned tool call cannot be dispatched as-is.

    Providers can return a tool call whose ``arguments`` string is truncated —
    generation degenerated and ran to the output-token cap mid-object. On the
    OpenAI Responses bridge this arrives indistinguishable from a complete call:
    the truncation is reported as ``status: "incomplete"`` on the raw response,
    but the transformed choice still carries ``finish_reason: "tool_calls"``.
    Validating the payload we were actually handed is provider-independent and
    does not depend on any upstream truncation signal surviving.
    """
    if not response.choices:
        return False
    choice = response.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    if not tool_calls:
        return False

    # A length-capped turn that still emitted tool calls was cut off mid-call.
    if choice.finish_reason == "length":
        return True

    for call in tool_calls:
        raw = getattr(getattr(call, "function", None), "arguments", None)
        if not isinstance(raw, str) or not raw:
            continue
        try:
            json.loads(raw)
        except ValueError:
            logger.warning(
                "Tool call %s returned unparseable arguments (%d chars)",
                getattr(getattr(call, "function", None), "name", "?"),
                len(raw),
            )
            return True
    return False


def check_safety_refusal(
    *,
    response: "ChatCompletion",
    kw: dict,
) -> Optional[str]:
    """Return the fallback model when the response is a safety refusal.

    Only models in ``REFUSAL_FALLBACK_MODELS`` carry safety classifiers;
    ``content_filter`` finish reasons from other models are left untouched.
    """
    from ..endpoints.anthropic import REFUSAL_FALLBACK_MODELS

    if _finish_reason(response) != "content_filter":
        return None
    return REFUSAL_FALLBACK_MODELS.get(str(kw.get("model") or ""))


def build_refusal_fallback_kw(*, kw: dict, fallback_model: str) -> dict:
    """Build retry kwargs that re-issue the unchanged request on the fallback.

    ``_unillm_transport_model`` steers the retry transport off the client's
    bound model; it is consumed by the retry executor and never sent to the
    provider.
    """
    retry_kw = dict(kw)
    retry_kw["model"] = fallback_model
    retry_kw["_unillm_transport_model"] = fallback_model
    return retry_kw


def _raise_if_still_refused(
    response: "ChatCompletion",
    *,
    original_model: str,
    fallback_model: str,
) -> None:
    if _finish_reason(response) == "content_filter":
        raise ModelRefusalError(
            f"{original_model} refused the request (safety classifier) and the "
            f"fallback model {fallback_model} also declined.",
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
        - RETRY_REASON_MALFORMED_TOOL_ARGUMENTS
        - RETRY_REASON_TOOL_CHOICE_REQUIRED
        - RETRY_REASON_INVALID_TOOL_NAME
    If needs_retry is False, retry_reason is None.

    This design allows the caller to handle the retry (sync or async) themselves.
    """
    # Checked first and for every provider: a tool call whose arguments do not
    # parse cannot be dispatched by anyone downstream, so there is nothing for
    # the provider-specific checks below to say about it.
    if check_malformed_tool_arguments(response):
        return True, RETRY_REASON_MALFORMED_TOOL_ARGUMENTS

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

    if retry_reason == RETRY_REASON_MALFORMED_TOOL_ARGUMENTS:
        # The rejected assistant turn is not replayed: its tool_call is
        # unresolvable, and an unanswered tool_call makes the retry request
        # itself invalid for most providers.
        retry_messages = list(kw.get("messages", []))
        retry_messages.append(
            {"role": "user", "content": MALFORMED_TOOL_ARGUMENTS_RETRY_NUDGE},
        )
    elif retry_reason == RETRY_REASON_INVALID_TOOL_NAME:
        retry_messages = _build_invalid_tool_name_retry_messages(
            kw=kw,
            msg=msg,
            assistant_content=assistant_content,
        )
    else:
        # Default: tool_choice_required case — rejected prose is referenced inside
        # a single user nudge, not replayed as an assistant turn.
        retry_messages = list(kw.get("messages", []))
        retry_messages.append(
            {
                "role": "user",
                "content": build_tool_choice_required_retry_nudge(assistant_content),
            },
        )

    # Create retry request
    retry_kw = dict(kw)
    retry_kw["messages"] = retry_messages
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
    del original_request_messages
    msg = response.choices[0].message

    if msg.tool_calls:
        valid_names = set(_get_valid_tool_names(tools))
        for tool_call in msg.tool_calls:
            called_name = _tool_call_name(tool_call)
            if called_name not in valid_names:
                return True, RETRY_REASON_INVALID_TOOL_NAME

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
    completion_mutator: Optional[CompletionMutator] = None,
) -> "ChatCompletion":
    """Run refusal fallback, healing, tool-choice retries, and response_format validation."""
    raw_tools = kw.get("tools")
    tools = list(raw_tools) if raw_tools is not None else None
    response_format_spec = get_response_format_spec(kw)

    refusal_fallback_model = check_safety_refusal(response=chat_completion, kw=kw)
    if refusal_fallback_model is not None:
        original_model = str(kw.get("model") or "")
        logger.warning(
            "%s refused the request (safety classifier); retrying on %s",
            original_model,
            refusal_fallback_model,
        )
        chat_completion = execute_retry(
            build_refusal_fallback_kw(kw=kw, fallback_model=refusal_fallback_model),
            "refusal-fallback",
        )
        _raise_if_still_refused(
            chat_completion,
            original_model=original_model,
            fallback_model=refusal_fallback_model,
        )

    chat_completion = apply_completion_mutator(
        chat_completion,
        completion_mutator=completion_mutator,
        provider=provider,
        original_tool_choice=original_tool_choice,
        request_kw=kw,
    )

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

    return promote_reasoning_content_to_content(chat_completion)


async def apply_postprocessing_pipeline_async(
    chat_completion: "ChatCompletion",
    *,
    kw: dict,
    provider: str,
    original_tool_choice: Optional[Any],
    reasoning_effort: Optional[str],
    original_request_messages: Optional[List[dict]] = None,
    execute_retry,
    completion_mutator: Optional[CompletionMutator] = None,
) -> "ChatCompletion":
    """Async variant of apply_postprocessing_pipeline."""
    raw_tools = kw.get("tools")
    tools = list(raw_tools) if raw_tools is not None else None
    response_format_spec = get_response_format_spec(kw)

    refusal_fallback_model = check_safety_refusal(response=chat_completion, kw=kw)
    if refusal_fallback_model is not None:
        original_model = str(kw.get("model") or "")
        logger.warning(
            "%s refused the request (safety classifier); retrying on %s",
            original_model,
            refusal_fallback_model,
        )
        chat_completion = await execute_retry(
            build_refusal_fallback_kw(kw=kw, fallback_model=refusal_fallback_model),
            "refusal-fallback",
        )
        _raise_if_still_refused(
            chat_completion,
            original_model=original_model,
            fallback_model=refusal_fallback_model,
        )

    chat_completion = apply_completion_mutator(
        chat_completion,
        completion_mutator=completion_mutator,
        provider=provider,
        original_tool_choice=original_tool_choice,
        request_kw=kw,
    )

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

    return promote_reasoning_content_to_content(chat_completion)


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
