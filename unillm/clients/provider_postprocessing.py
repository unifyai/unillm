"""
Provider-specific post-processing for LLM responses.

This module handles response transformations and fixes that need to happen
after the LLM call returns. It follows the same pattern as provider_preprocessing.py
but operates on responses rather than requests.

Currently handles:
- Anthropic: tool_choice="required" compliance with thinking mode
- Anthropic: invalid tool name detection (tool called not in schema)
- response_format schema validation with retry (all providers)
"""

import json
import logging
from typing import List, Optional, Tuple, Type, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

# Retry reason constants
RETRY_REASON_TOOL_CHOICE_REQUIRED = "tool_choice_required"
RETRY_REASON_INVALID_TOOL_NAME = "invalid_tool_name"

# Nudge message for retrying when model ignores tool_choice="required" instruction
TOOL_CHOICE_REQUIRED_RETRY_NUDGE = (
    "I understand you may not think a tool call is necessary on this step, but "
    "tool_choice is set to 'required' which means you MUST select the most "
    "appropriate tool with the most appropriate arguments. Please call a tool now."
)

# Nudge message templates for retrying when model calls tools not in the schema.
# These acknowledge the system message may list more tools than are currently callable,
# which happens when tool_policy restricts available tools on certain turns.
INVALID_TOOL_NAME_RETRY_NUDGE_SINGLE = (
    "You attempted to call '{invalid_tools}'. "
    "This tool may be mentioned in the system message, but it is not callable on this turn. "
    "The tools currently available are: {valid_tools}. "
    "Please select one of the available tools."
)
INVALID_TOOL_NAME_RETRY_NUDGE_PLURAL = (
    "You attempted to call {invalid_tools}. "
    "These tools may be mentioned in the system message, but they are not callable on this turn. "
    "The tools currently available are: {valid_tools}. "
    "Please select from the available tools only."
)

# Nudge for the special case where there are NO tools available at all.
# The model called tools (likely from descriptions in the system prompt) but
# zero tools are callable on this turn — it must respond with content directly.
# When response_format is set, the nudge asks for JSON matching the schema;
# otherwise it asks for plain text.
NO_TOOLS_RETRY_NUDGE_SINGLE = (
    "You attempted to call '{invalid_tools}', but there are no tools available on this turn. "
    "Do not call any tools. Respond with text content only."
)
NO_TOOLS_RETRY_NUDGE_PLURAL = (
    "You attempted to call {invalid_tools}, but there are no tools available on this turn. "
    "Do not call any tools. Respond with text content only."
)
NO_TOOLS_RETRY_NUDGE_SINGLE_WITH_SCHEMA = (
    "You attempted to call '{invalid_tools}', but there are no tools available on this turn. "
    "Do not call any tools. Respond with valid JSON that conforms to the following schema:\n{schema}\n\n"
    "Return ONLY the JSON object — no markdown, no commentary, no code fences."
)
NO_TOOLS_RETRY_NUDGE_PLURAL_WITH_SCHEMA = (
    "You attempted to call {invalid_tools}, but there are no tools available on this turn. "
    "Do not call any tools. Respond with valid JSON that conforms to the following schema:\n{schema}\n\n"
    "Return ONLY the JSON object — no markdown, no commentary, no code fences."
)


def check_needs_postprocessing(
    *,
    response: "ChatCompletion",
    provider: str,
    original_tool_choice: Optional[str],
    reasoning_effort: Optional[str],
    tools: Optional[List[dict]] = None,
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
    return False, None


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

    # Build retry messages: original messages + assistant response + nudge
    retry_messages = list(kw.get("messages", []))

    # Determine the nudge message based on retry reason
    if retry_reason == RETRY_REASON_INVALID_TOOL_NAME:
        # For invalid tool name, find ALL invalid tools and report them
        tool_calls = msg.tool_calls or []
        valid_tool_names = set(_get_valid_tool_names(kw.get("tools")))

        # Find all invalid tool names
        invalid_tools = []
        for tc in tool_calls:
            if tc.function.name not in valid_tool_names:
                invalid_tools.append(tc.function.name)

        if invalid_tools:
            if not valid_tool_names:
                # No tools available at all — use the dedicated no-tools nudge.
                # When response_format is set, direct the LLM to output JSON
                # matching the schema instead of "text content only" (which
                # would contradict the response_format constraint).
                rf_model = _get_response_format_model(kw)
                if rf_model is not None:
                    schema_str = json.dumps(
                        rf_model.model_json_schema(),
                        indent=2,
                    )
                    if len(invalid_tools) == 1:
                        nudge = NO_TOOLS_RETRY_NUDGE_SINGLE_WITH_SCHEMA.format(
                            invalid_tools=invalid_tools[0],
                            schema=schema_str,
                        )
                    else:
                        quoted = [f"'{t}'" for t in invalid_tools]
                        if len(quoted) == 2:
                            invalid_str = f"{quoted[0]} and {quoted[1]}"
                        else:
                            invalid_str = ", ".join(quoted[:-1]) + f", and {quoted[-1]}"
                        nudge = NO_TOOLS_RETRY_NUDGE_PLURAL_WITH_SCHEMA.format(
                            invalid_tools=invalid_str,
                            schema=schema_str,
                        )
                elif len(invalid_tools) == 1:
                    nudge = NO_TOOLS_RETRY_NUDGE_SINGLE.format(
                        invalid_tools=invalid_tools[0],
                    )
                else:
                    quoted = [f"'{t}'" for t in invalid_tools]
                    if len(quoted) == 2:
                        invalid_str = f"{quoted[0]} and {quoted[1]}"
                    else:
                        invalid_str = ", ".join(quoted[:-1]) + f", and {quoted[-1]}"
                    nudge = NO_TOOLS_RETRY_NUDGE_PLURAL.format(
                        invalid_tools=invalid_str,
                    )
            elif len(invalid_tools) == 1:
                valid_tools_str = ", ".join(sorted(valid_tool_names))
                nudge = INVALID_TOOL_NAME_RETRY_NUDGE_SINGLE.format(
                    invalid_tools=f"'{invalid_tools[0]}'",
                    valid_tools=valid_tools_str,
                )
            else:
                # Format multiple invalid tools as 'tool1', 'tool2', and 'tool3'
                valid_tools_str = ", ".join(sorted(valid_tool_names))
                quoted = [f"'{t}'" for t in invalid_tools]
                if len(quoted) == 2:
                    invalid_str = f"{quoted[0]} and {quoted[1]}"
                else:
                    invalid_str = ", ".join(quoted[:-1]) + f", and {quoted[-1]}"
                nudge = INVALID_TOOL_NAME_RETRY_NUDGE_PLURAL.format(
                    invalid_tools=invalid_str,
                    valid_tools=valid_tools_str,
                )
            # Add the assistant response with the invalid tool call
            # (content only, not thinking blocks, not the tool call itself)
            retry_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            )
        else:
            # Shouldn't happen, but fallback
            nudge = TOOL_CHOICE_REQUIRED_RETRY_NUDGE
            retry_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            )
    else:
        # Default: tool_choice_required case
        nudge = TOOL_CHOICE_REQUIRED_RETRY_NUDGE
        # Add the non-compliant assistant response (content only, not thinking blocks)
        retry_messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            },
        )

    # Add the nudge user message
    retry_messages.append(
        {
            "role": "user",
            "content": nudge,
        },
    )

    # Create retry request
    retry_kw = dict(kw)
    retry_kw["messages"] = retry_messages
    return retry_kw


def _check_anthropic_postprocessing(
    *,
    response: "ChatCompletion",
    original_tool_choice: Optional[str],
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
            called_name = tool_call.function.name
            if called_name not in valid_names:
                # Model called a tool not in the schema
                return True, RETRY_REASON_INVALID_TOOL_NAME

    # Check for tool_choice="required" non-compliance with thinking mode
    if reasoning_effort is not None and original_tool_choice == "required":
        if not msg.tool_calls:
            # Non-compliant: model responded with text only despite instruction
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


def _get_response_format_model(
    kw: dict,
) -> Optional[Type[BaseModel]]:
    """Extract the Pydantic model from the request kwargs, if present."""
    rf = kw.get("response_format")
    if rf is None:
        return None
    # response_format may be the Pydantic class directly
    if isinstance(rf, type) and issubclass(rf, BaseModel):
        return rf
    # Or it may be a dict with __pydantic_schema__ (unillm internal serialization)
    if isinstance(rf, dict) and "__pydantic_schema__" in rf:
        # We can't reconstruct the class from the schema dict alone, but the
        # original Pydantic class is stored on the *prompt* by callers.  Return
        # None here — validation will be skipped (no worse than today).
        return None
    return None


def check_response_format_compliance(
    *,
    response: "ChatCompletion",
    kw: dict,
) -> Tuple[bool, Optional[str], Optional[Type[BaseModel]]]:
    """Check whether the response satisfies the response_format Pydantic schema.

    Returns (needs_retry, validation_error_message, pydantic_model).
    If needs_retry is False the other values are None.
    """
    model_cls = _get_response_format_model(kw)
    if model_cls is None:
        return False, None, None

    msg = response.choices[0].message
    content = msg.content
    if content is None:
        # tool_calls response or empty — nothing to validate here
        return False, None, None

    # Try JSON parse then Pydantic validation
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        return True, f"Response is not valid JSON: {exc}", model_cls

    try:
        model_cls.model_validate(parsed)
    except Exception as exc:
        return True, str(exc), model_cls

    return False, None, None


def build_response_format_retry_kw(
    *,
    kw: dict,
    response: "ChatCompletion",
    validation_error: str,
    pydantic_model: Type[BaseModel],
) -> dict:
    """Build retry kwargs with a nudge explaining the schema violation."""
    msg = response.choices[0].message

    retry_messages = list(kw.get("messages", []))

    # Append the non-compliant assistant reply
    retry_messages.append(
        {
            "role": "assistant",
            "content": msg.content,
        },
    )

    # Build a compact schema representation for the nudge
    schema_str = json.dumps(pydantic_model.model_json_schema(), indent=2)

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
    # Remove tools and response_format from the retry.  The presence of
    # tools=[] interferes with response_format enforcement on some providers
    # (notably Anthropic), and response_format itself can be silently ignored
    # when the conversation context is tool-heavy.  We rely on the explicit
    # text-based nudge above to enforce the schema instead.
    retry_kw.pop("tools", None)
    retry_kw.pop("tool_choice", None)
    retry_kw.pop("response_format", None)
    return retry_kw
