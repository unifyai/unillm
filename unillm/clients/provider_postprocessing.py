"""
Provider-specific post-processing for LLM responses.

This module handles response transformations and fixes that need to happen
after the LLM call returns. It follows the same pattern as provider_preprocessing.py
but operates on responses rather than requests.

Currently handles:
- Anthropic: tool_choice="required" compliance with thinking mode
- Anthropic: invalid tool name detection (tool called not in schema)
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

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
# zero tools are callable on this turn — it must respond with text content.
NO_TOOLS_RETRY_NUDGE_SINGLE = (
    "You attempted to call '{invalid_tools}', but there are no tools available on this turn. "
    "Do not call any tools. Respond with text content only."
)
NO_TOOLS_RETRY_NUDGE_PLURAL = (
    "You attempted to call {invalid_tools}, but there are no tools available on this turn. "
    "Do not call any tools. Respond with text content only."
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
                # No tools available at all — use the dedicated no-tools nudge
                if len(invalid_tools) == 1:
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
                    "content": msg.content,
                },
            )
        else:
            # Shouldn't happen, but fallback
            nudge = TOOL_CHOICE_REQUIRED_RETRY_NUDGE
            retry_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                },
            )
    else:
        # Default: tool_choice_required case
        nudge = TOOL_CHOICE_REQUIRED_RETRY_NUDGE
        # Add the non-compliant assistant response (content only, not thinking blocks)
        retry_messages.append(
            {
                "role": "assistant",
                "content": msg.content,
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
