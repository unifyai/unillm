"""
Provider-specific post-processing for LLM responses.

This module handles response transformations and fixes that need to happen
after the LLM call returns. It follows the same pattern as provider_preprocessing.py
but operates on responses rather than requests.

Currently handles:
- Anthropic: tool_choice="required" compliance with thinking mode
"""

from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

# Nudge message for retrying when model ignores tool_choice="required" instruction
TOOL_CHOICE_REQUIRED_RETRY_NUDGE = (
    "I understand you may not think a tool call is necessary on this step, but "
    "tool_choice is set to 'required' which means you MUST select the most "
    "appropriate tool with the most appropriate arguments. Please call a tool now."
)


def check_needs_postprocessing(
    *,
    response: "ChatCompletion",
    provider: str,
    original_tool_choice: Optional[str],
    reasoning_effort: Optional[str],
) -> Tuple[bool, Optional[dict]]:
    """
    Check if a response needs post-processing (retry).

    Returns a tuple of (needs_retry, retry_kw).
    If needs_retry is True, retry_kw contains the kwargs for the retry call.
    If needs_retry is False, retry_kw is None.

    This design allows the caller to handle the retry (sync or async) themselves.
    """
    if provider == "anthropic":
        return _check_anthropic_postprocessing(
            response=response,
            original_tool_choice=original_tool_choice,
            reasoning_effort=reasoning_effort,
        )
    return False, None


def build_retry_kw(
    *,
    kw: dict,
    response: "ChatCompletion",
) -> dict:
    """
    Build the retry request kwargs with nudge messages appended.

    Call this only when check_needs_postprocessing returns True.
    """
    msg = response.choices[0].message

    # Build retry messages: original messages + assistant response + nudge
    retry_messages = list(kw.get("messages", []))

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
            "content": TOOL_CHOICE_REQUIRED_RETRY_NUDGE,
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
) -> Tuple[bool, Optional[dict]]:
    """
    Check if Anthropic response needs post-processing.

    When thinking mode is enabled (reasoning_effort is set), Anthropic's API
    doesn't support tool_choice="required". We work around this by:
    1. Preprocessing: downgrade to "auto" + add system instruction
    2. Postprocessing (here): if model ignored instruction, retry with nudge
    """
    # Only applies when thinking mode caused tool_choice downgrade
    if reasoning_effort is None or original_tool_choice != "required":
        return False, None

    # Check if response is compliant (has tool calls)
    msg = response.choices[0].message
    if msg.tool_calls:
        return False, None  # Compliant, no fix needed

    # Non-compliant: model responded with text only despite instruction
    return True, None  # Caller will build retry_kw
