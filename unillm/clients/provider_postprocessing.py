"""
Provider-specific post-processing for LLM responses.

This module handles response transformations and fixes that need to happen
after the LLM call returns. It follows the same pattern as provider_preprocessing.py
but operates on responses rather than requests.

Currently handles:
- Anthropic: tool_choice="required" compliance with thinking mode
"""

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

# Nudge message for retrying when model ignores tool_choice="required" instruction
TOOL_CHOICE_REQUIRED_RETRY_NUDGE = (
    "I understand you may not think a tool call is necessary on this step, but "
    "tool_choice is set to 'required' which means you MUST select the most "
    "appropriate tool with the most appropriate arguments. Please call a tool now."
)


def apply_provider_postprocessing(
    *,
    kw: dict,
    response: "ChatCompletion",
    provider: str,
    original_tool_choice: Optional[str],
    reasoning_effort: Optional[str],
    retry_fn: Callable[[dict], "ChatCompletion"],
) -> "ChatCompletion":
    """
    Apply provider-specific post-processing to an LLM response.

    This function may modify the response or trigger retries for provider-specific
    edge cases. Currently handles the Anthropic thinking mode + tool_choice="required"
    incompatibility.

    Args:
        kw: The request kwargs that were sent to the LLM.
        response: The LLM response to potentially fix.
        provider: The provider name (e.g., "anthropic", "openai").
        original_tool_choice: The tool_choice value before preprocessing.
        reasoning_effort: The reasoning_effort setting (indicates thinking mode).
        retry_fn: A function that takes retry_kw and returns a new ChatCompletion.

    Returns:
        The original response if compliant, or a fixed response from retry.
    """
    if provider == "anthropic":
        return _anthropic_postprocess(
            kw=kw,
            response=response,
            original_tool_choice=original_tool_choice,
            reasoning_effort=reasoning_effort,
            retry_fn=retry_fn,
        )
    return response


def _anthropic_postprocess(
    *,
    kw: dict,
    response: "ChatCompletion",
    original_tool_choice: Optional[str],
    reasoning_effort: Optional[str],
    retry_fn: Callable[[dict], "ChatCompletion"],
) -> "ChatCompletion":
    """
    Handle Anthropic-specific response fixes.

    When thinking mode is enabled (reasoning_effort is set), Anthropic's API
    doesn't support tool_choice="required". We work around this by:
    1. Preprocessing: downgrade to "auto" + add system instruction
    2. Postprocessing (here): if model ignored instruction, retry with nudge

    This ensures tool_choice="required" semantics are enforced even when the
    native API constraint can't be used.
    """
    # Only applies when thinking mode caused tool_choice downgrade
    if reasoning_effort is None or original_tool_choice != "required":
        return response

    # Check if response is compliant (has tool calls)
    msg = response.choices[0].message
    if msg.tool_calls:
        return response  # Compliant, no fix needed

    # Non-compliant: model responded with text only despite instruction
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

    return retry_fn(retry_kw)
