"""Provider-specific message preprocessing applied before cache lookup."""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

CONCURRENT_USER_MESSAGES_EXPLANATION = (
    "For all user messages which are represented in JSON format, please treat each "
    "item in the list as a separate message. The user message is shown in JSON format "
    "because this API does not natively support concurrent user messages (which is "
    "what actually occurred), and concurrent user messages are being represented this "
    "way instead."
)

THINKING_PREFILL_EXPLANATION = (
    "The conversation history up to this point is expressed below in JSON format "
    "due to schema constraints:\n"
)


def _move_system_messages_to_front(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Move all system messages to the beginning, preserving relative order."""
    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    return system + non_system


def _find_first_real_assistant_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    """
    Find the index of the first assistant message with thinking_blocks (real response).
    Returns None if no real assistant responses exist.
    """
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("thinking_blocks"):
            return i
    return None


def _has_prefilled_assistant_before_real(messages: List[Dict[str, Any]]) -> bool:
    """
    Check if there are prefilled assistant messages (without thinking_blocks)
    that appear before any real assistant response (with thinking_blocks).
    """
    first_real_idx = _find_first_real_assistant_index(messages)

    # Check messages up to (but not including) first real response
    check_until = first_real_idx if first_real_idx is not None else len(messages)

    for i in range(check_until):
        msg = messages[i]
        if msg.get("role") == "assistant" and not msg.get("thinking_blocks"):
            return True
    return False


def _convert_prefill_to_system_message(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert prefilled messages to a system message for Anthropic thinking mode.

    When thinking mode is enabled, Anthropic requires assistant messages to have
    thinking_blocks. For prefilled/synthetic assistant messages, we convert them
    to a JSON representation in the system message.

    Only messages BEFORE the first real assistant response (one with thinking_blocks)
    are converted. Messages from the first real response onwards are kept intact.
    """
    first_real_idx = _find_first_real_assistant_index(messages)

    # Determine which messages to convert vs keep
    if first_real_idx is not None:
        to_convert = messages[:first_real_idx]
        to_keep = messages[first_real_idx:]
    else:
        to_convert = messages
        to_keep = []

    # Extract existing system message content from the portion to convert
    system_content = ""
    non_system_to_convert = []
    for msg in to_convert:
        if msg.get("role") == "system":
            if system_content:
                system_content += "\n\n"
            system_content += msg.get("content", "")
        else:
            non_system_to_convert.append(msg)

    # Build result
    result = []

    # Add system message with converted content (if any non-system messages to convert)
    if non_system_to_convert:
        if system_content:
            system_content += "\n\n"
        system_content += THINKING_PREFILL_EXPLANATION
        system_content += json.dumps(non_system_to_convert, indent=4)

    if system_content:
        result.append({"role": "system", "content": system_content})

    # Add continuation prompt if we converted everything (no real messages to keep)
    if not to_keep:
        result.append({"role": "user", "content": "[continue]"})
    else:
        # Keep real messages intact
        result.extend(to_keep)

    return result


def _combine_adjacent_user_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Combine adjacent user messages into JSON content array format.
    Returns (result, whether_any_combining_occurred).
    """
    if not messages:
        return [], False

    result = []
    combined_any = False
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg.get("role") != "user":
            result.append(msg)
            i += 1
            continue

        # Collect adjacent user messages
        user_contents = []
        while i < len(messages) and messages[i].get("role") == "user":
            user_contents.append(messages[i].get("content", ""))
            i += 1

        if len(user_contents) == 1:
            result.append(msg)
        else:
            combined_any = True
            result.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": c} for c in user_contents
                            ],
                        },
                        indent=4,
                    ),
                },
            )

    return result, combined_any


def apply_provider_preprocessing(
    kw: Dict[str, Any],
    provider: Optional[str],
) -> Dict[str, Any]:
    """Apply provider-specific preprocessing to messages in kw dict (mutates kw)."""
    messages = kw.get("messages")
    if not messages:
        return kw

    # Only Anthropic preprocessing for now
    if provider != "anthropic":
        return kw

    messages = copy.deepcopy(messages)
    messages = _move_system_messages_to_front(messages)
    messages, combined_any = _combine_adjacent_user_messages(messages)

    if combined_any:
        # Insert explanation after system messages
        insert_pos = sum(1 for m in messages if m.get("role") == "system")
        messages.insert(
            insert_pos,
            {
                "role": "system",
                "content": CONCURRENT_USER_MESSAGES_EXPLANATION,
            },
        )

    # Handle prefilled assistant messages with thinking mode enabled.
    # Anthropic requires thinking_blocks on assistant messages when thinking is enabled.
    # Convert only prefilled messages (before first real response) to system message.
    if kw.get("reasoning_effort") is not None and _has_prefilled_assistant_before_real(
        messages,
    ):
        messages = _convert_prefill_to_system_message(messages)

    kw["messages"] = messages

    # Anthropic requires tools=[] (not None) when messages contain tool-related content.
    # This allows tool availability to change turn-by-turn while preserving history.
    if kw.get("tools") is None:
        kw["tools"] = []

    # Disable thinking mode if tool choice is required
    if kw.get("reasoning_effort") is not None and kw.get("tool_choice") == "required":
        del kw["reasoning_effort"]

    return kw
