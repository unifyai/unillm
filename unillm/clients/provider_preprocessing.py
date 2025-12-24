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


def _move_system_messages_to_front(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Move all system messages to the beginning, preserving relative order."""
    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    return system + non_system


def _needs_thinking_workaround(
    messages: List[Dict[str, Any]],
    reasoning_effort: Optional[str],
) -> bool:
    """
    Check if we need to inject a user message workaround for Anthropic thinking mode.

    Anthropic requires thinking_blocks in assistant messages when thinking mode is enabled.
    Since thinking_blocks are server-generated and cannot be faked, we inject a simple
    user message when:
    1. Thinking mode is enabled (via reasoning_effort)
    2. None of the assistant messages have thinking_blocks
    3. The final message is a tool result
    """
    # Check if thinking mode is enabled via reasoning_effort
    if not reasoning_effort:
        return False

    # Check if final message is a tool result
    if messages[-1].get("role") != "tool":
        return False

    # Check if any assistant message has thinking_blocks
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("thinking_blocks"):
            return False

    return True


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

    # Workaround for Anthropic thinking mode: when thinking is enabled but no
    # assistant messages have thinking_blocks (they're server-generated), and
    # the final message is a tool result, inject a user message to avoid API errors.
    if _needs_thinking_workaround(messages, kw.get("reasoning_effort", None)):
        messages.append({"role": "user", "content": "\n"})

    kw["messages"] = messages
    return kw
