"""Collapse LiteLLM Responses→Chat Completions bridge choice-splitting.

The Responses API may emit ``message`` and ``function_call`` items in one
``output`` list. LiteLLM's completion bridge maps each ``ResponseOutputMessage``
to its own ``choices[i]`` and appends tool calls as a later choice — even when
the request did not set ``n > 1``. Chat Completions semantics treat ``choices``
as independent samples for ``n``, so a single turn's text + tools belong on one
message (``content`` + ``tool_calls``).

Without this collapse, callers that only execute ``choices[0]`` drop the tools,
and prose→send mutators can turn the orphaned text choice into a send tool.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


def _requested_n(request_kw: Optional[dict]) -> Optional[int]:
    if not request_kw:
        return None
    raw = request_kw.get("n")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _message_tool_calls(message: Any) -> List[Any]:
    tool_calls = getattr(message, "tool_calls", None) or []
    return list(tool_calls)


def _message_text(message: Any) -> Optional[str]:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    return None


def maybe_collapse_responses_bridge_choices(
    chat_completion: "ChatCompletion",
    *,
    request_kw: Optional[dict] = None,
) -> "ChatCompletion":
    """Merge text-only + tool-bearing choices from a Responses bridge split.

    Heals only when all of the following hold:

    - Request ``n`` is absent or ``<= 1`` (real multi-sample must not be merged).
    - There are at least two choices.
    - At least one choice is text-only (content, no ``tool_calls``).
    - At least one choice has ``tool_calls``.

    That is the LiteLLM bridge signature for one Responses turn. Other
    multi-choice shapes (duplicate text-only, tool-only alternatives, ``n > 1``)
    are left untouched.
    """
    n = _requested_n(request_kw)
    if n is not None and n > 1:
        return chat_completion

    choices = list(getattr(chat_completion, "choices", None) or [])
    if len(choices) <= 1:
        return chat_completion

    text_only: List[Any] = []
    tool_bearing: List[Any] = []
    for choice in choices:
        message = choice.message
        has_tools = bool(_message_tool_calls(message))
        has_text = _message_text(message) is not None
        if has_tools:
            tool_bearing.append(choice)
        elif has_text:
            text_only.append(choice)

    if not text_only or not tool_bearing:
        return chat_completion

    content_parts: List[str] = []
    for choice in text_only + tool_bearing:
        text = _message_text(choice.message)
        if text is not None:
            content_parts.append(text)
    # Preserve order; drop exact consecutive duplicates from bridge quirks.
    deduped: List[str] = []
    for part in content_parts:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    merged_content = "\n\n".join(deduped) if deduped else None

    merged_tool_calls: List[Any] = []
    for choice in tool_bearing:
        merged_tool_calls.extend(_message_tool_calls(choice.message))

    reasoning_content = None
    reasoning_items = None
    for choice in text_only + tool_bearing:
        message = choice.message
        if reasoning_content is None:
            candidate = getattr(message, "reasoning_content", None)
            if isinstance(candidate, str) and candidate.strip():
                reasoning_content = candidate
        if reasoning_items is None:
            items = getattr(message, "reasoning_items", None)
            if items:
                reasoning_items = items

    # Prefer the first tool-bearing message as the surviving object so tool_calls
    # typing stays whatever the provider/SDK already produced.
    survivor = tool_bearing[0]
    msg = survivor.message
    msg.content = merged_content
    msg.tool_calls = merged_tool_calls
    if (
        reasoning_content is not None
        and hasattr(msg, "reasoning_content")
        and not getattr(msg, "reasoning_content", None)
    ):
        msg.reasoning_content = reasoning_content
    if (
        reasoning_items is not None
        and hasattr(msg, "reasoning_items")
        and not getattr(msg, "reasoning_items", None)
    ):
        msg.reasoning_items = reasoning_items

    survivor.index = 0
    survivor.finish_reason = "tool_calls"
    chat_completion.choices = [survivor]

    logger.info(
        "Collapsed Responses bridge choices: %d → 1 "
        "(text-only=%d, tool-bearing=%d, tools=%d)",
        len(choices),
        len(text_only),
        len(tool_bearing),
        len(merged_tool_calls),
    )
    return chat_completion
