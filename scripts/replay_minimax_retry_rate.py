#!/usr/bin/env python3
"""Replay saved unillm logs through healing/postprocessing to estimate retry rate."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from unillm.clients.provider_postprocessing import check_needs_postprocessing
from unillm.clients.response_format import canonicalize_response_format
from unillm.clients.response_healing import maybe_heal_tool_calls_in_completion
from pydantic import BaseModel, ConfigDict


class TextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thoughts: str


KNOWN_TOOLS = {
    "wait": {
        "type": "function",
        "function": {
            "name": "wait",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    "act": {
        "type": "function",
        "function": {
            "name": "act",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "requesting_contact_id": {"type": "integer"},
                },
                "required": ["query", "requesting_contact_id"],
            },
        },
    },
    "ask_about_contacts": {
        "type": "function",
        "function": {
            "name": "ask_about_contacts",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    "send_unify_message": {
        "type": "function",
        "function": {
            "name": "send_unify_message",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "contact_id": {"type": "integer"},
                },
                "required": ["content", "contact_id"],
            },
        },
    },
}


def _label(text: str) -> str:
    match = re.search(r"ConversationManager :: ([^\]]+)", text)
    return match.group(1) if match else ""


def _extract_response_content(text: str) -> str | None:
    part = text.split("LLM response")[-1]
    match = re.search(
        r'"content": "(\{.*?\})"\s*,\s*\n\s*"role": "assistant"',
        part,
        re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).encode("utf-8").decode("unicode_escape")


def _extract_tool_names(text: str) -> list[str]:
    request_part = text.split("LLM response")[0]
    names = re.findall(r'"name": "([a-zA-Z0-9_]+)"', request_part)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in KNOWN_TOOLS and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _completion(content: str) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=None)
    choice = Choice(index=0, message=message, finish_reason="stop")
    return ChatCompletion(
        id="replay",
        choices=[choice],
        created=0,
        model="replay",
        object="chat.completion",
    )


def _extract_request_messages(text: str) -> list[dict]:
    request_part = text.split("LLM response")[0]
    return [{"role": "user", "content": request_part}]


def _replay_file(path: Path) -> tuple[bool, str | None, str | None]:
    text = path.read_text(errors="replace")
    label = _label(text)
    if "minimax" not in label.lower() or "-retry" in label:
        return False, None, None

    content = _extract_response_content(text)
    if content is None:
        return False, "parse_error", None

    tool_names = _extract_tool_names(text)
    tools = [KNOWN_TOOLS[name] for name in tool_names]
    if not tools:
        return False, "no_tools", None

    response = _completion(content)
    response_format_spec = canonicalize_response_format(TextResponse)
    healed = maybe_heal_tool_calls_in_completion(
        response,
        provider="minimax",
        original_tool_choice="required",
        tools=tools,
        response_format_spec=response_format_spec,
        request_messages=_extract_request_messages(text),
    )
    needs_retry, reason = check_needs_postprocessing(
        response=healed,
        provider="minimax",
        original_tool_choice="required",
        reasoning_effort="high",
        tools=tools,
        request_messages=[],
        original_request_messages=[],
    )
    promoted = None
    if healed.choices[0].message.tool_calls:
        call = healed.choices[0].message.tool_calls[0]
        function = call.function if hasattr(call, "function") else call["function"]
        promoted = function.name if hasattr(function, "name") else function["name"]
    return True, reason if needs_retry else None, promoted


def main(argv: list[str]) -> int:
    log_dirs = [Path(arg) for arg in argv]
    if not log_dirs:
        print(
            "Usage: replay_minimax_retry_rate.py <unillm-log-dir> [...]",
            file=sys.stderr,
        )
        return 2

    counted = 0
    reasons: Counter[str] = Counter()
    promoted: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for log_dir in log_dirs:
        for path in sorted(log_dir.glob("*ConversationManager*.txt")):
            included, reason, tool_name = _replay_file(path)
            if not included:
                if reason:
                    skipped[reason] += 1
                continue
            counted += 1
            if reason:
                reasons[reason] += 1
            elif tool_name:
                promoted[tool_name] += 1

    retry_count = sum(reasons.values())
    retry_rate = (retry_count / counted * 100) if counted else 0.0
    print(f"MiniMax initial calls replayed: {counted}")
    print(f"Would still retry: {retry_count} ({retry_rate:.1f}%)")
    print(f"Healed without retry: {counted - retry_count}")
    if reasons:
        print("Retry reasons:", dict(reasons))
    if promoted:
        print("Promoted tools:", dict(promoted))
    if skipped:
        print("Skipped:", dict(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
