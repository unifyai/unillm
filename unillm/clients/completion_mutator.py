"""Generic pre-postprocessing completion mutation hook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


@dataclass(frozen=True)
class CompletionMutatorContext:
    """Read-only context passed to a completion mutator."""

    provider: str
    original_tool_choice: Any
    request_kw: dict[str, Any]


CompletionMutator = Callable[
    ["ChatCompletion", CompletionMutatorContext],
    "ChatCompletion",
]


def inject_tool_call(
    completion: "ChatCompletion",
    *,
    tool_name: str,
    arguments: dict[str, Any],
    clear_content: bool = True,
) -> "ChatCompletion":
    """Attach a synthetic tool call to a completion and mark it tool-complete."""
    msg = completion.choices[0].message
    if clear_content:
        msg.content = None
    msg.tool_calls = [
        ChatCompletionMessageToolCall(
            id="call_mutator_0",
            type="function",
            function=Function(
                name=tool_name,
                arguments=json.dumps(arguments),
            ),
        ).model_dump(warnings=False),
    ]
    completion.choices[0].finish_reason = "tool_calls"
    return completion


def apply_completion_mutator(
    chat_completion: "ChatCompletion",
    *,
    completion_mutator: Optional[CompletionMutator],
    provider: str,
    original_tool_choice: Any,
    request_kw: dict[str, Any],
) -> "ChatCompletion":
    if completion_mutator is None:
        return chat_completion
    context = CompletionMutatorContext(
        provider=provider,
        original_tool_choice=original_tool_choice,
        request_kw=request_kw,
    )
    return completion_mutator(chat_completion, context)
