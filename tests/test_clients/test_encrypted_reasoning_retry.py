"""Regression tests for invalid encrypted-reasoning strip-and-retry recovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from unillm.clients.encrypted_reasoning_retry import (
    is_invalid_encrypted_content_error,
    offending_reasoning_ids,
    strip_encrypted_reasoning_for_error,
    strip_encrypted_reasoning_from_messages,
)
from unillm.clients.uni_llm import (
    _acompletion_with_empty_responses_fallback,
    _completion_with_empty_responses_fallback,
)
from unillm.helpers import _is_retryable

RS_GOOD = "rs_0f1de4323a76f49e016a56cfc2328c8194b72ffd9d14c991dd"
RS_BAD = "rs_02ffae9afdf82d3b016a56cfd0750481908f1be8f1eaf95e79"


def _poison_exc() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(
        message=(
            "APIConnectionError: OpenrouterException - Error in response: "
            "{'code': 'invalid_prompt', 'message': 'The encrypted content for "
            f"item {RS_BAD} could not be verified. Reason: Encrypted content "
            "could not be decrypted or parsed.'}"
        ),
        llm_provider="openrouter",
        model="openrouter/responses/openai/gpt-5.6-sol",
    )


def _ok_completion() -> ModelResponse:
    response = ModelResponse(
        id="chatcmpl_enc_retry",
        choices=[
            Choices(
                message=Message(content="recovered"),
                finish_reason="stop",
                index=0,
            ),
        ],
    )
    response.model = "gpt-5.6-sol"
    response.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


def _history_with_both_items() -> list[dict]:
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                },
            ],
            "reasoning_items": [
                {
                    "type": "reasoning",
                    "id": RS_GOOD,
                    "encrypted_content": "good-blob",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                },
            ],
            "reasoning_items": [
                {
                    "type": "reasoning",
                    "id": RS_BAD,
                    "encrypted_content": "poison-blob",
                    "summary": [],
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
        {"role": "user", "content": "continue"},
    ]


def test_detects_invalid_encrypted_content_error_shapes():
    assert is_invalid_encrypted_content_error(_poison_exc())
    assert is_invalid_encrypted_content_error(
        litellm.BadRequestError(
            message="invalid_encrypted_content: could not be decrypted",
            model="gpt-5.6-sol",
            llm_provider="openai",
        ),
    )
    assert not is_invalid_encrypted_content_error(
        litellm.APIConnectionError(
            message="connection reset",
            llm_provider="openrouter",
            model="gpt-5.6-sol",
        ),
    )


def test_offending_reasoning_ids_parsed_from_error():
    assert offending_reasoning_ids(_poison_exc()) == [RS_BAD]


def test_strip_drops_only_offending_id_in_place():
    messages = _history_with_both_items()
    changed = strip_encrypted_reasoning_for_error(messages, _poison_exc())
    assert changed is True
    assert messages[1]["reasoning_items"][0]["id"] == RS_GOOD
    assert "reasoning_items" not in messages[3]


def test_strip_falls_back_to_all_encrypted_when_id_missing():
    messages = _history_with_both_items()
    exc = litellm.APIConnectionError(
        message="invalid_encrypted_content: encrypted content could not be verified",
        llm_provider="openrouter",
        model="gpt-5.6-sol",
    )
    assert strip_encrypted_reasoning_for_error(messages, exc) is True
    assert "reasoning_items" not in messages[1]
    assert "reasoning_items" not in messages[3]


def test_strip_targeted_id_not_found_strips_all_encrypted():
    messages = _history_with_both_items()
    changed = strip_encrypted_reasoning_from_messages(
        messages,
        drop_ids=["rs_does_not_exist"],
    )
    assert changed is True
    assert "reasoning_items" not in messages[1]
    assert "reasoning_items" not in messages[3]


def test_invalid_encrypted_content_is_not_blindly_retried():
    assert _is_retryable(_poison_exc()) is False


@pytest.mark.asyncio
async def test_acompletion_strips_offending_item_and_retries_once():
    messages = _history_with_both_items()
    calls: list[list[dict]] = []

    async def fake_acompletion(**kwargs):
        # Snapshot reasoning ids present on this attempt.
        present = []
        for msg in kwargs["messages"]:
            for item in msg.get("reasoning_items") or []:
                present.append(item["id"])
        calls.append(present)
        if len(calls) == 1:
            raise _poison_exc()
        return _ok_completion()

    with patch(
        "unillm.clients.uni_llm.litellm.acompletion",
        new=AsyncMock(side_effect=fake_acompletion),
    ):
        result = await _acompletion_with_empty_responses_fallback(
            shared_session=None,
            client=None,
            model="openrouter/responses/openai/gpt-5.6-sol",
            messages=messages,
        )

    assert result.choices[0].message.content == "recovered"
    assert calls == [[RS_GOOD, RS_BAD], [RS_GOOD]]
    # Live history mutated so later turns do not resend the poison item.
    assert "reasoning_items" not in messages[3]
    assert messages[1]["reasoning_items"][0]["id"] == RS_GOOD


def test_completion_strips_offending_item_and_retries_once():
    messages = _history_with_both_items()
    calls: list[list[str]] = []

    def fake_completion(**kwargs):
        present = []
        for msg in kwargs["messages"]:
            for item in msg.get("reasoning_items") or []:
                present.append(item["id"])
        calls.append(present)
        if len(calls) == 1:
            raise _poison_exc()
        return _ok_completion()

    with patch(
        "unillm.clients.uni_llm.litellm.completion",
        side_effect=fake_completion,
    ):
        result = _completion_with_empty_responses_fallback(
            model="openrouter/responses/openai/gpt-5.6-sol",
            messages=messages,
        )

    assert result.choices[0].message.content == "recovered"
    assert calls == [[RS_GOOD, RS_BAD], [RS_GOOD]]
    assert "reasoning_items" not in messages[3]
