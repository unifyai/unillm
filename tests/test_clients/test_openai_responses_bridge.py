"""Regression tests for OpenAI Responses bridge routing."""

from __future__ import annotations

from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import unillm
import unillm.logger as unillm_logger
from unillm.caching.local_cache import LocalCache
from unillm.llm_events import allm_event_hook_scope


def _tool_response(*, content: str | None = None, tool_call: bool = False):
    if tool_call:
        message = Message(
            content=None,
            tool_calls=[
                {
                    "id": "call_filter",
                    "type": "function",
                    "function": {
                        "name": "filter_contacts",
                        "arguments": '{"query":"alice"}',
                    },
                },
            ],
        )
        finish_reason = "tool_calls"
    else:
        message = Message(content=content or "Bridge response")
        finish_reason = "stop"

    response = ModelResponse(
        id="chatcmpl_bridge_test",
        choices=[Choices(message=message, finish_reason=finish_reason, index=0)],
    )
    response.model = "gpt-5.5"
    response.usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    return response


GPT_55_TOOL = {
    "type": "function",
    "strict": True,
    "function": {
        "name": "filter_contacts",
        "description": "Filter contacts by query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def test_sync_openai_tool_reasoning_call_uses_responses_bridge() -> None:
    """Sync GPT-5.5 tool calls with reasoning use the provider-qualified bridge."""
    captured: dict = {}
    prompt = f"Find Alice. request={uuid4()}"

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _tool_response(tool_call=True)

    with (
        patch("unillm.clients.uni_llm.litellm.completion", side_effect=fake_completion),
        patch("unillm.clients.uni_llm.compute_cost_from_response", return_value=None),
    ):
        client = unillm.Unify(
            "gpt-5.5@openai",
            reasoning_effort="low",
            api_key="test-key",
        )
        response = client.generate(
            messages=[{"role": "user", "content": prompt}],
            tools=[GPT_55_TOOL],
            tool_choice="required",
            parallel_tool_calls=False,
            return_full_completion=True,
        )

    assert captured["model"] == "openai/responses/gpt-5.5"
    assert captured["reasoning_effort"] == "low"
    assert captured["tool_choice"] == "required"
    assert captured["parallel_tool_calls"] is False
    assert set(captured["allowed_openai_params"]) >= {
        "parallel_tool_calls",
        "tool_choice",
    }
    assert captured["tools"][0]["function"]["strict"] is True
    assert "strict" not in captured["tools"][0]
    assert GPT_55_TOOL["strict"] is True
    assert "strict" not in GPT_55_TOOL["function"]
    assert response.choices[0].message.tool_calls[0].function.name == "filter_contacts"


@pytest.mark.asyncio
async def test_async_openai_responses_bridge_preserves_stateful_tool_history() -> None:
    """Async bridged tool calls still store ChatCompletion-shaped history."""
    captured: dict = {}
    prompt = f"Find Alice. request={uuid4()}"

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _tool_response(tool_call=True)

    with (
        patch(
            "unillm.clients.uni_llm.litellm.acompletion",
            side_effect=fake_acompletion,
        ),
        patch("unillm.clients.uni_llm.compute_cost_from_response", return_value=None),
    ):
        client = unillm.AsyncUnify(
            "gpt-5.5@openai",
            reasoning_effort="low",
            stateful=True,
            api_key="test-key",
        )
        await client.generate(
            messages=[{"role": "user", "content": prompt}],
            tools=[GPT_55_TOOL],
            tool_choice="required",
            return_full_completion=True,
        )

    assert captured["model"] == "openai/responses/gpt-5.5"
    assert captured["tool_choice"] == "required"
    assert client.messages[-1]["role"] == "assistant"
    assert client.messages[-1]["tool_calls"][0]["function"]["name"] == "filter_contacts"


@pytest.mark.asyncio
async def test_openai_responses_bridge_cost_events_use_canonical_model() -> None:
    """Billing and cost events use the provider model, not the bridge prefix."""
    captured: dict = {}
    prompt = f"Say done. request={uuid4()}"

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _tool_response(content="Done.")

    with (
        patch(
            "unillm.clients.uni_llm.litellm.acompletion",
            side_effect=fake_acompletion,
        ),
        patch(
            "unillm.clients.uni_llm.compute_cost_from_response",
            return_value=0.02,
        ) as compute_cost,
        patch("unillm.clients.uni_llm._safe_deduct_credits"),
    ):
        client = unillm.AsyncUnify(
            "gpt-5.5@openai",
            reasoning_effort="low",
            api_key="test-key",
        )
        llm_events = []
        with unillm.capture_costs() as events:
            async with allm_event_hook_scope(llm_events.append):
                await client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[GPT_55_TOOL],
                    tool_choice="auto",
                    return_full_completion=True,
                )

    assert captured["model"] == "openai/responses/gpt-5.5"
    assert compute_cost.call_args.args[0] == "gpt-5.5"
    assert len(events) == 1
    assert events[0].model == "gpt-5.5"
    assert events[0].provider_cost == 0.02
    assert llm_events[0].request["model"] == "gpt-5.5"
    assert llm_events[0].request["transport_model"] == "openai/responses/gpt-5.5"


@pytest.mark.asyncio
async def test_openai_responses_bridge_cache_and_logging_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Bridge responses serialize through cache and finalize request logs."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(LocalCache, "_cache_dir", str(cache_dir))
    monkeypatch.setattr(LocalCache, "_cache", None)
    monkeypatch.setattr(unillm_logger, "_LOG_DIR", None)
    monkeypatch.setattr(unillm_logger, "_LOG_DIR_CHECKED", False)
    monkeypatch.setenv("UNILLM_LOG_DIR", str(tmp_path / "logs"))
    unillm.configure_log_dir()

    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _tool_response(content="Cached bridge response.")

    with (
        patch(
            "unillm.clients.uni_llm.litellm.acompletion",
            side_effect=fake_acompletion,
        ),
        patch("unillm.clients.uni_llm.compute_cost_from_response", return_value=None),
    ):
        client = unillm.AsyncUnify(
            "gpt-5.5@openai",
            reasoning_effort="low",
            cache=True,
            api_key="test-key",
        )
        first = await client.generate(
            messages=[{"role": "user", "content": "Find Alice."}],
            tools=[GPT_55_TOOL],
            tool_choice="auto",
            return_full_completion=True,
        )
        second = await client.generate(
            messages=[{"role": "user", "content": "Find Alice."}],
            tools=[GPT_55_TOOL],
            tool_choice="auto",
            return_full_completion=True,
        )

    assert len(calls) == 1
    assert first.choices[0].message.content == second.choices[0].message.content

    log_text = "\n".join(path.read_text() for path in (tmp_path / "logs").glob("*.txt"))
    assert "openai/responses/gpt-5.5" in log_text
    assert "[cache: miss]" in log_text
    assert "[cache: hit]" in log_text
