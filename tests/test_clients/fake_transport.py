"""Capture the HTTP requests unillm sends, with no network and no real key.

``HTTPHandler.post`` is where LiteLLM hands a fully transformed request to
httpx for both the OpenRouter and the Anthropic transports, so replacing it
records exactly what a provider would receive: the URL, the headers and the
JSON body after every unillm and LiteLLM adaptation. Each request is answered
with a minimal, well-formed completion in the provider's own shape.

Provider keys are swapped for a placeholder for the duration, and credential
headers are dropped from what is recorded, so neither a real key nor the
placeholder ends up in an assertion message or a recording.
"""

import contextlib
import copy
import os
from json import loads
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import httpx
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from pydantic import SecretStr

import unillm

PLACEHOLDER_KEY = "sk-placeholder-not-a-key"  # pragma: allowlist secret
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})
_GATEWAY_VARIABLES = ("UNILLM_LLM_GATEWAY_URL", "UNILLM_LLM_GATEWAY_KEY")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# One endpoint per transport family unillm registers: OpenAI, Anthropic and
# xAI models behind OpenRouter, direct Anthropic, and the hosts pinned to a
# fixed OpenRouter provider list (DeepSeek, Kimi, GLM, MiniMax, MiMo, Llama,
# Mistral and gpt-oss).
ENDPOINTS = (
    "openai/gpt-5.6-sol@openrouter",
    "anthropic/claude-sonnet-4.6@openrouter",
    "claude-4.5-sonnet@bedrock",
    "claude-opus-5@anthropic",
    "grok-4@xai",
    "gemini-2.5-pro@vertex-ai",
    "gpt-oss-120b@togetherai",
    "deepseek-v4-max@deepseek",
    "kimi-k3@moonshotai",
    "glm-5.2@zai",
    "minimax-v3@minimax",
    "mimo-v2.5@xiaomi-mimo",
    "llama-3.3-70b-chat@groq",
    "mistral-large@mistral",
)

# Claude behind OpenRouter: requested markers are placed only once the
# client also has an affinity key.
CLAUDE_ON_OPENROUTER = (
    "anthropic/claude-sonnet-4.6@openrouter",
    "claude-4.5-sonnet@bedrock",
)

PROMPT_CACHING_VARIANTS = {
    "no-markers": None,
    "all-markers": ["system", "tools", "messages"],
}


def _reply(url: str) -> dict[str, Any]:
    if "anthropic.com" in url:
        return {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "claude-fake",
            "content": [{"type": "text", "text": "Sunny."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 2},
        }
    return {
        "id": "gen-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Sunny."},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    }


@contextlib.contextmanager
def captured_requests() -> Iterator[list[dict[str, Any]]]:
    """Record every provider request made inside the block."""
    from unillm.settings import SETTINGS

    sent: list[dict[str, Any]] = []

    def post(self, url, data=None, json=None, headers=None, **_):  # noqa: A002
        sent.append(
            {
                "url": url,
                "headers": {
                    name: value
                    for name, value in (headers or {}).items()
                    if name.lower() not in _CREDENTIAL_HEADERS
                },
                "body": loads(data) if data is not None else json,
            },
        )
        return httpx.Response(
            200,
            json=_reply(url),
            request=httpx.Request("POST", url),
        )

    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _GATEWAY_VARIABLES
    }
    placeholder = SecretStr(PLACEHOLDER_KEY)
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(SETTINGS, "OPENROUTER_API_KEY", placeholder),
        patch.object(SETTINGS, "ANTHROPIC_API_KEY", placeholder),
        patch.object(HTTPHandler, "post", post),
    ):
        yield sent


def send(endpoint: str, **client_kw: Any) -> list[dict[str, Any]]:
    """Send one tool-bearing turn from a fresh client; return what went out."""
    client = unillm.Unify(endpoint, cache=False, **client_kw)
    with captured_requests() as sent:
        client.generate(
            system_message="You answer weather questions briefly.",
            user_message="What is the weather in Paris?",
            tools=[copy.deepcopy(WEATHER_TOOL)],
            tool_choice="auto",
        )
    return sent


def record_off_path() -> dict[str, list[dict[str, Any]]]:
    """Every endpoint and marker variant, sent with no cache affinity."""
    return {
        f"{endpoint}|{variant}": send(endpoint, prompt_caching=prompt_caching)
        for endpoint in ENDPOINTS
        for variant, prompt_caching in PROMPT_CACHING_VARIANTS.items()
    }


_RECORDING = Path(__file__).parent / "recorded" / "off_path_requests.json"


def recorded(endpoint: str, variant: str = "no-markers") -> list[dict[str, Any]]:
    """What ``send`` put on the wire for this case at b1dced7, before the key."""
    return loads(_RECORDING.read_text())[f"{endpoint}|{variant}"]
