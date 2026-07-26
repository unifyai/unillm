"""Live probes for native image input on text-first provider endpoints.

These tests send a solid red PNG as an ``image_url`` block through the same
UniLLM endpoint strings Unity uses (``model@provider``). Under read-only CI
they replay from the LLM cache; locally they call OpenRouter when the cache
misses.

Run locally when deciding whether a slow-brain candidate can replace DeepSeek:

    cd unillm
    uv run pytest tests/test_clients/test_native_image_input.py -vv
"""

from __future__ import annotations

from .vision_probe_helpers import probe_native_image_input

MINIMAX_V3_ENDPOINT = "minimax-v3@minimax"
MIMO_V25_ENDPOINT = "mimo-v2.5@xiaomi-mimo"
GEMINI_3_PRO_ENDPOINT = "gemini-3-pro@vertex-ai"
KIMI_K3_ENDPOINT = "kimi-k3@moonshotai"
CLAUDE_OPUS_5_ENDPOINT = "claude-opus-5@anthropic"


def test_minimax_v3_accepts_native_image_input() -> None:
    probe_native_image_input(MINIMAX_V3_ENDPOINT)


def test_gemini_3_pro_accepts_native_image_input() -> None:
    probe_native_image_input(GEMINI_3_PRO_ENDPOINT)


def test_mimo_v25_accepts_native_image_input() -> None:
    probe_native_image_input(MIMO_V25_ENDPOINT)


def test_kimi_k3_accepts_native_image_input() -> None:
    probe_native_image_input(KIMI_K3_ENDPOINT)


def test_claude_opus_5_accepts_native_image_input() -> None:
    probe_native_image_input(CLAUDE_OPUS_5_ENDPOINT)
