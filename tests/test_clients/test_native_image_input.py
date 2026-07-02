"""Live probes for native image input on text-first provider endpoints.

These tests send a solid red PNG as an ``image_url`` block through the same
UniLLM endpoint strings Unity uses (``model@provider``). They are skipped when
no credentials are available for the configured transport path.

Run locally when deciding whether a slow-brain candidate can replace DeepSeek:

    cd unillm
    uv run pytest tests/test_clients/test_native_image_input.py -vv
"""

from __future__ import annotations

import os

import pytest

from .vision_probe_helpers import probe_native_image_input

MINIMAX_V3_ENDPOINT = "minimax-v3@minimax"
MIMO_V25_ENDPOINT = "mimo-v2.5@xiaomi-mimo"

_HAS_OPENROUTER_API_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
_HAS_MINIMAX_API_KEY = bool(os.environ.get("MINIMAX_API_KEY"))
_HAS_XIAOMI_MIMO_API_KEY = bool(os.environ.get("XIAOMI_MIMO_API_KEY"))

_CAN_PROBE_MINIMAX = _HAS_OPENROUTER_API_KEY or _HAS_MINIMAX_API_KEY
_CAN_PROBE_MIMO = _HAS_OPENROUTER_API_KEY or _HAS_XIAOMI_MIMO_API_KEY


@pytest.mark.skipif(
    not _CAN_PROBE_MINIMAX,
    reason="Set MINIMAX_API_KEY or OPENROUTER_API_KEY to probe MiniMax vision",
)
def test_minimax_v3_accepts_native_image_input() -> None:
    probe_native_image_input(MINIMAX_V3_ENDPOINT)


@pytest.mark.skipif(
    not _CAN_PROBE_MIMO,
    reason="Set XIAOMI_MIMO_API_KEY or OPENROUTER_API_KEY to probe MiMo vision",
)
def test_mimo_v25_accepts_native_image_input() -> None:
    probe_native_image_input(MIMO_V25_ENDPOINT)
