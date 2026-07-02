"""Shared helpers for live native-image-input probes."""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import litellm
import pytest

import unillm
from unillm.endpoints.utils import get_model_alias, get_transport_model_alias

IMAGE_INPUT_REJECTION_MARKERS = (
    "no endpoints found that support image input",
    "does not support image",
    "does not support vision",
    "image input is not supported",
    "multimodal input is not supported",
    "invalid image",
    "unsupported image",
)

NATIVE_IMAGE_PROMPT = (
    "The attached image is a solid color. Reply with exactly one word: the "
    "color name in English, lowercase, with no punctuation."
)


def solid_color_png_b64(
    width: int,
    height: int,
    rgb: tuple[int, int, int],
) -> str:
    """Return base64-encoded PNG bytes for a solid RGB rectangle."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def native_image_messages(*, color: str, png_b64: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": NATIVE_IMAGE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{png_b64}",
                    },
                },
            ],
        },
    ]


def probe_native_image_input(
    endpoint: str,
    *,
    expected_color: str = "red",
    temperature: float = 0.0,
    max_completion_tokens: int = 512,
) -> str:
    """Call *endpoint* with a solid-color PNG and return the model response.

    Raises ``pytest.fail`` when the provider rejects image input or when the
    model answer does not demonstrate that it saw the image.
    """
    png_b64 = solid_color_png_b64(32, 32, (255, 0, 0))
    messages = native_image_messages(color=expected_color, png_b64=png_b64)
    public_model = get_model_alias(endpoint)
    transport_model = get_transport_model_alias(endpoint)

    client = unillm.Unify(endpoint, temperature=temperature, cache=False)
    try:
        response = client.generate(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
    except litellm.exceptions.AuthenticationError as exc:
        pytest.skip(
            f"{endpoint} credentials unavailable for native image probe "
            f"(public={public_model!r}, transport={transport_model!r}): {exc}",
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in IMAGE_INPUT_REJECTION_MARKERS):
            pytest.fail(
                f"{endpoint} rejected native image input via "
                f"public={public_model!r} transport={transport_model!r}: "
                f"{type(exc).__name__}: {exc}",
            )
        raise

    if not (response or "").strip():
        pytest.fail(
            f"{endpoint} accepted image input but returned empty visible content "
            f"(public={public_model!r}, transport={transport_model!r}). "
            "Check the UniLLM log for reasoning-only output.",
        )

    normalized = response.strip().lower()
    if expected_color not in normalized:
        pytest.fail(
            f"{endpoint} accepted image input but did not identify "
            f"{expected_color!r} (public={public_model!r}, "
            f"transport={transport_model!r}). Response: {response!r}",
        )
    return response
