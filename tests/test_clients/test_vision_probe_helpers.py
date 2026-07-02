from tests.test_clients.vision_probe_helpers import (
    native_image_messages,
    solid_color_png_b64,
)


def test_solid_color_png_b64_is_valid_png() -> None:
    encoded = solid_color_png_b64(32, 32, (255, 0, 0))
    raw = __import__("base64").b64decode(encoded)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(raw) > 50


def test_native_image_messages_include_image_url_block() -> None:
    encoded = solid_color_png_b64(8, 8, (255, 0, 0))
    messages = native_image_messages(color="red", png_b64=encoded)
    content = messages[0]["content"]
    assert any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in content
    )
