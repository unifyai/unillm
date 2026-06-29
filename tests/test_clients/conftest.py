import os

import pytest

# minimax and xiaomi-mimo have no API key available in CI (or in the cache-refresh
# workflow that records responses), so under the read-only cache gate every call
# is a permanent miss. Skip them unless their key is present, mirroring the
# skipif + _HAS_*_API_KEY convention in test_minimax_v3.py / test_xiaomi_mimo_v25.py.
_HAS_MINIMAX_API_KEY = bool(os.environ.get("MINIMAX_API_KEY"))
_HAS_XIAOMI_MIMO_API_KEY = bool(os.environ.get("XIAOMI_MIMO_API_KEY"))

_TEST_MODELS = [
    "gpt-5.5@openai",
    "claude-4.8-opus@anthropic",
    "deepseek-v4-max@deepseek",
    pytest.param(
        "minimax-v3@minimax",
        marks=pytest.mark.skipif(
            not _HAS_MINIMAX_API_KEY,
            reason="No MiniMax API key available",
        ),
    ),
    pytest.param(
        "mimo-v2.5@xiaomi-mimo",
        marks=pytest.mark.skipif(
            not _HAS_XIAOMI_MIMO_API_KEY,
            reason="No Xiaomi MiMo API key available",
        ),
    ),
]


@pytest.fixture(params=_TEST_MODELS)
def model(request) -> str:
    return request.param
