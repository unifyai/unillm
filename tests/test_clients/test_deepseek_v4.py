import os

import pytest

import unillm
from unillm.endpoints.utils import get_model_alias, list_models

DEEPSEEK_V4_MAX_ENDPOINT = "deepseek-v4-max@deepseek"
DEEPSEEK_V4_MAX_PROVIDER_MODEL = "deepseek/deepseek-v4-pro"
_HAS_DEEPSEEK_API_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))


def test_deepseek_v4_max_alias_registered() -> None:
    assert get_model_alias(DEEPSEEK_V4_MAX_ENDPOINT) == DEEPSEEK_V4_MAX_PROVIDER_MODEL
    assert "deepseek-v4-max" in list_models("deepseek")


@pytest.mark.skipif(not _HAS_DEEPSEEK_API_KEY, reason="No DeepSeek API key available")
def test_sync_deepseek_v4_max_simple_message() -> None:
    client = unillm.Unify(DEEPSEEK_V4_MAX_ENDPOINT, cache=False, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=20,
    )

    assert "Paris" in response


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_DEEPSEEK_API_KEY, reason="No DeepSeek API key available")
async def test_async_deepseek_v4_max_simple_message() -> None:
    client = unillm.AsyncUnify(DEEPSEEK_V4_MAX_ENDPOINT, cache=False, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=20,
    )

    assert "Paris" in response
