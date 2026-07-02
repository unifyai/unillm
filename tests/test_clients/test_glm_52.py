import os

import pytest

import unillm
from unillm.clients.uni_llm import _prepare_provider_request_kw
from unillm.endpoints.utils import (
    get_model_alias,
    get_model_info,
    get_transport_model_alias,
    list_models,
)
from tests.test_clients.vision_probe_helpers import assert_native_image_input_rejected

GLM_52_ENDPOINT = "glm-5.2@zai"
GLM_52_PROVIDER_MODEL = "z-ai/glm-5.2"
_HAS_OPENROUTER_API_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))


def test_glm_52_alias_registered() -> None:
    assert get_model_alias(GLM_52_ENDPOINT) == GLM_52_PROVIDER_MODEL
    assert "glm-5.2" in list_models("zai")


def test_glm_52_model_info_registered() -> None:
    info = get_model_info(GLM_52_ENDPOINT)
    assert info["max_input_tokens"] == 1_048_576
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_glm_52_openrouter_transport_has_no_direct_api_base() -> None:
    transport_model = get_transport_model_alias(GLM_52_ENDPOINT)
    kw = {
        "model": transport_model,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _prepare_provider_request_kw(kw=kw, provider="zai", stream=False)

    assert transport_model.startswith("openrouter/")
    assert "api_base" not in kw


@pytest.mark.skipif(
    not _HAS_OPENROUTER_API_KEY,
    reason="No OpenRouter API key available",
)
def test_sync_glm_52_simple_message() -> None:
    client = unillm.Unify(GLM_52_ENDPOINT, temperature=0)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _HAS_OPENROUTER_API_KEY,
    reason="No OpenRouter API key available",
)
async def test_async_glm_52_simple_message() -> None:
    client = unillm.AsyncUnify(GLM_52_ENDPOINT, temperature=0)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_completion_tokens=300,
    )

    assert "paris" in response.lower()


@pytest.mark.skipif(
    not _HAS_OPENROUTER_API_KEY,
    reason="No OpenRouter API key available",
)
def test_glm_52_rejects_native_image_input_on_openrouter() -> None:
    assert_native_image_input_rejected(GLM_52_ENDPOINT)
