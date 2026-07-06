import pytest

_TEST_MODELS = [
    "gpt-5.5@openai",
    "claude-4.8-opus@anthropic",
    "deepseek-v4-max@deepseek",
    "minimax-v3@minimax",
    "mimo-v2.5@xiaomi-mimo",
    "glm-5.2@zai",
    "qwen3.7-plus@qwen",
]


@pytest.fixture(params=_TEST_MODELS)
def model(request) -> str:
    return request.param
