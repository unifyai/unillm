import pytest

_TEST_MODELS = [
    "gpt-5.5@openai",
    "claude-4.8-opus@anthropic",
    "deepseek-v4-max@deepseek",
]


@pytest.fixture(params=_TEST_MODELS)
def model(request) -> str:
    return request.param
