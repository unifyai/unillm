import pytest

_TEST_MODELS = [
    "gpt-5.2@openai",
    "claude-4.5-opus@anthropic",
]


@pytest.fixture(params=_TEST_MODELS)
def model(request) -> str:
    return request.param
