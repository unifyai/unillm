import pytest
from .helpers import new_llm_client


def test_simple_message(model):
    client = new_llm_client(model)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    assert "Paris" in response


@pytest.mark.asyncio
async def test_simple_message_async(model):
    client = new_llm_client(model, is_async=True)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    assert "Paris" in response
