import pytest
from unillm.clients.uni_llm import Unify, AsyncUnify

models = [
    "gpt-4o-mini@openai",
    "claude-3.5-haiku@anthropic",
]


@pytest.mark.parametrize("model", models)
def test_simple_message(model):
    client = Unify(
        model,
    )
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    assert "Paris" in response


@pytest.mark.parametrize("model", models)
@pytest.mark.asyncio
async def test_simple_message_async(model):
    client = AsyncUnify(
        model,
    )
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    assert "Paris" in response
