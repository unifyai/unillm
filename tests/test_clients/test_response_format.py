import json
import pytest
from unillm.clients.uni_llm import Unify, AsyncUnify
from pydantic import BaseModel

models = [
    "gpt-4o-mini@openai",
    "claude-4.5-haiku@anthropic",
]


class ResponseFormat(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model", models)
def test_response_format_pydantic(model):
    client = Unify(
        model,
    )
    response = client.generate(
        system_message="You are a helpful assistant that can answer questions about the user's name and age.",
        messages=[
            {"role": "user", "content": "My name is John and I am 30 years old"},
            {
                "role": "user",
                "content": "What is my name and age? respond in JSON format",
            },
        ],
        response_format=ResponseFormat,
    )

    response_json = json.loads(response)

    assert response_json["name"] == "John"
    assert response_json["age"] == 30


@pytest.mark.parametrize("model", models)
@pytest.mark.asyncio
async def test_response_format_pydantic_async(model):
    client = AsyncUnify(
        model,
        service_tier="priority",
    )
    response = await client.generate(
        system_message="You are a helpful assistant that can answer questions about the user's name and age.",
        messages=[
            {"role": "user", "content": "My name is John and I am 30 years old"},
            {
                "role": "user",
                "content": "What is my name and age? respond in JSON format",
            },
        ],
        response_format=ResponseFormat,
    )

    response_json = json.loads(response)

    assert response_json["name"] == "John"
    assert response_json["age"] == 30
