import json
import pytest
from .helpers import new_llm_client
from pydantic import BaseModel


class ResponseFormat(BaseModel):
    name: str
    age: int


def test_response_format_pydantic(model):
    client = new_llm_client(model)
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


@pytest.mark.asyncio
async def test_response_format_pydantic_async(model):
    client = new_llm_client(model, is_async=True)
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
