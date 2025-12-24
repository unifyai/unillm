import pytest
import json
from .helpers import new_llm_client


def test_tool_call(model):
    client = new_llm_client(model, stateful=True)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
        ],
        return_full_completion=True,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "add"
    assert json.loads(response.choices[0].message.tool_calls[0].function.arguments) == {
        "a": 1,
        "b": 1,
    }

    # add the result of the tool call to the messages
    client.append_messages(
        [
            {
                "role": "tool",
                "content": "2",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            },
        ],
    )

    response = client.generate()
    assert "2" in response


@pytest.mark.asyncio
async def test_tool_call_async(model):
    client = new_llm_client(model, stateful=True, is_async=True)
    response = await client.generate(
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
        ],
        return_full_completion=True,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "add"
    assert json.loads(response.choices[0].message.tool_calls[0].function.arguments) == {
        "a": 1,
        "b": 1,
    }

    # add the result of the tool call to the messages
    client.append_messages(
        [
            {
                "role": "tool",
                "content": "2",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            },
        ],
    )

    response = await client.generate()
    assert "2" in response
