import pytest
from .helpers import new_llm_client

# Use a value that looks like a raw ID that must be echoed verbatim
TEST_ID = "xK7-pQ9-mR2"

GET_ID_TOOL = {
    "type": "function",
    "function": {
        "name": "get_id",
        "description": "Returns a reference ID.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def test_tool_call(model):
    client = new_llm_client(model, stateful=True)
    response = client.generate(
        messages=[
            {
                "role": "user",
                "content": "Call get_id and tell me the exact ID returned, verbatim.",
            },
        ],
        tools=[GET_ID_TOOL],
        return_full_completion=True,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "get_id"

    # add the result of the tool call to the messages
    client.append_messages(
        [
            {
                "role": "tool",
                "content": TEST_ID,
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            },
        ],
    )

    response = client.generate()
    assert TEST_ID in response


@pytest.mark.asyncio
async def test_tool_call_async(model):
    client = new_llm_client(model, stateful=True, is_async=True)
    response = await client.generate(
        messages=[
            {
                "role": "user",
                "content": "Call get_id and tell me the exact ID returned, verbatim.",
            },
        ],
        tools=[GET_ID_TOOL],
        return_full_completion=True,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "get_id"

    # add the result of the tool call to the messages
    client.append_messages(
        [
            {
                "role": "tool",
                "content": TEST_ID,
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            },
        ],
    )

    response = await client.generate()
    assert TEST_ID in response
