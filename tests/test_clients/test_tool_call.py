import pytest
from .helpers import new_llm_client


CODEWORD = "ZEBRA_42_PHOENIX"

REVEAL_CODEWORD_TOOL = {
    "type": "function",
    "function": {
        "name": "reveal_codeword",
        "description": "Reveals the secret codeword. Must be called to obtain it.",
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
                "content": "Call the reveal_codeword tool and tell me the exact codeword.",
            },
        ],
        tools=[REVEAL_CODEWORD_TOOL],
        return_full_completion=True,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "reveal_codeword"

    # add the result of the tool call to the messages
    client.append_messages(
        [
            {
                "role": "tool",
                "content": CODEWORD,
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            },
        ],
    )

    response = client.generate()
    assert CODEWORD in response


@pytest.mark.asyncio
async def test_tool_call_async(model):
    client = new_llm_client(model, stateful=True, is_async=True)
    response = await client.generate(
        messages=[
            {
                "role": "user",
                "content": "Call the reveal_codeword tool and tell me the exact codeword.",
            },
        ],
        tools=[REVEAL_CODEWORD_TOOL],
        return_full_completion=True,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "reveal_codeword"

    # add the result of the tool call to the messages
    client.append_messages(
        [
            {
                "role": "tool",
                "content": CODEWORD,
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            },
        ],
    )

    response = await client.generate()
    assert CODEWORD in response
