from .helpers import new_llm_client


def test_prefill_assistant_message(model):
    client = new_llm_client(model)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
            {"role": "assistant", "content": "The answer is 2."},
            {"role": "user", "content": "What was the answer?"},
        ],
    )

    assert "2" in response


def test_prefill_assistant_message_no_user(model):
    client = new_llm_client(model)
    response = client.generate(
        messages=[
            {"role": "user", "content": "What is 1 + 1?"},
            {"role": "assistant", "content": "The answer is 2."},
        ],
    )


def test_prefill_tool_call(model):
    call_id = "fast_tool_123"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fast_tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]

    history = [
        {
            "role": "user",
            "content": (
                "Call the tool `fast_tool` (which just returns a token) and reply with the result only"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "fast_tool", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "fast_tool",
            "content": '"fast"',
        },
    ]

    client = new_llm_client(model)
    response = client.generate(messages=history, tools=tools)
    assert "fast" in response.choices[0].message.content
