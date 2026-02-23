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


# --- Tests for system_message constructor kwarg ---


def test_system_message_constructor_kwarg_state(model):
    """Test that system_message passed to constructor is preserved in client state."""
    system_msg = "You are a helpful assistant that speaks like a pirate."
    client = new_llm_client(model, system_message=system_msg)

    # Verify the system_message property is set
    assert client.system_message == system_msg

    # Verify the messages list has the system message at the front
    assert len(client.messages) >= 1
    assert client.messages[0]["role"] == "system"
    assert client.messages[0]["content"] == system_msg


@pytest.mark.asyncio
async def test_system_message_constructor_kwarg_state_async(model):
    """Test that system_message passed to async constructor is preserved in client state."""
    system_msg = "You are a helpful assistant that speaks like a pirate."
    client = new_llm_client(model, is_async=True, system_message=system_msg)

    # Verify the system_message property is set
    assert client.system_message == system_msg

    # Verify the messages list has the system message at the front
    assert len(client.messages) >= 1
    assert client.messages[0]["role"] == "system"
    assert client.messages[0]["content"] == system_msg


def test_system_message_constructor_used_in_generate(model):
    """Test that system_message from constructor is actually used when generating."""
    # Use a very specific instruction that's easy to verify
    client = new_llm_client(
        model,
        system_message="Always end your response with the word XYZZY.",
    )
    response = client.generate(user_message="Say hello")
    # The model should follow the system instruction
    assert "XYZZY" in response


def test_system_message_constructor_with_messages(model):
    """Test that system_message kwarg works alongside messages kwarg."""
    system_msg = "You are a helpful assistant."
    client = new_llm_client(
        model,
        system_message=system_msg,
        messages=[{"role": "user", "content": "Hello"}],
    )
    # The system_message should be prepended to messages
    assert client.messages[0]["role"] == "system"
    assert client.messages[0]["content"] == system_msg
    # User message should follow
    assert client.messages[1]["role"] == "user"
    assert client.messages[1]["content"] == "Hello"


def test_system_message_constructor_overrides_messages_system(model):
    """Test that explicit system_message kwarg overrides system message in messages."""
    override_msg = "OVERRIDE SYSTEM MESSAGE"
    client = new_llm_client(
        model,
        system_message=override_msg,
        messages=[
            {"role": "system", "content": "Original system message"},
            {"role": "user", "content": "Hello"},
        ],
    )
    # The explicit system_message should replace the one in messages
    assert client.messages[0]["role"] == "system"
    assert client.messages[0]["content"] == override_msg
    assert client.system_message == override_msg


def test_system_message_chained_still_works(model):
    """Test that chained set_system_message() still works after constructor."""
    client = new_llm_client(model)
    system_msg = "You are a helpful assistant."
    client.set_system_message(system_msg)

    assert client.system_message == system_msg
    assert client.messages[0]["role"] == "system"
    assert client.messages[0]["content"] == system_msg


def test_messages_only_no_spurious_system_message(model):
    """Test that passing only messages (no system_message) doesn't create a None system message."""
    client = new_llm_client(
        model,
        messages=[{"role": "user", "content": "What is 1+1?"}],
    )
    # Should not have a system message with None content
    if client.messages and client.messages[0]["role"] == "system":
        assert client.messages[0]["content"] is not None
    # The user message should be there
    assert any(m["role"] == "user" for m in client.messages)


def test_no_kwargs_no_system_message(model):
    """Test that client created with no kwargs has no system message."""
    client = new_llm_client(model)
    # system_message property should be None
    assert client.system_message is None
    # messages should be empty or not start with system
    if client.messages:
        # If there are messages, first one shouldn't be a system message with None content
        if client.messages[0]["role"] == "system":
            assert client.messages[0]["content"] is not None
