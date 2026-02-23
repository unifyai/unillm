import json
import pytest
from typing import List
from .helpers import new_llm_client
from pydantic import BaseModel, Field


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


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic response_format enforcement test
# ─────────────────────────────────────────────────────────────────────────────


class MeetingSummary(BaseModel):
    """Structured summary of a meeting."""

    topic: str = Field(..., description="Main topic of the meeting")
    participants: List[str] = Field(..., description="Names of participants")
    action_items: List[str] = Field(..., description="Action items from the meeting")
    summary: str = Field(..., description="Brief natural language summary")


def test_response_format_enforced_despite_contradictory_prompt():
    """response_format must be enforced even when the prompt explicitly asks for prose.

    The system message describes tools (as the production TranscriptManager
    does) but NO tools are passed in the API request.  Anthropic's Claude will
    attempt to call the described tools, triggering unillm's retry path which
    nudges with "Respond with text content only."  That nudge conflicts with
    response_format -- the response must still be valid JSON.

    This reproduces the exact failure pattern from the unity CI: the retry
    after a phantom tool-call returns unstructured prose instead of JSON.
    """
    import unillm

    client = unillm.Unify("claude-4.5-opus@anthropic", cache=False)
    response = client.generate(
        system_message=(
            "You are an assistant specialised in querying communication transcripts.\n"
            "Work strictly through the tools provided.\n\n"
            "Tools (name → argspec):\n"
            "{\n"
            '    "filter_messages": "(*, filter: str, offset: int = 0, limit: int = 100) -> Dict",\n'
            '    "search_messages": "(*, references: Dict[str, str], k: int = 10) -> Dict"\n'
            "}\n\n"
            "Use the tools to gather context before answering."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "What topics were discussed in recent emails? "
                    "Give me a thorough summary."
                ),
            },
        ],
        # No tools registered — any tool-call attempt triggers the retry nudge
        # "Respond with text content only", which contradicts response_format.
        tools=[],
        response_format=MeetingSummary,
    )

    # The response_format parameter should guarantee valid JSON matching the
    # MeetingSummary schema.  If the provider doesn't enforce response_format,
    # the LLM will instead follow the system prompt's tool descriptions and
    # return something unrelated (e.g. tool-call arguments, or free-form prose
    # after unillm's "Respond with text content only" retry nudge).
    parsed = json.loads(response)
    summary = MeetingSummary.model_validate(parsed)
    assert summary.topic, "topic is empty"
    assert isinstance(summary.participants, list), "participants is not a list"
    assert len(summary.participants) >= 1, "participants is empty"
    assert isinstance(summary.action_items, list), "action_items is not a list"
    assert summary.summary, "summary is empty"
