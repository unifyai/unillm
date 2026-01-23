"""
Test that Anthropic constrains tool names to the schema.

FINDING: Anthropic does NOT constrain tool names to the schema.
The model can call tools mentioned in the prompt even if they're not in the
`tools` array. This happens both with and without extended thinking.

The fix is to use strict mode (`strict: true`) on tool definitions.
"""

import json
import unillm


TOOL_A = {
    "type": "function",
    "function": {
        "name": "tool_a",
        "description": "A tool.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Semantic search over data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


def test_anthropic_no_tool_name_constraint():
    """
    Anthropic does NOT constrain tool names to schema.

    When tool_choice="required" and the prompt mentions tool_b,
    Claude calls tool_b even though only tool_a is in the schema.
    """
    client = unillm.Unify(
        "claude-4.5-opus@anthropic",
        cache=True,
        stateful=False,
    )

    system_prompt = """Tools:
- tool_a
- tool_b

You MUST call tool_b. Do not call tool_a.
"""

    response = client.generate(
        system_message=system_prompt,
        messages=[{"role": "user", "content": "Do it."}],
        tools=[TOOL_A],  # Only tool_a in schema
        tool_choice="required",
        return_full_completion=True,
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls and len(tool_calls) > 0

    called = tool_calls[0].function.name
    assert called == "tool_a", (
        f"Called '{called}' which is NOT in schema. "
        f"Anthropic does not constrain tool names. Fix: use strict=true."
    )


def test_anthropic_no_tool_name_constraint_with_thinking():
    """
    Anthropic does NOT constrain tool names even with extended thinking.

    Claude picks tool based on task semantics. Prompt describes both search
    and filter, but only search is in schema. For an exact-match task,
    Claude prefers filter - and calls it even though it's not available.

    The thinking block may correctly identify the conflict, but the tool
    call output is not constrained to the schema.
    """
    client = unillm.Unify(
        "claude-4.5-opus@anthropic",
        cache=True,
        stateful=False,
    )

    tools_json = json.dumps(
        {
            "search": "(query: str) - semantic search",
            "filter": "(expr: str) - exact match filter",
        },
        indent=2,
    )

    system_prompt = f"""You are an assistant with access to tools.

Tools (name → signature):
{tools_json}

Tool selection guidance:
- Use `search` for semantic/fuzzy queries
- Use `filter` for exact matches (names, emails, IDs)

Example:
- "Find John Smith" → filter(expr="name == 'John Smith'")
- "Find someone who works in finance" → search(query="works in finance")
"""

    response = client.generate(
        system_message=system_prompt,
        messages=[{"role": "user", "content": "Find Alice Smith."}],
        tools=[TOOL_SEARCH],  # Only search in schema, not filter
        tool_choice="required",
        reasoning_effort="high",
        return_full_completion=True,
    )

    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls or len(tool_calls) == 0:
        return  # Acceptable with tool_choice="auto" (downgraded)

    called = tool_calls[0].function.name
    assert called == "search", (
        f"Called '{called}' which is NOT in schema. "
        f"Anthropic does not constrain tool names even with thinking. "
        f"Fix: use strict=true."
    )
