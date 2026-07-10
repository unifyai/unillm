from pydantic import BaseModel, Field

from unillm.clients.response_format import (
    RESPONSE_FORMAT_SPEC_KEY,
    apply_response_format_transport,
    build_schema_instruction,
    canonicalize_response_format,
    ensure_response_format_spec,
    parse_structured_content,
    validate_against_spec,
)


class Answer(BaseModel):
    name: str
    age: int


class MeetingSummary(BaseModel):
    topic: str = Field(..., description="Main topic")
    summary: str = Field(..., description="Brief summary")


def test_canonicalize_pydantic_model():
    spec = canonicalize_response_format(Answer)
    assert spec is not None
    assert spec.pydantic_model is Answer
    assert spec.openai_type == "json_schema"
    assert "name" in spec.json_schema["properties"]


def test_canonicalize_json_schema_dict():
    envelope = {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "schema": Answer.model_json_schema(),
            "strict": True,
        },
    }
    spec = canonicalize_response_format(envelope)
    assert spec is not None
    assert spec.pydantic_model is None
    assert spec.openai_type == "json_schema"
    assert spec.json_schema == envelope["json_schema"]["schema"]


def test_canonicalize_json_object_dict():
    spec = canonicalize_response_format({"type": "json_object"})
    assert spec is not None
    assert spec.openai_type == "json_object"
    assert spec.json_schema == {"type": "object"}


def test_parse_structured_content_variants():
    parsed, error = parse_structured_content('{"name": "Ada", "age": 42}')
    assert error is None
    assert parsed == {"name": "Ada", "age": 42}

    fenced = 'Here you go:\n```json\n{"name": "Ada", "age": 42}\n```'
    parsed, error = parse_structured_content(fenced)
    assert error is None
    assert parsed["name"] == "Ada"

    prose = 'Sure! {"name": "Ada", "age": 42} thanks'
    parsed, error = parse_structured_content(prose)
    assert error is None
    assert parsed["age"] == 42


def test_validate_against_spec_pydantic():
    spec = canonicalize_response_format(Answer)
    assert validate_against_spec({"name": "Ada", "age": 42}, spec) is None
    assert validate_against_spec({"name": "Ada"}, spec) is not None


def test_apply_response_format_transport_native_is_noop():
    spec = canonicalize_response_format(Answer)
    kw = {
        "response_format": Answer,
        "messages": [{"role": "user", "content": "hello"}],
    }
    apply_response_format_transport(spec, "deepseek", kw)
    assert kw["response_format"] is Answer
    assert kw["messages"] == [{"role": "user", "content": "hello"}]


def test_apply_response_format_transport_hybrid_prompt():
    from unillm.clients import response_format as rf

    original = rf.provider_response_format_mode
    rf.provider_response_format_mode = lambda provider: "hybrid_prompt"
    try:
        spec = canonicalize_response_format(Answer)
        kw = {
            "response_format": Answer,
            "messages": [{"role": "user", "content": "hello"}],
        }
        apply_response_format_transport(spec, "deepseek", kw)
        assert kw["response_format"] == {"type": "json_object"}
        assert kw["messages"][0]["role"] == "system"
        assert "valid JSON only" in kw["messages"][0]["content"]
        assert '"name"' in kw["messages"][0]["content"]
        assert kw[RESPONSE_FORMAT_SPEC_KEY] is spec
    finally:
        rf.provider_response_format_mode = original


def test_apply_response_format_transport_is_idempotent():
    from unillm.clients import response_format as rf

    original = rf.provider_response_format_mode
    rf.provider_response_format_mode = lambda provider: "hybrid_prompt"
    try:
        spec = canonicalize_response_format(Answer)
        kw = {
            "response_format": Answer,
            "messages": [
                {
                    "role": "system",
                    "content": build_schema_instruction(spec.json_schema),
                },
                {"role": "user", "content": "hello"},
            ],
        }
        apply_response_format_transport(spec, "deepseek", kw)
        schema_messages = [
            message
            for message in kw["messages"]
            if message.get("role") == "system"
            and "valid JSON only" in message.get("content", "")
        ]
        assert len(schema_messages) == 1
    finally:
        rf.provider_response_format_mode = original


def test_ensure_response_format_spec_stashes_spec():
    kw = {"response_format": Answer}
    ensure_response_format_spec(kw)
    assert RESPONSE_FORMAT_SPEC_KEY in kw
    assert kw[RESPONSE_FORMAT_SPEC_KEY].pydantic_model is Answer
