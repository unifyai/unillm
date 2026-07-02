import json
from types import SimpleNamespace

from pydantic import BaseModel

from unillm.clients.provider_postprocessing import (
    build_response_format_retry_kw,
    check_response_format_compliance,
)
from unillm.clients.response_format import (
    RESPONSE_FORMAT_SPEC_KEY,
    canonicalize_response_format,
)


class Decision(BaseModel):
    delay: float
    content: str


def _completion(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_check_response_format_compliance_deepseek_style_kw():
    spec = canonicalize_response_format(Decision)
    kw = {
        "response_format": {"type": "json_object"},
        RESPONSE_FORMAT_SPEC_KEY: spec,
        "messages": [{"role": "user", "content": "hello"}],
    }
    needs_retry, error, returned_spec = check_response_format_compliance(
        response=_completion("not json at all"),
        kw=kw,
    )
    assert needs_retry is True
    assert error is not None
    assert returned_spec is spec


def test_check_response_format_compliance_valid_json():
    spec = canonicalize_response_format(Decision)
    kw = {RESPONSE_FORMAT_SPEC_KEY: spec}
    payload = json.dumps({"delay": 1.0, "content": "hello"})
    needs_retry, error, returned_spec = check_response_format_compliance(
        response=_completion(payload),
        kw=kw,
    )
    assert needs_retry is False
    assert error is None
    assert returned_spec is None


def test_build_response_format_retry_kw_restores_source():
    spec = canonicalize_response_format(Decision)
    kw = {
        "response_format": {"type": "json_object"},
        RESPONSE_FORMAT_SPEC_KEY: spec,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "tool_choice": "auto",
    }
    retry_kw = build_response_format_retry_kw(
        kw=kw,
        response=_completion("plain prose"),
        validation_error="Response is not valid JSON",
        response_format_spec=spec,
    )
    assert retry_kw["response_format"] is Decision
    assert retry_kw[RESPONSE_FORMAT_SPEC_KEY] is spec
    assert "tools" not in retry_kw
    assert retry_kw["messages"][-1]["role"] == "user"
