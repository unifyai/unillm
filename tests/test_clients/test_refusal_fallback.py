import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from unillm.clients.provider_postprocessing import (
    ModelRefusalError,
    apply_postprocessing_pipeline,
    apply_postprocessing_pipeline_async,
    build_refusal_fallback_kw,
    check_safety_refusal,
)

FABLE_5 = "anthropic/claude-fable-5"
OPUS_4_8 = "anthropic/claude-opus-4-8"


def _completion(
    *,
    content: str | None,
    finish_reason: str = "stop",
    model: str = FABLE_5,
):
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model=model,
        object="chat.completion",
    )


def _kw(model: str = FABLE_5) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_check_safety_refusal_maps_fable_to_opus():
    refusal = _completion(content=None, finish_reason="content_filter")
    assert check_safety_refusal(response=refusal, kw=_kw()) == OPUS_4_8


def test_check_safety_refusal_ignores_normal_stop():
    ok = _completion(content="hi", finish_reason="stop")
    assert check_safety_refusal(response=ok, kw=_kw()) is None


def test_check_safety_refusal_ignores_models_without_classifiers():
    refusal = _completion(
        content=None,
        finish_reason="content_filter",
        model=OPUS_4_8,
    )
    assert check_safety_refusal(response=refusal, kw=_kw(model=OPUS_4_8)) is None


def test_build_refusal_fallback_kw_swaps_model_and_transport():
    retry_kw = build_refusal_fallback_kw(kw=_kw(), fallback_model=OPUS_4_8)

    assert retry_kw["model"] == OPUS_4_8
    assert retry_kw["_unillm_transport_model"] == OPUS_4_8
    assert retry_kw["messages"] == _kw()["messages"]


def test_pipeline_retries_refusal_on_fallback_model():
    refusal = _completion(content=None, finish_reason="content_filter")
    fallback_answer = _completion(content="served by opus", model=OPUS_4_8)
    retries = []

    def execute_retry(retry_kw, label):
        retries.append((retry_kw, label))
        return fallback_answer

    result = apply_postprocessing_pipeline(
        refusal,
        kw=_kw(),
        provider="anthropic",
        original_tool_choice=None,
        reasoning_effort=None,
        execute_retry=execute_retry,
    )

    assert result.choices[0].message.content == "served by opus"
    assert len(retries) == 1
    retry_kw, label = retries[0]
    assert retry_kw["model"] == OPUS_4_8
    assert retry_kw["_unillm_transport_model"] == OPUS_4_8
    assert label == "refusal-fallback"


def test_pipeline_raises_when_fallback_also_refuses():
    refusal = _completion(content=None, finish_reason="content_filter")
    fallback_refusal = _completion(
        content=None,
        finish_reason="content_filter",
        model=OPUS_4_8,
    )

    with pytest.raises(ModelRefusalError):
        apply_postprocessing_pipeline(
            refusal,
            kw=_kw(),
            provider="anthropic",
            original_tool_choice=None,
            reasoning_effort=None,
            execute_retry=lambda retry_kw, label: fallback_refusal,
        )


def test_pipeline_leaves_compliant_responses_untouched():
    ok = _completion(content="hi", finish_reason="stop")

    def fail_retry(retry_kw, label):
        raise AssertionError("No retry expected for a compliant response")

    result = apply_postprocessing_pipeline(
        ok,
        kw=_kw(),
        provider="anthropic",
        original_tool_choice=None,
        reasoning_effort=None,
        execute_retry=fail_retry,
    )

    assert result.choices[0].message.content == "hi"


@pytest.mark.asyncio
async def test_async_pipeline_retries_refusal_on_fallback_model():
    refusal = _completion(content=None, finish_reason="content_filter")
    fallback_answer = _completion(content="served by opus", model=OPUS_4_8)

    async def execute_retry(retry_kw, label):
        assert retry_kw["model"] == OPUS_4_8
        assert label == "refusal-fallback"
        return fallback_answer

    result = await apply_postprocessing_pipeline_async(
        refusal,
        kw=_kw(),
        provider="anthropic",
        original_tool_choice=None,
        reasoning_effort=None,
        execute_retry=execute_retry,
    )

    assert result.choices[0].message.content == "served by opus"
