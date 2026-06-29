# global
import abc
import asyncio
import copy
import inspect
import logging
import os
import re

from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Type,
    Union,
)

import litellm

# local
import unify
from openai._types import Headers
from ..costs import compute_cost_from_response
from ..limit_hooks import (
    check_limits,
    check_limits_sync,
    is_limit_check_enabled,
    LimitCheckRequest,
    SpendingLimitExceededError,
)

_LOGGER = logging.getLogger("unillm")

_OPENAI_RESPONSES_BRIDGE_MODEL_PREFIX = "openai/responses/"
_OPENAI_RESPONSES_BRIDGE_ALLOWED_PARAMS = ("parallel_tool_calls", "tool_choice")
_OPENAI_GPT_MINOR_VERSION_RE = re.compile(r"^gpt-5\.(?P<minor>\d+)(?:[-.].*)?$")


def _enforce_parallel_tool_call_response_limit(
    chat_completion: Any,
    parallel_tool_calls: Optional[bool],
) -> bool:
    if parallel_tool_calls is not False or chat_completion is None:
        return False

    try:
        message = chat_completion.choices[0].message
        tool_calls = message.tool_calls or []
    except Exception:
        return False

    if len(tool_calls) <= 1:
        return False

    message.tool_calls = tool_calls[:1]
    return True


def _normalize_assistant_message_content(chat_completion: Any) -> bool:
    try:
        content = chat_completion.choices[0].message.content
    except Exception:
        return False

    if not isinstance(content, str):
        return False

    normalized = content.strip()
    if normalized == content:
        return False

    chat_completion.choices[0].message.content = normalized
    return True


def _safe_deduct_credits(
    amount: float,
    *,
    api_key: str | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    provider_cost: float | None = None,
) -> None:
    """Deduct credits with ledger metadata from billing context."""
    from ..billing_context import get_billing_context

    ctx = get_billing_context()
    detail: dict = {}
    if model:
        detail["model"] = model
    if prompt_tokens is not None:
        detail["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        detail["completion_tokens"] = completion_tokens
    if provider_cost is not None:
        detail["provider_cost"] = provider_cost
    if ctx.source:
        detail["source"] = ctx.source
    if ctx.label:
        detail["label"] = ctx.label

    try:
        unify.deduct_credits(
            amount,
            api_key=api_key,
            category="llm",
            assistant_id=ctx.assistant_id,
            user_id=ctx.user_id,
            organization_id=ctx.organization_id,
            description="Assistant work",
            detail=detail or None,
        )
    except Exception:
        _LOGGER.warning("Failed to deduct credits: $%.6f", amount, exc_info=True)


def _canonical_model_for_accounting(model: str | None) -> str:
    """Return the provider model name used for pricing, limits, and ledger metadata."""
    if not model:
        return ""
    if model.startswith(_OPENAI_RESPONSES_BRIDGE_MODEL_PREFIX):
        return model.removeprefix(_OPENAI_RESPONSES_BRIDGE_MODEL_PREFIX)
    return model


def _is_openai_gpt_responses_tool_model(model: str) -> bool:
    """Return whether an OpenAI model needs Responses for tools with reasoning."""
    match = _OPENAI_GPT_MINOR_VERSION_RE.match(model)
    return bool(match and int(match.group("minor")) >= 4)


def _copy_tool_for_responses_bridge(tool: Any) -> Any:
    """Copy a chat tool while placing strictness where LiteLLM's bridge reads it."""
    if not isinstance(tool, dict):
        return tool

    tool_copy = dict(tool)
    function = tool_copy.get("function")
    if tool_copy.get("type") == "function" and isinstance(function, dict):
        function_copy = dict(function)
        if "strict" in tool_copy and "strict" not in function_copy:
            function_copy["strict"] = tool_copy["strict"]
        tool_copy["function"] = function_copy
        tool_copy.pop("strict", None)
    return tool_copy


def _copy_tools_for_responses_bridge(tools: Iterable[Any] | None) -> list[Any] | None:
    """Return response-bridge-compatible tool definitions without mutating callers."""
    if tools is None:
        return None
    return [_copy_tool_for_responses_bridge(tool) for tool in tools]


def _allow_openai_params(kw: dict, params: Iterable[str]) -> None:
    """Preserve OpenAI-compatible params that LiteLLM provider metadata omits."""
    current = kw.get("allowed_openai_params")
    if current is None:
        allowed: set[str] = set()
    elif isinstance(current, str):
        allowed = {current}
    else:
        allowed = {str(param) for param in current}

    allowed.update(params)
    kw["allowed_openai_params"] = sorted(allowed)


def _allow_responses_bridge_params(kw: dict) -> None:
    """Preserve chat tool controls that LiteLLM otherwise drops before bridging."""
    _allow_openai_params(kw, _OPENAI_RESPONSES_BRIDGE_ALLOWED_PARAMS)


def _xiaomi_mimo_token_plan_api_base() -> str | None:
    api_key = os.environ.get("XIAOMI_MIMO_API_KEY", "")
    for region in ("sgp", "cn", "ams"):
        if api_key.startswith(f"tp-{region}"):
            return f"https://token-plan-{region}.xiaomimimo.com/v1"
    return None


def _apply_deepseek_v4_reasoning_effort(kw: dict, model: str) -> None:
    """Forward DeepSeek V4 graded ``reasoning_effort`` via ``extra_body``.

    WORKAROUND (BerriAI/litellm#27439): litellm's
    ``DeepSeekChatConfig.map_openai_params()`` discards the ``reasoning_effort``
    *value* for DeepSeek — it pops the param and only flips
    ``thinking={"type": "enabled"}``, so ``high`` / ``max`` / ``xhigh`` all
    collapse to DeepSeek's server default (``high``) with no way to request
    ``max``. The upstream fix is in flight via BerriAI/litellm#28702 (which
    supersedes #28138 / #28134; the earlier #27445 / #27829 were flawed) but is
    not yet released (still absent in litellm 1.88.0 and v1.89.0).

    Until that PR — or an equivalent — lands and we bump litellm, we forward the
    graded value ourselves through ``extra_body``, which litellm passes to the
    provider verbatim (bypassing ``map_openai_params``). ``thinking`` stays a
    top-level param so litellm still treats the turn as thinking-mode and keeps
    its multi-turn ``reasoning_content`` handling intact.

    REMOVE this function and its call site in ``_prepare_provider_request_kw``
    once litellm forwards ``reasoning_effort`` for DeepSeek natively.
    """
    effort = kw.get("reasoning_effort")
    if effort is None:
        return
    model_l = model.lower()
    # Always-on reasoners (deepseek-reasoner / R1) reject these fields.
    if "reasoner" in model_l or "r1" in model_l:
        return
    # Graded effort is a V4+ opt-in feature; older chat models ignore it.
    if "v4" not in model_l:
        return

    effort_l = str(effort).lower()
    extra_body = dict(kw.get("extra_body") or {})
    if effort_l == "none":
        # Disabled must go through extra_body: litellm only re-adds ``thinking``
        # when its type is "enabled", silently dropping the disabled form.
        kw.pop("reasoning_effort", None)
        extra_body["thinking"] = {"type": "disabled"}
        kw["extra_body"] = extra_body
        return

    # DeepSeek accepts only "high" and "max"; normalize per its docs.
    normalized = {
        "low": "high",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }.get(effort_l)
    if normalized is None:
        return  # unknown value: leave it for litellm to handle

    kw.pop("reasoning_effort", None)
    kw["thinking"] = {"type": "enabled"}
    extra_body["reasoning_effort"] = normalized
    kw["extra_body"] = extra_body


def _prepare_provider_request_kw(
    *,
    kw: dict,
    provider: str,
    stream: bool,
) -> str:
    """Apply provider transport adaptations and return the accounting model."""
    model = str(kw.get("model") or "")
    tools = kw.get("tools")

    if (
        provider == "openai"
        and not stream
        and tools
        and kw.get("reasoning_effort") is not None
        and _is_openai_gpt_responses_tool_model(model)
    ):
        kw["model"] = f"{_OPENAI_RESPONSES_BRIDGE_MODEL_PREFIX}{model}"
        kw["tools"] = _copy_tools_for_responses_bridge(tools)
        _allow_responses_bridge_params(kw)

    if provider == "minimax" and kw.get("api_base") is None:
        kw["api_base"] = "https://api.minimax.io/v1"

    if provider == "xiaomi-mimo" and kw.get("api_base") is None:
        api_base = _xiaomi_mimo_token_plan_api_base()
        if api_base is not None:
            kw["api_base"] = api_base
    if provider == "xiaomi-mimo" and tools:
        _allow_openai_params(kw, ("tools", "tool_choice"))
        kw.setdefault("tool_choice", "auto")
        extra_body = dict(kw.get("extra_body") or {})
        extra_body["thinking"] = {"type": "disabled"}
        kw["extra_body"] = extra_body

    if provider == "deepseek":
        _apply_deepseek_v4_reasoning_effort(kw, model)

    return _canonical_model_for_accounting(str(kw.get("model") or model))


def _request_kw_for_event(kw: dict, accounting_model: str) -> dict:
    """Return event metadata with the provider model as the primary model."""
    transport_model = kw.get("model")
    if transport_model == accounting_model:
        return kw

    event_kw = dict(kw)
    event_kw["model"] = accounting_model
    event_kw["transport_model"] = transport_model
    return event_kw


from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel
from typing_extensions import Self
from .provider_preprocessing import apply_provider_preprocessing

from ..caching import _get_cache, _write_to_cache, is_caching_enabled
from ..cache_events import _emit_cache_event
from ..cost_tracker import CostEvent, _emit_cost_event
from ..llm_events import _emit_llm_event, LLMEvent
from ..helpers import (
    _default,
    get_seed,
    retry_transient_400_async,
    retry_transient_400_sync,
    UNSET,
)
from ..clients.base import _Client
from ..endpoints.utils import get_model_alias
from ..logger import (
    write_request_pending,
    append_response_and_finalize,
    llm_span,
    set_span_response,
)
from ..types import Prompt, PromptCacheParam, VALID_CACHE_VALUES
from .shared_session import get_shared_session
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


class _UniClient(_Client, abc.ABC):
    def __init__(
        self,
        endpoint: str,
        *,
        system_message: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: Optional[bool] = False,
        stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = None,
        service_tier: Optional[str] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        api_key: Optional[str] = None,
        # python client arguments
        stateful: bool = False,
        return_full_completion: bool = False,
        cache: Optional[Union[bool, str]] = None,
        cache_backend: Optional[str] = None,
        prompt_caching: Optional[PromptCacheParam] = UNSET,  # type: ignore[assignment]
        origin: Optional[str] = None,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        **kwargs,
    ):
        """Initialize the Uni LLM Unify client.

        Args:
            endpoint: Endpoint name in OpenAI API format:
            <model_name>@<provider_name>
            Defaults to None.

            system_message: An optional string containing the system message. This
            always appears at the beginning of the list of messages.

            messages: A list of messages comprising the conversation so far. This will
            be appended to the system_message if it is not None, and any user_message
            will be appended if it is not None.

            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

            logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a JSON object that maps tokens (specified by their token
            ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior
            to sampling. The exact effect will vary per model, but values between -1 and
            1 should decrease or increase likelihood of selection; values like -100 or
            100 should result in a ban or exclusive selection of the relevant token.

            logprobs: Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token returned in the
            content of message.

            top_logprobs: An integer between 0 and 20 specifying the number of most
            likely tokens to return at each token position, each with an associated log
            probability. logprobs must be set to true if this parameter is used.

            max_completion_tokens: The maximum number of tokens that can be generated in
            the chat completion. The total length of input tokens and generated tokens
            is limited by the model's context length. Defaults to the provider's default
            max_completion_tokens when the value is None.

            n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep n as 1 to minimize costs.

            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied JSON
            schema. Learn more in the Structured Outputs guide. Setting to
            `{ "type": "json_object" }` enables JSON mode, which ensures the message the
            model generates is valid JSON.

            seed: If specified, a best effort attempt is made to sample
            deterministically, such that repeated requests with the same seed and
            parameters should return the same result. Determinism is not guaranteed, and
            you should refer to the system_fingerprint response parameter to monitor
            changes in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            stream: If True, generates content as a stream. If False, generates content
            as a single response. Defaults to False.

            stream_options: Options for streaming response. Only set this when you set
            stream: true.

            temperature:  What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            It is generally recommended to alter this or top_p, but not both.
            Defaults to the provider's default max_completion_tokens when the value is
            None.

            top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered. Generally recommended to alter this or temperature, but not
            both.

            tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the
            model. none means the model will not call any tool and instead generates a
            message. auto means the model can pick between generating a message or
            calling one or more tools. required means the model must call one or more
            tools. Specifying a particular tool via
            `{ "type": "function", "function": {"name": "my_function"} }`
            forces the model to call that tool.
            none is the default when no tools are present. auto is the default if tools
            are present.

            parallel_tool_calls: Whether to enable parallel function calling during tool
            use.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            return_full_completion: If False, only return the message content
            chat_completion.choices[0].message.content.strip(" ") from the OpenAI
            return. Otherwise, the full response chat_completion is returned.
            Defaults to False.

            cache: If True, then the arguments will be stored in a local cache file, and
            any future calls with identical arguments will read from the cache instead
            of running the LLM query. If "write" then the cache will only be written
            to, if "read" then the cache will be read from if a cache is available but
            will not write, and if "read-only" then the argument must be present in the
            cache, else an exception will be raised. Finally, an appending "-closest"
            will read the closest match from the cache, and overwrite it if cache writing
            is enabled. This argument only has any effect when stream=False.

            origin: An optional string tag for identifying the origin of LLM
            calls in log files, OTel spans, and events. Useful when multiple
            agents or subsystems share the same process and you need to tell
            their logs apart (e.g. ``"AgentA"``, ``"planner"``).

            extra_headers: Additional "passthrough" headers for the request which are
            provider-specific, and are not part of the OpenAI standard. They are handled
            by the provider-specific API.

            kwargs: Additional "passthrough" JSON properties for the body of the
            request, which are provider-specific, and are not part of the OpenAI
            standard. They will be handled by the provider-specific API.

        Raises:
            UnifyError: If the API key is missing.
        """
        self._base_constructor_args = dict(
            system_message=system_message,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            service_tier=service_tier,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            api_key=api_key,
            # python client arguments
            stateful=stateful,
            return_full_completion=return_full_completion,
            cache=cache,
            cache_backend=cache_backend,
            prompt_caching=None if prompt_caching is UNSET else prompt_caching,
            origin=origin,
            # passthrough arguments
            extra_headers=extra_headers,
            **kwargs,
        )
        super().__init__(**self._base_constructor_args)
        self._constructor_args = dict(
            endpoint=endpoint,
            **self._base_constructor_args,
        )
        self.set_endpoint(endpoint)

    # Settable Properties #
    # --------------------#

    @property
    def endpoint(self) -> str:
        """
        Get the endpoint name.

        Returns:
            The endpoint name.
        """
        return self._endpoint

    # Setters #
    # --------#

    def set_endpoint(self, value: str) -> Self:
        """
        Set the endpoint name.  # noqa: DAR101.

        Args:
            value: The endpoint name.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._model_alias = get_model_alias(value)
        self._endpoint = value
        self._model, self._provider = value.split("@")
        return self

    @staticmethod
    def _handle_kw(
        prompt,
        endpoint,
        stream,
        stream_options,
    ):
        prompt_dict = prompt.components
        kw = dict(
            model=get_model_alias(endpoint),
            **prompt_dict,
            stream=stream,
            stream_options=stream_options,
        )
        return {k: v for k, v in kw.items() if v is not None}

    # Representation #
    # ---------------#

    def __repr__(self):
        return "{}(endpoint={})".format(self.__class__.__name__, self._endpoint)

    def __str__(self):
        return "{}(endpoint={})".format(self.__class__.__name__, self._endpoint)

    # --------------------------------------------------------------------- #
    #  Helper(s) – keep the public surface of _UniClient unchanged          #
    # --------------------------------------------------------------------- #

    def _append_to_history(self, assistant_msg: dict) -> None:
        """Append a single assistant message to the internal history."""
        if self._messages is None:
            self._messages = []
        self._messages.append(assistant_msg)

    # --------------------------------------------------------------------- #
    #  Streaming wrappers                                                   #
    # --------------------------------------------------------------------- #

    def _wrap_sync_stream(
        self,
        stream: Generator[Any, None, None],
        *,
        stateful: bool,
        return_full_completion: bool,
    ) -> Generator[Any, None, None]:
        """
        Proxy a *synchronous* stream, collecting the emitted content so we can
        update or clear the history once (and only once) when the stream
        finishes.
        """
        collected: list[str] = []

        def _take(item: Any) -> str:
            if return_full_completion:
                # ChatCompletionChunk → extract incremental delta
                try:
                    delta = item.choices[0].delta.content
                    return delta or ""  # may be None
                except Exception:  # noqa: BLE001
                    return ""
            return str(item)

        try:
            for chunk in stream:
                if stateful:
                    piece = _take(chunk)
                    if piece:
                        collected.append(piece)
                yield chunk
        finally:  # executes on normal end *and* on .close()
            if stateful:
                if collected:
                    self._append_to_history(
                        {"role": "assistant", "content": "".join(collected).strip()},
                    )
            elif self._messages:
                self._messages.clear()

    def _wrap_async_stream(  # noqa: WPS231
        self,
        stream: AsyncGenerator[Any, None],
        *,
        stateful: bool,
        return_full_completion: bool,
    ) -> AsyncGenerator[Any, None]:
        """
        Same as `_wrap_sync_stream` but for *async* generators.
        """
        collected: list[str] = []

        async def _internal():
            async for chunk in stream:
                if stateful:
                    if return_full_completion:
                        try:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                collected.append(delta)
                        except Exception:  # noqa: BLE001
                            pass
                    else:
                        collected.append(str(chunk))
                yield chunk

            # async-generator exhausted
            if stateful:
                if collected:
                    self._append_to_history(
                        {"role": "assistant", "content": "".join(collected).strip()},
                    )
            elif self._messages:
                self._messages.clear()

        return _internal()

    # --------------------------------------------------------------------- #
    #  Single place that decides which helper to call                       #
    # --------------------------------------------------------------------- #

    def _apply_stateful_logic(  # noqa: WPS231,WPS211
        self,
        *,
        response: Any,
        stateful: bool,
        was_stream: bool,
        return_full_completion: bool,
    ) -> Any:
        """
        Ensures the conversation history is updated (or cleared) **once** per
        call, for all four modalities.
        """

        if was_stream:

            if inspect.iscoroutine(response):

                async def _await_then_wrap(coro):
                    inner = await coro  # real result from _generate

                    # inner is expected to be async-gen, but handle sync-gen too
                    if inspect.isasyncgen(inner):
                        return self._wrap_async_stream(
                            inner,
                            stateful=stateful,
                            return_full_completion=return_full_completion,
                        )
                    if isinstance(inner, (list, tuple)) or inspect.isgenerator(inner):
                        # rare case: provider gave back sync generator
                        return self._wrap_sync_stream(
                            inner,
                            stateful=stateful,
                            return_full_completion=return_full_completion,
                        )
                    # not a generator at all – treat like non-stream single result
                    return self._apply_stateful_logic(
                        response=inner,
                        stateful=stateful,
                        was_stream=False,
                        return_full_completion=return_full_completion,
                    )

                # Return *the coroutine itself* so the caller still needs to `await`
                return _await_then_wrap(response)

            # choose correct wrapper (sync vs async)
            if inspect.isasyncgen(response):
                return self._wrap_async_stream(
                    response,
                    stateful=stateful,
                    return_full_completion=return_full_completion,
                )
            return self._wrap_sync_stream(
                response,
                stateful=stateful,
                return_full_completion=return_full_completion,
            )

        # ───── coroutine (async-non-stream) path ──────────────────────────
        if inspect.iscoroutine(response):

            # 1. Capture the index where the assistant reply will go, but
            #    **do not** insert the placeholder yet.  This avoids sending an
            #    empty assistant message to the LLM while still fixing order.
            placeholder_idx: int | None = len(self._messages) if stateful else None

            # 2. Await the real coroutine and overwrite the placeholder in-place
            async def _await_and_process(coro: Coroutine[Any, Any, Any]):
                try:
                    res = await coro
                except Exception:
                    # nothing was inserted yet, just re-raise
                    raise

                if stateful and placeholder_idx is not None:
                    # Always store full message (preserves thinking blocks for Claude, etc.)
                    self._messages.insert(
                        placeholder_idx,
                        res.choices[0].message.model_dump(warnings=False),
                    )
                elif self._messages:
                    self._messages.clear()

                # Extract content if caller doesn't want full completion
                if not return_full_completion:
                    content = res.choices[0].message.content
                    return content.strip(" ") if content else ""
                return res

            return _await_and_process(response)

        # ---------- non-streaming path ----------
        if stateful:
            # Always store full message (preserves thinking blocks for Claude, etc.)
            assistant_dict = response.choices[0].message.model_dump(warnings=False)
            self._append_to_history(assistant_dict)
        elif self._messages:
            self._messages.clear()

        # Extract content if caller doesn't want full completion
        if not return_full_completion:
            content = response.choices[0].message.content
            return content.strip(" ") if content else ""
        return response

    # Generate #
    # ---------#

    def generate(
        self,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        *,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        # python client arguments
        stateful: Optional[bool] = None,
        return_full_completion: Optional[bool] = None,
        cache: Optional[Union[bool, str]] = None,
        cache_backend: Optional[str] = None,
        prompt_caching: Optional[PromptCacheParam] = None,
        origin: Optional[str] = None,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        service_tier: Optional[str] = None,
        **kwargs,
    ):
        """Generate a ChatCompletion response for the specified endpoint,
        from the provided query parameters.

        Args:
            user_message: A string containing the user message.
            If provided, messages must be None.

            system_message: An optional string containing the system message. This
            always appears at the beginning of the list of messages.

            messages: A list of messages comprising the conversation so far. This will
            be appended to the system_message if it is not None, and any user_message
            will be appended if it is not None.

            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

            logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a JSON object that maps tokens (specified by their token
            ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior
            to sampling. The exact effect will vary per model, but values between -1 and
            1 should decrease or increase likelihood of selection; values like -100 or
            100 should result in a ban or exclusive selection of the relevant token.

            logprobs: Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token returned in the
            content of message.

            top_logprobs: An integer between 0 and 20 specifying the number of most
            likely tokens to return at each token position, each with an associated log
            probability. logprobs must be set to true if this parameter is used.

            max_completion_tokens: The maximum number of tokens that can be generated in
            the chat completion. The total length of input tokens and generated tokens
            is limited by the model's context length. Defaults value is None. Uses the
            provider's default max_completion_tokens when None is explicitly passed.

            n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep n as 1 to minimize costs.

            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied JSON
            schema. Learn more in the Structured Outputs guide. Setting to
            `{ "type": "json_object" }` enables JSON mode, which ensures the message the
            model generates is valid JSON.

            seed: If specified, a best effort attempt is made to sample
            deterministically, such that repeated requests with the same seed and
            parameters should return the same result. Determinism is not guaranteed, and
            you should refer to the system_fingerprint response parameter to monitor
            changes in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            stream: If True, generates content as a stream. If False, generates content
            as a single response. Defaults to False.

            stream_options: Options for streaming response. Only set this when you set
            stream: true.

            temperature:  What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            It is generally recommended to alter this or top_p, but not both.
            Default value is 1.0. Defaults to the provider's default temperature when
            None is explicitly passed.

            top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered. Generally recommended to alter this or temperature, but not
            both.

            tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the
            model. none means the model will not call any tool and instead generates a
            message. auto means the model can pick between generating a message or
            calling one or more tools. required means the model must call one or more
            tools. Specifying a particular tool via
            `{ "type": "function", "function": {"name": "my_function"} }`
            forces the model to call that tool.
            none is the default when no tools are present. auto is the default if tools
            are present.

            parallel_tool_calls: Whether to enable parallel function calling during tool
            use.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            return_full_completion: If False, only return the message content
            chat_completion.choices[0].message.content.strip(" ") from the OpenAI
            return. Otherwise, the full response chat_completion is returned.
            Defaults to False.

            cache: If True, then the arguments will be stored in a local cache file, and
            any future calls with identical arguments will read from the cache instead
            of running the LLM query. If "write" then the cache will only be written
            to, if "read" then the cache will be read from if a cache is available but
            will not write, and if "read-only" then the argument must be present in the
            cache, else an exception will be raised. Finally, an appending "-closest"
            will read the closest match from the cache, and overwrite it if cache writing
            is enabled. This argument only has any effect when stream=False.

            origin: An optional string tag for identifying the origin of this
            LLM call in log files, OTel spans, and events.

            extra_headers: Additional "passthrough" headers for the request which are
            provider-specific, and are not part of the OpenAI standard. They are handled
            by the provider-specific API.

            kwargs: Additional "passthrough" JSON properties for the body of the
            request, which are provider-specific, and are not part of the OpenAI
            standard. They will be handled by the provider-specific API.

        Returns:
            If stream is True, returns a generator yielding chunks of content.
            If stream is False, returns a single string response.

        Raises:
            UnifyError: If an error occurs during content generation.
        """
        system_message = _default(system_message, self._system_message)
        messages = _default(messages, self._messages)
        stateful = _default(stateful, self._stateful)
        if messages:
            sys_msg_inside = any(msg["role"] == "system" for msg in messages)
            if not sys_msg_inside and system_message is not None:
                messages = [
                    {"role": "system", "content": system_message},
                ] + messages
            if user_message is not None:
                messages += [{"role": "user", "content": user_message}]
            self._messages = list(messages)  # Copy to avoid mutating user's list
        else:
            messages = list()
            if system_message is not None:
                messages += [{"role": "system", "content": system_message}]
            if user_message is not None:
                messages += [{"role": "user", "content": user_message}]
            self._messages = messages
        return_full_completion = (
            True
            if _default(tools, self._tools)
            else _default(return_full_completion, self._return_full_completion)
        )
        cache = _default(cache, self._cache)
        assert cache in VALID_CACHE_VALUES
        ret = self._generate(
            messages=messages,
            frequency_penalty=_default(frequency_penalty, self._frequency_penalty),
            logit_bias=_default(logit_bias, self._logit_bias),
            logprobs=_default(logprobs, self._logprobs),
            top_logprobs=_default(top_logprobs, self._top_logprobs),
            max_completion_tokens=_default(
                max_completion_tokens,
                self._max_completion_tokens,
            ),
            n=_default(n, self._n),
            presence_penalty=_default(presence_penalty, self._presence_penalty),
            response_format=_default(response_format, self._response_format),
            seed=_default(_default(seed, self._seed), get_seed()),
            stop=_default(stop, self._stop),
            stream=_default(stream, self._stream),
            stream_options=_default(stream_options, self._stream_options),
            temperature=_default(temperature, self._temperature),
            top_p=_default(top_p, self._top_p),
            service_tier=_default(service_tier, self._service_tier),
            tools=_default(tools, self._tools),
            tool_choice=_default(tool_choice, self._tool_choice),
            parallel_tool_calls=_default(
                parallel_tool_calls,
                self._parallel_tool_calls,
            ),
            reasoning_effort=_default(reasoning_effort, self._reasoning_effort),
            # python client arguments
            return_full_completion=return_full_completion,
            cache=_default(cache, is_caching_enabled()),
            cache_backend=_default(cache_backend, self._cache_backend),
            prompt_caching=_default(prompt_caching, self._prompt_caching),
            origin=_default(origin, self._origin),
            # passthrough arguments
            extra_headers=_default(extra_headers, self._extra_headers),
            **kwargs,
        )
        ret = self._apply_stateful_logic(
            response=ret,
            stateful=stateful,
            was_stream=_default(stream, self._stream),
            return_full_completion=return_full_completion,
        )
        return ret


class Unify(_UniClient):
    """Sync client for LLM inference via the model@provider endpoint format."""

    def _generate_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # stream
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        # python client arguments
        return_full_completion: bool,
        prompt_caching: Optional[PromptCacheParam],
        origin: Optional[str] = None,
    ) -> Generator[str, None, None]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=True,
            stream_options=stream_options,
        )
        # Apply provider-specific preprocessing (before cache, on a copy of messages)
        apply_provider_preprocessing(kw, self._provider, prompt_caching)
        accounting_model = _prepare_provider_request_kw(
            kw=kw,
            provider=self._provider,
            stream=True,
        )

        # Check spending limits before starting stream
        if is_limit_check_enabled():
            limit_request = LimitCheckRequest(
                model=accounting_model,
                endpoint=endpoint,
            )
            limit_result = check_limits_sync(limit_request)
            if not limit_result.allowed:
                raise SpendingLimitExceededError(limit_result)

        # Track usage from the stream for cost deduction
        usage_info = None
        llm_error: BaseException | None = None
        provider_cost: float | None = None
        billed_cost: float | None = None

        try:
            chat_completion = retry_transient_400_sync(
                lambda: litellm.completion(**kw),
            )
            for chunk in chat_completion:
                # Capture usage if present in the chunk (final chunk with include_usage)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

                if return_full_completion:
                    content = chunk
                else:
                    content = chunk.choices[0].delta.content  # type: ignore[union-attr]    # noqa: E501
                if content is not None:
                    yield content
        except litellm.exceptions.APIError as e:
            llm_error = Exception(e.message)
            raise llm_error
        except Exception as e:
            llm_error = e
            raise
        finally:
            # Deduct credits based on usage after streaming completes
            if usage_info is not None:
                prompt_tokens = getattr(usage_info, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage_info, "completion_tokens", 0) or 0
                if prompt_tokens > 0 or completion_tokens > 0:
                    from ..costs import compute_cost, get_cost_margin

                    provider_cost = compute_cost(
                        accounting_model,
                        prompt_tokens,
                        completion_tokens,
                    )
                    if provider_cost > 0:
                        billed_cost = provider_cost * get_cost_margin()
                        _safe_deduct_credits(
                            billed_cost,
                            api_key=self._api_key,
                            model=accounting_model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            provider_cost=provider_cost,
                        )

            # Emit LLM event (after streaming completes)
            _emit_llm_event(
                LLMEvent(
                    request=_request_kw_for_event(kw, accounting_model),
                    response=None,  # No single response for streams
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    origin=origin,
                ),
            )

            _emit_cost_event(
                CostEvent.from_completion(
                    model=accounting_model,
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    completion=usage_info,
                    cache_status="disabled",  # Streaming bypasses cache
                ),
            )

    def _execute_postprocessing_retry(
        self,
        retry_kw: dict,
        endpoint: str,
        label_suffix: str,
        origin: Optional[str] = None,
    ) -> "ChatCompletion":
        """Execute a single postprocessing retry: LLM call + logging + cost deduction."""
        label = f"{endpoint}-{label_suffix}"
        pending = write_request_pending(
            retry_kw,
            label=label,
            origin=origin,
            cache_enabled=False,
            client_id=self._client_id,
        )
        if pending and self._on_log_file_pending:
            self._on_log_file_pending(pending)
        completion = None
        try:
            with llm_span(
                label,
                self._model,
                provider=self._provider,
                origin=origin,
            ):
                completion = retry_transient_400_sync(
                    lambda: litellm.completion(**retry_kw),
                )
                _normalize_assistant_message_content(completion)
        finally:
            try:
                body = (
                    completion.model_dump(warnings=False)
                    if completion is not None and hasattr(completion, "model_dump")
                    else completion
                )
                final_path = append_response_and_finalize(
                    pending,
                    body,
                    "retry",
                    label=label,
                    origin=origin,
                )
                if final_path and self._on_log_file:
                    self._on_log_file(final_path)
            except Exception:
                pass
        if completion is not None:
            from ..costs import get_cost_margin

            accounting_model = _canonical_model_for_accounting(retry_kw.get("model"))
            cost = compute_cost_from_response(accounting_model, completion)
            if cost is not None and cost > 0:
                margin = get_cost_margin()
                billed = cost * margin
                _safe_deduct_credits(
                    billed,
                    api_key=self._api_key,
                    model=accounting_model,
                    provider_cost=cost,
                )

                _emit_cost_event(
                    CostEvent.from_completion(
                        model=accounting_model,
                        provider_cost=cost,
                        billed_cost=billed,
                        completion=completion,
                        cache_status="miss",
                    ),
                )
        return completion

    def _run_postprocessing(
        self,
        chat_completion: "ChatCompletion",
        kw: dict,
        endpoint: str,
        prompt: "Prompt",
        original_tool_choice: Optional[str],
        original_request_messages: Optional[List[dict]] = None,
        origin: Optional[str] = None,
    ) -> "ChatCompletion":
        """Run all postprocessing checks, retrying once per check if needed."""
        from .provider_postprocessing import (
            check_needs_postprocessing,
            build_retry_kw,
            check_response_format_compliance,
            build_response_format_retry_kw,
        )

        # Step 1: Provider-specific postprocessing (tool retries)
        raw_tools = kw.get("tools")
        needs_retry, retry_reason = check_needs_postprocessing(
            response=chat_completion,
            provider=self._provider,
            original_tool_choice=original_tool_choice,
            reasoning_effort=prompt.components.get("reasoning_effort"),
            tools=list(raw_tools) if raw_tools is not None else None,
            request_messages=kw.get("messages"),
            original_request_messages=original_request_messages,
        )
        if needs_retry:
            retry_kw = build_retry_kw(
                kw=kw,
                response=chat_completion,
                retry_reason=retry_reason,
            )
            chat_completion = self._execute_postprocessing_retry(
                retry_kw,
                endpoint,
                "retry",
                origin=origin,
            )

        # Step 2: response_format schema validation
        rf_needs_retry, rf_error, rf_model = check_response_format_compliance(
            response=chat_completion,
            kw=kw,
        )
        if rf_needs_retry and rf_model is not None:
            rf_retry_kw = build_response_format_retry_kw(
                kw=kw,
                response=chat_completion,
                validation_error=rf_error,
                pydantic_model=rf_model,
            )
            chat_completion = self._execute_postprocessing_retry(
                rf_retry_kw,
                endpoint,
                "rf-retry",
                origin=origin,
            )

        return chat_completion

    def _generate_non_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # python client arguments
        cache: Union[bool, str],
        cache_backend: str,
        prompt_caching: Optional[PromptCacheParam],
        origin: Optional[str] = None,
    ) -> Union[str, ChatCompletion]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=False,
            stream_options=None,
        )
        # Capture original tool_choice before preprocessing may modify it
        original_tool_choice = kw.get("tool_choice")
        original_request_messages = copy.deepcopy(kw.get("messages"))

        # Apply provider-specific preprocessing (before cache, on a copy of messages)
        apply_provider_preprocessing(kw, self._provider, prompt_caching)
        accounting_model = _prepare_provider_request_kw(
            kw=kw,
            provider=self._provider,
            stream=False,
        )

        # Write request to log file (before LLM call) so we don't lose it if call hangs
        pending_path = write_request_pending(
            kw,
            label=endpoint,
            origin=origin,
            cache_enabled=cache not in (False, None),
            client_id=self._client_id,
        )
        if pending_path and self._on_log_file_pending:
            self._on_log_file_pending(pending_path)

        if isinstance(cache, str) and cache.endswith("-closest"):
            cache = cache.removesuffix("-closest")
            read_closest = True
        else:
            read_closest = False

        # Initialize before try block so finally can access them
        chat_completion = None
        is_cache_enabled = cache in [True, "both", "read", "read-only"]
        cache_status = "pending" if is_cache_enabled else "disabled"
        in_cache = False
        llm_error: BaseException | None = None
        provider_cost: float | None = None
        billed_cost: float | None = None

        # Wrap in OTel span with try/finally to guarantee log finalization
        try:
            with llm_span(
                endpoint,
                self._model,
                provider=self._provider,
                origin=origin,
            ) as span:
                if is_cache_enabled:
                    chat_completion = _get_cache(
                        fn_name="chat.completions.create",
                        kw=kw,
                        raise_on_empty=cache == "read-only",
                        read_closest=read_closest,
                        delete_closest=read_closest,
                        backend=cache_backend,
                    )
                    in_cache = True if chat_completion is not None else False
                if chat_completion is None:
                    # Check spending limits before making LLM call (cache miss)
                    if is_limit_check_enabled():
                        limit_request = LimitCheckRequest(
                            model=accounting_model,
                            endpoint=endpoint,
                        )
                        limit_result = check_limits_sync(limit_request)
                        if not limit_result.allowed:
                            raise SpendingLimitExceededError(limit_result)

                    try:
                        chat_completion = retry_transient_400_sync(
                            lambda: litellm.completion(**kw),
                        )
                        _normalize_assistant_message_content(chat_completion)
                    except litellm.exceptions.APIError as e:
                        llm_error = Exception(e.message)
                        raise llm_error
                else:
                    _normalize_assistant_message_content(chat_completion)

                # Determine cache status after resolution
                if is_cache_enabled:
                    cache_status = "hit" if in_cache else "miss"

                # Set span response attributes
                set_span_response(span, cache_status, chat_completion)

                _emit_cache_event(
                    {
                        "cache_status": cache_status,
                        "endpoint": endpoint,
                        "request_kw": kw,
                    },
                )
        except BaseException as e:
            if llm_error is None:
                llm_error = e
            if cache_status == "pending":
                cache_status = "error"
            raise
        finally:
            # Finalize log file with response and cache status (always runs)
            try:
                resp_body = (
                    chat_completion.model_dump(warnings=False)
                    if chat_completion is not None
                    and hasattr(chat_completion, "model_dump")
                    else chat_completion
                )
                # For logging, include error info if present
                log_body = resp_body
                if llm_error is not None:
                    error_info = {
                        "type": type(llm_error).__name__,
                        "message": str(llm_error),
                    }
                    log_body = {"response": resp_body, "error": error_info}
                final_path = append_response_and_finalize(
                    pending_path,
                    log_body,
                    cache_status,
                    label=endpoint,
                    origin=origin,
                )
                if final_path and self._on_log_file:
                    self._on_log_file(final_path)
            except Exception:
                pass

            # Compute costs for event (only for cache misses - cache hits are free)
            if not in_cache and chat_completion is not None:
                from ..costs import get_cost_margin

                provider_cost = compute_cost_from_response(
                    accounting_model,
                    chat_completion,
                )
                if provider_cost is not None and provider_cost > 0:
                    billed_cost = provider_cost * get_cost_margin()

            # Emit LLM event (after LLM call, always runs)
            # Use unwrapped resp_body for LLM event (not the error-wrapped log_body)
            _emit_llm_event(
                LLMEvent(
                    request=_request_kw_for_event(kw, accounting_model),
                    response=resp_body,
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    origin=origin,
                ),
            )

            _emit_cost_event(
                CostEvent.from_completion(
                    model=accounting_model,
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    completion=chat_completion,
                    cache_status=cache_status,
                ),
            )

        # Apply postprocessing checks (tool retries + response_format validation)
        original_completion = chat_completion
        if chat_completion is not None:
            chat_completion = self._run_postprocessing(
                chat_completion,
                kw,
                endpoint,
                prompt,
                original_tool_choice,
                original_request_messages,
                origin=origin,
            )
            _enforce_parallel_tool_call_response_limit(
                chat_completion,
                prompt.components.get("parallel_tool_calls"),
            )

        # Cache the FINAL response (after any post-processing), not intermediate ones
        did_postprocess = chat_completion is not original_completion
        if (chat_completion is not None or read_closest) and cache in [
            True,
            "both",
            "write",
        ]:
            # Write to cache if it wasn't a cache hit, or if post-processing changed it
            if not in_cache or cache == "write" or did_postprocess:
                _write_to_cache(
                    fn_name="chat.completions.create",
                    kw=kw,
                    response=chat_completion,
                    backend=cache_backend,
                )

        # Deduct credits for cache misses (use already-computed billed_cost).
        if billed_cost is not None and billed_cost > 0:
            _safe_deduct_credits(
                billed_cost,
                api_key=self._api_key,
                model=accounting_model,
                provider_cost=provider_cost,
            )

        # Always return full completion; _apply_stateful_logic handles extraction
        return chat_completion

    def _generate(  # noqa: WPS234, WPS211
        self,
        messages: Optional[List[ChatCompletionMessageParam]],
        *,
        frequency_penalty: Optional[float],
        logit_bias: Optional[Dict[str, int]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        max_completion_tokens: Optional[int],
        n: Optional[int],
        presence_penalty: Optional[float],
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]],
        seed: Optional[int],
        stop: Union[Optional[str], List[str]],
        stream: Optional[bool],
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        temperature: Optional[float],
        top_p: Optional[float],
        service_tier: Optional[str],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam],
        parallel_tool_calls: Optional[bool],
        reasoning_effort: Optional[str],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        cache_backend: str,
        prompt_caching: Optional[PromptCacheParam],
        origin: Optional[str] = None,
        # passthrough arguments
        extra_headers: Optional[Headers],
        **kwargs,
    ) -> Union[Generator[str, None, None], str]:  # noqa: DAR101, DAR201, DAR401
        prompt = Prompt(
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            service_tier=service_tier,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            extra_headers=extra_headers,
        )
        if stream:
            return self._generate_stream(
                self._endpoint,
                prompt,
                # stream
                stream_options=stream_options,
                # python client arguments
                return_full_completion=return_full_completion,
                prompt_caching=prompt_caching,
                origin=origin,
            )
        return self._generate_non_stream(
            self._endpoint,
            prompt,
            # python client arguments
            cache=cache,
            cache_backend=cache_backend,
            prompt_caching=prompt_caching,
            origin=origin,
        )

    def to_async_client(self):
        """
        Return an asynchronous version of the client (`AsyncUnify` instance), with the
        exact same configuration as this synchronous (`Unify`) client.

        Returns:
            An `AsyncUnify` instance with the same configuration as this `Unify`
            instance.
        """
        return AsyncUnify(**self._constructor_args)


class AsyncUnify(_UniClient):
    """Async client for LLM inference via the model@provider endpoint format."""

    # Providers whose litellm handler expects an OpenAI SDK client (AsyncOpenAI)
    # as the ``client`` kwarg.  We must NOT pass an AsyncHTTPHandler for these.
    _OPENAI_SDK_PROVIDERS = frozenset({"openai", "azure", "azure_ai", "xiaomi-mimo"})

    _async_http_client: Optional[AsyncHTTPHandler] = None
    _async_http_client_session = None  # tracks which aiohttp session the handler wraps

    def _get_async_http_client(self) -> Optional[AsyncHTTPHandler]:
        """Return an ``AsyncHTTPHandler`` backed by the shared aiohttp session,
        but **only** for providers whose litellm handler accepts one (e.g.
        Anthropic).  For OpenAI-SDK providers the ``client`` kwarg has a
        different meaning (``AsyncOpenAI``), so we return ``None`` to avoid
        interfering."""
        if self._provider in self._OPENAI_SDK_PROVIDERS:
            return None

        session = get_shared_session()
        if (
            self._async_http_client is None
            or self._async_http_client_session is not session
        ):
            self._async_http_client = AsyncHTTPHandler(shared_session=session)
            self._async_http_client_session = session
        return self._async_http_client

    async def _generate_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # stream
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        # python client arguments
        return_full_completion: bool,
        prompt_caching: Optional[PromptCacheParam],
        origin: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=True,
            stream_options=stream_options,
        )
        # Apply provider-specific preprocessing (before cache, on a copy of messages)
        apply_provider_preprocessing(kw, self._provider, prompt_caching)
        accounting_model = _prepare_provider_request_kw(
            kw=kw,
            provider=self._provider,
            stream=True,
        )

        # Write request to log file (before LLM call) so we don't lose it if call hangs
        pending_path = write_request_pending(
            kw,
            label=endpoint,
            origin=origin,
            cache_enabled=False,
            client_id=self._client_id,
        )
        if pending_path and self._on_log_file_pending:
            self._on_log_file_pending(pending_path)

        # Start limit check and stream connection in parallel for in-flight cancellation
        limit_task: asyncio.Task | None = None
        if is_limit_check_enabled():
            limit_request = LimitCheckRequest(
                model=accounting_model,
                endpoint=endpoint,
            )
            limit_task = asyncio.create_task(
                check_limits(limit_request),
                name="spending_limit_check_stream",
            )

        # Track usage from the stream for cost deduction
        usage_info = None
        llm_error: BaseException | None = None
        provider_cost: float | None = None
        billed_cost: float | None = None
        async_stream = None
        collected_content: list[str] = []

        try:
            # Start stream connection (this initiates the LLM call)
            stream_task = asyncio.create_task(
                retry_transient_400_async(
                    lambda: litellm.acompletion(
                        shared_session=get_shared_session(),
                        client=self._get_async_http_client(),
                        **kw,
                    ),
                ),
                name="llm_stream_init",
            )

            # Wait for limit check (fast) while stream connects
            if limit_task is not None:
                limit_result = await limit_task
                limit_task = None
                if not limit_result.allowed:
                    # Cancel in-flight stream connection
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                    raise SpendingLimitExceededError(limit_result)

            # Limit passed, get the stream
            async_stream = await stream_task

            async for chunk in async_stream:  # type: ignore[union-attr]
                # Capture usage if present in the chunk (final chunk with include_usage)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

                if return_full_completion:
                    yield chunk
                else:
                    text = chunk.choices[0].delta.content or ""
                    collected_content.append(text)
                    yield text
        except litellm.exceptions.APIError as e:
            llm_error = Exception(e.message)
            raise llm_error
        except Exception as e:
            llm_error = e
            raise
        finally:
            # Finalize log file with collected response content
            try:
                log_body: dict | str | None = "".join(collected_content) or None
                if llm_error is not None:
                    error_info = {
                        "type": type(llm_error).__name__,
                        "message": str(llm_error),
                    }
                    log_body = {"response": log_body, "error": error_info}
                final_path = append_response_and_finalize(
                    pending_path,
                    log_body,
                    "error" if llm_error else "disabled",
                    label=endpoint,
                    origin=origin,
                )
                if final_path and self._on_log_file:
                    self._on_log_file(final_path)
            except BaseException:
                pass

            # Deduct credits based on usage after streaming completes
            if usage_info is not None:
                prompt_tokens = getattr(usage_info, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage_info, "completion_tokens", 0) or 0
                if prompt_tokens > 0 or completion_tokens > 0:
                    from ..costs import compute_cost, get_cost_margin

                    provider_cost = compute_cost(
                        accounting_model,
                        prompt_tokens,
                        completion_tokens,
                    )
                    if provider_cost > 0:
                        billed_cost = provider_cost * get_cost_margin()
                        asyncio.create_task(
                            asyncio.to_thread(
                                _safe_deduct_credits,
                                billed_cost,
                                api_key=self._api_key,
                                model=accounting_model,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                provider_cost=provider_cost,
                            ),
                            name="unillm_deduct_credits_stream",
                        )

            # Emit LLM event (after streaming completes)
            _emit_llm_event(
                LLMEvent(
                    request=_request_kw_for_event(kw, accounting_model),
                    response=None,  # No single response for streams
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    origin=origin,
                ),
            )

            _emit_cost_event(
                CostEvent.from_completion(
                    model=accounting_model,
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    completion=usage_info,
                    cache_status="disabled",  # Streaming bypasses cache
                ),
            )

    async def _execute_postprocessing_retry(
        self,
        retry_kw: dict,
        endpoint: str,
        label_suffix: str,
        origin: Optional[str] = None,
    ) -> "ChatCompletion":
        """Execute a single postprocessing retry: LLM call + logging + cost deduction."""
        label = f"{endpoint}-{label_suffix}"
        pending = write_request_pending(
            retry_kw,
            label=label,
            origin=origin,
            cache_enabled=False,
            client_id=self._client_id,
        )
        if pending and self._on_log_file_pending:
            self._on_log_file_pending(pending)
        completion = None
        try:
            with llm_span(
                label,
                self._model,
                provider=self._provider,
                origin=origin,
            ):
                completion = await retry_transient_400_async(
                    lambda: litellm.acompletion(
                        shared_session=get_shared_session(),
                        client=self._get_async_http_client(),
                        **retry_kw,
                    ),
                )
                _normalize_assistant_message_content(completion)
        finally:
            try:
                body = (
                    completion.model_dump(warnings=False)
                    if completion is not None and hasattr(completion, "model_dump")
                    else completion
                )
                final_path = append_response_and_finalize(
                    pending,
                    body,
                    "retry",
                    label=label,
                    origin=origin,
                )
                if final_path and self._on_log_file:
                    self._on_log_file(final_path)
            except Exception:
                pass
        if completion is not None:
            from ..costs import get_cost_margin

            accounting_model = _canonical_model_for_accounting(retry_kw.get("model"))
            cost = compute_cost_from_response(accounting_model, completion)
            if cost is not None and cost > 0:
                margin = get_cost_margin()
                billed = cost * margin
                asyncio.create_task(
                    asyncio.to_thread(
                        _safe_deduct_credits,
                        billed,
                        api_key=self._api_key,
                        model=accounting_model,
                        provider_cost=cost,
                    ),
                    name=f"unillm_deduct_credits_{label_suffix}",
                )

                _emit_cost_event(
                    CostEvent.from_completion(
                        model=accounting_model,
                        provider_cost=cost,
                        billed_cost=billed,
                        completion=completion,
                        cache_status="miss",
                    ),
                )
        return completion

    async def _run_postprocessing(
        self,
        chat_completion: "ChatCompletion",
        kw: dict,
        endpoint: str,
        prompt: "Prompt",
        original_tool_choice: Optional[str],
        original_request_messages: Optional[List[dict]] = None,
        origin: Optional[str] = None,
    ) -> "ChatCompletion":
        """Run all postprocessing checks, retrying once per check if needed."""
        from .provider_postprocessing import (
            check_needs_postprocessing,
            build_retry_kw,
            check_response_format_compliance,
            build_response_format_retry_kw,
        )

        # Step 1: Provider-specific postprocessing (tool retries)
        raw_tools = kw.get("tools")
        needs_retry, retry_reason = check_needs_postprocessing(
            response=chat_completion,
            provider=self._provider,
            original_tool_choice=original_tool_choice,
            reasoning_effort=prompt.components.get("reasoning_effort"),
            tools=list(raw_tools) if raw_tools is not None else None,
            request_messages=kw.get("messages"),
            original_request_messages=original_request_messages,
        )
        if needs_retry:
            retry_kw = build_retry_kw(
                kw=kw,
                response=chat_completion,
                retry_reason=retry_reason,
            )
            chat_completion = await self._execute_postprocessing_retry(
                retry_kw,
                endpoint,
                "retry",
                origin=origin,
            )

        # Step 2: response_format schema validation
        rf_needs_retry, rf_error, rf_model = check_response_format_compliance(
            response=chat_completion,
            kw=kw,
        )
        if rf_needs_retry and rf_model is not None:
            rf_retry_kw = build_response_format_retry_kw(
                kw=kw,
                response=chat_completion,
                validation_error=rf_error,
                pydantic_model=rf_model,
            )
            chat_completion = await self._execute_postprocessing_retry(
                rf_retry_kw,
                endpoint,
                "rf-retry",
                origin=origin,
            )

        return chat_completion

    async def _generate_non_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # python client arguments
        cache: Union[bool, str],
        cache_backend: str,
        prompt_caching: Optional[PromptCacheParam],
        origin: Optional[str] = None,
    ) -> Union[str, ChatCompletion]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=False,
            stream_options=None,
        )
        # Capture original tool_choice before preprocessing may modify it
        original_tool_choice = kw.get("tool_choice")
        original_request_messages = copy.deepcopy(kw.get("messages"))

        # Apply provider-specific preprocessing (before cache, on a copy of messages)
        apply_provider_preprocessing(kw, self._provider, prompt_caching)
        accounting_model = _prepare_provider_request_kw(
            kw=kw,
            provider=self._provider,
            stream=False,
        )

        # Write request to log file (before LLM call) so we don't lose it if call hangs
        pending_path = write_request_pending(
            kw,
            label=endpoint,
            origin=origin,
            cache_enabled=cache not in (False, None),
            client_id=self._client_id,
        )
        if pending_path and self._on_log_file_pending:
            self._on_log_file_pending(pending_path)

        if isinstance(cache, str) and cache.endswith("-closest"):
            cache = cache.removesuffix("-closest")
            read_closest = True
        else:
            read_closest = False

        # Initialize before try block so finally can access them
        chat_completion = None
        is_cache_enabled = cache in [True, "both", "read", "read-only"]
        cache_status = "pending" if is_cache_enabled else "disabled"
        in_cache = False
        llm_error: BaseException | None = None
        provider_cost: float | None = None
        billed_cost: float | None = None

        # Task tracking for cleanup
        limit_task: asyncio.Task | None = None
        llm_task: asyncio.Task | None = None

        # Wrap in OTel span with try/finally to guarantee log finalization
        try:
            with llm_span(
                endpoint,
                self._model,
                provider=self._provider,
                origin=origin,
            ) as span:
                if is_cache_enabled:
                    chat_completion = _get_cache(
                        fn_name="chat.completions.create",
                        kw=kw,
                        raise_on_empty=cache == "read-only",
                        read_closest=read_closest,
                        delete_closest=read_closest,
                        backend=cache_backend,
                    )
                    in_cache = True if chat_completion is not None else False
                if chat_completion is None:
                    # Start limit check and LLM call in parallel for true in-flight
                    # cancellation. Limit check is fast (~50ms), LLM call is slow.
                    if is_limit_check_enabled():
                        limit_request = LimitCheckRequest(
                            model=accounting_model,
                            endpoint=endpoint,
                        )
                        limit_task = asyncio.create_task(
                            check_limits(limit_request),
                            name="spending_limit_check",
                        )

                    # Start LLM call immediately (don't wait for limit check)
                    llm_task = asyncio.create_task(
                        retry_transient_400_async(
                            lambda: litellm.acompletion(
                                shared_session=get_shared_session(),
                                client=self._get_async_http_client(),
                                **kw,
                            ),
                        ),
                        name="llm_call",
                    )

                    try:
                        # Wait for limit check first (fast) while LLM runs in background
                        if limit_task is not None:
                            limit_result = await limit_task
                            limit_task = None  # Mark as consumed
                            if not limit_result.allowed:
                                # Cancel in-flight LLM call
                                llm_task.cancel()
                                try:
                                    await llm_task
                                except asyncio.CancelledError:
                                    pass
                                llm_task = None
                                raise SpendingLimitExceededError(limit_result)

                        # Limit check passed (or disabled), wait for LLM result
                        chat_completion = await llm_task
                        _normalize_assistant_message_content(chat_completion)
                        llm_task = None  # Mark as consumed
                    except litellm.exceptions.APIError as e:
                        llm_error = Exception(e.message)
                        raise llm_error
                else:
                    _normalize_assistant_message_content(chat_completion)

                # Determine cache status after resolution
                if is_cache_enabled:
                    cache_status = "hit" if in_cache else "miss"

                # Set span response attributes
                set_span_response(span, cache_status, chat_completion)

                _emit_cache_event(
                    {
                        "cache_status": cache_status,
                        "endpoint": endpoint,
                        "request_kw": kw,
                    },
                )
        except BaseException as e:
            # Capture the error for the response event
            if llm_error is None:
                llm_error = e
            if cache_status == "pending":
                cache_status = "error"
            raise
        finally:
            # Cancel any unconsumed tasks (e.g., cache hit or error)
            for task in [limit_task, llm_task]:
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

            # Finalize log file with response and cache status (always runs)
            try:
                resp_body = (
                    chat_completion.model_dump(warnings=False)
                    if chat_completion is not None
                    and hasattr(chat_completion, "model_dump")
                    else chat_completion
                )
                # For logging, include error info if present
                log_body = resp_body
                if llm_error is not None:
                    error_info = {
                        "type": type(llm_error).__name__,
                        "message": str(llm_error),
                    }
                    log_body = {"response": resp_body, "error": error_info}
                final_path = append_response_and_finalize(
                    pending_path,
                    log_body,
                    cache_status,
                    label=endpoint,
                    origin=origin,
                )
                if final_path and self._on_log_file:
                    self._on_log_file(final_path)
            except BaseException:
                pass

            # Compute costs for event (only for cache misses - cache hits are free)
            if not in_cache and chat_completion is not None:
                from ..costs import get_cost_margin

                provider_cost = compute_cost_from_response(
                    accounting_model,
                    chat_completion,
                )
                if provider_cost is not None and provider_cost > 0:
                    billed_cost = provider_cost * get_cost_margin()

            # Emit LLM event (after LLM call, always runs)
            # Use unwrapped resp_body for LLM event (not the error-wrapped log_body)
            _emit_llm_event(
                LLMEvent(
                    request=_request_kw_for_event(kw, accounting_model),
                    response=resp_body,
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    origin=origin,
                ),
            )

            _emit_cost_event(
                CostEvent.from_completion(
                    model=accounting_model,
                    provider_cost=provider_cost,
                    billed_cost=billed_cost,
                    completion=chat_completion,
                    cache_status=cache_status,
                ),
            )

        # Deduct credits for cache misses (use already-computed billed_cost)
        if billed_cost is not None and billed_cost > 0:
            asyncio.create_task(
                asyncio.to_thread(
                    _safe_deduct_credits,
                    billed_cost,
                    api_key=self._api_key,
                    model=accounting_model,
                    provider_cost=provider_cost,
                ),
                name="unillm_deduct_credits",
            )

        # Apply postprocessing checks (tool retries + response_format validation)
        original_completion = chat_completion
        if chat_completion is not None:
            chat_completion = await self._run_postprocessing(
                chat_completion,
                kw,
                endpoint,
                prompt,
                original_tool_choice,
                original_request_messages,
                origin=origin,
            )
            _enforce_parallel_tool_call_response_limit(
                chat_completion,
                prompt.components.get("parallel_tool_calls"),
            )

        # Cache the FINAL response (after any post-processing), not intermediate ones
        did_postprocess = chat_completion is not original_completion
        if (chat_completion is not None or read_closest) and cache in [
            True,
            "both",
            "write",
        ]:
            # Write to cache if it wasn't a cache hit, or if post-processing changed it
            if not in_cache or cache == "write" or did_postprocess:
                _write_to_cache(
                    fn_name="chat.completions.create",
                    kw=kw,
                    response=chat_completion,
                    backend=cache_backend,
                )

        # Always return full completion; _apply_stateful_logic handles extraction
        return chat_completion

    async def _generate(  # noqa: WPS234, WPS211
        self,
        messages: Optional[List[ChatCompletionMessageParam]],
        *,
        frequency_penalty: Optional[float],
        logit_bias: Optional[Dict[str, int]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        max_completion_tokens: Optional[int],
        n: Optional[int],
        presence_penalty: Optional[float],
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]],
        seed: Optional[int],
        stop: Union[Optional[str], List[str]],
        stream: Optional[bool],
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        temperature: Optional[float],
        top_p: Optional[float],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam],
        parallel_tool_calls: Optional[bool],
        reasoning_effort: Optional[str],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        cache_backend: str,
        prompt_caching: Optional[PromptCacheParam],
        origin: Optional[str] = None,
        # passthrough arguments
        extra_headers: Optional[Headers],
        service_tier: Optional[str] = None,
        **kwargs,
    ) -> Union[AsyncGenerator[str, None], str]:  # noqa: DAR101, DAR201, DAR401
        prompt = Prompt(
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_headers=extra_headers,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
        )
        if stream:
            return self._generate_stream(
                self._endpoint,
                prompt,
                # stream
                stream_options=stream_options,
                # python client arguments
                return_full_completion=return_full_completion,
                prompt_caching=prompt_caching,
                origin=origin,
            )
        return await self._generate_non_stream(
            self._endpoint,
            prompt,
            # python client arguments
            cache=cache,
            cache_backend=cache_backend,
            prompt_caching=prompt_caching,
            origin=origin,
        )

    def to_sync_client(self):
        """
        Return a synchronous version of the client (`Unify` instance), with the
        exact same configuration as this asynchronous (`AsyncUnify`) client.

        Returns:
            A `Unify` instance with the same configuration as this `AsyncUnify`
            instance.
        """
        return Unify(**self._constructor_args)
