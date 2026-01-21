# global
import copy
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Literal, Optional, Type, Union

import requests

# noinspection PyProtectedMember
from openai._types import Headers
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, create_model
from typing_extensions import Self

# local
from unify import BASE_URL
from unify.utils import http

from unify.utils.helpers import _create_request_header, _validate_api_key
from unillm.types import PromptCacheParam


class _Client(ABC):
    """Base Abstract class for interacting with the Unify chat completions endpoint."""

    def __init__(
        self,
        *,
        system_message: Optional[str],
        messages: Optional[List[ChatCompletionMessageParam]],
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
        # platform arguments
        api_key: Optional[str],
        # python client arguments
        stateful: bool,
        return_full_completion: bool,
        cache: Union[bool, str],
        cache_backend: str,
        prompt_caching: PromptCacheParam,
        # passthrough arguments
        extra_headers: Optional[Headers],
        **kwargs,
    ) -> None:

        # initial values
        self._api_key = _validate_api_key(api_key)
        self._system_message = None
        self._messages = None
        self._frequency_penalty = None
        self._logit_bias = None
        self._logprobs = None
        self._top_logprobs = None
        self._max_completion_tokens = None
        self._n = None
        self._presence_penalty = None
        self._response_format = None
        self._seed = None
        self._stop = None
        self._stream = None
        self._stream_options = None
        self._temperature = None
        self._top_p = None
        self._service_tier = None
        self._tools = None
        self._tool_choice = None
        self._parallel_tool_calls = None
        self._reasoning_effort = None
        self._stateful = None
        self._return_full_completion = None
        self._cache = None
        self._cache_backend = None
        self._prompt_caching = None
        self._extra_headers = None

        # set based on arguments
        self.set_system_message(system_message)
        self.set_messages(messages)
        self.set_frequency_penalty(frequency_penalty)
        self.set_logit_bias(logit_bias)
        self.set_logprobs(logprobs)
        self.set_top_logprobs(top_logprobs)
        self.set_max_completion_tokens(max_completion_tokens)
        self.set_n(n)
        self.set_presence_penalty(presence_penalty)
        self.set_response_format(response_format)
        self.set_seed(seed)
        self.set_stop(stop)
        self.set_stream(stream)
        self.set_stream_options(stream_options)
        self.set_temperature(temperature)
        self.set_top_p(top_p)
        self.set_service_tier(service_tier)
        self.set_tools(tools)
        self.set_tool_choice(tool_choice)
        self.set_parallel_tool_calls(parallel_tool_calls)
        self.set_reasoning_effort(reasoning_effort)
        # python client arguments
        self.set_stateful(stateful)
        self.set_return_full_completion(return_full_completion)
        self.set_cache(cache)
        self.set_cache_backend(cache_backend)
        self.set_prompt_caching(prompt_caching)
        # passthrough arguments
        self.set_extra_headers(extra_headers)

        # Store defaults
        self._defaults = {
            "system_message": system_message,
            "messages": messages,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "max_completion_tokens": max_completion_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "stream_options": stream_options,
            "temperature": temperature,
            "top_p": top_p,
            "service_tier": service_tier,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "reasoning_effort": reasoning_effort,
        }

    # Properties #
    # -----------#

    @property
    def system_message(self) -> Optional[str]:
        """
        Get the default system message, if set.

        Returns:
            The default system message.
        """
        return self._system_message

    @property
    def messages(
        self,
    ) -> Optional[List[ChatCompletionMessageParam]]:
        """
        Get the default messages, if set.

        Returns:
            The default messages.
        """
        return self._messages

    @property
    def frequency_penalty(self) -> Optional[float]:
        """
        Get the default frequency penalty, if set.

        Returns:
            The default frequency penalty.
        """
        return self._frequency_penalty

    @property
    def logit_bias(self) -> Optional[Dict[str, int]]:
        """
        Get the default logit bias, if set.

        Returns:
            The default logit bias.
        """
        return self._logit_bias

    @property
    def logprobs(self) -> Optional[bool]:
        """
        Get the default logprobs, if set.

        Returns:
            The default logprobs.
        """
        return self._logprobs

    @property
    def top_logprobs(self) -> Optional[int]:
        """
        Get the default top logprobs, if set.

        Returns:
            The default top logprobs.
        """
        return self._top_logprobs

    @property
    def max_completion_tokens(self) -> Optional[int]:
        """
        Get the default max tokens, if set.

        Returns:
            The default max tokens.
        """
        return self._max_completion_tokens

    @property
    def n(self) -> Optional[int]:
        """
        Get the default n, if set.

        Returns:
            The default n value.
        """
        return self._n

    @property
    def presence_penalty(self) -> Optional[float]:
        """
        Get the default presence penalty, if set.

        Returns:
            The default presence penalty.
        """
        return self._presence_penalty

    @property
    def response_format(self) -> Optional[Union[Type[BaseModel], Dict[str, str]]]:
        """
        Get the default response format, if set.

        Returns:
            The default response format.
        """
        return self._response_format

    @property
    def seed(self) -> Optional[int]:
        """
        Get the default seed value, if set.

        Returns:
            The default seed value.
        """
        return self._seed

    @property
    def stop(self) -> Union[Optional[str], List[str]]:
        """
        Get the default stop value, if set.

        Returns:
            The default stop value.
        """
        return self._stop

    @property
    def stream(self) -> Optional[bool]:
        """
        Get the default stream bool, if set.

        Returns:
            The default stream bool.
        """
        return self._stream

    @property
    def stream_options(self) -> Optional[ChatCompletionStreamOptionsParam]:
        """
        Get the default stream options, if set.

        Returns:
            The default stream options.
        """
        return self._stream_options

    @property
    def temperature(self) -> Optional[float]:
        """
        Get the default temperature, if set.

        Returns:
            The default temperature.
        """
        return self._temperature

    @property
    def top_p(self) -> Optional[float]:
        """
        Get the default top p value, if set.

        Returns:
            The default top p value.
        """
        return self._top_p

    @property
    def service_tier(self) -> Optional[str]:
        """
        Get the default service tier, if set.

        Returns:
            The default service tier.
        """
        return self._service_tier

    @property
    def tools(self) -> Optional[Iterable[ChatCompletionToolParam]]:
        """
        Get the default tools, if set.

        Returns:
            The default tools.
        """
        return self._tools

    @property
    def tool_choice(self) -> Optional[ChatCompletionToolChoiceOptionParam]:
        """
        Get the default tool choice, if set.

        Returns:
            The default tool choice.
        """
        return self._tool_choice

    @property
    def parallel_tool_calls(self) -> Optional[bool]:
        """
        Get the default parallel tool calls bool, if set.

        Returns:
            The default parallel tool calls bool.
        """
        return self._parallel_tool_calls

    @property
    def reasoning_effort(self) -> Optional[str]:
        """
        Get the default reasoning, if set.

        Returns:
            The default reasoning.
        """
        return self._reasoning_effort

    @property
    def stateful(self) -> bool:
        """
        Get the default stateful bool, if set.

        Returns:
            The default stateful bool.
        """
        return self._stateful

    @property
    def return_full_completion(self) -> bool:
        """
        Get the default return full completion bool.

        Returns:
            The default return full completion bool.
        """
        return self._return_full_completion

    @property
    def cache(self) -> bool:
        """
        Get default the cache bool.

        Returns:
            The default cache bool.
        """
        return self._cache

    @property
    def prompt_caching(self) -> Optional[List[Literal["tools", "system", "user"]]]:
        """
        Get the default prompt caching settings for Anthropic.

        Returns:
            List of cache breakpoint locations, or None if not set.
        """
        return self._prompt_caching

    @property
    def extra_headers(self) -> Optional[Headers]:
        """
        Get the default extra headers, if set.

        Returns:
            The default extra headers.
        """
        return self._extra_headers

    # Setters #
    # --------#

    def set_system_message(self, value: str) -> Self:
        """
        Set the default system message.

        Args:
            value: The default system message.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._system_message = value
        if self._messages is None or self._messages == []:
            self._messages = [
                {
                    "role": "system",
                    "content": value,
                },
            ]
        elif self._messages[0]["role"] != "system":
            self._messages = [
                {
                    "role": "system",
                    "content": value,
                },
            ] + self._messages
        else:
            self._messages[0] = {
                "role": "system",
                "content": value,
            }
        return self

    def set_messages(
        self,
        value: List[ChatCompletionMessageParam],
    ) -> Self:
        """
        Set the default messages.

        Args:
            value: The default messages.

        Returns:
            This client, useful for chaining inplace calls.
        """
        if value is None:
            value = []
        self._messages = value
        if value and value[0]["role"] == "system":
            self.set_system_message(value[0]["content"])
        return self

    def append_messages(
        self,
        value: List[ChatCompletionMessageParam],
    ) -> Self:
        """
        Append to the default messages.

        Args:
            value: The messages to append to the default.

        Returns:
            This client, useful for chaining inplace calls.
        """
        if self._messages is None:
            self._messages = []
        self._messages += value
        return self

    def set_frequency_penalty(self, value: float) -> Self:
        """
        Set the default frequency penalty.

        Args:
            value: The default frequency penalty.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._frequency_penalty = value
        return self

    def set_logit_bias(self, value: Dict[str, int]) -> Self:
        """
        Set the default logit bias.

        Args:
            value: The default logit bias.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._logit_bias = value
        return self

    def set_logprobs(self, value: bool) -> Self:
        """
        Set the default logprobs.

        Args:
            value: The default logprobs.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._logprobs = value
        return self

    def set_top_logprobs(self, value: int) -> Self:
        """
        Set the default top logprobs.

        Args:
            value: The default top logprobs.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._top_logprobs = value
        return self

    def set_max_completion_tokens(self, value: int) -> Self:
        """
        Set the default max tokens.

        Args:
            value: The default max tokens.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._max_completion_tokens = value
        return self

    def set_n(self, value: int) -> Self:
        """
        Set the default n value.

        Args:
            value: The default n value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._n = value
        return self

    def set_presence_penalty(self, value: float) -> Self:
        """
        Set the default presence penalty.

        Args:
            value: The default presence penalty.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._presence_penalty = value
        return self

    def set_response_format(
        self,
        value: Optional[Union[Type[BaseModel], Dict[str, str]]],
    ) -> Self:
        """
        Set the default response format.

        Args:
            value: The default response format.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._response_format = value
        return self

    def set_seed(self, value: Optional[int]) -> Self:
        """
        Set the default seed value.

        Args:
            value: The default seed value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._seed = value
        return self

    def set_stop(self, value: Union[str, List[str]]) -> Self:
        """
        Set the default stop value.

        Args:
            value: The default stop value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stop = value
        return self

    def set_stream(self, value: bool) -> Self:
        """
        Set the default stream bool.

        Args:
            value: The default stream bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stream = value
        return self

    def set_stream_options(self, value: ChatCompletionStreamOptionsParam) -> Self:
        """
        Set the default stream options.

        Args:
            value: The default stream options.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stream_options = value
        return self

    def set_temperature(self, value: float) -> Self:
        """
        Set the default temperature.

        Args:
            value: The default temperature.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._temperature = value
        return self

    def set_top_p(self, value: float) -> Self:
        """
        Set the default top p value.

        Args:
            value: The default top p value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._top_p = value
        return self

    def set_service_tier(self, value: Optional[str]) -> Self:
        """
        Set the default service tier.

        Args:
            value: The default service tier.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._service_tier = value
        return self

    def set_tools(self, value: Iterable[ChatCompletionToolParam]) -> Self:
        """
        Set the default tools.

        Args:
            value: The default tools.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._tools = value
        return self

    def set_tool_choice(self, value: ChatCompletionToolChoiceOptionParam) -> Self:
        """
        Set the default tool choice.

        Args:
            value: The default tool choice.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._tool_choice = value
        return self

    def set_parallel_tool_calls(self, value: bool) -> Self:
        """
        Set the default parallel tool calls bool.

        Args:
            value: The default parallel tool calls bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._parallel_tool_calls = value
        return self

    def set_reasoning_effort(self, value: str) -> Self:
        """
        Set the default reasoning effort.

        Args:
            value: The default reasoning effort.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._reasoning_effort = value
        return self

    def set_stateful(self, value: bool) -> Self:
        """
        Set the default stateful bool.

        Args:
            value: The default stateful bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stateful = value
        return self

    def set_return_full_completion(self, value: bool) -> Self:
        """
        Set the default return full completion bool.

        Args:
            value: The default return full completion bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._return_full_completion = value
        return self

    def set_cache(self, value: bool) -> Self:
        """
        Set the default cache bool.

        Args:
            value: The default cache bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._cache = value
        return self

    def set_cache_backend(self, value: str) -> Self:
        """
        Set the default cache backend.

        Args:
            value: The default cache backend.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._cache_backend = value
        return self

    def set_prompt_caching(
        self,
        value: Optional[List[Literal["tools", "system", "user"]]],
    ) -> Self:
        """
        Set the prompt caching settings for Anthropic models.

        Args:
            value: List of locations to insert cache breakpoints.
                   Valid values: "tools", "system", "user".

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._prompt_caching = value
        return self

    def set_extra_headers(self, value: Headers) -> Self:
        """
        Set the default extra headers.

        Args:
            value: The default extra headers.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._extra_headers = value
        return self

    # Reset Methods #
    # -------------#

    def reset_system_message(self) -> Self:
        """
        Reset the system message to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_system_message(self._defaults["system_message"])

    def reset_messages(self) -> Self:
        """
        Reset the messages to their default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_messages(self._defaults["messages"])

    def reset_frequency_penalty(self) -> Self:
        """
        Reset the frequency penalty to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_frequency_penalty(self._defaults["frequency_penalty"])

    def reset_logit_bias(self) -> Self:
        """
        Reset the logit bias to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_logit_bias(self._defaults["logit_bias"])

    def reset_logprobs(self) -> Self:
        """
        Reset the logprobs to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_logprobs(self._defaults["logprobs"])

    def reset_top_logprobs(self) -> Self:
        """
        Reset the top logprobs to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_top_logprobs(self._defaults["top_logprobs"])

    def reset_max_completion_tokens(self) -> Self:
        """
        Reset the max completion tokens to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_max_completion_tokens(self._defaults["max_completion_tokens"])

    def reset_n(self) -> Self:
        """
        Reset n to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_n(self._defaults["n"])

    def reset_presence_penalty(self) -> Self:
        """
        Reset the presence penalty to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_presence_penalty(self._defaults["presence_penalty"])

    def reset_response_format(self) -> Self:
        """
        Reset the response format to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_response_format(self._defaults["response_format"])

    def reset_seed(self) -> Self:
        """
        Reset the seed to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_seed(self._defaults["seed"])

    def reset_stop(self) -> Self:
        """
        Reset the stop value to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_stop(self._defaults["stop"])

    def reset_stream(self) -> Self:
        """
        Reset the stream value to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_stream(self._defaults["stream"])

    def reset_stream_options(self) -> Self:
        """
        Reset the stream options to their default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_stream_options(self._defaults["stream_options"])

    def reset_temperature(self) -> Self:
        """
        Reset the temperature to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_temperature(self._defaults["temperature"])

    def reset_top_p(self) -> Self:
        """
        Reset the top p value to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_top_p(self._defaults["top_p"])

    def reset_service_tier(self) -> Self:
        """
        Reset the service tier to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_service_tier(self._defaults["service_tier"])

    def reset_tools(self) -> Self:
        """
        Reset the tools to their default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_tools(self._defaults["tools"])

    def reset_tool_choice(self) -> Self:
        """
        Reset the tool choice to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_tool_choice(self._defaults["tool_choice"])

    def reset_parallel_tool_calls(self) -> Self:
        """
        Reset the parallel tool calls to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_parallel_tool_calls(self._defaults["parallel_tool_calls"])

    def reset_reasoning_effort(self) -> Self:
        """
        Reset the reasoning effort to its default value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        return self.set_reasoning_effort(self._defaults["reasoning_effort"])

    def reset_all(self) -> Self:
        """
        Reset base client properties to their default values.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self.reset_system_message()
        self.reset_messages()
        self.reset_frequency_penalty()
        self.reset_logit_bias()
        self.reset_logprobs()
        self.reset_top_logprobs()
        self.reset_max_completion_tokens()
        self.reset_n()
        self.reset_presence_penalty()
        self.reset_response_format()
        self.reset_seed()
        self.reset_stop()
        self.reset_stream()
        self.reset_stream_options()
        self.reset_temperature()
        self.reset_top_p()
        self.reset_service_tier()
        self.reset_tools()
        self.reset_tool_choice()
        self.reset_parallel_tool_calls()
        self.reset_reasoning_effort()
        return self

    # Credits #
    # --------#

    def get_credit_balance(self) -> Union[float, None]:
        """
        Get the remaining credits left on your account.

        Returns:
            The remaining credits on the account if successful, otherwise None.
        Raises:
            BadRequestError: If there was an HTTP error.
            ValueError: If there was an error parsing the JSON response.
        """
        url = f"{BASE_URL}/credits"
        headers = _create_request_header(self._api_key)
        try:
            response = http.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise Exception(response.json())
            return response.json()["credits"]
        except requests.RequestException as e:
            raise requests.RequestException(
                "There was an error with the request.",
            ) from e
        except (KeyError, ValueError) as e:
            raise ValueError("Error parsing JSON response.") from e

    # Methods #
    # --------#

    def copy(self):
        # noinspection PyUnresolvedReferences,PyArgumentList
        return type(self)(
            **copy.deepcopy(
                {
                    **self._constructor_args,
                    **dict(
                        system_message=self._system_message,
                        messages=self._messages,
                        frequency_penalty=self._frequency_penalty,
                        logit_bias=self._logit_bias,
                        logprobs=self._logprobs,
                        top_logprobs=self._top_logprobs,
                        max_completion_tokens=self._max_completion_tokens,
                        n=self._n,
                        presence_penalty=self._presence_penalty,
                        response_format=self._response_format,
                        seed=self._seed,
                        stop=self._stop,
                        stream=self._stream,
                        stream_options=self._stream_options,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        service_tier=self._service_tier,
                        tools=self._tools,
                        tool_choice=self._tool_choice,
                        parallel_tool_calls=self._parallel_tool_calls,
                        # platform arguments
                        api_key=self._api_key,
                        # python client arguments
                        stateful=self._stateful,
                        return_full_completion=self._return_full_completion,
                        cache=self._cache,
                        prompt_caching=self._prompt_caching,
                        # passthrough arguments
                        extra_headers=self._extra_headers,
                    ),
                },
            ),
        )

    def json(self):
        model = create_model(type(self).__name__, __config__={"extra": "allow"})
        instance = model(
            **{
                "type": type(self).__name__,
                **self._constructor_args,
                **dict(
                    system_message=self._system_message,
                    messages=self._messages,
                    frequency_penalty=self._frequency_penalty,
                    logit_bias=self._logit_bias,
                    logprobs=self._logprobs,
                    top_logprobs=self._top_logprobs,
                    max_completion_tokens=self._max_completion_tokens,
                    n=self._n,
                    presence_penalty=self._presence_penalty,
                    response_format=self._response_format,
                    seed=self._seed,
                    stop=self._stop,
                    stream=self._stream,
                    stream_options=self._stream_options,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    service_tier=self._service_tier,
                    tools=self._tools,
                    tool_choice=self._tool_choice,
                    parallel_tool_calls=self._parallel_tool_calls,
                    # platform arguments
                    api_key=self._api_key,
                    # python client arguments
                    stateful=self._stateful,
                    return_full_completion=self._return_full_completion,
                    cache=self._cache,
                    prompt_caching=self._prompt_caching,
                    # passthrough arguments
                    extra_headers=self._extra_headers,
                ),
            },
        )
        return instance.model_dump()

    # Abstract Methods #
    # -----------------#

    @abstractmethod
    def _generate(
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
        # passthrough arguments
        extra_headers: Optional[Headers],
        **kwargs,
    ):
        raise NotImplementedError
