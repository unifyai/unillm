from . import base, uni_llm

__all__ = ["uni_llm", "base"]

import litellm

from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
from .shared_session import SHARED_SESSION

litellm.drop_params = True

litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(client_session=SHARED_SESSION)
