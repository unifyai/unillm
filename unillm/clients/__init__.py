from . import base, uni_llm

__all__ = ["uni_llm", "base"]

import litellm
import aiohttp
import asyncio
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

litellm.drop_params = True


# Optimized for high throughput
async def make_session():
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
        connector=aiohttp.TCPConnector(
            limit=1000,  # High connection limit
            limit_per_host=200,  # Per host limit
            ttl_dns_cache=600,  # DNS cache
            keepalive_timeout=60,  # Keep connections alive
            enable_cleanup_closed=True,
        ),
    )


session = asyncio.run(make_session())
litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(client_session=session)
