import aiohttp
import asyncio


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


SHARED_SESSION = asyncio.run(make_session())
