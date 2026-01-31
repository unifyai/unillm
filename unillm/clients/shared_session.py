import aiohttp


def make_session() -> aiohttp.ClientSession:
    """
    Factory function to create an optimized aiohttp ClientSession.

    This is a synchronous factory (not async) because LiteLLM's
    LiteLLMAiohttpTransport calls it to create sessions in the correct
    event loop when needed.

    IMPORTANT: This must be a factory function, not a pre-created session.
    Pre-created sessions (via asyncio.run()) are bound to a closed event loop
    and cause configuration to be lost when litellm recreates the session.

    The session is optimized for high throughput with:
    - 300 second total timeout
    - 1000 max connections (200 per host)
    - DNS caching for 600 seconds
    - 60 second keepalive
    """
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
        connector=aiohttp.TCPConnector(
            limit=1000,  # High connection limit
            limit_per_host=200,  # Per host limit
            ttl_dns_cache=600,  # DNS cache
            keepalive_timeout=60,  # Keep connections alive
        ),
    )


# Export the factory function, not a pre-created session.
# LiteLLM's LiteLLMAiohttpTransport detects callables and stores them
# as _client_factory, using them to create sessions in the correct
# event loop when needed.
SHARED_SESSION = make_session
