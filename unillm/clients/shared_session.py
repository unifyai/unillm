import asyncio

import aiohttp

_shared_session: aiohttp.ClientSession | None = None
_session_loop: asyncio.AbstractEventLoop | None = None


# Optimized for high throughput
def get_shared_session() -> aiohttp.ClientSession | None:
    """Lazily create and return a shared aiohttp session.

    The session is created on first access so it binds to the
    currently running asyncio event loop instead of a temporary one.

    Returns None when called from a sync context (no running event loop),
    since sync litellm calls don't use aiohttp anyway.
    """
    global _shared_session, _session_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop (sync context) — can't create a session,
        # and sync litellm.completion() doesn't need one.
        return None

    if (
        _shared_session is None
        or _shared_session.closed
        or _session_loop is not current_loop
    ):
        _shared_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(
                limit=1000,  # High connection limit
                limit_per_host=200,  # Per host limit
                ttl_dns_cache=600,  # DNS cache
                keepalive_timeout=60,  # Keep connections alive
                enable_cleanup_closed=False,
            ),
        )
        _session_loop = current_loop

    return _shared_session
