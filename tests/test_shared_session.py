"""
Tests for shared session configuration and event loop compatibility.

These tests verify that the shared aiohttp session works correctly when used
from different event loops (e.g., pytest's event loop vs import-time loop).
"""

import pytest


@pytest.mark.asyncio
async def test_shared_session_config_preserved_across_event_loops():
    """
    Verify that session configuration is preserved when the session
    is used from a different event loop than it was created in.

    This is a regression test for a bug where:
    1. SHARED_SESSION was created via asyncio.run() at import time
    2. When used from pytest's event loop, litellm detected the loop mismatch
    3. Litellm recreated the session but WITHOUT our custom configuration
    4. This caused connection limits to drop (1000 -> 100) and other settings
       to be lost, potentially causing performance issues or hangs

    The fix is to make SHARED_SESSION a factory function instead of a
    pre-created session, so litellm can create properly configured sessions
    in the correct event loop when needed.
    """
    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

    from unillm.clients.shared_session import SHARED_SESSION

    # Create transport with our shared session (as litellm does internally)
    transport = LiteLLMAiohttpTransport(client=SHARED_SESSION)

    # Get a valid session for the current event loop.
    # This is what litellm does internally before making requests.
    # If SHARED_SESSION was created in a different/closed event loop,
    # this method will detect the mismatch and recreate the session.
    session = transport._get_valid_client_session()

    try:
        # The session MUST have our timeout configuration (300 seconds).
        assert session.timeout is not None, (
            "Session has no timeout configured! "
            "This will cause LLM calls to hang indefinitely."
        )
        assert session.timeout.total == 300, (
            f"Expected 300s timeout, got {session.timeout.total}. "
            "Timeout configuration was lost during session recreation."
        )

        # The session MUST have our connector configuration.
        # These settings are critical for high-throughput performance.
        # If these fail, the session was recreated without our custom config.
        assert session.connector is not None, "Session has no connector!"

        assert session.connector.limit == 1000, (
            f"Expected connector limit=1000, got {session.connector.limit}. "
            "Connector configuration was lost during session recreation."
        )

        assert session.connector.limit_per_host == 200, (
            f"Expected connector limit_per_host=200, got {session.connector.limit_per_host}. "
            "Connector configuration was lost during session recreation."
        )
    finally:
        # Only close if we created a new session (factory pattern)
        # Don't close if it's the original shared session
        if callable(SHARED_SESSION):
            await session.close()
