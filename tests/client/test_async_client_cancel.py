"""
Regression tests for response cleanup on async cancel and timeout.

Verifies AC-1, AC-2, and AC-4.
"""

from __future__ import annotations

import asyncio

import pytest

import httpx


@pytest.fixture(scope="module")
def anyio_backend():
    # asyncio.wait_for is the cancel path under test; it is not a trio API.
    return "asyncio"


def _inflight_requests(client: httpx.AsyncClient) -> int:
    """Requests still checked out of the httpcore connection pool."""
    pool = getattr(client._transport, "_pool")
    return len(pool._requests)


@pytest.mark.anyio
async def test_cancel_during_streaming_read(server):
    """
    Cancel via asyncio.wait_for during a streaming read closes the response
    and returns the connection before the client exits.
    """
    url = server.url.copy_with(path="/slow_response")

    async with httpx.AsyncClient() as client:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.get(url), timeout=0.2)

        assert _inflight_requests(client) == 0


@pytest.mark.anyio
async def test_client_timeout_during_streaming_read(server):
    """
    A client read timeout during a streaming read closes the response
    and returns the connection before the client exits.
    """
    url = server.url.copy_with(path="/slow_response")
    timeout = httpx.Timeout(5.0, read=0.2)

    async with httpx.AsyncClient(timeout=timeout) as client:
        with pytest.raises(httpx.TimeoutException):
            await client.get(url)

        assert _inflight_requests(client) == 0


@pytest.mark.anyio
async def test_response_closed_after_cancel(server):
    """Response.is_closed is set when a streaming aread() is cancelled."""
    url = server.url.copy_with(path="/slow_response")
    response = None

    async with httpx.AsyncClient() as client:
        response = await client.send(client.build_request("GET", url), stream=True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(response.aread(), timeout=0.2)

        assert response.is_closed
        assert _inflight_requests(client) == 0

    assert response is not None
    assert response.is_closed


@pytest.mark.anyio
async def test_connection_returned_to_pool_after_cancel(server):
    """The pool has no in-flight request after a streaming read is cancelled."""
    url = server.url.copy_with(path="/slow_response")

    async with httpx.AsyncClient() as client:
        try:
            await asyncio.wait_for(client.get(url), timeout=0.2)
        except asyncio.TimeoutError:
            pass

        assert _inflight_requests(client) == 0


@pytest.mark.anyio
async def test_double_close_idempotency(server):
    """aclose() is idempotent."""
    async with httpx.AsyncClient() as client:
        response = await client.get(server.url)
        await response.aclose()
        await response.aclose()


@pytest.mark.anyio
async def test_cancel_during_context_manager(server):
    """
    async with response does not conflict with cleanup inside aread()/aclose().
    """
    url = server.url.copy_with(path="/slow_response")

    async with httpx.AsyncClient() as client:
        with pytest.raises(asyncio.TimeoutError):
            async with client.stream("GET", url) as response:
                await asyncio.wait_for(response.aread(), timeout=0.2)

        assert _inflight_requests(client) == 0
