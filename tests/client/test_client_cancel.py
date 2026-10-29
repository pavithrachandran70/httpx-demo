"""
Regression tests for response cleanup on sync cancel and timeout.

Verifies AC-3: sync Client matches async cleanup when a streaming read
is interrupted. Sync has no asyncio.wait_for equivalent; the client read
timeout is the interrupt, and it must close the response and return the
connection to the pool.
"""

from __future__ import annotations

import pytest

import httpx


def _inflight_requests(client: httpx.Client) -> int:
    """Requests still checked out of the httpcore connection pool."""
    pool = getattr(client._transport, "_pool")
    return len(pool._requests)


def test_sync_cancel_during_streaming_read(server):
    """A sync read timeout during streaming returns the connection."""
    url = server.url.copy_with(path="/slow_response")
    timeout = httpx.Timeout(5.0, read=0.2)

    with httpx.Client(timeout=timeout) as client:
        with pytest.raises(httpx.TimeoutException):
            client.get(url)

        assert _inflight_requests(client) == 0


def test_sync_response_closed_after_cancel(server):
    """response.is_closed is set when a streaming read() times out."""
    url = server.url.copy_with(path="/slow_response")
    timeout = httpx.Timeout(5.0, read=0.2)
    response = None

    with httpx.Client(timeout=timeout) as client:
        response = client.send(client.build_request("GET", url), stream=True)
        with pytest.raises(httpx.TimeoutException):
            response.read()

        assert response.is_closed
        assert _inflight_requests(client) == 0

    assert response is not None
    assert response.is_closed


def test_sync_double_close_idempotency(server):
    """close() is idempotent."""
    with httpx.Client() as client:
        response = client.get(server.url)
        response.close()
        response.close()


def test_sync_cancel_during_context_manager(server):
    """
    with response does not conflict with cleanup inside read()/close().
    """
    url = server.url.copy_with(path="/slow_response")
    timeout = httpx.Timeout(5.0, read=0.2)

    with httpx.Client(timeout=timeout) as client:
        with pytest.raises(httpx.TimeoutException):
            with client.stream("GET", url) as response:
                response.read()

        assert _inflight_requests(client) == 0
