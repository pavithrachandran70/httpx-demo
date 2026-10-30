from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
import typing
from datetime import timedelta

import pytest

import httpx


@pytest.mark.anyio
async def test_get(server):
    url = server.url
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.get(url)
    assert response.status_code == 200
    assert response.text == "Hello, world!"
    assert response.http_version == "HTTP/1.1"
    assert response.headers
    assert repr(response) == "<Response [200 OK]>"
    assert response.elapsed > timedelta(seconds=0)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("invalid://example.org", id="scheme-not-http(s)"),
        pytest.param("://example.org", id="no-scheme"),
        pytest.param("http://", id="no-host"),
    ],
)
@pytest.mark.anyio
async def test_get_invalid_url(server, url):
    async with httpx.AsyncClient() as client:
        with pytest.raises((httpx.UnsupportedProtocol, httpx.LocalProtocolError)):
            await client.get(url)


@pytest.mark.anyio
async def test_build_request(server):
    url = server.url.copy_with(path="/echo_headers")
    headers = {"Custom-header": "value"}
    async with httpx.AsyncClient() as client:
        request = client.build_request("GET", url)
        request.headers.update(headers)
        response = await client.send(request)

    assert response.status_code == 200
    assert response.url == url

    assert response.json()["Custom-header"] == "value"


@pytest.mark.anyio
async def test_post(server):
    url = server.url
    async with httpx.AsyncClient() as client:
        response = await client.post(url, content=b"Hello, world!")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_post_json(server):
    url = server.url
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"text": "Hello, world!"})
    assert response.status_code == 200


@pytest.mark.anyio
async def test_stream_response(server):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", server.url) as response:
            body = await response.aread()

    assert response.status_code == 200
    assert body == b"Hello, world!"
    assert response.content == b"Hello, world!"


@pytest.mark.anyio
async def test_access_content_stream_response(server):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", server.url) as response:
            pass

    assert response.status_code == 200
    with pytest.raises(httpx.ResponseNotRead):
        response.content  # noqa: B018


@pytest.mark.anyio
async def test_stream_request(server):
    async def hello_world() -> typing.AsyncIterator[bytes]:
        yield b"Hello, "
        yield b"world!"

    async with httpx.AsyncClient() as client:
        response = await client.post(server.url, content=hello_world())
    assert response.status_code == 200


@pytest.mark.anyio
async def test_cannot_stream_sync_request(server):
    def hello_world() -> typing.Iterator[bytes]:  # pragma: no cover
        yield b"Hello, "
        yield b"world!"

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError):
            await client.post(server.url, content=hello_world())


@pytest.mark.anyio
async def test_raise_for_status(server):
    async with httpx.AsyncClient() as client:
        for status_code in (200, 400, 404, 500, 505):
            response = await client.request(
                "GET", server.url.copy_with(path=f"/status/{status_code}")
            )

            if 400 <= status_code < 600:
                with pytest.raises(httpx.HTTPStatusError) as exc_info:
                    response.raise_for_status()
                assert exc_info.value.response == response
            else:
                assert response.raise_for_status() is response


@pytest.mark.anyio
async def test_options(server):
    async with httpx.AsyncClient() as client:
        response = await client.options(server.url)
    assert response.status_code == 200
    assert response.text == "Hello, world!"


@pytest.mark.anyio
async def test_head(server):
    async with httpx.AsyncClient() as client:
        response = await client.head(server.url)
    assert response.status_code == 200
    assert response.text == ""


@pytest.mark.anyio
async def test_put(server):
    async with httpx.AsyncClient() as client:
        response = await client.put(server.url, content=b"Hello, world!")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_patch(server):
    async with httpx.AsyncClient() as client:
        response = await client.patch(server.url, content=b"Hello, world!")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_delete(server):
    async with httpx.AsyncClient() as client:
        response = await client.delete(server.url)
    assert response.status_code == 200
    assert response.text == "Hello, world!"


@pytest.mark.anyio
async def test_100_continue(server):
    headers = {"Expect": "100-continue"}
    content = b"Echo request body"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            server.url.copy_with(path="/echo_body"), headers=headers, content=content
        )

    assert response.status_code == 200
    assert response.content == content


@pytest.mark.anyio
async def test_context_managed_transport():
    class Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.events: list[str] = []

        async def aclose(self):
            # The base implementation of httpx.AsyncBaseTransport just
            # calls into `.aclose`, so simple transport cases can just override
            # this method for any cleanup, where more complex cases
            # might want to additionally override `__aenter__`/`__aexit__`.
            self.events.append("transport.aclose")

        async def __aenter__(self):
            await super().__aenter__()
            self.events.append("transport.__aenter__")

        async def __aexit__(self, *args):
            await super().__aexit__(*args)
            self.events.append("transport.__aexit__")

    transport = Transport()
    async with httpx.AsyncClient(transport=transport):
        pass

    assert transport.events == [
        "transport.__aenter__",
        "transport.aclose",
        "transport.__aexit__",
    ]


@pytest.mark.anyio
async def test_context_managed_transport_and_mount():
    class Transport(httpx.AsyncBaseTransport):
        def __init__(self, name: str) -> None:
            self.name: str = name
            self.events: list[str] = []

        async def aclose(self):
            # The base implementation of httpx.AsyncBaseTransport just
            # calls into `.aclose`, so simple transport cases can just override
            # this method for any cleanup, where more complex cases
            # might want to additionally override `__aenter__`/`__aexit__`.
            self.events.append(f"{self.name}.aclose")

        async def __aenter__(self):
            await super().__aenter__()
            self.events.append(f"{self.name}.__aenter__")

        async def __aexit__(self, *args):
            await super().__aexit__(*args)
            self.events.append(f"{self.name}.__aexit__")

    transport = Transport(name="transport")
    mounted = Transport(name="mounted")
    async with httpx.AsyncClient(
        transport=transport, mounts={"http://www.example.org": mounted}
    ):
        pass

    assert transport.events == [
        "transport.__aenter__",
        "transport.aclose",
        "transport.__aexit__",
    ]
    assert mounted.events == [
        "mounted.__aenter__",
        "mounted.aclose",
        "mounted.__aexit__",
    ]


def hello_world(request):
    return httpx.Response(200, text="Hello, world!")


@pytest.mark.anyio
async def test_client_closed_state_using_implicit_open():
    client = httpx.AsyncClient(transport=httpx.MockTransport(hello_world))

    assert not client.is_closed
    await client.get("http://example.com")

    assert not client.is_closed
    await client.aclose()

    assert client.is_closed
    # Once we're close we cannot make any more requests.
    with pytest.raises(RuntimeError):
        await client.get("http://example.com")

    # Once we're closed we cannot reopen the client.
    with pytest.raises(RuntimeError):
        async with client:
            pass  # pragma: no cover


@pytest.mark.anyio
async def test_client_closed_state_using_with_block():
    async with httpx.AsyncClient(transport=httpx.MockTransport(hello_world)) as client:
        assert not client.is_closed
        await client.get("http://example.com")

    assert client.is_closed
    with pytest.raises(RuntimeError):
        await client.get("http://example.com")


def unmounted(request: httpx.Request) -> httpx.Response:
    data = {"app": "unmounted"}
    return httpx.Response(200, json=data)


def mounted(request: httpx.Request) -> httpx.Response:
    data = {"app": "mounted"}
    return httpx.Response(200, json=data)


@pytest.mark.anyio
async def test_mounted_transport():
    transport = httpx.MockTransport(unmounted)
    mounts = {"custom://": httpx.MockTransport(mounted)}

    async with httpx.AsyncClient(transport=transport, mounts=mounts) as client:
        response = await client.get("https://www.example.com")
        assert response.status_code == 200
        assert response.json() == {"app": "unmounted"}

        response = await client.get("custom://www.example.com")
        assert response.status_code == 200
        assert response.json() == {"app": "mounted"}


@pytest.mark.anyio
async def test_async_mock_transport():
    async def hello_world(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Hello, world!")

    transport = httpx.MockTransport(hello_world)

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://www.example.com")
        assert response.status_code == 200
        assert response.text == "Hello, world!"


@pytest.mark.anyio
async def test_cancellation_during_stream():
    """
    If any BaseException is raised during streaming the response, then the
    stream should be closed.

    This includes:

    * `asyncio.CancelledError` (A subclass of BaseException from Python 3.8 onwards.)
    * `trio.Cancelled`
    * `KeyboardInterrupt`
    * `SystemExit`

    See https://github.com/encode/httpx/issues/2139
    """
    stream_was_closed = False

    def response_with_cancel_during_stream(request):
        class CancelledStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> typing.AsyncIterator[bytes]:
                yield b"Hello"
                raise KeyboardInterrupt()
                yield b", world"  # pragma: no cover

            async def aclose(self) -> None:
                nonlocal stream_was_closed
                stream_was_closed = True

        return httpx.Response(
            200, headers={"Content-Length": "12"}, stream=CancelledStream()
        )

    transport = httpx.MockTransport(response_with_cancel_during_stream)

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(KeyboardInterrupt):
            await client.get("https://www.example.com")
        assert stream_was_closed


@pytest.mark.anyio
async def test_server_extensions(server):
    url = server.url
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.get(url)
    assert response.status_code == 200
    assert response.extensions["http_version"] == b"HTTP/1.1"


def _start_slow_server() -> str:
    """Local server that sends headers, then pauses before the body.

    The pause is long enough for a client timeout or ``asyncio.wait_for`` to
    fire while the response body is still being read.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    def serve() -> None:
        try:
            conn, _addr = sock.accept()
            with conn:
                conn.recv(65536)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/octet-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
                time.sleep(2)
                with contextlib.suppress(OSError):
                    conn.sendall(b"6\r\nchunk0\r\n0\r\n\r\n")
        finally:
            sock.close()

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{port}/stream"


def _inflight_requests(client: httpx.AsyncClient) -> list[typing.Any]:
    """Requests the connection pool still considers active."""
    pool = client._transport._pool  # type: ignore[attr-defined]
    return list(pool._requests)


def _deadline_error(backend_name: str) -> type[BaseException]:
    """Exception raised when a mid-read deadline fires.

    AC1 cancels with ``asyncio.wait_for`` on asyncio. Trio applies the same
    deadline with ``trio.fail_after``, which raises ``TooSlowError``.
    """
    if backend_name == "trio":
        import trio

        return trio.TooSlowError
    return asyncio.TimeoutError


async def _await_with_deadline(
    coro: typing.Awaitable[typing.Any],
    seconds: float,
    backend_name: str,
) -> None:
    """Cancel `coro` after `seconds` (``asyncio.wait_for`` or trio equivalent)."""
    if backend_name == "trio":
        import trio

        with trio.fail_after(seconds):
            await coro
    else:
        await asyncio.wait_for(coro, timeout=seconds)


@pytest.mark.anyio
async def test_async_cancel_during_stream_closes_response(anyio_backend_name):
    """
    AC1: Cancelling via asyncio.wait_for timeout during a streaming read
    closes the response and returns the connection to the pool.
    """
    url = _start_slow_server()
    seen: list[httpx.Response] = []

    async def on_response(response: httpx.Response) -> None:
        seen.append(response)

    async with httpx.AsyncClient(
        timeout=None, trust_env=False, event_hooks={"response": [on_response]}
    ) as client:
        with pytest.raises(_deadline_error(anyio_backend_name)):
            await _await_with_deadline(client.get(url), 0.4, anyio_backend_name)
        assert seen[-1].is_closed
        assert _inflight_requests(client) == []


@pytest.mark.anyio
async def test_async_timeout_during_stream_closes_response():
    """
    AC2: A client read timeout during a streaming read closes the response
    and returns the connection to the pool.
    """
    url = _start_slow_server()
    seen: list[httpx.Response] = []

    async def on_response(response: httpx.Response) -> None:
        seen.append(response)

    timeout = httpx.Timeout(5.0, read=0.2)
    async with httpx.AsyncClient(
        timeout=timeout, trust_env=False, event_hooks={"response": [on_response]}
    ) as client:
        with pytest.raises((httpx.ReadTimeout, httpx.TimeoutException)):
            await client.get(url)
        assert seen[-1].is_closed
        assert _inflight_requests(client) == []


@pytest.mark.anyio
async def test_async_cancel_during_body_iteration_closes_response(anyio_backend_name):
    """
    AC1 edge case: Cancellation during explicit body iteration closes the
    response and returns the connection to the pool.
    """
    url = _start_slow_server()
    seen: list[httpx.Response] = []

    async def consume_with_timeout() -> None:
        async with client.stream("GET", url) as response:
            seen.append(response)
            async for _chunk in response.aiter_bytes():
                pass

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        with pytest.raises(_deadline_error(anyio_backend_name)):
            await _await_with_deadline(consume_with_timeout(), 0.4, anyio_backend_name)
        assert seen[-1].is_closed
        assert _inflight_requests(client) == []
