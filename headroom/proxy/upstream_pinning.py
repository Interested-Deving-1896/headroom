"""Connect-time half of the SSRF guard: dial the address that was validated.

:mod:`headroom.proxy.upstream_guard` resolves a caller-supplied upstream and
judges the answer -- and then hands the *hostname* to httpx, which resolves it
again when it opens the socket. Two resolutions mean two chances to answer, and
an attacker who controls the name's authoritative DNS takes both: a public
address for the check, ``169.254.169.254`` for the connection. The verdict was
never wrong; it just never reached the socket (DNS rebinding / TOCTOU).

This module carries it there. It replaces the connection pool's network backend
with one that substitutes a validated address for the hostname at ``connect_tcp``
time, and *only* that: the request still carries the hostname in its URL, so

  * httpx still sends ``Host: <hostname>`` -- an upstream that routes by Host
    (every gateway, every multi-tenant provider) still routes correctly; and
  * httpcore still passes the hostname to ``start_tls`` as ``server_hostname``,
    so SNI and certificate hostname verification are untouched.

That is the whole reason the substitution happens down at the socket instead of
by rewriting the URL to an IP literal. A rewritten URL takes the Host header and
the certificate identity down with it -- the upstream misroutes, and the
handshake either fails or has to be weakened to an unverified one to work at
all. It would also collapse two hostnames sharing an address into one pooled
origin, letting a connection whose certificate was checked for one host serve
requests for the other.

Scope, deliberately narrow: this layer answers only for names the guard has just
validated. Every configured provider upstream -- Anthropic, OpenAI, Gemini,
Bedrock, Copilot, LiteLLM, a Kong gateway -- is operator-configured rather than
client-supplied, never passes through the guard, and so resolves exactly as it
did before. Redirects and retries re-enter through the same path and are treated
the same way. When an HTTP/SOCKS proxy is configured the pool dials the
*proxy's* host, which is never a pinned name, so proxied deployments are
likewise unchanged (their DNS was always the proxy's to do).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import httpcore
import httpx

from headroom.proxy.upstream_guard import validated_addresses


class PinnedAddressBackend(httpcore.AsyncNetworkBackend):
    """Network backend that dials a validated address in place of a pinned name.

    Everything else is delegated untouched, including the ``connect_tcp``
    keyword arguments -- ``local_address`` and ``socket_options`` are how
    operators bind egress to a chosen interface, so dropping them here would
    quietly change which source address the proxy connects from.
    """

    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        # httpcore hands us the origin host as `str`; be tolerant anyway, since
        # a bytes host would silently miss every pin and fail open.
        name = host.decode("ascii", "ignore") if isinstance(host, bytes) else host
        addresses = validated_addresses(name)

        async def dial(target: str) -> httpcore.AsyncNetworkStream:
            return await self._inner.connect_tcp(
                target,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )

        if not addresses:
            # Not a checked destination (or the pin aged out): resolve as before.
            return await dial(host)

        # Try each validated address in turn, in the resolver's own order. A
        # later one is reached only when an earlier one could not be connected
        # to at all -- the answer is unreachable on this host's network, e.g. an
        # AAAA record where there is no IPv6 route. That is the fallback the OS
        # resolver would have made had we handed it the name, and it costs
        # nothing in safety because every address here passed the same check.
        # Only connect failures fall through: once a socket is open, whatever
        # happens on it belongs to the caller. The last address is dialled
        # outside the loop so its failure propagates as itself.
        for address in addresses[:-1]:
            try:
                return await dial(address)
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                continue
        return await dial(addresses[-1])

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._inner.connect_unix_socket(
            path, timeout=timeout, socket_options=socket_options
        )

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _pools(client: httpx.AsyncClient) -> Iterator[Any]:
    """Yield the httpcore pool behind every transport ``client`` may dial through.

    Mounted transports matter as much as the primary one: httpx builds them for
    ``HTTPS_PROXY``/``NO_PROXY``-style environments, and a request routed to one
    of those would otherwise leave through an unpinned pool.
    """
    transports = [getattr(client, "_transport", None)]
    transports.extend(getattr(client, "_mounts", {}).values())
    for transport in transports:
        pool = getattr(transport, "_pool", None)
        if pool is not None:
            yield pool


def install_upstream_pinning(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Make ``client`` dial validated addresses for checked hosts. Returns it.

    Raises ``RuntimeError`` when httpx/httpcore no longer expose the connection
    pool this hooks into. That is fail-closed on purpose: the alternative is a
    proxy that starts happily and silently re-opens the rebinding hole, and the
    versions involved are locked in ``uv.lock``, so it can only fire on a
    deliberate dependency change -- exactly when someone should be looking.
    """
    hooked = 0
    for pool in _pools(client):
        backend = getattr(pool, "_network_backend", None)
        if backend is None:
            continue
        if isinstance(backend, PinnedAddressBackend):  # already installed
            hooked += 1
            continue
        pool._network_backend = PinnedAddressBackend(backend)
        hooked += 1
    if not hooked:
        raise RuntimeError(
            "cannot pin validated upstream addresses: this httpx/httpcore "
            "exposes no connection pool to hook. Refusing to run unpinned — "
            "see headroom/proxy/upstream_pinning.py."
        )
    return client
