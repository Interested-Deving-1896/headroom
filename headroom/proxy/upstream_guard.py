"""SSRF guard for client-supplied upstream base URLs (WEB-01).

Clients may redirect the proxy's upstream via the ``x-headroom-base-url`` header
(BYOK / custom OpenAI-compatible endpoints). Without validation this lets a
caller turn the proxy into a confused deputy — reaching cloud-metadata
(``169.254.169.254``) or internal RFC1918 hosts the caller cannot reach directly.

Policy:
  * Default: reject destinations that resolve to private, loopback, link-local,
    or otherwise non-public addresses. Public hosts (api.openai.com, api.x.ai,
    Azure, ...) are allowed so ordinary BYOK keeps working.
  * When ``HEADROOM_ALLOWED_BASE_URLS`` is set (comma-separated hosts or URLs),
    bare hosts permit every safe scheme/port for that host, while URLs permit
    only their exact normalized origin. Because that is an explicit operator
    choice, allowlisted destinations may point at internal/on-prem endpoints.

Answering "is this safe?" is only half the job: the answer has to survive to the
socket. Every accepted destination therefore has the addresses it was judged on
recorded here (see :func:`validated_addresses`), and the proxy's HTTP client
dials one of *those* instead of resolving the name a second time —
``headroom/proxy/upstream_pinning.py`` is the connect-time half.

This module intentionally depends only on the standard library so it is safe to
import from any handler without risking an import cycle.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from urllib.parse import urlparse

ALLOWED_BASE_URLS_ENV = "HEADROOM_ALLOWED_BASE_URLS"

# `socket.getaddrinfo` has no timeout parameter and runs on whatever thread
# calls it -- which, for the proxy, is the event loop. A caller-supplied host
# that resolves slowly therefore stalls every other in-flight request, so the
# lookup is bounded here and fails closed when it overruns. Callers already in
# async context should prefer `is_safe_upstream_url_async`, which keeps the
# wait off the loop entirely.
RESOLVE_TIMEOUT_ENV = "HEADROOM_UPSTREAM_RESOLVE_TIMEOUT_S"
_DEFAULT_RESOLVE_TIMEOUT_S = 3.0
_RESOLVER_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hr-upstream-dns")


def _resolve_timeout_seconds() -> float:
    raw = (os.environ.get(RESOLVE_TIMEOUT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_RESOLVE_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RESOLVE_TIMEOUT_S
    return value if value > 0 else _DEFAULT_RESOLVE_TIMEOUT_S


_SAFE_SCHEMES = {"http", "https", "ws", "wss"}

# Pinning the addresses a check was made against.
#
# `socket.getaddrinfo` below and the HTTP client's own lookup at connect time
# are two separate resolutions of the same name, and an attacker who controls
# its authoritative DNS answers them differently: a public address for the
# check, 169.254.169.254 (or an RFC1918 host) for the connection. That is DNS
# rebinding, and it makes a verdict based only on the first answer advisory --
# the socket never sees it. So the addresses that were actually judged are
# recorded here for the connection to dial.
#
# The window is short on purpose, and deliberately not configurable: a pin only
# has to bridge the microseconds between a check and the connection it
# authorises, while an operator who stretched it to hours would be pinning
# legitimate upstreams to addresses they have since moved off. An expired pin is
# simply absent, which returns that host to ordinary resolution -- exactly the
# behaviour every unchecked host already has -- and the next request through the
# guard re-validates and re-pins it.
_PIN_TTL_SECONDS = 60.0
# Hostnames here are caller-supplied, so the store is bounded; least recently
# written out first.
_PIN_MAX_ENTRIES = 512
_PIN_LOCK = threading.Lock()
_PINS: dict[str, tuple[float, tuple[str, ...]]] = {}


def _record_validated_addresses(host: str, addresses: tuple[str, ...]) -> None:
    """Remember what ``host`` was judged on, for the connection that follows."""
    if not host or not addresses:
        return
    expires_at = time.monotonic() + _PIN_TTL_SECONDS
    with _PIN_LOCK:
        # Re-insert rather than update so dict order stays write order, which is
        # what the eviction below pops from.
        _PINS.pop(host, None)
        _PINS[host] = (expires_at, addresses)
        while len(_PINS) > _PIN_MAX_ENTRIES:
            _PINS.pop(next(iter(_PINS)))


def validated_addresses(host: str) -> tuple[str, ...] | None:
    """Return the addresses a recent check accepted for ``host``, if any.

    ``None`` means "not this module's business", never "unsafe": the host may
    simply never have been checked -- configured provider upstreams are not
    client-supplied and never pass through here -- or its pin may have aged out.
    Whether a destination is allowed at all is :func:`is_safe_upstream_url`'s
    answer, not this one's.
    """
    # Every upstream connection the proxy opens asks this question, while only
    # deployments that actually accept `x-headroom-base-url` ever have an answer
    # -- so get the empty case out of the way without taking the lock. Reading a
    # dict's truthiness is atomic, and a pin landing concurrently is one the
    # caller could not have been waiting on anyway.
    if not _PINS:
        return None
    key = (host or "").strip().lower()
    if not key:
        return None
    with _PIN_LOCK:
        entry = _PINS.get(key)
        if entry is None:
            return None
        expires_at, addresses = entry
        if expires_at <= time.monotonic():
            del _PINS[key]
            return None
    return addresses


def clear_validated_addresses() -> None:
    """Drop every pin. For tests and for a proxy restarting its HTTP clients."""
    with _PIN_LOCK:
        _PINS.clear()


def _allowlisted_destinations() -> tuple[set[str], set[tuple[str, str, int]]] | None:
    raw = os.environ.get(ALLOWED_BASE_URLS_ENV)
    if not raw or not raw.strip():
        return None
    hosts: set[str] = set()
    origins: set[tuple[str, str, int]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "://" not in item:
            parsed = urlparse(f"//{item}")
            if parsed.hostname:
                hosts.add(parsed.hostname.lower())
            continue
        parsed = urlparse(item)
        if parsed.scheme.lower() not in _SAFE_SCHEMES or not parsed.hostname:
            continue
        try:
            port = parsed.port
        except ValueError:
            continue
        if port is None:
            port = 443 if parsed.scheme.lower() in {"https", "wss"} else 80
        origins.add((parsed.scheme.lower(), parsed.hostname.lower(), port))
    return hosts, origins


# RFC 6052 / RFC 8215: these IPv6 prefixes embed an IPv4 address in their low
# 32 bits, and `ipaddress` reports the well-known one as globally routable. On a
# NAT64 network `64:ff9b::7f00:1` reaches 127.0.0.1, so the embedded address is
# what has to be judged. 6to4, Teredo and IPv4-mapped forms are already caught
# by the `is_global` test below.
_NAT64_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("64:ff9b:1::/48"),
)


def _nat64_embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if not any(addr in prefix for prefix in _NAT64_PREFIXES):
        return None
    try:
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    except (ipaddress.AddressValueError, ValueError):  # pragma: no cover - defensive
        return None


def _is_internal_address(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable (e.g. scoped link-local) -> treat as unsafe
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    ):
        return True
    # Anything not globally routable. This is what catches RFC 6598 shared
    # address space (100.64.0.0/10) -- which `is_private` does not flag, and
    # which reaches ISP and cloud-internal infrastructure -- along with
    # benchmarking (198.18/15), TEST-NET, 240/4, 6to4 and Teredo tunnels that
    # embed an internal IPv4, and any future special-use range the stdlib
    # learns about.
    if not addr.is_global:
        return True
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = _nat64_embedded_ipv4(addr)
        if embedded is not None and _is_internal_address(str(embedded)):
            return True
    return False


def is_safe_upstream_url(url: str) -> bool:
    """Return True if ``url`` is a safe client-chosen upstream destination.

    In allowlist mode only allowlisted hosts pass. Otherwise the host is
    resolved and rejected if any resolved address is internal/metadata, which
    also catches DNS names that point at private space.

    A destination accepted on its addresses also has them pinned, so the
    connection that follows cannot be re-pointed by a second DNS answer.
    Allowlist mode pins nothing: it admits a host by name without resolving it
    at all, precisely so split-horizon and on-prem endpoints -- whose addresses
    are the operator's business, and may legitimately move or round-robin --
    keep working.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False

    allow = _allowlisted_destinations()
    if allow is not None:
        hosts, origins = allow
        if host.lower() in hosts:
            return True
        try:
            port = parsed.port
        except ValueError:
            return False
        if port is None:
            port = 443 if parsed.scheme.lower() in {"https", "wss"} else 80
        return (parsed.scheme.lower(), host.lower(), port) in origins

    try:
        infos = _RESOLVER_POOL.submit(
            socket.getaddrinfo, host, None, 0, 0, socket.IPPROTO_TCP
        ).result(timeout=_resolve_timeout_seconds())
    except (OSError, _FutureTimeout):
        # Resolution and connection are separate operations, so allowing a DNS
        # miss here would fail open if the name resolves on the later lookup.
        # A lookup that overruns the budget is treated the same way.
        # Operators can explicitly allowlist split-horizon/internal endpoints.
        return False
    addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
    if any(_is_internal_address(address) for address in addresses):
        return False
    # Every address in this answer passed, so pin them all: the connection may
    # dial any one of them and still be dialling something this check accepted.
    # Keeping the whole set (in resolver order) rather than a single winner
    # leaves the connection somewhere to go when the first address is an AAAA on
    # a host with no IPv6 route -- the fallback the OS resolver would otherwise
    # have done for us.
    _record_validated_addresses(host.lower(), addresses)
    return True


async def is_safe_upstream_url_async(url: str) -> bool:
    """Async form of :func:`is_safe_upstream_url` for event-loop callers.

    Same policy; the blocking resolution runs off the loop so a hostile or
    slow-resolving hostname cannot stall unrelated in-flight requests.
    """
    return await asyncio.to_thread(is_safe_upstream_url, url)
