"""Air-gap / no-egress master switch (``HEADROOM_OFFLINE``).

A single predicate the individual egress paths consult so a regulated or
air-gapped deployment can disable **all** outbound network access with one
flag: the telemetry beacon, the update check, the license/usage reporter, and
HuggingFace model downloads. Each of those already had its own opt-out; this
is the one switch that turns them all off together and fails closed.

:func:`guard_egress` is the chokepoint version of that predicate: a path that
is about to open an outbound connection calls it and gets a loud
:class:`OfflineEgressBlocked` instead of a socket. Prefer it over a bare
``if is_offline(): return`` at any call site that actually dials out — the
meta-test in ``tests/test_offline_egress_chokepoint.py`` enumerates the HTTP
clients in the tree and requires each one to be behind the guard or carry a
written allowlist reason, so the guard is what keeps a newly-added egress path
from silently escaping the air-gap.

Kept at the top level (depends only on the stdlib) so any layer — telemetry,
proxy, model code — can import it without creating a package cycle.
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

# Whitespace trimmed off the raw value before matching. Enumerated rather than
# left to ``str.strip()``'s default, because that default is Python's
# ``str.isspace()`` — which includes U+001C-U+001F (the ASCII file/group/record
# /unit separators) — while Rust's ``str::trim()`` is the Unicode White_Space
# property, which does not. ``HEADROOM_OFFLINE=$'\x1c1'`` therefore read as
# offline to Python and online to Rust: one process air-gapped, the other not,
# from one environment variable. Neither side's default is more right than the
# other, so both now trim exactly this set. Mirrored by ``TRIM_CHARS`` in
# ``crates/headroom-core/src/offline.rs``; the parity test compares them.
#
# Case folding needs no such treatment: Python's ``str.lower()`` and Rust's
# ``to_ascii_lowercase()`` differ only on non-ASCII input, and no non-ASCII
# character lowercases into any character of "1"/"true"/"yes"/"on".
_TRIM_CHARS = " \t\n\r\x0b\x0c"

OFFLINE_ENV = "HEADROOM_OFFLINE"


def is_offline() -> bool:
    """Return True when ``HEADROOM_OFFLINE`` selects fully-offline operation."""
    return os.environ.get(OFFLINE_ENV, "").strip(_TRIM_CHARS).lower() in _TRUE_VALUES


def apply_offline_env() -> None:
    """Force HuggingFace/Transformers offline so model code uses only locally
    cached artifacts and never reaches the Hub.

    Idempotent and uses ``setdefault`` so an explicit operator override (e.g.
    ``HF_HUB_OFFLINE=0``) still wins. Call once early in startup.
    """
    if is_offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class OfflineEgressBlocked(RuntimeError):
    """Raised by :func:`guard_egress` when ``HEADROOM_OFFLINE`` is in force.

    A distinct, named type so callers can tell "the operator air-gapped this
    box" apart from "the network was flaky". That distinction matters because
    several Headroom egress paths deliberately fail OPEN on network errors
    (remote Kompress passes content through verbatim, the license reporter
    falls back to a cached grant). Fail-open is right for a flaky endpoint and
    WRONG for a policy refusal: swallowing this exception would turn the
    air-gap switch back into a suggestion. Anything that catches broad
    ``Exception`` around an egress call should re-raise this.
    """

    def __init__(self, purpose: str, destination: str | None = None) -> None:
        self.purpose = purpose
        self.destination = destination
        where = f" to {destination}" if destination else ""
        super().__init__(
            f"{OFFLINE_ENV} is set: refusing outbound network access for "
            f"{purpose}{where}. Unset {OFFLINE_ENV}, or turn off the feature "
            f"that needs this connection."
        )


def guard_egress(purpose: str, destination: str | None = None) -> None:
    """The single chokepoint every Headroom-initiated egress path must call.

    Raise :class:`OfflineEgressBlocked` when ``HEADROOM_OFFLINE`` selects
    offline operation; return silently otherwise.

    Call it BEFORE the socket exists — before constructing the client, not
    just before the request — so a pooled/keep-alive connection is never even
    opened. ``purpose`` and ``destination`` land verbatim in the message, so
    an operator who trips this learns which feature to turn off.

    Why raise instead of returning a no-op result: a silent skip is
    indistinguishable from success at the call site, so a future refactor can
    quietly reintroduce egress and nothing fails. A loud, named exception is
    what makes the meta-test in ``tests/test_offline_egress_chokepoint.py``
    able to assert "every egress path is behind this or allowlisted".

    Deliberately NOT exempted:

    * Loopback / in-cluster destinations. The guard cannot reliably tell an
      in-cluster collector from an internet host (DNS, proxies and sidecars
      all blur it), and ``HEADROOM_OFFLINE`` is documented as "no outbound
      traffic". Paths that only ever talk to ``127.0.0.1`` — the readiness
      probes, the proxy's own ``/livez`` check — simply do not call the guard
      and are allowlisted by name in the meta-test instead.
    * The proxy's own request forwarding to the caller's configured upstream.
      That is the caller's traffic, not Headroom phoning home; an air-gapped
      deployment points it at an on-prem endpoint and still needs it to work.
    """
    if is_offline():
        raise OfflineEgressBlocked(purpose, destination)
