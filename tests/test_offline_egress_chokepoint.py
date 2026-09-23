"""A-2: ``HEADROOM_OFFLINE=1`` must actually stop outbound connections.

The switch was documented as an air-gap guarantee but several egress paths
never consulted it, so a customer who set it and believed they were air-gapped
was wrong. These tests pin the guarantee down in two complementary ways:

* **Per-path socket tests.** Each previously-unguarded path is exercised with
  ``socket.socket.connect`` / ``socket.create_connection`` booby-trapped, so
  the test fails if a single connection is attempted. Asserting "no socket"
  rather than "raises" matters: a guard placed after the client is constructed
  would still pass a raises-check while leaking a connection.

* **A meta-test.** The per-path tests only cover the paths we already know
  about, and the whole point of a chokepoint is that path number four cannot
  forget it. ``test_every_egress_site_is_guarded_or_allowlisted`` enumerates
  the outbound HTTP clients in ``headroom/`` and requires each file either to
  route through ``headroom.offline.guard_egress`` or to carry a written reason
  in ``_EGRESS_ALLOWLIST``.

The Rust half of the switch (``crates/headroom-core/src/offline.rs``) is
runtime-tested by ``cargo test -p headroom-core``; what lives here is the
cross-language parity assertion, because the two implementations silently
drifting apart is the failure mode a Python-only test suite cannot see.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from headroom.offline import OfflineEgressBlocked, guard_egress

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "headroom"


# ──────────────────────────── socket booby trap ────────────────────────────


class SocketOpened(AssertionError):
    """Raised from the patched socket entry points.

    An ``AssertionError`` subclass so that a path which swallows broad
    ``Exception`` still surfaces this as a test failure rather than being
    mistaken for the network error it is imitating.
    """


@pytest.fixture
def no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to open a TCP connection fail the test.

    We patch the three entry points that every client in this tree bottoms out
    in: ``socket.create_connection`` (urllib, httpcore's sync backend),
    ``socket.socket.connect`` and ``socket.socket.connect_ex`` (everything
    else, including anyio's async backend). Creating a socket object is
    harmless — connecting is what leaves the box — so we trap the connect, not
    the constructor.
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise SocketOpened(f"outbound connection attempted while offline: {args!r}")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(socket.socket, "connect_ex", _boom)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")


# ─────────────────────────── the guard's own contract ───────────────────────


class TestGuardEgress:
    def test_raises_when_offline(self, offline: None) -> None:
        with pytest.raises(OfflineEgressBlocked) as excinfo:
            guard_egress("widget sync", "https://widgets.example.com")
        message = str(excinfo.value)
        # The operator has to be able to act on this without reading source:
        # which switch, which feature, which host.
        assert "HEADROOM_OFFLINE" in message
        assert "widget sync" in message
        assert "https://widgets.example.com" in message

    def test_is_a_no_op_when_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        assert guard_egress("widget sync", "https://widgets.example.com") is None

    def test_is_not_a_bare_runtime_error(self) -> None:
        # Callers need to distinguish a policy refusal from a flaky network so
        # their fail-open handlers can re-raise it. A bare RuntimeError would
        # be indistinguishable.
        assert issubclass(OfflineEgressBlocked, RuntimeError)
        assert OfflineEgressBlocked is not RuntimeError


# ──────────────────────── path 1: remote Kompress ───────────────────────────


class TestRemoteKompressOffline:
    def test_constructing_opens_no_socket(self, offline: None, no_sockets: None) -> None:
        from headroom.transforms.kompress_remote import RemoteKompressCompressor

        with pytest.raises(OfflineEgressBlocked):
            RemoteKompressCompressor(endpoint="https://kompress.example.com", token="secret")

    def test_compress_refuses_when_the_flag_flips_after_construction(
        self, monkeypatch: pytest.MonkeyPatch, no_sockets: None
    ) -> None:
        """The ContentRouter caches one compressor per instance, so an object
        built before the switch was set outlives it. The second guard, inside
        ``compress``, is what covers that window."""
        from headroom.transforms.kompress_remote import RemoteKompressCompressor

        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        compressor = RemoteKompressCompressor(endpoint="https://kompress.example.com")

        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        # Comfortably over KompressConfig.min_input_words (64), so we are past
        # the short-input passthrough and genuinely on the POST path.
        content = "alpha beta gamma delta " * 40
        with pytest.raises(OfflineEgressBlocked):
            compressor.compress(content)

    def test_compress_guard_is_outside_the_fail_open_handler(
        self, monkeypatch: pytest.MonkeyPatch, no_sockets: None
    ) -> None:
        """``compress`` turns every exception into a silent passthrough. If the
        guard sat inside that ``try``, the air-gap refusal would degrade to
        "compression just stopped working" with no signal — which is how the
        defect hid in the first place."""
        from headroom.transforms.kompress_remote import RemoteKompressCompressor

        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        compressor = RemoteKompressCompressor(endpoint="https://kompress.example.com")
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        content = "alpha beta gamma delta " * 40
        try:
            result = compressor.compress(content)
        except OfflineEgressBlocked:
            return
        pytest.fail(
            "compress() swallowed the offline refusal and passed content "
            f"through (compressed == original: {result.compressed == content})"
        )


# ────────────────────── path 2: OTLP metric exporter ────────────────────────


class TestOtlpExporterOffline:
    def test_configure_opens_no_socket(self, offline: None, no_sockets: None) -> None:
        from headroom.observability.metrics import OTelMetricsConfig, configure_otel_metrics

        config = OTelMetricsConfig(
            enabled=True,
            exporter="otlp_http",
            endpoint="http://collector.example.com:4318/v1/metrics",
        )
        with pytest.raises(OfflineEgressBlocked):
            configure_otel_metrics(config)

    def test_default_endpoint_is_blocked_too(self, offline: None, no_sockets: None) -> None:
        """With no explicit endpoint the OTEL SDK falls back to
        ``OTEL_EXPORTER_OTLP_ENDPOINT`` or ``localhost:4318``. "We did not name
        a host" is not the same as "we will not connect", so the guard must
        fire on the unset case as well."""
        from headroom.observability.metrics import OTelMetricsConfig, configure_otel_metrics

        with pytest.raises(OfflineEgressBlocked):
            configure_otel_metrics(OTelMetricsConfig(enabled=True, exporter="otlp_http"))

    def test_disabled_config_is_untouched(self, offline: None, no_sockets: None) -> None:
        # enabled=False never had an exporter; it must stay a quiet no-op
        # rather than becoming a new startup failure.
        from headroom.observability.metrics import OTelMetricsConfig, configure_otel_metrics

        assert configure_otel_metrics(OTelMetricsConfig(enabled=False)) is not None

    def test_console_exporter_still_works_offline(self, offline: None, no_sockets: None) -> None:
        """The escape hatch we point operators at. If this ever starts raising,
        an air-gapped deployment has no metrics story at all."""
        from headroom.observability import metrics as metrics_mod
        from headroom.observability.metrics import OTelMetricsConfig, configure_otel_metrics

        previous = metrics_mod._global_metrics
        try:
            configured = configure_otel_metrics(OTelMetricsConfig(enabled=True, exporter="console"))
            assert configured is not None
        finally:
            # configure_otel_metrics installs a process-global MeterProvider
            # with a background export timer. Left running, it keeps writing to
            # pytest's captured stdout after the test closes it. Tear it down
            # and put the previous facade back.
            provider = metrics_mod._owned_meter_provider
            if provider is not None:
                provider.shutdown()
            with metrics_mod._metrics_lock:
                metrics_mod._owned_meter_provider = None
                metrics_mod._owned_metrics_config = None
                metrics_mod._global_metrics = previous


# ───────────────── path 3: the Rust HuggingFace fetch (parity) ──────────────

_RUST_OFFLINE = REPO_ROOT / "crates" / "headroom-core" / "src" / "offline.rs"
_RUST_HF = REPO_ROOT / "crates" / "headroom-core" / "src" / "tokenizer" / "hf_impl.rs"


class TestRustOfflineParity:
    """The Rust core reads the same switch from the same process environment.

    The runtime assertion ("``from_pretrained`` returns ``Offline`` and never
    reaches the Hub") lives in ``hf_impl.rs``'s own ``#[cfg(test)]`` module,
    because a socket-level assertion has to run inside the process that would
    open the socket. What pytest can usefully add is the part cargo cannot
    see: that the two halves of one switch still agree.
    """

    def test_rust_guard_exists_and_reads_the_same_env_var(self) -> None:
        source = _RUST_OFFLINE.read_text(encoding="utf-8")
        assert 'pub const OFFLINE_ENV: &str = "HEADROOM_OFFLINE";' in source
        assert "pub fn guard_egress(" in source

    def test_truthy_values_match_the_python_side(self) -> None:
        from headroom.offline import _TRUE_VALUES

        source = _RUST_OFFLINE.read_text(encoding="utf-8")
        match = re.search(r"const TRUE_VALUES: \[&str; \d+\] = \[(.*?)\];", source, re.DOTALL)
        assert match, "TRUE_VALUES not found in crates/headroom-core/src/offline.rs"
        rust_values = set(re.findall(r'"([^"]+)"', match.group(1)))
        assert rust_values == set(_TRUE_VALUES), (
            "HEADROOM_OFFLINE truthiness has drifted between Python and Rust. "
            f"python={sorted(_TRUE_VALUES)} rust={sorted(rust_values)}. A "
            "deployment that reads as offline to one runtime and online to the "
            "other is exactly the hole this switch is supposed to close."
        )

    def test_hf_fetch_guards_before_it_builds_a_client(self) -> None:
        source = _RUST_HF.read_text(encoding="utf-8")
        assert "guard_egress(" in source, "hf_impl.rs does not consult the offline guard"
        guard_at = source.index("guard_egress(")
        api_at = source.index("hf_hub::api::sync::Api::new()")
        assert guard_at < api_at, (
            "the offline guard runs after Api::new(), which already resolves the "
            "Hub endpoint and builds the ureq agent — guard before the client "
            "exists, not before the request"
        )


# ──────────────────────────────── meta-test ─────────────────────────────────

# Anything that can open an outbound connection. Kept deliberately broad and
# textual: a regex over the tree catches a new egress path in review even when
# it lands in a module nobody thought to wire into the guard, which an
# import-graph check would not.
_EGRESS_PATTERNS = re.compile(
    r"""
      httpx\.(?:Async)?Client\(       # httpx sync/async client
    | (?<![\w.])requests\.(?:get|post|put|patch|delete|head|request|Session)\(
    | urlopen\(                       # urllib.request.urlopen, bare, or a
                                      # module-local `_urlopen` wrapper — the
                                      # `def` line itself is skipped by the
                                      # scanner, so only call sites count
    | aiohttp\.ClientSession\(
    | (?<![\w.])urllib3\.PoolManager\(
    """,
    re.VERBOSE,
)

# Every file below opens an outbound connection WITHOUT calling guard_egress,
# and each needs a reason that survives review. Categories used:
#
#   loopback      talks only to 127.0.0.1 — it never leaves the box, so the
#                 air-gap switch has nothing to protect.
#   gated         the caller already checks is_offline() before this code can
#                 run; routing it through guard_egress would be redundant.
#   user-traffic  the caller's own request being forwarded, not Headroom
#                 phoning home. An air-gapped deployment still needs it.
#   unguarded     genuinely still reachable under HEADROOM_OFFLINE. Recorded
#                 here on purpose rather than quietly ignored — out of scope
#                 for A-2, which covers the three paths the audit confirmed.
#
# Value is (number of egress sites in the file, reason). The count is part of
# the assertion so that ADDING a second client to an already-listed file trips
# this test too — otherwise the allowlist becomes a blanket exemption for the
# whole file forever.
_EGRESS_ALLOWLIST: dict[str, tuple[int, str]] = {
    "proxy/server.py": (
        2,
        "user-traffic: the proxy forwarding the caller's request to the "
        "upstream they configured. Blocking this would break every air-gapped "
        "deployment that points Headroom at an on-prem model endpoint, which "
        "is the main reason such a deployment exists.",
    ),
    "update_check.py": (
        1,
        "gated: is_update_check_enabled() returns False when is_offline(), so "
        "the request is never built. See headroom/update_check.py.",
    ),
    "telemetry/session.py": (
        1,
        "gated: the beacon upload is reached only via is_telemetry_enabled(), "
        "which returns False when is_offline(). See headroom/telemetry/beacon.py.",
    ),
    "telemetry/reporter.py": (
        1,
        "gated: UsageReporter is only constructed and started when "
        "`not (config.offline or is_offline())` — see the license-key branch "
        "in headroom/proxy/server.py.",
    ),
    "install/health.py": (
        1,
        "loopback: readiness/health probes against the operator's own proxy "
        "URL, used by the installers and `headroom doctor`.",
    ),
    "cli/wrap.py": (
        2,
        "loopback: both sites are hard-coded http://127.0.0.1:<port> calls to "
        "the locally running proxy (/health and /admin/runtime-env).",
    ),
    "cli/learn.py": (
        1,
        "loopback: hard-coded http://127.0.0.1:<port>/admin/runtime-env on the local proxy.",
    ),
    "providers/copilot/wrap.py": (
        1,
        "loopback: hard-coded http://127.0.0.1:<port>/health on the local proxy.",
    ),
    "testing/harness.py": (
        1,
        "loopback: the test harness waiting for its own subprocess proxy on "
        "127.0.0.1 to become ready. Test-only code.",
    ),
    "ccr/mcp_server.py": (
        3,
        "loopback: retrieval and liveness calls against the operator's local "
        "proxy_url (defaults to 127.0.0.1). The MCP server is a sidecar to the "
        "proxy, not an internet client.",
    ),
    "memory/adapters/embedders.py": (
        1,
        "loopback: OllamaEmbedder against the operator's own Ollama base_url, "
        "which defaults to 127.0.0.1:11434.",
    ),
    "copilot_auth.py": (
        6,
        "unguarded: GitHub Copilot device-flow auth and token exchange. "
        "Interactive, user-initiated `headroom auth` egress rather than "
        "background phone-home, and a Copilot subscription is unusable on an "
        "air-gapped box regardless. Out of scope for A-2 (which covers the "
        "three paths the audit confirmed); needs its own guard + CLI message.",
    ),
    "subscription/client.py": (
        1,
        "unguarded: Anthropic subscription usage polling. Same reasoning as "
        "copilot_auth.py — out of scope for A-2, needs its own guard.",
    ),
    "subscription/codex_rate_limits.py": (
        1,
        "unguarded: Codex rate-limit polling. Out of scope for A-2.",
    ),
    "subscription/copilot_quota.py": (
        1,
        "unguarded: Copilot quota polling. Out of scope for A-2.",
    ),
    "binaries.py": (
        1,
        "unguarded: release-binary downloads for `headroom install`. Install "
        "time, not proxy runtime; already refuses non-https and verifies the "
        "download. Out of scope for A-2.",
    ),
    "graph/installer.py": (
        1,
        "unguarded: codebase-memory-mcp release download during install. Same "
        "reasoning as binaries.py. Out of scope for A-2.",
    ),
    "evals/datasets.py": (
        2,
        "unguarded: BFCL eval-dataset download. Developer/benchmark tooling, "
        "never reached by the proxy at runtime. Out of scope for A-2.",
    ),
    "integrations/asgi.py": (
        1,
        "unguarded: Headroom Cloud compression mode, which is opt-in by "
        "definition (an air-gapped deployment does not configure a cloud API "
        "URL). Out of scope for A-2.",
    ),
    "integrations/litellm_callback.py": (
        1,
        "unguarded: Headroom Cloud compression mode. Same reasoning as "
        "integrations/asgi.py. Out of scope for A-2.",
    ),
}


def _egress_sites() -> dict[str, list[tuple[int, str]]]:
    """Map ``headroom/``-relative path -> [(line number, source line)]."""
    found: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        hits = []
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            # A `def _urlopen(...)` wrapper is a definition, not a call; the
            # call sites through it are counted separately.
            if stripped.startswith(("def ", "async def ")):
                continue
            if _EGRESS_PATTERNS.search(line):
                hits.append((number, stripped))
        if hits:
            found[path.relative_to(PACKAGE_ROOT).as_posix()] = hits
    return found


class TestEgressChokepointCoverage:
    """Fails when someone adds an outbound HTTP client that nothing checked.

    The failure message names the file and the exact lines, and tells the
    author the two acceptable resolutions, so the test is a code-review aid
    rather than a puzzle.
    """

    def test_every_egress_site_is_guarded_or_allowlisted(self) -> None:
        problems: list[str] = []
        for relpath, hits in _egress_sites().items():
            source = (PACKAGE_ROOT / relpath).read_text(encoding="utf-8")
            if "guard_egress" in source:
                continue
            entry = _EGRESS_ALLOWLIST.get(relpath)
            if entry is None:
                shown = "\n".join(f"        line {n}: {text}" for n, text in hits)
                problems.append(f"  headroom/{relpath} — not guarded, not allowlisted\n{shown}")
                continue
            expected, _reason = entry
            if len(hits) != expected:
                shown = "\n".join(f"        line {n}: {text}" for n, text in hits)
                problems.append(
                    f"  headroom/{relpath} — allowlist records {expected} egress "
                    f"site(s), found {len(hits)}\n{shown}"
                )

        assert not problems, (
            "New or changed outbound HTTP egress found in headroom/.\n\n"
            + "\n".join(problems)
            + "\n\nHEADROOM_OFFLINE=1 is documented as an air-gap switch, so "
            "every path that opens a connection must either:\n"
            "  1. call headroom.offline.guard_egress(purpose, destination) "
            "BEFORE the client is constructed, or\n"
            "  2. be added to _EGRESS_ALLOWLIST in this file with a written "
            "reason (loopback / gated / user-traffic / unguarded) and the "
            "number of egress sites in the file.\n"
            "Do not silence this by widening the regex."
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        """A stale entry is worse than a missing one: it reads as a reviewed
        decision about code that no longer exists, and it hides the next real
        addition to that file behind a count that was never re-checked."""
        sites = _egress_sites()
        stale = sorted(set(_EGRESS_ALLOWLIST) - set(sites))
        assert not stale, (
            f"_EGRESS_ALLOWLIST entries no longer have any egress site; delete them: {stale}"
        )

    def test_allowlist_reasons_are_written_out(self) -> None:
        """Guards the guard: an entry with an empty or placeholder reason is an
        exemption nobody justified."""
        for relpath, (count, reason) in _EGRESS_ALLOWLIST.items():
            assert count > 0, f"{relpath}: egress-site count must be positive"
            assert len(reason) >= 40, f"{relpath}: allowlist reason is too thin to review"
            assert reason.split(":")[0] in {
                "loopback",
                "gated",
                "user-traffic",
                "unguarded",
            }, f"{relpath}: reason must start with a known category, got {reason!r}"
