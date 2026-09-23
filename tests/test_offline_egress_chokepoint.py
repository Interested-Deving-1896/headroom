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
  the outbound clients in ``headroom/`` and requires each **site** either to
  have a ``guard_egress`` call that dominates it or to be counted in the
  allowlist with a written reason. Per site, not per file: a file-wide
  "contains the string guard_egress" check — the first version of this test —
  exempts a module because of a word in its docstring, and makes the
  allowlist's site counts unreachable for every file that guards anything.
  ``TestSiteScannerRules`` pins each of those bypasses.

* **The same sweep over ``crates/``.** ``TestRustEgressChokepointCoverage``
  applies a text-level version of the rule to the Rust sources. ``crates/``
  was outside the Python scan entirely, which is how the Kompress model and
  fastembed weight downloads stayed open in the same change that guarded the
  Rust tokenizer fetch for precisely the reason that applied to all three.

The runtime behaviour of the Rust switch (``crates/headroom-core/src/
offline.rs``) is tested by ``cargo test -p headroom-core``; what lives here is
the cross-language parity assertion, because the two implementations silently
drifting apart is the failure mode a Python-only test suite cannot see.
"""

from __future__ import annotations

import ast
import re
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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

    def test_is_outside_the_exception_hierarchy(self) -> None:
        """The refusal must survive a ``except Exception`` fail-open handler.

        It used to be a ``RuntimeError`` and the module docstring asked every
        broad handler to re-raise it. Nothing did: four reachable
        ``except Exception`` blocks turned the refusal into silent degradation,
        and a full ``/v1/messages`` request under ``HEADROOM_OFFLINE=1`` with a
        remote Kompress endpoint returned 200 with the content uncompressed
        while the guard had fired twice. The convention was the bug; the type
        is the fix.
        """
        assert issubclass(OfflineEgressBlocked, BaseException)
        assert not issubclass(OfflineEgressBlocked, Exception), (
            "OfflineEgressBlocked must stay outside Exception, or every "
            "`except Exception:` fail-open handler in the tree silently "
            "downgrades an air-gap refusal to 'that feature stopped working'."
        )

    def test_a_broad_exception_handler_cannot_swallow_it(self, offline: None) -> None:
        """The property above, exercised rather than asserted about."""
        swallowed = False
        try:
            try:
                guard_egress("widget sync", "https://widgets.example.com")
            except Exception:  # noqa: BLE001 — the whole point of the test
                swallowed = True
        except OfflineEgressBlocked:
            pass
        assert not swallowed


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


# ───────────── path 2b: the Langfuse OTLP trace exporter ────────────────────


class TestLangfuseExporterOffline:
    """The metric exporter's twin, missed when the metric one was guarded.

    Same shape — an OTLP/HTTP exporter plus a background batch timer — but
    pointed at ``cloud.langfuse.com`` by default rather than at whatever the
    operator configured, so if anything it is the more clear-cut egress of the
    two.
    """

    def test_configure_opens_no_socket(self, offline: None, no_sockets: None) -> None:
        from headroom.observability.tracing import (
            LangfuseTracingConfig,
            configure_langfuse_tracing,
        )

        config = LangfuseTracingConfig(enabled=True, public_key="pk", secret_key="sk")
        with pytest.raises(OfflineEgressBlocked) as excinfo:
            configure_langfuse_tracing(config)
        assert "cloud.langfuse.com" in str(excinfo.value)

    def test_disabled_config_is_untouched(self, offline: None, no_sockets: None) -> None:
        from headroom.observability.tracing import (
            LangfuseTracingConfig,
            configure_langfuse_tracing,
        )

        assert configure_langfuse_tracing(LangfuseTracingConfig(enabled=False)) is not None


# ────────── path 4: the Python half of the HuggingFace download ─────────────


class TestHuggingFaceDownloadOffline:
    """``onnx_runtime.hf_hub_download_local_first`` is the Python twin of the
    Rust Hub fetch this PR guarded, and it was left open — every ONNX model in
    the tree (Kompress, the image router, the memory embedders) resolves
    through it.

    Both halves of the contract are pinned here, because the interesting part
    is what is NOT refused: the guard sits on the network fallback only, so a
    pre-seeded air-gapped cache keeps working. A guard at the top of the
    function would pass a "raises" test and break every air-gapped deployment
    that did the thing we tell operators to do.
    """

    @staticmethod
    def _fake_hub(monkeypatch: pytest.MonkeyPatch, *, cached: str | None) -> list[bool]:
        import huggingface_hub
        from huggingface_hub.errors import LocalEntryNotFoundError

        seen: list[bool] = []

        def fake_download(
            repo_id: str,
            filename: str,
            *,
            revision: str | None = None,
            local_files_only: bool = False,
        ) -> str:
            seen.append(local_files_only)
            if local_files_only and cached is None:
                raise LocalEntryNotFoundError("cold cache")
            return cached or "/downloaded/from/the/hub"

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        return seen

    def test_a_cold_cache_is_refused(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.onnx_runtime import hf_hub_download_local_first

        seen = self._fake_hub(monkeypatch, cached=None)
        with pytest.raises(OfflineEgressBlocked):
            hf_hub_download_local_first("acme/model", "model.onnx")
        # The cache lookup ran; the network download never did.
        assert seen == [True]

    def test_a_warm_cache_still_resolves_offline(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.onnx_runtime import hf_hub_download_local_first

        self._fake_hub(monkeypatch, cached="/cache/acme/model.onnx")
        assert hf_hub_download_local_first("acme/model", "model.onnx") == "/cache/acme/model.onnx"

    def test_allow_network_false_still_raises_the_cache_error(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``allow_network=False`` never reaches the guard, so the caller keeps
        seeing the local-lookup error it already handles."""
        from huggingface_hub.errors import LocalEntryNotFoundError

        from headroom.onnx_runtime import hf_hub_download_local_first

        self._fake_hub(monkeypatch, cached=None)
        with pytest.raises(LocalEntryNotFoundError):
            hf_hub_download_local_first("acme/model", "model.onnx", allow_network=False)


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

    def test_trim_chars_match_the_python_side(self) -> None:
        """Value parity is only half of it.

        The old pair diffed the accepted token sets and nothing else, so it was
        green while Python used ``str.strip()`` (which strips U+001C-U+001F)
        and Rust used ``str::trim()`` (which does not). Normalisation is part
        of the contract; this pins the set both sides trim.
        """
        from headroom.offline import _TRIM_CHARS

        source = _RUST_OFFLINE.read_text(encoding="utf-8")
        match = re.search(r"const TRIM_CHARS: \[char; \d+\] = \[(.*?)\];", source, re.DOTALL)
        assert match, "TRIM_CHARS not found in crates/headroom-core/src/offline.rs"
        rust_chars = set()
        for literal in re.findall(r"'((?:\\u\{[0-9a-fA-F]+\}|\\.|[^'])+)'", match.group(1)):
            escape = re.fullmatch(r"\\u\{([0-9a-fA-F]+)\}", literal)
            if escape:
                rust_chars.add(chr(int(escape.group(1), 16)))
            else:
                rust_chars.add({"\\t": "\t", "\\n": "\n", "\\r": "\r"}.get(literal, literal))
        assert rust_chars == set(_TRIM_CHARS), (
            "HEADROOM_OFFLINE trimming has drifted between Python and Rust. "
            f"python={sorted(map(ord, _TRIM_CHARS))} rust={sorted(map(ord, rust_chars))}. "
            "A value that normalises differently in the two runtimes air-gaps "
            "one half of the process and not the other."
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", True),
            (" 1 ", True),
            ("\t1\n", True),
            ("\r\n TRUE \r\n", True),
            ("\x0byes\x0c", True),
            # U+001C-U+001F: whitespace to str.strip(), not to str::trim().
            # This pair is the regression the token-set diff could not see.
            ("\x1c1", False),
            ("1\x1f", False),
            # Unicode spaces: whitespace to str::trim(), and (NBSP) to
            # str.strip() as well. Neither trims them now.
            ("\xa01", False),
            ("\u2007on", False),
            ("", False),
            (" ", False),
        ],
    )
    def test_normalisation_matches_the_rust_side(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        """Same table as ``offline::tests::normalisation_matches_python``.

        Duplicated rather than shared because the point is that two separate
        implementations agree; a shared fixture would only prove one of them
        reads the fixture.
        """
        from headroom.offline import is_offline

        monkeypatch.setenv("HEADROOM_OFFLINE", raw)
        assert is_offline() is expected

    def test_the_rust_normalisation_table_covers_the_same_cases(self) -> None:
        """If one side's table grows a case the other lacks, the pair stops
        being a parity test and becomes two independent tests that happen to
        share a name."""
        source = _RUST_OFFLINE.read_text(encoding="utf-8")
        assert "fn normalisation_matches_python()" in source
        for needle in ('"\\u{1c}1"', '"1\\u{1f}"', '"\\u{a0}1"', '"\\u{2007}on"'):
            assert needle in source, f"rust normalisation table is missing {needle}"

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


# ─────────── the refusal has to survive the fail-open handlers ──────────────


class TestRefusalSurvivesFailOpenHandlers:
    """Raising was never the hard part; being heard was.

    Under ``HEADROOM_OFFLINE=1`` + ``HEADROOM_KOMPRESS_ENDPOINT``, a full
    ``/v1/messages`` request used to return **200 with the content
    uncompressed** while ``guard_egress`` had fired twice: every fail-open
    ``except Exception`` between the guard and the response logged a warning
    and passed the content through. Each site below is one of those handlers,
    exercised with a compressor that refuses.
    """

    def test_thinking_compactor_does_not_swallow_it(self) -> None:
        """``_memo_compact`` wraps ``kompress.compress(...)`` — the very call
        the in-``compress()`` guard protects — in ``except Exception``."""
        from headroom.transforms.thinking_compactor import _memo_compact

        class _Refusing:
            def compress(self, text: str, allow_download: bool = False) -> object:
                raise OfflineEgressBlocked("remote Kompress inference", "https://k.example.com")

        with pytest.raises(OfflineEgressBlocked):
            _memo_compact("a2 offline probe, unique so the memo cache misses", _Refusing())

    def test_kompress_model_ready_does_not_report_a_refusal_as_ready(self) -> None:
        """``_kompress_model_ready`` answered ``except Exception: return True``
        — reporting a policy refusal as "the model is ready"."""
        from headroom.transforms.content_router import ContentRouter

        class _Stub:
            config = SimpleNamespace(enable_kompress=True)
            _runtime_kompress_model = None
            _kompress_model_ready = ContentRouter._kompress_model_ready

            def _get_kompress(self) -> object:
                raise OfflineEgressBlocked("remote Kompress inference", "https://k.example.com")

        with pytest.raises(OfflineEgressBlocked):
            _Stub()._kompress_model_ready()

    def test_the_router_reaching_for_remote_kompress_propagates(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.transforms.content_router import ContentRouter

        monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://kompress.example.com")

        class _Stub:
            config = SimpleNamespace(ccr_inject_marker=True)
            _kompress_remote = None
            _get_remote_kompress = ContentRouter._get_remote_kompress

        with pytest.raises(OfflineEgressBlocked):
            _Stub()._get_remote_kompress()

    def test_the_native_detector_fallback_reraises_it(self) -> None:
        """``_detect_content``'s ``except BaseException`` degrades a native
        panic to the pure-Python detector. It re-raises the control-flow
        BaseExceptions; the air-gap refusal is now on that list."""
        from headroom.transforms import content_router

        source = Path(content_router.__file__).read_text(encoding="utf-8")
        assert (
            "except (KeyboardInterrupt, SystemExit, GeneratorExit, OfflineEgressBlocked):" in source
        )


class TestBackgroundDownloadThreads:
    """The two daemon threads that fetch the Kompress model are the one place
    where propagating is the wrong answer.

    They are background *refreshes*, not the request path, and an unhandled
    ``BaseException`` in a thread reaches ``threading.excepthook`` as a bare
    traceback — on every air-gapped startup with a cold cache. Both handle the
    refusal explicitly and report it with the switch named, which is what
    ``is_offline()``'s "skip an optional refresh" case is for.
    """

    def test_prefetch_reports_the_refusal_and_stops(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        import huggingface_hub
        from huggingface_hub.errors import LocalEntryNotFoundError

        from headroom.transforms import kompress_compressor

        attempts: list[str] = []

        def fake_download(repo_id, filename, *, revision=None, local_files_only=False):
            attempts.append(filename)
            raise LocalEntryNotFoundError("cold cache")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        monkeypatch.setattr(kompress_compressor, "_kompress_cache", {})

        with caplog.at_level("WARNING"):
            assert kompress_compressor.prefetch_kompress_artifacts("acme/model") is False
        assert "HEADROOM_OFFLINE" in caplog.text
        # One candidate tried, then it stops: every other candidate would be
        # refused for the same reason.
        assert len(attempts) == 1

    def test_background_download_reports_the_refusal(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        from headroom.transforms import kompress_compressor

        def refuse(*args: object, **kwargs: object) -> None:
            raise OfflineEgressBlocked("HuggingFace download of acme/model", "huggingface.co")

        monkeypatch.setattr(kompress_compressor, "_load_kompress", refuse)
        with caplog.at_level("WARNING"):
            kompress_compressor._background_download("acme/model", "cpu")
        assert "refused" in caplog.text
        assert "HEADROOM_OFFLINE" in caplog.text


class TestBroadHandlerSweep:
    """``except Exception`` can no longer swallow the refusal — the type sees
    to that. ``except BaseException`` and bare ``except:`` still can, so they
    are enumerated here and each one has to either re-raise or carry a reason.

    This is the test the brief asked for: "fails if a new broad handler
    swallows it". It is deliberately a whole-tree sweep rather than a list of
    the four handlers that were found, because the four were found by hand and
    the fifth will not be.
    """

    # file -> (line count, reason). Each of these hands the caught exception
    # back to another thread that re-raises it, so the refusal is delayed but
    # never lost.
    _ALLOWED: dict[str, tuple[int, str]] = {
        "tokenizers/huggingface.py": (
            1,
            "relayed: the handler appends to `error` and the calling thread "
            "re-raises it after join(). Not a swallow, a hand-off.",
        ),
        "tokenizers/tiktoken_counter.py": (
            1,
            "relayed: stores into box['err'], re-raised in the calling thread.",
        ),
        "transforms/content_router.py": (
            2,
            "relayed: both are watchdog-thread bodies that store into a box "
            "the caller re-raises from. The third handler in this file, the "
            "native-detect degrade path, re-raises OfflineEgressBlocked "
            "explicitly and so does not appear here.",
        ),
    }

    @staticmethod
    def _broad_handlers() -> dict[str, list[tuple[int, str]]]:
        found: dict[str, list[tuple[int, str]]] = {}
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (UnicodeDecodeError, SyntaxError):  # pragma: no cover - defensive
                continue
            lines = source.splitlines()
            hits: list[tuple[int, str]] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                # A sibling handler that catches the refusal first makes every
                # later handler on this try safe.
                sibling_reraises = any(
                    handler.type is not None
                    and "OfflineEgressBlocked" in ast.unparse(handler.type)
                    and any(isinstance(stmt, ast.Raise) for stmt in handler.body)
                    for handler in node.handlers
                )
                for handler in node.handlers:
                    caught = "" if handler.type is None else ast.unparse(handler.type)
                    if handler.type is not None and "BaseException" not in caught:
                        continue
                    if sibling_reraises:
                        continue
                    if isinstance(handler.body[-1], ast.Raise):
                        continue  # unconditional re-raise
                    if "OfflineEgressBlocked" in ast.unparse(handler):
                        continue  # handled explicitly inside
                    hits.append((handler.lineno, lines[handler.lineno - 1].strip()))
            if hits:
                found[path.relative_to(PACKAGE_ROOT).as_posix()] = sorted(hits)
        return found

    def test_no_broad_handler_swallows_the_refusal(self) -> None:
        problems: list[str] = []
        found = self._broad_handlers()
        for relpath, hits in found.items():
            entry = self._ALLOWED.get(relpath)
            if entry is None:
                shown = "\n".join(f"        line {n}: {text}" for n, text in hits)
                problems.append(f"  headroom/{relpath}\n{shown}")
            elif len(hits) != entry[0]:
                shown = "\n".join(f"        line {n}: {text}" for n, text in hits)
                problems.append(
                    f"  headroom/{relpath} — allowed {entry[0]} handler(s), found {len(hits)}"
                    f"\n{shown}"
                )
        assert not problems, (
            "A broad `except BaseException:` / bare `except:` can swallow "
            "OfflineEgressBlocked.\n\n"
            + "\n".join(problems)
            + "\n\nOfflineEgressBlocked derives from BaseException so that no "
            "`except Exception:` can degrade an air-gap refusal into 'that "
            "feature stopped working'. A handler that catches BaseException "
            "undoes that. Either:\n"
            "  1. re-raise unconditionally (`raise` as the last statement), or\n"
            "  2. add `except OfflineEgressBlocked: raise` ahead of it, or\n"
            "  3. record it in _ALLOWED here with a written reason."
        )

    def test_allowed_reasons_are_written_out(self) -> None:
        for relpath, (count, reason) in self._ALLOWED.items():
            assert count > 0, relpath
            assert len(reason) >= 40, f"{relpath}: reason is too thin to review"

    def test_allowed_has_no_stale_entries(self) -> None:
        found = self._broad_handlers()
        stale = sorted(set(self._ALLOWED) - set(found))
        assert not stale, f"entries with no broad handler left; delete them: {stale}"


# ───────────────── startup refuses a contradictory configuration ────────────


class TestStartupRefusal:
    """An air-gap contradiction is a configuration error, so it is settled at
    startup with the same shape ``_check_rust_core`` uses: say what, say how to
    fix it, exit 78 (``EX_CONFIG``).

    Before this, ``configure_otel_metrics`` was called OUTSIDE the lifespan's
    ``try``, above the line that sets ``app.state.startup_error`` — so the
    refusal escaped as an unhandled error out of ``lifespan`` and took down a
    proxy that had been serving traffic, with a traceback instead of an
    explanation. Only operators who set ``HEADROOM_OTEL_METRICS_ENABLED=1``
    (default off) ever saw it, which is exactly the population that should not
    have to read a stack trace to learn they set two contradictory flags.
    """

    def test_remote_kompress_contradiction_exits_78(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from headroom.proxy import server

        monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://kompress.example.com")
        with pytest.raises(SystemExit) as excinfo:
            server._preflight_offline_egress()
        assert excinfo.value.code == 78
        message = capsys.readouterr().err
        assert "HEADROOM_KOMPRESS_ENDPOINT" in message
        assert "kompress.example.com" in message

    def test_otlp_metrics_contradiction_exits_78(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from headroom.proxy import server

        monkeypatch.delenv("HEADROOM_KOMPRESS_ENDPOINT", raising=False)
        monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENABLED", "1")
        with pytest.raises(SystemExit) as excinfo:
            server._configure_observability_or_refuse()
        assert excinfo.value.code == 78
        message = capsys.readouterr().err
        # The operator has to leave with a next action, not just a refusal.
        assert "HEADROOM_OTEL_METRICS_EXPORTER=console" in message

    def test_langfuse_contradiction_exits_78(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from headroom.proxy import server

        monkeypatch.delenv("HEADROOM_KOMPRESS_ENDPOINT", raising=False)
        monkeypatch.delenv("HEADROOM_OTEL_METRICS_ENABLED", raising=False)
        monkeypatch.setenv("HEADROOM_LANGFUSE_ENABLED", "1")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        with pytest.raises(SystemExit) as excinfo:
            server._configure_observability_or_refuse()
        assert excinfo.value.code == 78
        assert "HEADROOM_LANGFUSE_ENABLED" in capsys.readouterr().err

    def test_an_air_gapped_proxy_with_no_egress_configured_starts(
        self, offline: None, no_sockets: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The common case must stay a no-op — this is a refusal, not a new
        reason for an air-gapped proxy to fail to boot."""
        from headroom.proxy import server

        for name in (
            "HEADROOM_KOMPRESS_ENDPOINT",
            "HEADROOM_OTEL_METRICS_ENABLED",
            "HEADROOM_LANGFUSE_ENABLED",
        ):
            monkeypatch.delenv(name, raising=False)
        assert server._configure_observability_or_refuse() is None

    def test_the_lifespan_routes_through_the_refusing_wrapper(self) -> None:
        """The defect was one of placement, not of logic: the exporter call sat
        outside the lifespan's own try. Pin that it now goes through the
        wrapper, so a future edit cannot quietly move it back."""
        from headroom.proxy import server

        source = Path(server.__file__).read_text(encoding="utf-8")
        assert "_configure_observability_or_refuse()" in source
        lifespan_at = source.index("async def lifespan(")
        body = source[lifespan_at:]
        assert "configure_otel_metrics(" not in body, (
            "lifespan calls configure_otel_metrics directly again; the "
            "OfflineEgressBlocked it can raise is a BaseException and will "
            "escape uvicorn as an unhandled error"
        )


# ──────────────────────────────── meta-test ─────────────────────────────────
#
# The per-path tests above only cover the paths we already know about. This
# half is the standing guarantee: a NEW egress path cannot land without either
# calling the guard or being written into the allowlist with a reason.
#
# It decides per **egress site**, not per file. The first cut of this test
# skipped any file whose text contained "guard_egress" anywhere — including in
# a docstring — which made every allowlist count unreachable for guarded files
# and let a second, unguarded client slip into an already-guarded module. The
# scan is therefore built on the AST (so comments and docstrings cannot vouch
# for anything) and a site counts as guarded only when a guard_egress call
# DOMINATES it: same block or an enclosing block, textually earlier, in the
# same function. A guard in a sibling branch, in a nested function, or in an
# except: arm the site does not sit in is not a guard for that site.


@dataclass(frozen=True)
class _Site:
    """One place that can open an outbound connection."""

    line: int
    text: str
    callee: str
    guarded: bool


# Callee names (as ``ast.unparse`` renders them) that can open a connection.
# Matched against the whole dotted name, so ``self.client.post`` does not match
# ``requests.post`` and a local variable named ``urlopen`` does.
_EGRESS_CALLEES = re.compile(
    r"""^(?:
        httpx\.(?:Async)?Client                        # httpx sync/async client
      | httpx\.(?:get|post|put|patch|delete|head|request|stream)  # module-level verbs
      | requests\.(?:get|post|put|patch|delete|head|request|Session)
      | (?:urllib\.request\.)?urlopen | _urlopen       # urllib, bare or wrapped
      | aiohttp\.ClientSession
      | urllib3\.PoolManager
      | (?:huggingface_hub\.)?hf_hub_download          # the Python half of the HF fetch
      | (?:fastembed\.)?TextEmbedding                  # fastembed pulls ONNX weights from HF
      | OTLP(?:Metric|Span|Log)Exporter                # OTEL export, incl. its background timer
      | (?:openai\.)?(?:Async)?(?:OpenAI|AzureOpenAI)  # provider SDKs build their own
      | (?:anthropic\.)?(?:Async)?Anthropic(?:Bedrock|Vertex)?
    )$""",
    re.VERBOSE,
)

# Statement fields that hold a nested block. ``handlers``/``cases`` hold nodes
# that own a block rather than a block, so they are unwrapped separately.
_BLOCK_OWNERS: tuple[type, ...] = (ast.excepthandler,) + (
    (ast.match_case,) if hasattr(ast, "match_case") else ()
)

# A position in the statement tree: one (block identity, index) pair per level
# of nesting. Comparing two of these is how dominance is decided.
_Chain = tuple[tuple[int, int], ...]


def _child_blocks(node: ast.AST) -> Iterator[list[ast.stmt]]:
    """Yield the statement lists nested directly inside ``node``."""
    for _field, value in ast.iter_fields(node):
        if not isinstance(value, list) or not value:
            continue
        if isinstance(value[0], ast.stmt):
            yield value  # type: ignore[misc]
        else:
            for item in value:
                if isinstance(item, _BLOCK_OWNERS):
                    yield from _child_blocks(item)


def _statement_chains(module: ast.Module) -> list[tuple[ast.stmt, _Chain]]:
    """Every statement in the module, paired with its position chain."""
    out: list[tuple[ast.stmt, _Chain]] = []

    def walk(block: list[ast.stmt], prefix: _Chain) -> None:
        for index, statement in enumerate(block):
            chain: _Chain = (*prefix, (id(block), index))
            out.append((statement, chain))
            for nested in _child_blocks(statement):
                walk(nested, chain)

    walk(module.body, ())
    return out


def _own_expressions(statement: ast.stmt) -> Iterator[ast.AST]:
    """Expression nodes belonging to ``statement`` itself, not to its block.

    Descending into nested statements here would attribute an inner call to
    the outer compound statement and give it the wrong position, which is what
    dominance is computed from.
    """
    stack: list[ast.AST] = []
    for _field, value in ast.iter_fields(statement):
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, ast.AST) and not isinstance(item, (ast.stmt, *_BLOCK_OWNERS)):
                stack.append(item)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.stmt, *_BLOCK_OWNERS)):
                stack.append(child)


def _dominates(guard: _Chain, site: _Chain) -> bool:
    """True when a guard at ``guard`` always runs before a site at ``site``.

    The guard's own block must be the site's block or an ancestor of it, and
    the guard must come earlier in that block. That rejects, on purpose:

    * a guard inside an ``if``/``except``/nested ``def`` the site is not in —
      the guard's block is not on the site's ancestor chain;
    * a guard that appears later in the same block;
    * a guard anywhere in the file that simply shares a module with the site,
      which is all the first version of this test ever checked.

    It also (conservatively) rejects a guard that lives in a helper the site's
    function calls. That is the intended trade: the guard is cheap, and
    "somebody up the stack probably guards this" is how egress paths get lost.
    """
    if len(guard) > len(site):
        return False
    depth = len(guard) - 1
    if guard[:depth] != site[:depth]:
        return False
    guard_block, guard_index = guard[depth]
    site_block, site_index = site[depth]
    return guard_block == site_block and guard_index < site_index


def _is_egress_call(call: ast.Call) -> str | None:
    """The matched callee name, or None when this call cannot leave the box."""
    try:
        name = ast.unparse(call.func)
    except Exception:  # pragma: no cover - defensive
        return None
    if not _EGRESS_CALLEES.match(name):
        return None
    if name.endswith("hf_hub_download"):
        # ``local_files_only=True`` is a pure cache lookup: huggingface_hub
        # raises rather than dialling, so it opens no socket and needs no
        # guard. Counting it would force a guard onto the cache-hit path and
        # break exactly the pre-seeded air-gapped deployment we want to keep
        # working. (The network fallback beside it still counts.)
        for keyword in call.keywords:
            if (
                keyword.arg == "local_files_only"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                return None
    return name


def _python_egress_sites(source: str) -> list[_Site]:
    """Every egress site in one Python module, each marked guarded or not."""
    module = ast.parse(source)
    lines = source.splitlines()
    guards: list[_Chain] = []
    candidates: list[tuple[ast.Call, str, _Chain]] = []

    for statement, chain in _statement_chains(module):
        for node in _own_expressions(statement):
            if not isinstance(node, ast.Call):
                continue
            try:
                func_name = ast.unparse(node.func)
            except Exception:  # pragma: no cover - defensive
                func_name = ""
            if func_name.split(".")[-1] == "guard_egress":
                guards.append(chain)
                continue
            callee = _is_egress_call(node)
            if callee is not None:
                candidates.append((node, callee, chain))

    sites: list[_Site] = []
    for node, callee, chain in candidates:
        guarded = any(_dominates(guard, chain) for guard in guards)
        text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
        sites.append(_Site(line=node.lineno, text=text, callee=callee, guarded=guarded))
    return sorted(sites, key=lambda site: site.line)


# Every file below has at least one egress site that does NOT call
# guard_egress, and each needs a reason that survives review. Categories:
#
#   loopback      talks only to 127.0.0.1 — it never leaves the box, so the
#                 air-gap switch has nothing to protect.
#   gated         the caller already checks is_offline() before this code can
#                 run; routing it through guard_egress would be redundant.
#   user-traffic  the caller's own request being forwarded, not Headroom
#                 phoning home. An air-gapped deployment still needs it.
#   unguarded     genuinely still reachable under HEADROOM_OFFLINE. Recorded
#                 here on purpose rather than quietly ignored — out of scope
#                 for A-2, which covers the paths the audit confirmed.
#
# Value is (number of UNGUARDED egress sites in the file, reason). Guarded
# sites are not counted and do not need an entry, so a file can legitimately
# appear here and still route some of its egress through the chokepoint. The
# count is part of the assertion: adding a second unguarded client to a listed
# file trips this test, and so does adding one to a file that is fully guarded
# today — that file simply has no entry, so the new site is unallowlisted.
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
    "cli/mcp.py": (
        1,
        "loopback: `headroom mcp status` probing <proxy_url>/health, which is "
        "the operator's own local proxy (defaults to 127.0.0.1:8787).",
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
        2,
        "loopback: OllamaEmbedder against the operator's own Ollama base_url "
        "(defaults to 127.0.0.1:11434). The second site is OpenAIEmbedder's "
        "AsyncOpenAI client, which is unguarded and opt-in-cloud by "
        "configuration — same reasoning as memory/backends/direct_mem0.py, "
        "out of scope for A-2. (The HF model fetches in this file go through "
        "onnx_runtime.hf_hub_download_local_first, which now guards.)",
    ),
    "copilot_auth.py": (
        4,
        "unguarded: GitHub Copilot device-flow auth and token exchange. "
        "Interactive, user-initiated `headroom auth` egress rather than "
        "background phone-home, and a Copilot subscription is unusable on an "
        "air-gapped box regardless. Out of scope for A-2 (which covers the "
        "paths the audit confirmed); needs its own guard + CLI message.",
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
    "evals/batch_compression_eval.py": (
        2,
        "unguarded: the Anthropic/OpenAI SDK clients the batch compression "
        "benchmark drives. Developer tooling run by hand, never imported by "
        "the proxy. Out of scope for A-2.",
    ),
    "evals/memory/judge.py": (
        4,
        "unguarded: the OpenAI/Anthropic LLM-judge clients used to score eval "
        "runs (two ternaries, so four constructor sites). Developer tooling, "
        "same reasoning as evals/datasets.py. Out of scope for A-2.",
    ),
    "evals/html_extraction.py": (
        4,
        "unguarded: the OpenAI/Anthropic clients the HTML-extraction eval "
        "drives. Developer tooling run by hand, never imported by the proxy. "
        "Out of scope for A-2.",
    ),
    "evals/prompt_comparison.py": (
        1,
        "unguarded: the OpenAI client the prompt-comparison eval drives. "
        "Developer tooling, same reasoning as evals/html_extraction.py. Out "
        "of scope for A-2.",
    ),
    "evals/runners/before_after.py": (
        3,
        "unguarded: the Anthropic/OpenAI clients the before-after eval runner "
        "builds. Developer tooling, same reasoning as evals/datasets.py. Out "
        "of scope for A-2.",
    ),
    "memory/backends/direct_mem0.py": (
        1,
        "unguarded: the OpenAI embedder the direct mem0 backend builds. An "
        "opt-in memory backend that is configured with a cloud embedding "
        "endpoint by definition. Out of scope for A-2.",
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
    "relevance/embedding.py": (
        2,
        "unguarded: fastembed's TextEmbedding pulls ONNX weights from HF on a "
        "cache miss. Unlike onnx_runtime.hf_hub_download_local_first there is "
        "no cache-first/network split to hang the guard on, so guarding it "
        "would also refuse a warm, pre-seeded cache. Needs that split first; "
        "out of scope for A-2. The Rust twin (relevance/embedding.rs) IS "
        "guarded because its caller degrades to BM25 either way.",
    ),
}


def _python_sites_by_file() -> dict[str, list[_Site]]:
    """Map ``headroom/``-relative path -> egress sites, guarded or not."""
    found: dict[str, list[_Site]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        try:
            sites = _python_egress_sites(source)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        if sites:
            found[path.relative_to(PACKAGE_ROOT).as_posix()] = sites
    return found


def _unguarded_by_file() -> dict[str, list[_Site]]:
    out = {}
    for relpath, sites in _python_sites_by_file().items():
        unguarded = [site for site in sites if not site.guarded]
        if unguarded:
            out[relpath] = unguarded
    return out


def _render(relpath: str, sites: list[_Site], *, root: str = "headroom") -> str:
    shown = "\n".join(f"        line {site.line}: {site.text}" for site in sites)
    return f"  {root}/{relpath}\n{shown}"


_RESOLUTIONS = (
    "\n\nHEADROOM_OFFLINE=1 is documented as an air-gap switch, so every path "
    "that opens a connection must either:\n"
    "  1. call guard_egress(purpose, destination) BEFORE the client is "
    "constructed, in a position that dominates the call site (same block or "
    "an enclosing one, earlier in that block), or\n"
    "  2. be added to the allowlist in this file with a written reason "
    "(loopback / gated / user-traffic / unguarded) and the number of "
    "UNGUARDED egress sites in the file.\n"
    "Do not silence this by widening the regex, and do not rely on a guard "
    "somewhere else in the same file — it has to dominate the site."
)


class TestEgressChokepointCoverage:
    """Fails when someone adds an outbound client that nothing checked.

    The failure message names the file and the exact lines, and tells the
    author the two acceptable resolutions, so the test is a code-review aid
    rather than a puzzle.
    """

    def test_every_egress_site_is_guarded_or_allowlisted(self) -> None:
        problems: list[str] = []
        unguarded = _unguarded_by_file()
        for relpath, sites in unguarded.items():
            entry = _EGRESS_ALLOWLIST.get(relpath)
            if entry is None:
                problems.append(
                    _render(relpath, sites) + "\n        (not guarded, not allowlisted)"
                )
                continue
            expected, _reason = entry
            if len(sites) != expected:
                problems.append(
                    _render(relpath, sites)
                    + f"\n        (allowlist records {expected} unguarded egress "
                    f"site(s), found {len(sites)})"
                )

        assert not problems, (
            "New or changed outbound egress found in headroom/.\n\n"
            + "\n".join(problems)
            + _RESOLUTIONS
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        """A stale entry is worse than a missing one: it reads as a reviewed
        decision about code that no longer exists, and it hides the next real
        addition to that file behind a count that was never re-checked."""
        unguarded = _unguarded_by_file()
        stale = sorted(set(_EGRESS_ALLOWLIST) - set(unguarded))
        assert not stale, (
            "allowlist entries no longer have any UNGUARDED egress site; "
            f"delete them (or they were just fixed — delete them anyway): {stale}"
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

    def test_the_guarded_paths_are_actually_seen_as_guarded(self) -> None:
        """The scanner has to recognise the guards this PR added, or "no
        unguarded sites" would be a vacuous pass for those files."""
        sites = _python_sites_by_file()
        for relpath in (
            "transforms/kompress_remote.py",
            "observability/metrics.py",
            "onnx_runtime.py",
        ):
            assert relpath in sites, f"{relpath} has no detected egress site at all"
            assert any(site.guarded for site in sites[relpath]), (
                f"{relpath} routes through guard_egress but the scanner does not "
                "see any site as guarded — the dominance check has drifted"
            )


# ───────────────── the meta-test's own dominance rules ──────────────────────


class TestSiteScannerRules:
    """Tests for the scanner itself.

    The defect this replaces was not a missing rule, it was a rule that never
    ran: a file-wide ``"guard_egress" in source`` check meant the word in a
    docstring exempted the whole module. A meta-test nobody tests is just a
    comment, so each bypass that was demonstrated against the old version is
    pinned here.
    """

    def test_a_guard_in_a_docstring_guards_nothing(self) -> None:
        source = '"""This module would call guard_egress if it were real."""\nimport httpx\nc = httpx.Client()\n'
        sites = _python_egress_sites(source)
        assert [site.guarded for site in sites] == [False]

    def test_a_second_client_in_a_guarded_file_is_unguarded(self) -> None:
        source = (
            "def one():\n"
            "    guard_egress('a', 'b')\n"
            "    return httpx.Client()\n"
            "\n"
            "def two():\n"
            "    return httpx.Client()\n"
        )
        sites = _python_egress_sites(source)
        assert [site.guarded for site in sites] == [True, False]

    def test_a_guard_in_a_sibling_branch_does_not_count(self) -> None:
        source = (
            "def f(flag):\n"
            "    if flag:\n"
            "        guard_egress('a', 'b')\n"
            "    return httpx.Client()\n"
        )
        assert [site.guarded for site in _python_egress_sites(source)] == [False]

    def test_a_guard_in_an_enclosing_block_does_count(self) -> None:
        source = (
            "def f(flag):\n"
            "    guard_egress('a', 'b')\n"
            "    if flag:\n"
            "        with open('x') as fh:\n"
            "            return httpx.Client()\n"
        )
        assert [site.guarded for site in _python_egress_sites(source)] == [True]

    def test_a_guard_in_a_nested_function_does_not_count(self) -> None:
        source = (
            "def f():\n"
            "    def inner():\n"
            "        guard_egress('a', 'b')\n"
            "    return httpx.Client()\n"
        )
        assert [site.guarded for site in _python_egress_sites(source)] == [False]

    def test_a_guard_after_the_site_does_not_count(self) -> None:
        source = "def f():\n    c = httpx.Client()\n    guard_egress('a', 'b')\n    return c\n"
        assert [site.guarded for site in _python_egress_sites(source)] == [False]

    def test_a_one_line_def_is_not_invisible(self) -> None:
        """The previous scanner skipped any line starting with ``def``/``async
        def`` to avoid counting a ``def _urlopen(...)`` wrapper, which also hid
        every egress packed onto a one-line body."""
        source = "def f(): return httpx.Client().post('https://x')\n"
        assert [site.guarded for site in _python_egress_sites(source)] == [False]

    def test_a_urlopen_wrapper_definition_is_still_not_a_call_site(self) -> None:
        source = "def _urlopen(url):\n    return urllib.request.urlopen(url)\n"
        sites = _python_egress_sites(source)
        assert [site.callee for site in sites] == ["urllib.request.urlopen"]

    def test_a_commented_out_client_is_not_a_site(self) -> None:
        source = "# c = httpx.Client()\nx = 1\n"
        assert _python_egress_sites(source) == []

    def test_a_cache_only_hf_download_is_not_a_site(self) -> None:
        source = (
            "def f():\n"
            "    a = hf_hub_download(r, f, local_files_only=True)\n"
            "    return hf_hub_download(r, f)\n"
        )
        sites = _python_egress_sites(source)
        assert [site.line for site in sites] == [3]

    def test_a_method_named_post_is_not_requests_post(self) -> None:
        source = "def f(self):\n    return self.client.post('/x')\n"
        assert _python_egress_sites(source) == []

    def test_module_level_httpx_verbs_are_sites(self) -> None:
        source = "def f():\n    return httpx.get('https://x')\n"
        assert [site.callee for site in _python_egress_sites(source)] == ["httpx.get"]


# ───────────────── the documented claim vs the actual guarantee ─────────────

_AIR_GAP_DOCS = (
    "docs/metrics-technical-guide.md",
    "docs/content/docs/proxy.mdx",
)

# Phrases that promise a whole-process egress kill switch. Fine to write once
# the allowlist has no `unguarded` entries left; false until then.
_OVERCLAIMS = (
    "disables all outbound traffic",
    "disables all egress",
    "blocks all outbound",
    "hard-disable **all** egress",
    "hard-disables all egress",
    "no outbound traffic at all",
)


class TestDocsMatchTheGuarantee:
    """The docs and the allowlist have to agree about what the switch does.

    The change that introduced the chokepoint also strengthened
    `docs/metrics-technical-guide.md` to say `HEADROOM_OFFLINE=1` "disables all
    outbound traffic" — while its own allowlist recorded seven paths as
    `unguarded`, plus two Rust downloads and the Python HuggingFace fetch. An
    operator reading that sentence and skipping the firewall rule would have
    been wrong. This test makes the sentence and the allowlist move together:
    the strong claim is allowed again the moment the last `unguarded` entry
    goes away, and not before.
    """

    def test_no_doc_claims_more_than_the_allowlist_admits(self) -> None:
        still_unguarded = sorted(
            relpath
            for relpath, (_count, reason) in _EGRESS_ALLOWLIST.items()
            if reason.startswith("unguarded")
        )
        if not still_unguarded:
            pytest.skip("nothing is recorded as unguarded; the strong claim would be fair")
        for relative in _AIR_GAP_DOCS:
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            for claim in _OVERCLAIMS:
                assert claim not in text, (
                    f"{relative} says {claim!r}, but _EGRESS_ALLOWLIST still "
                    f"records these as reachable under HEADROOM_OFFLINE: "
                    f"{still_unguarded}. Guard them or soften the sentence — "
                    "an operator who believes the sentence skips the firewall rule."
                )

    def test_the_docs_still_describe_the_switch(self) -> None:
        """The cheap way to pass the test above is to delete the paragraph."""
        for relative in _AIR_GAP_DOCS:
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            assert "HEADROOM_OFFLINE" in text, f"{relative} no longer documents the switch"


# ─────────────────────── the Rust half of the same sweep ────────────────────
#
# `crates/` was outside the Python scan entirely, which is how two Rust
# downloads (the Kompress model and the fastembed weights) sat unguarded in the
# same PR that guarded the Rust tokenizer for exactly the stated reason. This
# is a text scan, not an AST one: there is no Rust parser here, so "guarded"
# means a guard_egress call earlier in the same `fn` at no deeper indentation.
# Weaker than the Python dominance check, and deliberately so — it is a
# review-time tripwire for a new egress path, and the runtime assertions live
# in each crate's own `#[test]`s.

_RUST_EGRESS_PATTERNS = re.compile(
    r"""
      hf_hub::api::(?:sync|tokio)::Api(?:Builder)?::new\(
    | (?<![\w:])Api(?:Builder)?::new\(
    | (?<![\w:])TextEmbedding::try_new\w*\(
    | (?<![\w:])reqwest::(?:Client::(?:new|builder)|get|post)\(
    | (?<![\w:])ureq::(?:agent|builder|get|post|put|delete|request)\(
    """,
    re.VERBOSE,
)

_RUST_FN = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?"
    r"(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+\w"
)

_RUST_ALLOWLIST: dict[str, tuple[int, str]] = {
    "headroom-proxy/src/proxy.rs": (
        1,
        "user-traffic: the reqwest client the Rust proxy forwards the caller's "
        "own request through, the exact counterpart of the Python proxy's "
        "allowlist entry. An air-gapped deployment points it at an on-prem "
        "endpoint and still needs it to work.",
    ),
}


def _rust_egress_sites() -> dict[str, list[_Site]]:
    """Map ``crates/``-relative path -> egress sites, guarded or not."""
    crates_root = REPO_ROOT / "crates"
    found: dict[str, list[_Site]] = {}
    for path in sorted(crates_root.glob("*/src/**/*.rs")):
        lines = path.read_text(encoding="utf-8").splitlines()
        # Everything from the module-level `#[cfg(test)] mod tests` on is test
        # code: it is expected to build clients, and it never ships. Match the
        # `mod` too — a bare `#[cfg(test)]` also decorates test-only `use`
        # lines near the top of a file, and truncating there would blind the
        # sweep to the entire module (it did, for headroom-proxy/src/proxy.rs).
        for index, line in enumerate(lines):
            if line.rstrip() != "#[cfg(test)]":
                continue
            following = next((nxt for nxt in lines[index + 1 :] if nxt.strip()), "")
            if following.lstrip().startswith("mod "):
                lines = lines[:index]
                break
        sites: list[_Site] = []
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith(("//", "*", "#[")):
                continue
            match = _RUST_EGRESS_PATTERNS.search(line)
            if not match:
                continue
            start = _enclosing_rust_fn(lines, number)
            if start is None:
                continue
            body = lines[start : number - 1]
            indent = len(line) - len(line.lstrip())
            guarded = any(
                "guard_egress(" in candidate
                and (len(candidate) - len(candidate.lstrip())) <= indent
                for candidate in body
            )
            sites.append(_Site(line=number, text=stripped, callee=match.group(0), guarded=guarded))
        if sites:
            found[path.relative_to(crates_root).as_posix()] = sites
    return found


def _enclosing_rust_fn(lines: list[str], number: int) -> int | None:
    """Index of the line after the `fn` header enclosing line ``number``.

    None when the site is inside a `#[test]`/`#[cfg(test)]` function, which the
    sweep ignores, or when no `fn` header precedes it at all.
    """
    for index in range(number - 1, -1, -1):
        if not _RUST_FN.match(lines[index]):
            continue
        attributes = "\n".join(lines[max(0, index - 5) : index])
        if "#[test]" in attributes or "#[tokio::test]" in attributes:
            return None
        if "#[cfg(test)]" in attributes:
            return None
        return index + 1
    return None


class TestRustEgressChokepointCoverage:
    def test_every_rust_egress_site_is_guarded_or_allowlisted(self) -> None:
        problems: list[str] = []
        unguarded = {
            relpath: [site for site in sites if not site.guarded]
            for relpath, sites in _rust_egress_sites().items()
        }
        unguarded = {relpath: sites for relpath, sites in unguarded.items() if sites}
        for relpath, sites in unguarded.items():
            entry = _RUST_ALLOWLIST.get(relpath)
            if entry is None:
                problems.append(
                    _render(relpath, sites, root="crates")
                    + "\n        (not guarded, not allowlisted)"
                )
                continue
            expected, _reason = entry
            if len(sites) != expected:
                problems.append(
                    _render(relpath, sites, root="crates")
                    + f"\n        (allowlist records {expected}, found {len(sites)})"
                )
        assert not problems, (
            "New or changed outbound egress found in crates/.\n\n"
            + "\n".join(problems)
            + _RESOLUTIONS
        )

    def test_the_guarded_rust_paths_are_seen_as_guarded(self) -> None:
        sites = _rust_egress_sites()
        for relpath in (
            "headroom-core/src/tokenizer/hf_impl.rs",
            "headroom-core/src/transforms/kompress.rs",
            "headroom-core/src/relevance/embedding.rs",
        ):
            assert relpath in sites, f"{relpath} has no detected egress site at all"
            assert all(site.guarded for site in sites[relpath]), (
                f"{relpath} has an egress site the Rust sweep does not see as guarded"
            )

    def test_rust_allowlist_has_no_stale_entries(self) -> None:
        unguarded = {
            relpath
            for relpath, sites in _rust_egress_sites().items()
            if any(not site.guarded for site in sites)
        }
        stale = sorted(set(_RUST_ALLOWLIST) - unguarded)
        assert not stale, f"Rust allowlist entries with no unguarded site; delete them: {stale}"
