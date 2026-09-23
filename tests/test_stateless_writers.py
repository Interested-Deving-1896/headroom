"""Tests for the complete stateless write guarantee.

Covers the process-wide stateless flag and the opt-in serving writers gated by
it: the output-savings recorder, persistent memory, and the CCR compression
store. (Savings tracker and TOIN are covered in their own test modules.)
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

from headroom import paths
from headroom.proxy.output_savings import SavingsRecorder
from headroom.relevance.embedding import (
    _DEFAULT_MODEL_PINNED_REVISION,
    DEFAULT_MODEL_NAME,
    _pinned_revision,
)


@pytest.fixture(autouse=True)
def _reset_stateless_globals():
    """The stateless flag and TOIN singleton are process-global — never leak."""
    yield
    paths.set_process_stateless(None)
    try:
        from headroom.telemetry.toin import reset_toin

        reset_toin()
    except Exception:
        pass


# ---- process-wide stateless flag ------------------------------------------


def test_process_stateless_flag_set_and_clear(monkeypatch):
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    paths.set_process_stateless(False)
    assert paths.process_is_stateless() is False
    paths.set_process_stateless(True)
    assert paths.process_is_stateless() is True


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("on", True), ("off", False), ("", False)]
)
def test_process_stateless_env(monkeypatch, value, expected):
    """With no explicit call recorded, the env var decides."""
    paths.set_process_stateless(None)
    monkeypatch.setenv("HEADROOM_STATELESS", value)
    assert paths.process_is_stateless() is expected


def test_explicit_false_clears_the_env_latch(monkeypatch):
    """`HEADROOM_STATELESS=true` must not be an unclearable process-wide latch.

    ``headroom proxy --stateless`` *exports* the env var so uvicorn workers
    inherit the mode. If the flag could only ever be OR-ed in, every later
    proxy built in the same process would silently keep the stateless
    behaviour, and ``set_process_stateless(False)`` — which
    ``HeadroomProxy.__init__`` calls for a non-stateless config — could not
    undo it.
    """
    monkeypatch.setenv("HEADROOM_STATELESS", "true")
    paths.set_process_stateless(None)
    assert paths.process_is_stateless() is True

    paths.set_process_stateless(False)
    assert paths.process_is_stateless() is False

    # ...and None hands the decision back to the environment.
    paths.set_process_stateless(None)
    assert paths.process_is_stateless() is True


def test_non_stateless_proxy_after_stateless_export_is_not_stateless(monkeypatch, tmp_path):
    """The end-to-end shape of the latch: CLI export, then a normal proxy."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    from headroom.cache.compression_store import _create_default_ccr_backend
    from headroom.proxy.server import ProxyConfig, create_app

    # What `--stateless` leaves behind in the environment.
    monkeypatch.setenv("HEADROOM_STATELESS", "true")
    paths.set_process_stateless(None)

    config = ProxyConfig(stateless=False)
    assert config.stateless is False
    create_app(config)

    assert paths.process_is_stateless() is False
    # The observable consequence: this proxy gets its real CCR backend back.
    backend = _create_default_ccr_backend()
    assert backend is not None
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def test_proxy_config_stateless_defaults_from_env(monkeypatch):
    """An embedder that sets only the env var still gets a stateless config.

    This is what keeps the explicit-False-wins rule above from *disabling*
    stateless for callers who never passed the flag.
    """
    from headroom.proxy.server import ProxyConfig

    monkeypatch.setenv("HEADROOM_STATELESS", "true")
    assert ProxyConfig().stateless is True
    monkeypatch.setenv("HEADROOM_STATELESS", "off")
    assert ProxyConfig().stateless is False


# ---- output-savings recorder ----------------------------------------------


def test_output_savings_flush_writes_nothing_when_stateless(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    path = tmp_path / "output_savings.json"
    paths.set_process_stateless(True)
    rec = SavingsRecorder(path)
    rec.flush()
    assert not path.exists()


def test_output_savings_flush_persists_when_not_stateless(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    path = tmp_path / "output_savings.json"
    paths.set_process_stateless(False)
    rec = SavingsRecorder(path)
    rec.flush()
    assert path.exists()


# ---- persistent memory ----------------------------------------------------


def test_memory_disabled_under_stateless(tmp_path, monkeypatch):
    """A stateless proxy with --memory must not initialize memory or write a DB."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(memory_enabled=True, stateless=True))
    proxy = app.state.proxy
    assert proxy.memory_handler is None
    assert not (tmp_path / ".headroom" / "memory.db").exists()


# ---- fastembed model pinning ----------------------------------------------


def test_fastembed_default_model_is_pinned_to_sha(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    rev = _pinned_revision(DEFAULT_MODEL_NAME)
    assert rev == _DEFAULT_MODEL_PINNED_REVISION
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_fastembed_custom_model_not_pinned(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    assert _pinned_revision("intfloat/e5-small-v2") is None


def test_fastembed_pin_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    assert _pinned_revision(DEFAULT_MODEL_NAME) is None


# ---- CCR compression store -------------------------------------------------
#
# The CCR store's default backend is a SQLite file in the workspace that holds
# the *verbatim originals* of compressed tool results. A stateless deployment
# asking for "no request content on disk" must not get that file -- and getting
# it must not cost the operators who share that file across workers.


@pytest.fixture
def _reset_ccr_store():
    """The CCR singleton is process-global; never leak one between tests."""
    from headroom.cache.compression_store import reset_compression_store

    reset_compression_store()
    yield
    reset_compression_store()


def _sqlite_rows(db_path) -> list[str]:
    """Read the stored entry blobs through a *fresh* connection.

    Reading the file bytes is not enough: the backend runs in WAL mode, so a
    just-written row lives in ``-wal`` and a file-content grep on the main
    database silently passes.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT entry_json FROM ccr_entries").fetchall()]
    finally:
        conn.close()


def test_ccr_backend_is_in_memory_under_stateless(monkeypatch):
    """process_is_stateless() beats a local-disk backend choice, env included."""
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "sqlite")
    from headroom.cache.compression_store import _create_default_ccr_backend

    paths.set_process_stateless(True)
    assert _create_default_ccr_backend() is None


def test_ccr_backend_is_sqlite_when_not_stateless(monkeypatch, tmp_path):
    """Control: the persistent default is unchanged outside stateless mode."""
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    from headroom.cache.backends.sqlite import SQLiteBackend
    from headroom.cache.compression_store import _create_default_ccr_backend

    paths.set_process_stateless(False)
    backend = _create_default_ccr_backend()
    assert isinstance(backend, SQLiteBackend)
    backend.close()


# ---- (4) an external backend is not a local-disk backend -------------------


class _FakeExternalBackend:
    """Stand-in for the registered `redis` CCR adapter: nothing on local disk."""

    is_process_local = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _register_fake_external_backend(monkeypatch, name="redis"):
    import importlib.metadata as md

    class _EP:
        def __init__(self, name):
            self.name = name

        def load(self):
            return _FakeExternalBackend

    monkeypatch.setattr(md, "entry_points", lambda group=None: [_EP(name)])


def test_stateless_keeps_external_ccr_backend(monkeypatch):
    """`--stateless` is a promise about the local filesystem, not a downgrade.

    A redis backend writes nothing to this machine's disk and is the only way
    to share retrieval across workers. Forcing it to a per-worker dict would
    break retrieval to fix nothing.
    """
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    _register_fake_external_backend(monkeypatch)
    from headroom.cache.compression_store import _create_default_ccr_backend

    paths.set_process_stateless(True)
    assert isinstance(_create_default_ccr_backend(), _FakeExternalBackend)


def test_ccr_backend_writes_local_disk_classification():
    from headroom.cache.compression_store import ccr_backend_writes_local_disk

    assert ccr_backend_writes_local_disk(None) is True
    assert ccr_backend_writes_local_disk("") is True
    assert ccr_backend_writes_local_disk("sqlite") is True
    assert ccr_backend_writes_local_disk(" SQLite ") is True
    assert ccr_backend_writes_local_disk("memory") is False
    assert ccr_backend_writes_local_disk("redis") is False


# ---- (1) the stateless swap must not destroy the shared database -----------


def test_stateless_apply_does_not_wipe_the_shared_ccr_database(tmp_path, _reset_ccr_store):
    """ccr_store.db is shared across workers and outlives this process.

    Building a stateless proxy in a process that already holds a SQLite-backed
    store must not delete rows the other live workers are serving.
    """
    from headroom.cache.backends.sqlite import SQLiteBackend
    from headroom.cache.compression_store import get_compression_store
    from headroom.proxy.server import ProxyConfig, _apply_stateless_persistence

    db = tmp_path / "ccr_store.db"
    store = get_compression_store(backend=SQLiteBackend(db))
    store.store("original-from-another-worker", "[compressed]", tool_name="Bash")
    assert len(_sqlite_rows(db)) == 1

    _apply_stateless_persistence(ProxyConfig(stateless=True))

    rows = _sqlite_rows(db)
    assert len(rows) == 1, "stateless startup deleted another worker's entries"
    assert "original-from-another-worker" in rows[0]


# ---- (2) ...and must actually stop the writes it exists for ----------------


def test_stateless_apply_stops_an_already_captured_store_writing_to_disk(
    tmp_path, _reset_ccr_store
):
    """Dropping the module global is not enough.

    CompressionFeedback (and anything else holding the singleton) keeps
    its own reference and would go on writing verbatim originals to SQLite
    after the stateless proxy is up.
    """
    from headroom.cache.backends.sqlite import SQLiteBackend
    from headroom.cache.compression_feedback import CompressionFeedback
    from headroom.cache.compression_store import get_compression_store
    from headroom.proxy.server import ProxyConfig, _apply_stateless_persistence

    db = tmp_path / "ccr_store.db"
    get_compression_store(backend=SQLiteBackend(db))
    tracker = CompressionFeedback()
    captured = tracker.store  # the capture that the old reset could not reach

    _apply_stateless_persistence(ProxyConfig(stateless=True))

    secret = "SECRET-AFTER-STATELESS-" + ("x" * 256)
    captured.store(secret, "[compressed]", tool_name="Bash")
    assert not any(secret in row for row in _sqlite_rows(db)), (
        "the captured store went on writing verbatim originals to disk"
    )
    assert captured.backend_is_process_local is True


def test_use_process_local_backend_is_idempotent(_reset_ccr_store):
    from headroom.cache.compression_store import CompressionStore

    store = CompressionStore()
    assert store.backend_is_process_local is True
    assert store.use_process_local_backend() is False


# ---- (5) multi-worker diagnostics ------------------------------------------


def test_miss_detail_names_the_in_process_store(monkeypatch, _reset_ccr_store):
    """A cross-worker miss must not be reported as a TTL problem."""
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    from headroom.cache.compression_store import (
        PROCESS_LOCAL_STORE_DETAIL,
        format_retrieval_miss_detail,
        get_compression_store,
    )

    store = get_compression_store()
    detail = format_retrieval_miss_detail(store.get_entry_status("deadbeef"))
    assert PROCESS_LOCAL_STORE_DETAIL in detail
    assert "in-process only" in detail


def test_miss_detail_stays_ttl_only_for_a_shared_store(tmp_path, _reset_ccr_store):
    from headroom.cache.backends.sqlite import SQLiteBackend
    from headroom.cache.compression_store import (
        PROCESS_LOCAL_STORE_DETAIL,
        format_retrieval_miss_detail,
        get_compression_store,
    )

    store = get_compression_store(backend=SQLiteBackend(tmp_path / "ccr_store.db"))
    detail = format_retrieval_miss_detail(store.get_entry_status("deadbeef"))
    assert PROCESS_LOCAL_STORE_DETAIL not in detail


def test_expired_entry_detail_is_unchanged(_reset_ccr_store):
    """An entry that was found and aged out really is a TTL story."""
    from headroom.cache.compression_store import format_retrieval_miss_detail

    detail = format_retrieval_miss_detail(
        {"status": "expired", "ttl_seconds": 1800, "age_seconds": 2000.0}
    )
    assert detail == "Entry expired (CCR TTL: 1800 seconds; age: 2000 seconds)"


def test_ccr_miss_message_does_not_blame_ttl_alone():
    from headroom.cache.compression_store import CCR_MISS_MESSAGE

    assert "re-read" in CCR_MISS_MESSAGE
    assert "re-run" in CCR_MISS_MESSAGE
    assert "in-process only" in CCR_MISS_MESSAGE


# ---- CLI surface -----------------------------------------------------------


def _run_proxy_cli(monkeypatch, args, *, run_server=None, home=None):
    """Invoke the real `headroom proxy` command with the server stubbed out."""
    from click.testing import CliRunner

    from headroom.cli import proxy as proxy_cli
    from headroom.proxy import server as proxy_server

    if home is not None:
        monkeypatch.setenv("HOME", str(home))
    # The CLI imports run_server from headroom.proxy.server inside the command
    # body, so patch it at the source module.
    monkeypatch.setattr(proxy_server, "run_server", run_server or (lambda config, **kw: None))
    result = CliRunner().invoke(proxy_cli.proxy, args)
    output = result.output + (result.stderr if result.stderr_bytes else "")
    return result, output


def test_stateless_cli_forces_sqlite_to_memory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "sqlite")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    result, _ = _run_proxy_cli(monkeypatch, ["--stateless", "--port", "18791"])
    assert result.exit_code == 0, result.output
    assert os.environ["HEADROOM_CCR_BACKEND"] == "memory"
    assert os.environ["HEADROOM_STATELESS"] == "true"


def test_stateless_cli_keeps_an_external_ccr_backend(monkeypatch, tmp_path):
    """The operator's shared, no-local-disk backend survives `--stateless`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    result, _ = _run_proxy_cli(monkeypatch, ["--stateless", "--port", "18792"])
    assert result.exit_code == 0, result.output
    assert os.environ["HEADROOM_CCR_BACKEND"] == "redis"


def test_stateless_multi_worker_warns_about_per_worker_ccr(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    _, output = _run_proxy_cli(monkeypatch, ["--stateless", "--workers", "2", "--port", "18793"])
    assert "--workers 2" in output
    assert "different worker" in output


def test_stateless_single_worker_does_not_warn(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    _, output = _run_proxy_cli(monkeypatch, ["--stateless", "--port", "18794"])
    assert "different worker" not in output


def test_stateless_multi_worker_with_external_backend_does_not_warn(monkeypatch, tmp_path):
    """The external backend is shared, so there is nothing to warn about."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    _, output = _run_proxy_cli(monkeypatch, ["--stateless", "--workers", "2", "--port", "18795"])
    assert "different worker" not in output


def test_stateless_banner_does_not_claim_no_filesystem_writes(monkeypatch, tmp_path):
    """The banner asserted something a real run contradicts. It must not."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("HEADROOM_WORKSPACE_DIR", raising=False)
    _, output = _run_proxy_cli(monkeypatch, ["--stateless", "--port", "18796"], home=home)
    assert "Stateless:    YES" in output
    assert "no filesystem writes" not in output
    # It names what a real run does still write.
    assert "proxy-18796.log" in output
    assert "beacon lock" in output


# ---- end-to-end ------------------------------------------------------------


def test_stateless_proxy_writes_nothing_but_logs_under_home(tmp_path, monkeypatch):
    """`headroom proxy --stateless` against a fresh HOME leaves only logs behind.

    Runs the real CLI with ``run_server`` swapped for a stand-in that builds the
    app *and runs its lifespan* (so the beacon lock, the subscription tracker
    and the install_id writers are actually exercised), then pushes a
    tool-result original through the CCR store and retrieves it again -- the
    two writes the on-disk backend and the retrieval log would have made.
    """
    pytest.importorskip("click")
    import logging

    from starlette.testclient import TestClient

    from headroom.cache.compression_store import get_compression_store, reset_compression_store
    from headroom.proxy import server as proxy_server

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HEADROOM_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("HEADROOM_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.chdir(tmp_path)

    secret = "BEGIN-SECRET-TOOL-OUTPUT-" + ("x" * 512)
    headroom_logger = logging.getLogger("headroom")
    handlers_before = list(headroom_logger.handlers)

    def _fake_run_server(config, **kwargs):
        app = proxy_server.create_app(config)
        # The context manager runs the lifespan, which is where the beacon
        # lock, the subscription tracker and install_id writers live.
        with TestClient(app) as client:
            store = get_compression_store()
            hash_key = store.store(secret, "[compressed]", tool_name="Bash")
            # Retrieval is the path that logs a payload preview.
            assert store.retrieve(hash_key) is not None
            client.get("/health")

    reset_compression_store()
    try:
        result, _ = _run_proxy_cli(
            monkeypatch,
            ["--stateless", "--port", "18799"],
            run_server=_fake_run_server,
            home=home,
        )
        assert result.exit_code == 0, result.output
        assert os.environ.get("HEADROOM_CCR_BACKEND") == "memory"
        # The env-only gates (TTL observations, update-check cache) and worker
        # subprocesses see the mode only if the flag exports it.
        assert os.environ.get("HEADROOM_STATELESS") == "true"
    finally:
        reset_compression_store()
        # create_app attaches a RotatingFileHandler to the `headroom` logger.
        # Leaving it attached keeps the rest of the session writing into a
        # reaped tmp dir.
        for handler in list(headroom_logger.handlers):
            if handler not in handlers_before:
                headroom_logger.removeHandler(handler)
                handler.close()

    workspace = home / ".headroom"

    # Every workspace file a stateless run is still expected to produce, named
    # deliberately so a NEW writer fails this test instead of slipping in. Each
    # entry is in the PR's audit table; none of them carries request content.
    #
    #   logs/                 always-on runtime log (_setup_file_logging)
    #   .beacon_lock_<port>   worker-election lock: a PID, unlinked on shutdown
    #   subscription_state.json  usage counters (subscription/tracker.py)
    #   config/install_id     beacon install id; gone under --offline
    def _expected(path) -> bool:
        rel = path.relative_to(workspace)
        return (
            rel.parts[0] == "logs"
            or path.name.startswith(".beacon_lock")
            or rel.as_posix() == "subscription_state.json"
            or rel.as_posix() == "config/install_id"
        )

    stray = [p for p in workspace.rglob("*") if p.is_file() and not _expected(p)]
    assert stray == [], f"stateless proxy wrote an unaudited file: {stray}"
    # Belt and braces: the content itself is nowhere under HOME -- including
    # the runtime log, which is why the retrieval payload preview is off under
    # stateless.
    for path in home.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), f"tool-result content leaked to {path}"
