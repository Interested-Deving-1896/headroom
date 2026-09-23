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
    paths.set_process_stateless(False)
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
    paths.set_process_stateless(False)
    monkeypatch.setenv("HEADROOM_STATELESS", value)
    assert paths.process_is_stateless() is expected


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
# asking for "no filesystem writes" must not get that file.


def test_ccr_backend_is_in_memory_under_stateless(monkeypatch):
    """process_is_stateless() beats every backend choice, env included."""
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
    assert isinstance(_create_default_ccr_backend(), SQLiteBackend)


def test_stateless_proxy_writes_nothing_but_logs_under_home(tmp_path, monkeypatch):
    """`headroom proxy --stateless` against a fresh HOME leaves only logs behind.

    Runs the real CLI with ``run_server`` swapped for a stand-in that builds the
    app and pushes a tool-result original through the CCR store — the write the
    on-disk backend would have made.
    """
    pytest.importorskip("click")
    from click.testing import CliRunner

    from headroom.cache.compression_store import get_compression_store, reset_compression_store
    from headroom.cli import proxy as proxy_cli
    from headroom.proxy import server as proxy_server

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HEADROOM_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("HEADROOM_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.chdir(tmp_path)

    secret = "BEGIN-SECRET-TOOL-OUTPUT-" + ("x" * 512)

    def _fake_run_server(config, **kwargs):
        proxy_server.create_app(config)
        get_compression_store().store(secret, "[compressed]", tool_name="Bash")

    # The CLI imports run_server from headroom.proxy.server inside the command
    # body, so patch it at the source module.
    monkeypatch.setattr(proxy_server, "run_server", _fake_run_server)

    reset_compression_store()
    try:
        result = CliRunner().invoke(proxy_cli.proxy, ["--stateless", "--port", "18799"])
        assert result.exit_code == 0, result.output
        assert os.environ.get("HEADROOM_CCR_BACKEND") == "memory"
        # The env-only gates (TTL observations, update-check cache) and worker
        # subprocesses see the mode only if the flag exports it.
        assert os.environ.get("HEADROOM_STATELESS") == "true"
    finally:
        reset_compression_store()

    workspace = home / ".headroom"
    # Logs are allowed (they are always-on, see _setup_file_logging) and so is
    # the empty worker-election lock, which lifespan creates and deletes.
    stray = [
        p
        for p in workspace.rglob("*")
        if p.is_file() and p.parent != workspace / "logs" and not p.name.startswith(".beacon_lock")
    ]
    assert stray == [], f"stateless proxy wrote {stray}"
    # Belt and braces: the content itself is nowhere under HOME.
    for path in home.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), f"tool-result content leaked to {path}"
