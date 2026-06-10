"""Tests for the SQLite CCR backend and session-scale TTL defaults.

The SQLite backend is the default for `get_compression_store()` because the
30-minute TTL assumes entries survive proxy restarts and are visible across
worker processes — neither holds for the in-memory dict.
"""

from __future__ import annotations

import time

import pytest

from headroom.cache.backends.sqlite import SQLiteBackend
from headroom.cache.compression_store import CompressionEntry, CompressionStore


def make_entry(hash_key: str = "h1", content: str = "x" * 600, ttl: int = 1800) -> CompressionEntry:
    return CompressionEntry(
        hash=hash_key,
        original_content=content,
        compressed_content="c",
        original_tokens=100,
        compressed_tokens=10,
        original_item_count=50,
        compressed_item_count=5,
        tool_name="Read",
        tool_call_id="t1",
        query_context=None,
        created_at=time.time(),
        ttl=ttl,
    )


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "ccr_test.db"


class TestSQLiteBackend:
    def test_crud_roundtrip(self, db_path):
        b = SQLiteBackend(db_path)
        entry = make_entry()
        b.set("h1", entry)

        got = b.get("h1")
        assert got is not None
        assert got.original_content == entry.original_content
        assert got.tool_name == "Read"
        assert got.ttl == 1800

        assert b.exists("h1")
        assert b.count() == 1
        assert b.keys() == ["h1"]
        assert b.delete("h1")
        assert not b.exists("h1")
        assert not b.delete("h1")

    def test_survives_reopen(self, db_path):
        """The restart-survival property the default flip exists for."""
        SQLiteBackend(db_path).set("h1", make_entry())

        reopened = SQLiteBackend(db_path)
        got = reopened.get("h1")
        assert got is not None
        assert got.original_content == "x" * 600

    def test_two_connections_share_data(self, db_path):
        """Multi-worker property: a second live connection sees writes."""
        writer = SQLiteBackend(db_path)
        reader = SQLiteBackend(db_path)
        writer.set("h1", make_entry())
        assert reader.get("h1") is not None

    def test_items_and_stats(self, db_path):
        b = SQLiteBackend(db_path)
        b.set("h1", make_entry("h1"))
        b.set("h2", make_entry("h2"))

        items = dict(b.items())
        assert set(items) == {"h1", "h2"}
        stats = b.get_stats()
        assert stats["backend_type"] == "sqlite"
        assert stats["entry_count"] == 2
        assert stats["bytes_used"] > 0

    def test_clear(self, db_path):
        b = SQLiteBackend(db_path)
        b.set("h1", make_entry())
        b.clear()
        assert b.count() == 0

    def test_unknown_json_fields_tolerated(self, db_path):
        """Forward-compat: entries written by a newer headroom version
        (extra fields) must still load."""
        b = SQLiteBackend(db_path)
        b.set("h1", make_entry())
        with b._lock:
            row = b._conn.execute("SELECT entry_json FROM ccr_entries WHERE hash='h1'").fetchone()
            doctored = row[0][:-1] + ', "field_from_the_future": 7}'
            b._conn.execute("UPDATE ccr_entries SET entry_json=? WHERE hash='h1'", (doctored,))
            b._conn.commit()

        got = b.get("h1")
        assert got is not None
        assert got.original_content == "x" * 600

    def test_store_ttl_enforcement_via_compression_store(self, db_path):
        """TTL checks stay in CompressionStore; expired entries miss."""
        store = CompressionStore(backend=SQLiteBackend(db_path))
        expired = make_entry(ttl=1)
        expired.created_at = time.time() - 10
        store._backend.set("h1", expired)

        assert store.retrieve("h1") is None

    def test_retrieval_count_persists(self, db_path):
        """record_access mutations are re-persisted (store re-sets the
        entry after mutating), so feedback counts survive reopen."""
        store = CompressionStore(backend=SQLiteBackend(db_path))
        store._backend.set("h1", make_entry())
        store.retrieve("h1", query="foo")

        reopened = SQLiteBackend(db_path)
        got = reopened.get("h1")
        assert got is not None
        assert got.retrieval_count == 1


class TestDefaults:
    def test_session_scale_ttl_lockstep(self):
        """CCRConfig, CompressionEntry, and CompressionStore must agree."""
        from headroom.config import CCRConfig

        assert CCRConfig().store_ttl_seconds == 1800
        assert CompressionEntry.__dataclass_fields__["ttl"].default == 1800
        assert CompressionStore()._default_ttl == 1800

    def test_default_backend_is_sqlite(self, monkeypatch, tmp_path):
        from headroom.cache.compression_store import _create_default_ccr_backend

        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        monkeypatch.setenv("HEADROOM_CCR_SQLITE_PATH", str(tmp_path / "d.db"))
        backend = _create_default_ccr_backend()
        assert backend is not None
        assert backend.get_stats()["backend_type"] == "sqlite"

    def test_memory_opt_out(self, monkeypatch):
        from headroom.cache.compression_store import _create_default_ccr_backend

        monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
        assert _create_default_ccr_backend() is None

    def test_miss_message_is_actionable(self):
        from headroom.cache.compression_store import CCR_MISS_MESSAGE

        assert "re-read" in CCR_MISS_MESSAGE
        assert "re-run" in CCR_MISS_MESSAGE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
