"""Conversation age must share the compression store's lifetime and durability."""

import concurrent.futures
import threading

import pytest

from headroom.cache.backends import InMemoryBackend, SQLiteBackend
from headroom.cache.compression_store import CompressionStore


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteBackend(db_path=tmp_path / "context.sqlite")
    return InMemoryBackend()


def _observe(backend, conversation, hashes):
    observe = getattr(backend, "observe_context_turn", None)
    assert callable(observe), "Backend must retain conversation clock independently of payloads"
    return observe(conversation, hashes)


def test_shared_hash_has_independent_first_turn_in_each_conversation(backend):
    store = CompressionStore(backend=backend)
    key = store.store("shared original", "shared sample")
    other = store.store("other original", "other sample")
    first = _observe(backend, "A", [key])
    assert first.current_turn == 1
    assert first.compression_turns[key][1] == 1
    for _ in range(4):
        _observe(backend, "B", [other])
    shared_in_b = _observe(backend, "B", [key])
    assert shared_in_b.current_turn == 5
    assert shared_in_b.compression_turns[key][1] == 5
    again = _observe(backend, "A", [key])
    assert again.current_turn == 2
    assert again.compression_turns[key][1] == 1
    assert store.get_stats()["entry_count"] == 2


def test_missing_marker_does_not_reset_original_event_turn(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    other = store.store("other original", "other sample")
    _observe(backend, "A", [key])
    assert _observe(backend, "A", ["not-owned"]) is None
    for _ in range(4):
        _observe(backend, "A", [other])
    again = _observe(backend, "A", [key])
    assert again.current_turn == 6
    assert again.compression_turns[key][1] == 1


def test_fresh_same_hash_recompression_has_new_first_turn(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    before = _observe(backend, "A", [key])
    for _ in range(4):
        _observe(backend, "A", [key])
    assert store.store("original", "fresh sample") == key
    after = _observe(backend, "A", [key])
    assert after.current_turn == 6
    assert after.compression_turns[key][1] == 6
    assert after.compression_turns[key][0] != before.compression_turns[key][0]


def test_sqlite_second_handle_preserves_clock_and_event_age(tmp_path):
    path = tmp_path / "context.sqlite"
    first = SQLiteBackend(db_path=path)
    store = CompressionStore(backend=first)
    key = store.store("original", "sample")
    for _ in range(5):
        _observe(first, "A", [key])
    reopened = SQLiteBackend(db_path=path)
    snapshot = _observe(reopened, "A", [key])
    assert snapshot.current_turn == 6
    assert snapshot.compression_turns[key][1] == 1
    assert CompressionStore(backend=reopened).retrieve(key).original_content == "original"


def test_sqlite_concurrent_handles_do_not_lose_turns(tmp_path):
    path = tmp_path / "context.sqlite"
    backends = [SQLiteBackend(db_path=path), SQLiteBackend(db_path=path)]
    key = CompressionStore(backend=backends[0]).store("original", "sample")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_observe, backends[index % 2], "A", [key]) for index in range(12)
        ]
        snapshots = [future.result() for future in futures]
    assert sorted(snapshot.current_turn for snapshot in snapshots) == list(range(1, 13))
    assert {snapshot.compression_turns[key][1] for snapshot in snapshots} == {1}
    assert _observe(SQLiteBackend(db_path=path), "A", [key]).current_turn == 13


def test_sqlite_concurrent_same_hash_stores_have_distinct_events(tmp_path, monkeypatch):
    path = tmp_path / "context.sqlite"
    backends = [SQLiteBackend(db_path=path), SQLiteBackend(db_path=path)]
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    key = CompressionStore(backend=backends[0]).store("original", "sample")
    stores = [CompressionStore(backend=backend) for backend in backends]
    barrier = threading.Barrier(2)
    timestamps = []
    for backend in backends:
        get = backend.get
        write_name = "set_new" if hasattr(backend, "set_new") else "set"
        write = getattr(backend, write_name)

        def synchronized_get(hash_key, get=get):
            entry = get(hash_key)
            barrier.wait(timeout=10)
            return entry

        def capture_write(hash_key, entry, write=write):
            write(hash_key, entry)
            timestamps.append(entry.created_at)

        monkeypatch.setattr(backend, "get", synchronized_get)
        monkeypatch.setattr(backend, write_name, capture_write)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.store, "original", "fresh") for store in stores]
        assert [future.result() for future in futures] == [key, key]
    assert len(set(timestamps)) == 2


def test_corrupt_retained_clock_skips_expansion_without_losing_payload(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    _observe(backend, "A", [key])
    if isinstance(backend, SQLiteBackend):
        backend._conn.execute(
            "UPDATE ccr_context_states SET state_json = ? WHERE conversation_key = ?",
            ('{"version":1,"turn":-1,"events":{}}', "A"),
        )
        backend._conn.commit()
    else:
        state, expiry = backend._context_states["A"]
        state["turn"] = -1
        backend._context_states["A"] = (state, expiry)
    assert _observe(backend, "A", [key]) is None
    assert store.retrieve(key).original_content == "original"


def test_expired_clock_is_cleaned_and_fresh_payload_gets_new_anchor(backend, monkeypatch):
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample", ttl=1)
    _observe(backend, "A", [key])
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1002.0)
    assert _observe(backend, "A", [key]) is None
    store.store("original", "fresh", ttl=10)
    snapshot = _observe(backend, "A", [key])
    assert snapshot.current_turn == 1
    assert snapshot.compression_turns[key][1] == 1
