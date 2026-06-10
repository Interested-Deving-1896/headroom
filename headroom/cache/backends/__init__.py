"""Storage backends for CompressionStore.

This module provides pluggable storage backends for CCR (Compress-Cache-Retrieve).
The default is SQLite (restart-safe, shared across workers); in-memory is
available via ``HEADROOM_CCR_BACKEND=memory``, and alternative backends can
be implemented for:
- Distributed caching (Redis, MongoDB via entry points)
- Custom storage solutions

Usage:
    from headroom.cache.backends import SQLiteBackend, CompressionStoreBackend
    from headroom.cache.compression_store import CompressionStore

    # Default (SQLite at ~/.headroom/ccr_store.db)
    store = CompressionStore()

    # Use custom backend
    class MyBackend:
        # Implement CompressionStoreBackend protocol
        ...
    store = CompressionStore(backend=MyBackend())
"""

from .base import CompressionStoreBackend
from .memory import InMemoryBackend
from .sqlite import SQLiteBackend

__all__ = [
    "CompressionStoreBackend",
    "InMemoryBackend",
    "SQLiteBackend",
]
