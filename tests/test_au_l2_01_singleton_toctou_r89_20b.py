"""R89-20b AU-L2-01 regression — Qdrant client singleton TOCTOU.

Pre-fix: ``_get_client()`` in ``ingest/index.py`` used a plain
``if _client is None`` check. Two threads racing on cold start could
both see ``None`` and both instantiate; the second assignment overwrote
the first, orphaning in-flight queries.

Post-fix: double-checked locking with ``threading.Lock`` — fast path
preserved (no lock on hot read), but only one thread enters the
``QdrantClient(...)`` call.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def qdrant_stub(monkeypatch):
    """Stub qdrant_client; monkeypatch auto-cleans sys.modules after test."""
    factory = MagicMock(side_effect=lambda *a, **kw: object())
    stub_qc = types.ModuleType("qdrant_client")
    stub_qc.QdrantClient = factory  # type: ignore[attr-defined]
    stub_models = types.ModuleType("qdrant_client.models")
    monkeypatch.setitem(sys.modules, "qdrant_client", stub_qc)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", stub_models)
    # Also evict cached arastirma_ussu.ingest.index so the next import picks
    # up the stub at the lazy `from qdrant_client import QdrantClient as _QC`.
    monkeypatch.delitem(sys.modules, "arastirma_ussu.ingest.index", raising=False)
    return factory


def test_au_l2_01_concurrent_get_client_returns_same_instance(qdrant_stub) -> None:
    """Concurrent _get_client() callers must all receive the SAME instance."""
    from arastirma_ussu.ingest import index as idx

    idx._client = None  # type: ignore[attr-defined]

    instances: list[object] = []
    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()  # maximise contention at the if-None check
        instances.append(idx._get_client())

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 16, "AU-L2-01: lost a thread in the race harness"
    first = instances[0]
    assert all(inst is first for inst in instances), (
        "AU-L2-01: TOCTOU regression — concurrent _get_client() returned "
        "multiple distinct QdrantClient instances"
    )


def test_au_l2_01_qdrant_client_constructed_at_most_once(qdrant_stub) -> None:
    """The underlying QdrantClient(...) constructor must be called ONCE
    across N concurrent first-time _get_client() calls."""
    from arastirma_ussu.ingest import index as idx

    idx._client = None  # type: ignore[attr-defined]
    qdrant_stub.reset_mock()

    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()
        idx._get_client()

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert qdrant_stub.call_count == 1, (
        f"AU-L2-01: QdrantClient(...) instantiated {qdrant_stub.call_count}x "
        "under concurrent race — expected exactly 1 (double-checked lock failed)"
    )


def test_au_l2_01_threading_lock_imported_in_index_module() -> None:
    """Structural guard: threading must be imported (lock primitive present)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "arastirma_ussu" / "ingest" / "index.py"
    text = src.read_text(encoding="utf-8")
    assert "import threading" in text, (
        "AU-L2-01: 'import threading' missing from ingest/index.py"
    )
    assert "threading.Lock()" in text, (
        "AU-L2-01: threading.Lock() primitive missing — DCL incomplete"
    )
