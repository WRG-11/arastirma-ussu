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


def _install_qdrant_stub() -> tuple[types.ModuleType, MagicMock]:
    """Stub the qdrant_client module so tests don't need a real Qdrant.

    Returns (stub_module, qclient_factory_mock). The factory counts
    instantiations — we assert it was called exactly once across N
    concurrent _get_client() calls.
    """
    factory = MagicMock(side_effect=lambda *a, **kw: object())
    stub_qc = types.ModuleType("qdrant_client")
    stub_qc.QdrantClient = factory  # type: ignore[attr-defined]
    sys.modules["qdrant_client"] = stub_qc
    # Stub qdrant_client.models (used elsewhere; safe to be empty here)
    stub_models = types.ModuleType("qdrant_client.models")
    sys.modules["qdrant_client.models"] = stub_models
    return stub_qc, factory


def test_au_l2_01_concurrent_get_client_returns_same_instance() -> None:
    """Concurrent _get_client() callers must all receive the SAME instance."""
    _install_qdrant_stub()

    # Fresh import of index module (so _client starts at None)
    if "arastirma_ussu.ingest.index" in sys.modules:
        del sys.modules["arastirma_ussu.ingest.index"]
    from arastirma_ussu.ingest import index as idx

    # Reset module state defensively (in case prior tests primed it)
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


def test_au_l2_01_qdrant_client_constructed_at_most_once() -> None:
    """The underlying QdrantClient(...) constructor must be called ONCE
    across N concurrent first-time _get_client() calls."""
    _, factory = _install_qdrant_stub()

    if "arastirma_ussu.ingest.index" in sys.modules:
        del sys.modules["arastirma_ussu.ingest.index"]
    from arastirma_ussu.ingest import index as idx

    idx._client = None  # type: ignore[attr-defined]
    factory.reset_mock()

    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()
        idx._get_client()

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert factory.call_count == 1, (
        f"AU-L2-01: QdrantClient(...) instantiated {factory.call_count}x "
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
