"""R89-20b AU-L2-03b regression — memory.store ConversationMemory singleton TOCTOU.

Sister to AU-L2-01 (Qdrant) + AU-L2-03a (embed). ``get_memory()`` was
guarded by a plain ``if _memory is None`` check; concurrent first
callers could each instantiate ConversationMemory (which opens a
Qdrant connection + may create a collection), the second silently
replacing the first reference + orphaning the loser's connection.

Fix: double-checked locking.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def qdrant_stub(monkeypatch):
    """Stub qdrant_client; monkeypatch auto-cleans sys.modules."""
    stub_qc = types.ModuleType("qdrant_client")
    stub_qc.QdrantClient = MagicMock()  # type: ignore[attr-defined]

    class _Distance:
        COSINE = "cosine"

    class _PayloadSchemaType:
        KEYWORD = "keyword"

    class _VectorParams:
        def __init__(self, *a, **kw) -> None:
            pass

    stub_models = types.ModuleType("qdrant_client.models")
    stub_models.Distance = _Distance  # type: ignore[attr-defined]
    stub_models.PayloadSchemaType = _PayloadSchemaType  # type: ignore[attr-defined]
    stub_models.VectorParams = _VectorParams  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qdrant_client", stub_qc)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", stub_models)
    monkeypatch.delitem(sys.modules, "arastirma_ussu.memory.store", raising=False)


def test_au_l2_03b_concurrent_get_memory_returns_same_instance(
    qdrant_stub, monkeypatch
) -> None:
    from arastirma_ussu.memory import store as mem

    mem._memory = None  # type: ignore[attr-defined]

    instances: list[object] = []
    instantiation_count = [0]

    def _counting_init(self, *args, **kwargs):
        instantiation_count[0] += 1
        # Minimal init: just set the attributes get_memory's caller might touch
        self._client = MagicMock()
        self._collection = "test"

    monkeypatch.setattr(mem.ConversationMemory, "__init__", _counting_init)

    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()
        instances.append(mem.get_memory(client=MagicMock()))

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 16
    first = instances[0]
    assert all(inst is first for inst in instances), (
        "AU-L2-03b: TOCTOU regression — concurrent get_memory() returned "
        "multiple distinct ConversationMemory instances"
    )
    assert instantiation_count[0] == 1, (
        f"AU-L2-03b: ConversationMemory(...) instantiated {instantiation_count[0]}x "
        "under concurrent race — expected exactly 1 (DCL failed)"
    )


def test_au_l2_03b_threading_lock_imported_in_store_module() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "arastirma_ussu" / "memory" / "store.py"
    text = src.read_text(encoding="utf-8")
    assert "import threading" in text, "AU-L2-03b: 'import threading' missing"
    assert "_memory_lock = threading.Lock()" in text, "AU-L2-03b: _memory_lock missing"
