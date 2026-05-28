"""R89-65b AU-L2-01 re-do — index.py Qdrant client singleton TOCTOU.

Re-do of the reverted R89-20b fix. Pattern 47 stdlib reuse: ``functools.lru_cache``
replaces manual double-checked locking + ``threading.Lock()``.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def qdrant_stub(monkeypatch):
    factory = MagicMock(side_effect=lambda *a, **kw: MagicMock(name="QdrantClient"))
    stub = types.ModuleType("qdrant_client")
    stub.QdrantClient = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", stub)
    monkeypatch.delitem(sys.modules, "arastirma_ussu.ingest.index", raising=False)

    yield factory

    try:
        from arastirma_ussu.ingest import index as i

        i._get_client.cache_clear()
    except Exception:  # pragma: no cover
        pass


def test_au_l2_01_concurrent_get_client_returns_same_instance(qdrant_stub) -> None:
    from arastirma_ussu.ingest import index as i

    i._get_client.cache_clear()
    qdrant_stub.reset_mock()

    instances: list[object] = []
    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()
        instances.append(i._get_client())

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 16
    first = instances[0]
    assert all(inst is first for inst in instances), (
        "AU-L2-01 re-do: concurrent _get_client() returned distinct instances"
    )
    assert qdrant_stub.call_count == 1, (
        f"AU-L2-01 re-do: QdrantClient instantiated {qdrant_stub.call_count}x "
        "under concurrent race — expected exactly 1"
    )


def test_au_l2_01_lru_cache_decorator_in_index_module() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "arastirma_ussu" / "ingest" / "index.py"
    text = src.read_text(encoding="utf-8")
    assert "from functools import lru_cache" in text
    assert "@lru_cache" in text


def test_au_l2_01_cache_clear_resets_singleton(qdrant_stub) -> None:
    from arastirma_ussu.ingest import index as i

    i._get_client.cache_clear()
    first = i._get_client()
    second = i._get_client()
    assert first is second

    i._get_client.cache_clear()
    third = i._get_client()
    assert third is not first
    assert qdrant_stub.call_count == 2
