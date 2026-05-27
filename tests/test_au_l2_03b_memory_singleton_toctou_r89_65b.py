"""R89-65b AU-L2-03b re-do — memory/store.py ConversationMemory singleton TOCTOU.

Re-do of the reverted R89-20b fix. Pattern 47 stdlib reuse: ``functools.lru_cache``
on a zero-arg helper ``_get_default_memory``; ``get_memory(client=None)`` routes
through it, ``get_memory(client=<x>)`` constructs fresh (explicit client = explicit
identity; QdrantClient instances aren't hashable so they can't be a cache key).
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def cm_factory_stub(monkeypatch):
    """Patch ConversationMemory.__init__ to count instantiations + skip Qdrant.

    We don't stub qdrant_client globally here — instead we replace
    ConversationMemory's collection-ensure path with a no-op, so __init__
    succeeds without a real Qdrant connection.
    """
    # Stub qdrant_client so the default-client construction path inside
    # ConversationMemory.__init__ doesn't try a real connection.
    qstub = types.ModuleType("qdrant_client")
    qstub.QdrantClient = MagicMock(  # type: ignore[attr-defined]
        side_effect=lambda *a, **kw: MagicMock(name="QdrantClient")
    )
    monkeypatch.setitem(sys.modules, "qdrant_client", qstub)

    qmstub = types.ModuleType("qdrant_client.models")
    qmstub.Distance = MagicMock()  # type: ignore[attr-defined]
    qmstub.PayloadSchemaType = MagicMock()  # type: ignore[attr-defined]
    qmstub.VectorParams = MagicMock()  # type: ignore[attr-defined]
    qmstub.PointStruct = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client.models", qmstub)

    monkeypatch.delitem(sys.modules, "arastirma_ussu.memory.store", raising=False)

    from arastirma_ussu.memory import store as s

    factory = MagicMock(wraps=s.ConversationMemory.__init__, return_value=None)
    monkeypatch.setattr(s.ConversationMemory, "__init__", factory)
    # Also skip the Qdrant collection-ensure call inside __init__ — but our
    # wraps factory returns None and doesn't actually call into the real
    # __init__ body; instead instances are constructed empty. That's fine for
    # identity assertions.
    yield s, factory

    try:
        s._get_default_memory.cache_clear()
    except Exception:  # pragma: no cover
        pass


def test_au_l2_03b_concurrent_get_memory_returns_same_default_instance(
    cm_factory_stub,
) -> None:
    s, factory = cm_factory_stub
    s._get_default_memory.cache_clear()
    factory.reset_mock()

    instances: list[object] = []
    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()
        instances.append(s.get_memory())

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 16
    first = instances[0]
    assert all(inst is first for inst in instances), (
        "AU-L2-03b re-do: concurrent get_memory() returned distinct singletons"
    )
    assert factory.call_count == 1, (
        f"AU-L2-03b re-do: ConversationMemory instantiated {factory.call_count}x "
        "under concurrent race — expected exactly 1"
    )


def test_au_l2_03b_lru_cache_decorator_in_store_module() -> None:
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "arastirma_ussu"
        / "memory"
        / "store.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "from functools import lru_cache" in text
    assert "@lru_cache" in text
    assert "_get_default_memory" in text, (
        "AU-L2-03b re-do: missing default-memory cache helper — singleton "
        "wiring uses a zero-arg cached function because QdrantClient is unhashable"
    )


def test_au_l2_03b_explicit_client_bypasses_singleton_cache(cm_factory_stub) -> None:
    """Passing an explicit client constructs a fresh ConversationMemory.

    This is the semantically-correct behaviour: each explicit client is its
    own identity. Only the default ``client=None`` path is cached.
    """
    s, factory = cm_factory_stub
    s._get_default_memory.cache_clear()
    factory.reset_mock()

    default_a = s.get_memory()
    default_b = s.get_memory()
    assert default_a is default_b, "default-path singleton broken"

    explicit_client = MagicMock(name="custom_client")
    explicit_a = s.get_memory(client=explicit_client)
    explicit_b = s.get_memory(client=explicit_client)
    assert explicit_a is not explicit_b, (
        "AU-L2-03b re-do: explicit client should NOT be singleton-cached "
        "(each explicit client identity gets a fresh memory)"
    )
    assert explicit_a is not default_a, (
        "AU-L2-03b re-do: explicit-client memory should not equal default memory"
    )
