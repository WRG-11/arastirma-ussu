"""R89-21b AU-L2-08 regression — memory.store Qdrant calls wrapped.

Sister to AU-L2-07 (same module-class: Qdrant call without try/except).
Five sites in ``ConversationMemory``:
  - save()  -> _client.upsert
  - search() -> _client.query_points
  - count() -> _client.count
  - clear() -> _client.delete_collection
  - _evict_oldest() -> _client.scroll + _client.delete

Each must catch Exception, log server-side, raise ``MemoryStoreError``
without including the raw exception text (PII / injection guard).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def memory_with_mocked_client(monkeypatch):
    """Build a ConversationMemory with an entirely mocked Qdrant client.

    R89-71b (amend): the previous incarnation of this fixture stubbed
    ``qdrant_client`` + ``qdrant_client.models`` via
    ``monkeypatch.setitem(sys.modules, ...)`` + ``delitem`` + re-import to
    accommodate environments without the real qdrant-client installed.
    That worked in isolation but polluted later tests (specifically
    ``test_memory.py::test_lru_eviction``) on CI: the re-imported
    ``arastirma_ussu.memory.store`` module shadow-effected the original
    in a way that caused LRU eviction to silently no-op when the real
    ``memory_client`` fixture ran subsequently. Root cause: cumulative
    delitem/re-import cycles fragmented module identity even after
    monkeypatch restored ``sys.modules``.

    Fix: assume qdrant-client is installed (it is in CI + project deps),
    use ``ConversationMemory.__new__`` to bypass ``__init__``, and rely
    purely on attribute-level mocks. No sys.modules manipulation -> no
    cross-test pollution. This preserves the AU-L2-08 wrap assertions
    (exception text never reaches caller, MemoryStoreError raised, log
    emitted) without the unsafe import-machinery dance.
    """
    # Stub embed so we don't load SentenceTransformer in save(). Patched
    # via monkeypatch.setattr -> auto-restores cleanly at teardown.
    import arastirma_ussu.ingest.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: [0.1, 0.2, 0.3])

    from arastirma_ussu.memory.store import ConversationMemory

    # Bypass __init__'s collection-ensure path (no real Qdrant connection).
    instance = ConversationMemory.__new__(ConversationMemory)
    instance._client = MagicMock()
    instance._collection = "test_collection"
    return instance


def _boom() -> Exception:
    return RuntimeError(
        "PROXY_USER=admin:s3cret /home/qdrant/cluster.toml -- leaked"
    )


def test_au_l2_08_save_qdrant_exception_wrapped(
    memory_with_mocked_client, caplog
):
    from arastirma_ussu.memory.store import MemoryStoreError

    mem = memory_with_mocked_client
    mem._client.count.return_value = MagicMock(count=0)
    mem._client.upsert.side_effect = _boom()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(MemoryStoreError) as excinfo:
            mem.save("q", "a")

    # Domain exception message is generic, no PII
    assert "PROXY_USER" not in str(excinfo.value)
    assert "leaked" not in str(excinfo.value)
    assert "save failed" in str(excinfo.value)
    # Original chained for debug
    assert excinfo.value.__cause__ is not None
    # Log carries diagnostics
    assert any("memory upsert failed" in r.message for r in caplog.records)


def test_au_l2_08_search_qdrant_exception_wrapped(
    memory_with_mocked_client, caplog
):
    from arastirma_ussu.memory.store import MemoryStoreError

    mem = memory_with_mocked_client
    mem._client.query_points.side_effect = _boom()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(MemoryStoreError) as excinfo:
            mem.search("q")

    assert "PROXY_USER" not in str(excinfo.value)
    assert "search failed" in str(excinfo.value)


def test_au_l2_08_count_qdrant_exception_wrapped(memory_with_mocked_client):
    from arastirma_ussu.memory.store import MemoryStoreError

    mem = memory_with_mocked_client
    mem._client.count.side_effect = _boom()

    with pytest.raises(MemoryStoreError) as excinfo:
        mem.count()

    assert "PROXY_USER" not in str(excinfo.value)
    assert "count failed" in str(excinfo.value)


def test_au_l2_08_clear_qdrant_exception_wrapped(memory_with_mocked_client):
    from arastirma_ussu.memory.store import MemoryStoreError

    mem = memory_with_mocked_client
    mem._client.delete_collection.side_effect = _boom()

    with pytest.raises(MemoryStoreError) as excinfo:
        mem.clear()

    assert "PROXY_USER" not in str(excinfo.value)
    assert "clear failed" in str(excinfo.value)


def test_au_l2_08_evict_oldest_scroll_exception_wrapped(
    memory_with_mocked_client,
):
    from arastirma_ussu.memory.store import MemoryStoreError

    mem = memory_with_mocked_client
    mem._client.scroll.side_effect = _boom()

    with pytest.raises(MemoryStoreError) as excinfo:
        mem._evict_oldest()

    assert "PROXY_USER" not in str(excinfo.value)
    assert "LRU eviction failed" in str(excinfo.value)


def test_au_l2_08_memory_store_error_is_exception_subclass():
    from arastirma_ussu.memory.store import MemoryStoreError

    assert issubclass(MemoryStoreError, Exception)


def test_au_l2_08_happy_paths_unchanged(memory_with_mocked_client):
    """Sanity: success paths preserved."""
    mem = memory_with_mocked_client
    mem._client.count.return_value = MagicMock(count=5)
    mem._client.upsert.return_value = None

    # save returns point id
    point_id = mem.save("question?", "answer.")
    assert isinstance(point_id, str) and len(point_id) > 10

    # count returns int
    assert mem.count() == 5

    # search empty result list
    mem._client.query_points.return_value = MagicMock(points=[])
    assert mem.search("query") == []
