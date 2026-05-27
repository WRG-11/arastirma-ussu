"""R89-65b AU-L2-03a re-do — embed.py SentenceTransformer singleton TOCTOU.

Re-do of the reverted R89-20b fix (f873ad48 -> 5ab7225 revert PR #9). The
prior fix used double-checked locking with ``threading.Lock()`` which was
race-correct but leaked stubbed objects into the module-level ``_model``
global across tests, breaking 11 downstream tests with AttributeError on
``.encode()`` / ``.get_sentence_embedding_dimension()``.

R89-65b fix: ``functools.lru_cache(maxsize=1)`` (Pattern 47 stdlib reuse strict).
CPython GIL makes cache fill atomic — concurrent first callers serialize.

Test discipline (SB-66 sister):
- ``monkeypatch.setitem(sys.modules, ...)`` (auto-restore on teardown)
- ``monkeypatch.delitem(...)`` to force a fresh import
- Explicit ``_get_model.cache_clear()`` in fixture teardown — no module-state leak
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def st_stub(monkeypatch):
    """Stub sentence_transformers + clear embed.py lru_cache on teardown.

    SB-66 discipline: monkeypatch.setitem auto-restores sys.modules; we ALSO
    explicitly clear the lru_cache so the next test gets a fresh load (no
    bleed of the stub return into other tests).
    """
    factory = MagicMock(side_effect=lambda *a, **kw: MagicMock(name="STModel"))
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    monkeypatch.delitem(sys.modules, "arastirma_ussu.ingest.embed", raising=False)

    yield factory

    # Belt-and-suspenders: even if the module was already imported pre-fixture,
    # clear its lru_cache so subsequent tests don't see the stubbed instance.
    try:
        from arastirma_ussu.ingest import embed as e

        e._get_model.cache_clear()
    except Exception:  # pragma: no cover — defensive on teardown
        pass


def test_au_l2_03a_concurrent_get_model_returns_same_instance(st_stub) -> None:
    from arastirma_ussu.ingest import embed as e

    e._get_model.cache_clear()
    st_stub.reset_mock()

    instances: list[object] = []
    barrier = threading.Barrier(16)

    def _race() -> None:
        barrier.wait()
        instances.append(e._get_model())

    threads = [threading.Thread(target=_race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 16
    first = instances[0]
    assert all(inst is first for inst in instances), (
        "AU-L2-03a re-do: TOCTOU regression — concurrent _get_model() returned "
        "multiple distinct SentenceTransformer instances"
    )
    assert st_stub.call_count == 1, (
        f"AU-L2-03a re-do: SentenceTransformer(...) instantiated {st_stub.call_count}x "
        "under concurrent race — expected exactly 1 (lru_cache fill failed)"
    )


def test_au_l2_03a_lru_cache_decorator_in_embed_module() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "arastirma_ussu" / "ingest" / "embed.py"
    text = src.read_text(encoding="utf-8")
    assert "from functools import lru_cache" in text, (
        "AU-L2-03a re-do: 'from functools import lru_cache' missing"
    )
    assert "@lru_cache" in text, (
        "AU-L2-03a re-do: @lru_cache decorator missing — singleton not cached"
    )


def test_au_l2_03a_cache_clear_resets_singleton(st_stub) -> None:
    """Regression guard: explicit cache_clear() yields a fresh instance.

    This is the key property the reverted DCL fix lacked — the module-level
    ``_model = None`` reset path was incomplete because the stubbed factory
    re-populated the global with a bare object that survived monkeypatch
    sys.modules cleanup.
    """
    from arastirma_ussu.ingest import embed as e

    e._get_model.cache_clear()
    first = e._get_model()
    second = e._get_model()
    assert first is second, "lru_cache should return same instance on repeat calls"

    e._get_model.cache_clear()
    third = e._get_model()
    assert third is not first, (
        "AU-L2-03a re-do: cache_clear() did not invalidate the singleton — "
        "test cleanup discipline broken"
    )
    assert st_stub.call_count == 2, (
        f"factory should be called once per cache fill (got {st_stub.call_count})"
    )
