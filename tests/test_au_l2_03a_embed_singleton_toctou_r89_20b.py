"""R89-20b AU-L2-03a regression — embed.py SentenceTransformer singleton TOCTOU.

Sister to AU-L2-01 (Qdrant). Same lazy-singleton anti-pattern: cold-start
race could load the model twice. Cost is meaningful here: SentenceTransformer
load is multi-second + multi-MB.

Fix: double-checked lock around the lazy load.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock


def _install_st_stub() -> MagicMock:
    """Stub sentence_transformers — factory counts instantiations."""
    factory = MagicMock(side_effect=lambda *a, **kw: object())
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = factory  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub
    return factory


def test_au_l2_03a_concurrent_get_model_returns_same_instance() -> None:
    factory = _install_st_stub()

    if "arastirma_ussu.ingest.embed" in sys.modules:
        del sys.modules["arastirma_ussu.ingest.embed"]
    from arastirma_ussu.ingest import embed as e

    e._model = None  # type: ignore[attr-defined]
    factory.reset_mock()

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
        "AU-L2-03a: TOCTOU regression — concurrent _get_model() returned "
        "multiple distinct SentenceTransformer instances"
    )
    assert factory.call_count == 1, (
        f"AU-L2-03a: SentenceTransformer(...) instantiated {factory.call_count}x "
        "under concurrent race — expected exactly 1 (DCL failed)"
    )


def test_au_l2_03a_threading_lock_imported_in_embed_module() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "arastirma_ussu" / "ingest" / "embed.py"
    text = src.read_text(encoding="utf-8")
    assert "import threading" in text, "AU-L2-03a: 'import threading' missing"
    assert "threading.Lock()" in text, "AU-L2-03a: threading.Lock() missing — DCL incomplete"
