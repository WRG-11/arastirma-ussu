"""R89-21b AU-L2-07 regression — query_index Qdrant exception sentinel.

Pre-fix: ``query_index`` in ``ingest/index.py`` called ``c.query_points(...)``
without any exception handling. Backend failures (Qdrant unreachable,
schema drift, auth expiry) propagated as raw exceptions to the agent
tool wrapper, which does ``except Exception as e: observation = f"Tool error: {e}"``.
Sister leak class to AU-L2-05: raw exception text round-trips into
the LLM context (PII / injection risk).

Post-fix: catch Exception inside query_index, log server-side, return
stable Turkish sentinel.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock


def _make_fake_client(*, query_points_raises: Exception | None = None) -> MagicMock:
    """Build a mock Qdrant client with controllable failure modes."""
    client = MagicMock()
    client.collection_exists.return_value = True
    if query_points_raises is not None:
        client.query_points.side_effect = query_points_raises
    else:
        client.query_points.return_value = MagicMock(points=[])
    return client


def test_au_l2_07_query_points_exception_returns_sentinel(
    caplog, monkeypatch
) -> None:
    """Adversarial Qdrant exception with PII payload -> sentinel returned."""
    # Avoid loading the real SentenceTransformer
    import arastirma_ussu.ingest.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: [0.1, 0.2, 0.3])

    from arastirma_ussu.ingest.index import query_index

    boom = RuntimeError(
        "PROXY_USER=admin:s3cret /home/qdrant/cluster.toml -- leaked"
    )
    client = _make_fake_client(query_points_raises=boom)

    with caplog.at_level(logging.WARNING):
        result = query_index("test query", client=client)

    assert result == "Belge sorgusu su anda yapilamadi.", (
        f"AU-L2-07: expected sentinel; got {result!r}"
    )
    # PII keywords absent from return value
    assert "PROXY_USER" not in result
    assert "/home/qdrant" not in result
    assert "leaked" not in result

    # Diagnostic log present
    assert any(
        "query_points failed" in r.message for r in caplog.records
    ), "AU-L2-07: expected 'query_points failed' warning log"


def test_au_l2_07_query_points_success_unchanged(monkeypatch) -> None:
    """Sanity: happy path still returns formatted results (no regression)."""
    import arastirma_ussu.ingest.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: [0.1, 0.2, 0.3])

    from arastirma_ussu.ingest.index import query_index

    point = MagicMock()
    point.payload = {"file_name": "doc.txt", "text": "matched chunk text"}
    point.score = 0.95
    client = _make_fake_client()
    client.query_points.return_value = MagicMock(points=[point])

    result = query_index("test query", client=client)

    assert "doc.txt" in result
    assert "matched chunk text" in result
