"""Qdrant-backed conversation memory."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING

from arastirma_ussu.config import QdrantConfig

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_cfg = QdrantConfig()
MAX_POINTS = _cfg.max_conversation_points


class MemoryStoreError(Exception):
    """Raised when a Qdrant-backed memory operation fails.

    R89-21b AU-L2-08 (re-do R89-65b Phase B): domain-specific exception so
    callers can catch memory-layer failures without also catching unrelated
    errors AND without exposing the underlying Qdrant exception's PII /
    stack payload to the LLM. The original exception is logged server-side
    via ``logger.warning(...)`` but never included in this exception's
    message (Pattern 48 honest-discipline: comment explains the deliberate
    information hiding).
    """


class ConversationMemory:
    """Store and retrieve past Q&A pairs via Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str = _cfg.conversations_collection,
    ) -> None:
        if client is None:
            from qdrant_client import QdrantClient as _QC

            client = _QC(host=_cfg.host, port=_cfg.port, prefer_grpc=_cfg.prefer_grpc)
        self._client = client
        self._collection = collection
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create collection if it does not exist (idempotent)."""
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        from arastirma_ussu.ingest.embed import get_embedding_dim

        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=get_embedding_dim(),
                distance=Distance.COSINE,
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="timestamp",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, question: str, answer: str) -> str:
        """Save a Q&A pair. Returns the point ID. Evicts oldest if at cap."""
        from qdrant_client.models import PointStruct

        from arastirma_ussu.ingest.embed import embed_query

        # LRU eviction
        if self.count() >= MAX_POINTS:
            self._evict_oldest()

        # Hybrid embed: question + truncated answer
        embed_text = f"Q: {question}\nA: {answer[:200]}"
        vector = embed_query(embed_text)

        point_id = str(uuid.uuid4())
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                ],
            )
        except Exception as exc:
            # R89-21b AU-L2-08: Qdrant upsert exception was uncaught,
            # leaking raw library detail to caller. Wrap in domain
            # exception with no PII; log original server-side.
            logger.warning("memory upsert failed: %s", exc)
            raise MemoryStoreError("memory save failed") from exc
        return point_id

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = _cfg.memory_score_threshold,
    ) -> list[dict]:
        """Search memory for similar past Q&A pairs."""
        from arastirma_ussu.ingest.embed import embed_query

        vector = embed_query(query)
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            # R89-21b AU-L2-08: Qdrant query exception was uncaught.
            # Sister to AU-L2-07 ingest/index.py query_points. Wrap.
            logger.warning("memory query_points failed: %s", exc)
            raise MemoryStoreError("memory search failed") from exc

        out: list[dict] = []
        for point in results.points:
            out.append({
                "question": point.payload.get("question", ""),
                "answer": point.payload.get("answer", ""),
                "score": point.score,
                "timestamp": point.payload.get("timestamp", ""),
            })
        return out

    def format_results(self, results: list[dict]) -> str:
        """Format search results for agent consumption."""
        if not results:
            return "Hafizada eslesen konusma bulunamadi."

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            ts = r.get("timestamp", "")[:10]  # date only
            score = f"{r['score']:.3f}" if r.get("score") is not None else "N/A"
            parts.append(
                f"[{i}] (benzerlik: {score}, tarih: {ts})\n"
                f"S: {r['question']}\n"
                f"Y: {r['answer']}"
            )
        return "\n\n---\n\n".join(parts)

    def count(self) -> int:
        """Return number of stored conversations."""
        try:
            return self._client.count(collection_name=self._collection).count
        except Exception as exc:
            # R89-21b AU-L2-08: called from save() LRU check; raw
            # exception would surface to the user-facing save path.
            logger.warning("memory count failed: %s", exc)
            raise MemoryStoreError("memory count failed") from exc

    def clear(self) -> None:
        """Delete and recreate the collection."""
        try:
            self._client.delete_collection(self._collection)
            self._ensure_collection()
        except Exception as exc:
            # R89-21b AU-L2-08: explicit user action; surface as
            # domain error without raw Qdrant detail.
            logger.warning("memory clear failed: %s", exc)
            raise MemoryStoreError("memory clear failed") from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Delete the oldest point by timestamp (LRU eviction)."""
        # R89-21b AU-L2-08: scroll + delete each had no exception handling.
        # Called from save() -- failure here was crashing save()
        # opaquely. Wrap both calls under one MemoryStoreError mapping.
        try:
            # Scroll to find the point with the earliest timestamp
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=1,
                order_by="timestamp",
            )
            if points:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=[points[0].id],
                )
        except Exception as exc:
            logger.warning("memory _evict_oldest failed: %s", exc)
            raise MemoryStoreError("memory LRU eviction failed") from exc


# R89-65b AU-L2-03b re-do: TOCTOU-safe lazy singleton via functools.lru_cache.
# Sister to embed.py ``_get_model`` (see that module for the full rationale).
#
# Note on the ``client`` parameter: lru_cache keys on positional + keyword args,
# so callers passing an explicit ``client`` get their own instance (semantically
# correct — a different client means a different memory). The singleton
# behaviour only applies to the default ``client=None`` path, which is the
# production call site (``agent/graph.py``, ``memory/tool.py``). Tests pass
# ``client=<mock>``; each gets a fresh memory + Qdrant client.
#
# Because ``QdrantClient`` instances are not hashable, we route the default
# path through a separate zero-arg helper (``_get_default_memory``) and only
# cache that. Custom-client calls construct a fresh ``ConversationMemory``
# every time (no cache; explicit client = explicit identity).
@lru_cache(maxsize=1)
def _get_default_memory() -> ConversationMemory:
    """Singleton ConversationMemory backed by the default Qdrant client."""
    return ConversationMemory(client=None)


def get_memory(client: QdrantClient | None = None) -> ConversationMemory:
    """Return a ConversationMemory instance.

    For the default ``client=None`` path the same singleton is returned across
    concurrent first callers (thread-safe via CPython's GIL + lru_cache fill
    atomicity). When ``client`` is provided explicitly, a fresh instance is
    constructed — each explicit client gets its own memory.
    """
    if client is None:
        return _get_default_memory()
    return ConversationMemory(client=client)
