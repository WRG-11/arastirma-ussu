"""Qdrant-backed conversation memory."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from arastirma_ussu.config import QdrantConfig

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_cfg = QdrantConfig()
MAX_POINTS = _cfg.max_conversation_points

# Module-level singleton
_memory: ConversationMemory | None = None
_memory_lock = threading.Lock()


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
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

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
        return self._client.count(collection_name=self._collection).count

    def clear(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection)
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Delete the oldest point by timestamp (LRU eviction)."""
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


def get_memory(client: QdrantClient | None = None) -> ConversationMemory:
    """Return the singleton ConversationMemory instance (thread-safe).

    R89-20b AU-L2-03b: pre-fix this used a plain ``if _memory is None``
    TOCTOU check. ConversationMemory ``__init__`` opens a Qdrant
    connection and may create a collection; concurrent first-call
    races could spawn two instances, both touching the same collection
    metadata + leaving the loser orphaned (its connection never
    closed). Double-checked locking eliminates the race while keeping
    the hot read lock-free.
    """
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                _memory = ConversationMemory(client=client)
    return _memory
