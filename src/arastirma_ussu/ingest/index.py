"""Vector index backed by Qdrant (Layer 3) or in-memory LlamaIndex (fallback)."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from arastirma_ussu.config import IngestConfig, QdrantConfig
from arastirma_ussu.ingest.loader import load_documents

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_icfg = IngestConfig()
_qcfg = QdrantConfig()

# Module-level state
_client: QdrantClient | None = None
_collection_ready: bool = False


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------

def _get_client() -> QdrantClient:
    """Return a cached Qdrant client."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient as _QC

        _client = _QC(host=_qcfg.host, port=_qcfg.port, prefer_grpc=_qcfg.prefer_grpc)
    return _client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient | None = None) -> bool:
    """Ensure the documents collection exists (idempotent).

    Returns True if collection is ready, False if no documents and no collection.
    """
    global _collection_ready
    if _collection_ready:
        return True

    c = client or _get_client()
    col = _qcfg.documents_collection

    if c.collection_exists(col):
        _collection_ready = True
        return True

    # No collection yet — don't create empty
    return False


def _build_collection(
    doc_dir: str | Path = _icfg.doc_dir,
    client: QdrantClient | None = None,
    force_rebuild: bool = False,
) -> bool:
    """(Re)build the documents collection from data/documents/.

    Returns True if collection was built, False if no documents found.
    """
    global _collection_ready
    from llama_index.core.node_parser import SentenceSplitter
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from arastirma_ussu.ingest.embed import embed_texts, get_embedding_dim

    c = client or _get_client()
    col = _qcfg.documents_collection

    # Delete if force rebuild
    if force_rebuild and c.collection_exists(col):
        c.delete_collection(col)
        _collection_ready = False

    # Load documents
    documents = load_documents(doc_dir)
    if not documents:
        logger.warning("Belge bulunamadi: %s", doc_dir)
        return False

    # Chunk
    splitter = SentenceSplitter(
        chunk_size=_icfg.chunk_size,
        chunk_overlap=_icfg.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    if not nodes:
        return False

    # Create collection
    dim = get_embedding_dim()
    if not c.collection_exists(col):
        c.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # Embed and upsert in batches
    batch_size = 32
    for start in range(0, len(nodes), batch_size):
        batch_nodes = nodes[start : start + batch_size]
        texts = [n.get_content() for n in batch_nodes]
        vectors = embed_texts(texts, batch_size=batch_size)

        points = []
        for j, (node, vec) in enumerate(zip(batch_nodes, vectors)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{node.node_id}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "text": node.get_content(),
                        "file_name": node.metadata.get("file_name", ""),
                        "file_path": node.metadata.get("file_path", ""),
                        "chunk_index": start + j,
                    },
                )
            )
        c.upsert(collection_name=col, points=points)

    _collection_ready = True
    logger.info(
        "Qdrant '%s' koleksiyonu olusturuldu: %d chunk", col, len(nodes)
    )
    return True


# ---------------------------------------------------------------------------
# Public API (same interface as Layer 2)
# ---------------------------------------------------------------------------

def ensure_index(
    doc_dir: str | Path = _icfg.doc_dir,
    force_rebuild: bool = False,
    client: QdrantClient | None = None,
) -> bool:
    """Ensure the documents collection is ready.

    Compatible with Layer 2 signature for graph.py REPL command.
    Returns True if ready, False if no documents.
    """
    if not force_rebuild and ensure_collection(client):
        return True
    return _build_collection(doc_dir=doc_dir, client=client, force_rebuild=force_rebuild)


def query_index(
    query: str,
    top_k: int = _icfg.top_k,
    client: QdrantClient | None = None,
) -> str:
    """Query the documents collection, return formatted chunk text."""
    from arastirma_ussu.ingest.embed import embed_query

    c = client or _get_client()
    col = _qcfg.documents_collection

    if not c.collection_exists(col):
        return "Indekslenmis belge bulunamadi. data/documents/ dizinine dosya ekleyin."

    vector = embed_query(query)
    try:
        results = c.query_points(
            collection_name=col,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:
        # R89-21b AU-L2-07: previously no try/except around the Qdrant
        # query call. Backend failures (Qdrant down, schema drift,
        # auth expiry) propagated as raw exceptions to the caller --
        # which is the agent tool wrapper that does
        # `except Exception as e: observation = f"Tool error: {e}"`.
        # That round-trips the raw exception string back into the LLM
        # context (sister leak to AU-L2-05 web_search). Fix: catch,
        # log server-side, return a stable Turkish sentinel. Do NOT
        # re-raise -- caller's catch leaks exception text.
        logger.warning("query_points failed for collection %s: %s", col, exc)
        return "Belge sorgusu su anda yapilamadi."

    if not results.points:
        return "Sorguyla eslesen belge parcasi bulunamadi."

    parts: list[str] = []
    for i, point in enumerate(results.points, 1):
        source = point.payload.get("file_name", "bilinmeyen")
        score = f"{point.score:.3f}" if point.score is not None else "N/A"
        text = point.payload.get("text", "").strip()
        parts.append(f"[{i}] (kaynak: {source}, skor: {score})\n{text}")

    return "\n\n---\n\n".join(parts)
