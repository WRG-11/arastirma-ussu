"""Vector index: chunking, embedding, persistence, and querying."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from arastirma_ussu.config import IngestConfig
from arastirma_ussu.ingest.loader import load_documents

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex

logger = logging.getLogger(__name__)

_cfg = IngestConfig()

# Module-level lazy singleton
_index: VectorStoreIndex | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_embed_model():
    """CPU-only HuggingFace embedding model."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=_cfg.embedding_model,
        device="cpu",
        embed_batch_size=32,
    )


def _apply_settings() -> None:
    """Configure LlamaIndex global Settings once."""
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter

    Settings.llm = None  # all LLM calls go through LangGraph
    Settings.embed_model = _build_embed_model()
    Settings.node_parser = SentenceSplitter(
        chunk_size=_cfg.chunk_size,
        chunk_overlap=_cfg.chunk_overlap,
    )


def _build_index(
    doc_dir: str | Path = _cfg.doc_dir,
    persist_dir: str | Path = _cfg.index_dir,
    force_rebuild: bool = False,
) -> VectorStoreIndex | None:
    """Build a fresh index or load a persisted one."""
    from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

    _apply_settings()
    persist_path = Path(persist_dir)

    # Try loading persisted index
    if not force_rebuild and persist_path.exists() and any(persist_path.iterdir()):
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_path))
            index = load_index_from_storage(storage_ctx)
            logger.info("Persisted index yuklendi: %s", persist_path)
            return index
        except Exception:
            logger.warning("Persisted index bozuk, yeniden olusturuluyor", exc_info=True)

    # Build from documents
    documents = load_documents(doc_dir)
    if not documents:
        return None

    index = VectorStoreIndex.from_documents(documents)

    # Persist
    persist_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_path))
    logger.info("Index olusturuldu ve kaydedildi: %s", persist_path)

    return index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_index(
    doc_dir: str | Path = _cfg.doc_dir,
    persist_dir: str | Path = _cfg.index_dir,
    force_rebuild: bool = False,
) -> VectorStoreIndex | None:
    """Return the singleton index, building it on first call."""
    global _index
    if _index is not None and not force_rebuild:
        return _index
    _index = _build_index(doc_dir, persist_dir, force_rebuild)
    return _index


def query_index(query: str, top_k: int = _cfg.top_k) -> str:
    """Query the index, return formatted chunk text."""
    index = ensure_index()
    if index is None:
        return "Indekslenmis belge bulunamadi. data/documents/ dizinine dosya ekleyin."

    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    if not nodes:
        return "Sorguyla eslesen belge parcasi bulunamadi."

    parts: list[str] = []
    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("file_name", "bilinmeyen")
        score = f"{node.score:.3f}" if node.score is not None else "N/A"
        text = node.get_content().strip()
        parts.append(f"[{i}] (kaynak: {source}, skor: {score})\n{text}")

    return "\n\n---\n\n".join(parts)
