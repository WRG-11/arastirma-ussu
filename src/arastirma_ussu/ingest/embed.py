"""Shared embedding utility — sentence-transformers, CPU-only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arastirma_ussu.config import IngestConfig

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_cfg = IngestConfig()

# Module-level lazy singleton
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load embedding model on first call, cache thereafter."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_cfg.embedding_model, device="cpu")
    return _model


def get_embedding_dim() -> int:
    """Return the embedding dimensionality (model-swap-safe)."""
    return _get_model().get_embedding_dimension()


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of texts into float vectors."""
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
