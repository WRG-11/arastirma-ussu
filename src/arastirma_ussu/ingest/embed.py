"""Shared embedding utility — sentence-transformers, CPU-only."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from arastirma_ussu.config import IngestConfig

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_cfg = IngestConfig()


# R89-65b AU-L2-03a re-do: TOCTOU-safe lazy singleton via functools.lru_cache.
#
# Pre-fix (reverted f873ad48 R89-20b): a manual ``_model: SentenceTransformer | None
# = None`` module global with double-checked locking + ``threading.Lock()``. The
# DCL itself was correct, but interacted badly with tests that stubbed
# ``sentence_transformers`` to return a bare ``object()`` — the module-level
# ``_model`` cache survived sys.modules cleanup, leaking the bare object into
# 11 downstream tests with AttributeError on ``.encode()`` /
# ``.get_sentence_embedding_dimension()``.
#
# Fix (Pattern 47 reuse>new strict): ``functools.lru_cache(maxsize=1)``. CPython's
# GIL makes the cache lookup atomic — concurrent first callers serialize on
# the cache fill, no manual ``threading.Lock()`` needed. Tests reset state via
# ``_get_model.cache_clear()`` (explicit, leakage-free), which the prior global
# ``_model = None`` reset only half-did.
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load embedding model on first call, cache thereafter (thread-safe)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_cfg.embedding_model, device="cpu")


def get_embedding_dim() -> int:
    """Return the embedding dimensionality (model-swap-safe)."""
    return _get_model().get_sentence_embedding_dimension()


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of texts into float vectors."""
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
