"""Shared test fixtures."""

import urllib.request

import pytest


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture
def skip_no_ollama():
    """Skip test if Ollama is not running."""
    if not ollama_available():
        pytest.skip("Ollama not running")


# ---------------------------------------------------------------------------
# LlamaIndex (Layer 2+)
# ---------------------------------------------------------------------------

def llamaindex_available() -> bool:
    """Check if llama-index-core is importable."""
    try:
        import llama_index.core  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def skip_no_llamaindex():
    """Skip test if LlamaIndex is not installed."""
    if not llamaindex_available():
        pytest.skip("llama-index not installed")


@pytest.fixture
def tmp_doc_dir(tmp_path):
    """Temp documents directory with a sample .txt file."""
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()
    (doc_dir / "sample.txt").write_text(
        "Python programlama dili Guido van Rossum tarafindan gelistirilmistir. "
        "Ilk surum 1991 yilinda yayinlanmistir. "
        "Python acik kaynakli bir dildir ve genis bir topluluga sahiptir.",
        encoding="utf-8",
    )
    return doc_dir


# ---------------------------------------------------------------------------
# Qdrant (Layer 3+)
# ---------------------------------------------------------------------------

def qdrant_server_available() -> bool:
    """Check if Qdrant server is reachable on localhost:6333."""
    try:
        req = urllib.request.Request("http://localhost:6333/readyz")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def qdrant_client_available() -> bool:
    """Check if qdrant-client is importable."""
    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def skip_no_qdrant():
    """Skip test if Qdrant server is not running."""
    if not qdrant_server_available():
        pytest.skip("Qdrant not running")


@pytest.fixture
def skip_no_qdrant_client():
    """Skip test if qdrant-client is not installed."""
    if not qdrant_client_available():
        pytest.skip("qdrant-client not installed")


@pytest.fixture
def memory_client():
    """In-memory Qdrant client for unit tests (no Docker needed)."""
    from qdrant_client import QdrantClient

    return QdrantClient(location=":memory:")
