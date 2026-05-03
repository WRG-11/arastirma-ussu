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
