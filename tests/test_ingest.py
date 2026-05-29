"""Layer 2 smoke + integration tests for document ingestion."""

import pytest

from arastirma_ussu.ingest.tool import doc_search

# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests — no LlamaIndex needed for import/registry checks
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestDocSearchSmoke:
    def test_import(self):
        """doc_search is importable without LlamaIndex installed."""
        assert doc_search is not None

    def test_callable(self):
        assert callable(doc_search)

    def test_returns_string(self):
        """Calling doc_search always returns a string, never raises."""
        result = doc_search("test query")
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.smoke
class TestToolRegistryDocSearch:
    def test_registry_has_doc_search(self):
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        assert "doc_search" in registry

    def test_registry_doc_search_callable(self):
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        assert callable(registry["doc_search"].func)


@pytest.mark.smoke
class TestLoaderSmoke:
    def test_supported_extensions(self, skip_no_llamaindex):
        from arastirma_ussu.ingest.loader import SUPPORTED_EXTENSIONS

        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_empty_dir(self, skip_no_llamaindex, tmp_path):
        from arastirma_ussu.ingest.loader import load_documents

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = load_documents(empty_dir)
        assert result == []

    def test_missing_dir(self, skip_no_llamaindex):
        from arastirma_ussu.ingest.loader import load_documents

        result = load_documents("/nonexistent/path/docs")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — need LlamaIndex installed
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestIndexIntegration:
    # Layer 3 (Qdrant) integration tests.  memory_client provides an in-memory
    # Qdrant instance — no running server needed.  skip_no_llamaindex guards the
    # SentenceSplitter import inside _build_collection.

    def test_build_collection_from_txt(self, skip_no_llamaindex, tmp_doc_dir, memory_client):
        from arastirma_ussu.ingest.index import _build_collection

        result = _build_collection(doc_dir=tmp_doc_dir, client=memory_client, force_rebuild=True)
        assert result is True

    def test_build_collection_stores_chunks(self, skip_no_llamaindex, tmp_doc_dir, memory_client):
        from arastirma_ussu.ingest.index import _build_collection
        from arastirma_ussu.config import QdrantConfig

        _build_collection(doc_dir=tmp_doc_dir, client=memory_client, force_rebuild=True)
        col_name = QdrantConfig().documents_collection
        count = memory_client.count(collection_name=col_name)
        assert count.count > 0

    def test_rebuild_clears_and_recreates(self, skip_no_llamaindex, tmp_doc_dir, memory_client):
        from arastirma_ussu.ingest.index import _build_collection

        r1 = _build_collection(doc_dir=tmp_doc_dir, client=memory_client, force_rebuild=True)
        r2 = _build_collection(doc_dir=tmp_doc_dir, client=memory_client, force_rebuild=True)
        assert r1 is True
        assert r2 is True

    def test_empty_documents_graceful(self, skip_no_llamaindex, tmp_path, memory_client):
        from arastirma_ussu.ingest.index import _build_collection

        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        result = _build_collection(doc_dir=empty_dir, client=memory_client, force_rebuild=True)
        assert result is False

    def test_doc_search_end_to_end(self, skip_no_qdrant):
        result = doc_search("Python programlama")
        assert isinstance(result, str)
        assert len(result) > 10
