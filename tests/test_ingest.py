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
    def test_build_index_from_txt(self, skip_no_llamaindex, tmp_doc_dir, tmp_path):
        from arastirma_ussu.ingest.index import _build_index

        index_dir = tmp_path / "index"
        index = _build_index(
            doc_dir=tmp_doc_dir,
            persist_dir=index_dir,
            force_rebuild=True,
        )
        assert index is not None

    def test_query_returns_content(self, skip_no_llamaindex, tmp_doc_dir, tmp_path):
        from arastirma_ussu.ingest.index import _build_index, query_index, _apply_settings
        import arastirma_ussu.ingest.index as idx_mod

        _apply_settings()
        index_dir = tmp_path / "index"
        idx_mod._index = _build_index(
            doc_dir=tmp_doc_dir,
            persist_dir=index_dir,
            force_rebuild=True,
        )
        assert idx_mod._index is not None

        result = query_index("Python yaraticisi")
        assert "Guido" in result or "Python" in result

    def test_persist_and_reload(self, skip_no_llamaindex, tmp_doc_dir, tmp_path):
        from arastirma_ussu.ingest.index import _build_index, _apply_settings
        import arastirma_ussu.ingest.index as idx_mod

        _apply_settings()
        index_dir = tmp_path / "index"

        # Build and persist
        idx1 = _build_index(doc_dir=tmp_doc_dir, persist_dir=index_dir, force_rebuild=True)
        assert idx1 is not None
        assert (index_dir / "docstore.json").exists()

        # Clear singleton, reload from disk
        idx_mod._index = None
        idx2 = _build_index(doc_dir=tmp_doc_dir, persist_dir=index_dir, force_rebuild=False)
        assert idx2 is not None

    def test_empty_documents_graceful(self, skip_no_llamaindex, tmp_path):
        from arastirma_ussu.ingest.index import _build_index, query_index, _apply_settings
        import arastirma_ussu.ingest.index as idx_mod

        _apply_settings()
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        index_dir = tmp_path / "index"

        idx_mod._index = _build_index(
            doc_dir=empty_dir, persist_dir=index_dir, force_rebuild=True
        )
        # _index is None because no documents
        assert idx_mod._index is None

        result = query_index("anything")
        assert "bulunamadi" in result.lower() or "ekleyin" in result.lower()

    def test_doc_search_end_to_end(self, skip_no_llamaindex, tmp_doc_dir, tmp_path):
        from arastirma_ussu.ingest.index import _build_index, _apply_settings
        import arastirma_ussu.ingest.index as idx_mod

        _apply_settings()
        index_dir = tmp_path / "index"
        idx_mod._index = _build_index(
            doc_dir=tmp_doc_dir,
            persist_dir=index_dir,
            force_rebuild=True,
        )

        result = doc_search("Python programlama")
        assert isinstance(result, str)
        assert len(result) > 10
