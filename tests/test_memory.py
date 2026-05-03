"""Layer 3 smoke + integration tests for Qdrant memory and document index."""

import pytest

from arastirma_ussu.memory.tool import memory_search

# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests — qdrant-client :memory: mode, no Docker needed
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestMemorySearchSmoke:
    def test_import(self):
        assert memory_search is not None

    def test_callable(self):
        assert callable(memory_search)

    def test_returns_string(self):
        result = memory_search("test query")
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.smoke
class TestToolRegistryMemorySearch:
    def test_registry_has_memory_search(self):
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        assert "memory_search" in registry

    def test_registry_callable(self):
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        assert callable(registry["memory_search"].func)


@pytest.mark.smoke
class TestQdrantConfig:
    def test_defaults(self):
        from arastirma_ussu.config import QdrantConfig

        cfg = QdrantConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6333
        assert cfg.documents_collection == "documents"
        assert cfg.conversations_collection == "conversations"
        assert cfg.max_conversation_points == 5000
        assert cfg.memory_score_threshold == 0.65

    def test_in_app_config(self):
        from arastirma_ussu.config import AppConfig, QdrantConfig

        app = AppConfig()
        assert isinstance(app.qdrant, QdrantConfig)


@pytest.mark.smoke
class TestEmbedSmoke:
    def test_embed_query(self, skip_no_qdrant_client):
        from arastirma_ussu.ingest.embed import embed_query

        vec = embed_query("test embedding")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_get_embedding_dim(self, skip_no_qdrant_client):
        from arastirma_ussu.ingest.embed import get_embedding_dim

        dim = get_embedding_dim()
        assert dim > 0


@pytest.mark.smoke
class TestConversationMemorySmoke:
    """Uses qdrant-client :memory: mode — no Docker needed."""

    def test_save_and_search(self, memory_client):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(client=memory_client, collection="test_conv")
        mem.save("Python ne zaman cikti?", "Python 1991'de yayinlandi.")
        results = mem.search("Python tarihi", score_threshold=0.0)
        assert len(results) >= 1
        assert "1991" in results[0]["answer"]

    def test_empty_search(self, memory_client):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(client=memory_client, collection="test_empty")
        results = mem.search("herhangi bir sey")
        assert results == []

    def test_format_results_empty(self, memory_client):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(client=memory_client, collection="test_fmt")
        formatted = mem.format_results([])
        assert "bulunamadi" in formatted.lower()

    def test_format_results_with_data(self, memory_client):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(client=memory_client, collection="test_fmt2")
        results = [
            {"question": "Soru?", "answer": "Cevap.", "score": 0.85, "timestamp": "2026-05-03T12:00:00"},
        ]
        formatted = mem.format_results(results)
        assert "Soru?" in formatted
        assert "Cevap." in formatted
        assert "0.850" in formatted

    def test_count(self, memory_client):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(client=memory_client, collection="test_cnt")
        mem.save("q1", "a1")
        mem.save("q2", "a2")
        assert mem.count() == 2

    def test_clear(self, memory_client):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(client=memory_client, collection="test_clr")
        mem.save("q1", "a1")
        assert mem.count() == 1
        mem.clear()
        assert mem.count() == 0


@pytest.mark.smoke
class TestDocumentIndexSmoke:
    """Document index with Qdrant :memory: client."""

    def test_build_and_query(self, skip_no_llamaindex, memory_client, tmp_doc_dir):
        import arastirma_ussu.ingest.index as idx_mod

        # Reset module state
        idx_mod._client = memory_client
        idx_mod._collection_ready = False

        built = idx_mod._build_collection(
            doc_dir=tmp_doc_dir, client=memory_client, force_rebuild=True
        )
        assert built is True

        result = idx_mod.query_index("Python yaraticisi", client=memory_client)
        assert "Python" in result or "Guido" in result

    def test_ensure_collection_idempotent(self, skip_no_llamaindex, memory_client, tmp_doc_dir):
        import arastirma_ussu.ingest.index as idx_mod

        idx_mod._client = memory_client
        idx_mod._collection_ready = False

        idx_mod._build_collection(doc_dir=tmp_doc_dir, client=memory_client, force_rebuild=True)
        idx_mod._collection_ready = False  # reset flag to test ensure

        # Should find existing collection, not rebuild
        ready = idx_mod.ensure_collection(client=memory_client)
        assert ready is True

    def test_empty_dir_returns_false(self, skip_no_llamaindex, memory_client, tmp_path):
        import arastirma_ussu.ingest.index as idx_mod

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        idx_mod._collection_ready = False

        built = idx_mod._build_collection(
            doc_dir=empty_dir, client=memory_client, force_rebuild=True
        )
        assert built is False


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — need Docker Qdrant running
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQdrantServerIntegration:
    def test_memory_save_and_search(self, skip_no_qdrant):
        from arastirma_ussu.memory.store import ConversationMemory

        mem = ConversationMemory(collection="test_integration_conv")
        try:
            mem.clear()
            mem.save("Qdrant nedir?", "Qdrant bir vektor veritabanidir.")
            results = mem.search("vektor veritabani", score_threshold=0.0)
            assert len(results) >= 1
        finally:
            mem.clear()

    def test_document_index_rebuild(self, skip_no_qdrant, skip_no_llamaindex, tmp_doc_dir):
        import arastirma_ussu.ingest.index as idx_mod

        old_client = idx_mod._client
        idx_mod._client = None  # force reconnect to real server
        idx_mod._collection_ready = False

        try:
            built = idx_mod._build_collection(
                doc_dir=tmp_doc_dir, force_rebuild=True
            )
            assert built is True

            result = idx_mod.query_index("Python")
            assert len(result) > 10
        finally:
            idx_mod._client = old_client
            idx_mod._collection_ready = False
