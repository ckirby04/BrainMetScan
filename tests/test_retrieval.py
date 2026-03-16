"""Tests for hybrid retrieval engine."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.rag.retrieval import HybridRetriever, RetrievalResult


class TestRRFFusion:
    """Test Reciprocal Rank Fusion logic."""

    def _make_retriever(self):
        """Create a retriever with no backing stores."""
        with patch("chromadb.PersistentClient"):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.dense_weight = 0.7
            retriever.sparse_weight = 0.3
            retriever.rrf_k = 60
            retriever._collection = None
            retriever._bm25 = None
            retriever._bm25_ids = []
            retriever._bm25_metadata = {}
            retriever.embedder = None
            retriever.collection_name = "test"
            retriever._client = MagicMock()
        return retriever

    def test_rrf_documents_in_both_rank_higher(self):
        """Documents appearing in both dense and sparse should rank highest."""
        retriever = self._make_retriever()

        dense = [("doc_A", 0.95), ("doc_B", 0.80), ("doc_C", 0.70)]
        sparse = [("doc_B", 5.0), ("doc_D", 3.0), ("doc_A", 2.0)]

        fused = retriever._reciprocal_rank_fusion(dense, sparse, k=60)

        # doc_A and doc_B appear in both lists, should have highest RRF scores
        fused_dict = dict(fused)
        assert fused_dict["doc_A"] > fused_dict.get("doc_C", 0)
        assert fused_dict["doc_B"] > fused_dict.get("doc_C", 0)
        assert fused_dict["doc_B"] > fused_dict.get("doc_D", 0)

    def test_rrf_scores_positive(self):
        """All RRF scores should be positive."""
        retriever = self._make_retriever()

        dense = [("doc_A", 0.9), ("doc_B", 0.5)]
        sparse = [("doc_C", 3.0)]

        fused = retriever._reciprocal_rank_fusion(dense, sparse, k=60)
        for _, score in fused:
            assert score > 0


class TestSparseSearch:
    def test_sparse_no_index(self):
        """Graceful empty result when no BM25 index loaded."""
        with patch("chromadb.PersistentClient"):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._bm25 = None
            retriever._bm25_ids = []

        results = retriever._sparse_search("brain metastasis", k=10)
        assert results == []


class TestDenseSearch:
    def test_dense_search_mock(self):
        """Mock ChromaDB query and verify result format."""
        with patch("chromadb.PersistentClient"):
            retriever = HybridRetriever.__new__(HybridRetriever)

            mock_collection = MagicMock()
            mock_collection.count.return_value = 100
            mock_collection.query.return_value = {
                "ids": [["chunk_1", "chunk_2"]],
                "distances": [[0.1, 0.3]],
                "documents": [["text 1", "text 2"]],
                "metadatas": [[{"section": "abstract"}, {"section": "results"}]],
            }
            retriever._collection = mock_collection

            query_emb = np.random.randn(512).astype(np.float32)
            results = retriever._dense_search(query_emb, k=5)

            assert len(results) == 2
            assert results[0][0] == "chunk_1"
            assert results[0][1] == pytest.approx(0.9, abs=0.01)  # 1 - 0.1

    def test_dense_search_no_collection(self):
        """Dense search with no collection returns empty."""
        with patch("chromadb.PersistentClient"):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._collection = None

        results = retriever._dense_search(np.zeros(512), k=5)
        assert results == []


class TestRetrievalResult:
    def test_result_fields(self):
        result = RetrievalResult(
            chunk_id="12345_abstract_0",
            text="Brain metastases are common.",
            score=0.85,
            section="abstract",
            title="Test Paper",
            authors="Smith J",
            journal="Neuro-oncology",
            year="2024",
            doi="10.1234/test",
            retrieval_method="hybrid",
        )
        assert result.chunk_id == "12345_abstract_0"
        assert result.score == 0.85
        assert result.retrieval_method == "hybrid"
