"""
Integration tests for the RAG pipeline.
End-to-end: embed -> ingest mock paper -> chunk -> store -> retrieve -> verify.
"""

import os
import tempfile

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_open_clip():
    """Mock open_clip to avoid downloading model in tests."""
    import torch

    mock_model = MagicMock()

    def _fake_encode(x):
        batch_size = x.shape[0]
        emb = torch.randn(batch_size, 512)
        return emb

    mock_model.encode_text.side_effect = _fake_encode
    mock_model.encode_image.side_effect = _fake_encode
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None

    mock_preprocess = lambda img: torch.randn(3, 224, 224)
    mock_tokenizer = MagicMock(
        side_effect=lambda texts, context_length=256: torch.randint(
            0, 1000, (len(texts), context_length)
        )
    )

    with patch("open_clip.create_model_and_transforms") as mock_create, \
         patch("open_clip.get_tokenizer") as mock_get_tok:
        mock_create.return_value = (mock_model, mock_preprocess, mock_preprocess)
        mock_get_tok.return_value = mock_tokenizer
        yield


@pytest.fixture
def mock_papers():
    """Create mock papers for testing."""
    from src.rag.ingestion import Paper

    return [
        Paper(
            pmid="10000001",
            title="Deep Learning for Brain Metastasis Segmentation",
            abstract="We developed a U-Net model for automated brain metastasis "
            "segmentation achieving 0.85 Dice score on multi-sequence MRI.",
            authors="Smith J, Doe A",
            journal="Neuro-oncology",
            year="2024",
            doi="10.1234/neuro.2024.001",
            mesh_terms=["Brain Neoplasms", "Deep Learning"],
        ),
        Paper(
            pmid="10000002",
            title="Stereotactic Radiosurgery Outcomes in Brain Metastases",
            abstract="This study evaluates stereotactic radiosurgery outcomes "
            "in 200 patients with brain metastases from non-small cell lung cancer.",
            authors="Jones B, Kim C",
            journal="Radiology",
            year="2023",
            doi="10.5678/rad.2023.002",
            mesh_terms=["Radiosurgery", "Brain Neoplasms"],
            sections={
                "introduction": "Brain metastases are the most common intracranial tumors.",
                "methods": "Retrospective analysis of 200 patients treated with SRS.",
                "results": "Local control rate was 89% at 12 months.",
                "discussion": "SRS provides excellent local control for brain metastases.",
            },
            has_full_text=True,
        ),
    ]


class TestEndToEndPipeline:
    """Integration tests covering the full RAG pipeline."""

    def test_chunk_embed_store_retrieve(self, mock_open_clip, mock_papers):
        """Full pipeline: chunk papers -> embed -> store in ChromaDB -> retrieve."""
        from src.rag.chunking import SemanticChunker
        from src.rag.embeddings import BiomedCLIPEmbedder
        from src.rag.retrieval import HybridRetriever

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Chunk papers
            chunker = SemanticChunker()
            chunks = chunker.chunk_corpus(mock_papers)
            assert len(chunks) > 0

            # Embed and store
            embedder = BiomedCLIPEmbedder(device="cpu")
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)

            import chromadb

            db_path = os.path.join(tmpdir, "chromadb_v2")
            client = chromadb.PersistentClient(path=db_path)
            collection = client.create_collection(
                name="literature_chunks_v2",
                metadata={"hnsw:space": "cosine"},
            )

            ids = [c.chunk_id for c in chunks]
            metadatas = [
                {
                    "section": c.section,
                    "title": c.title,
                    "authors": c.authors,
                    "journal": c.journal,
                    "year": c.year,
                    "doi": c.doi,
                    "paper_id": c.paper_id,
                }
                for c in chunks
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

            assert collection.count() == len(chunks)

            # Build BM25 index
            bm25_path = os.path.join(tmpdir, "bm25_index.pkl")
            retriever = HybridRetriever(
                db_path=db_path,
                embedder=embedder,
                collection_name="literature_chunks_v2",
            )
            retriever.build_bm25_index(chunks, bm25_path)

            # Reload retriever with BM25
            retriever2 = HybridRetriever(
                db_path=db_path,
                embedder=embedder,
                bm25_index_path=bm25_path,
                collection_name="literature_chunks_v2",
            )

            # Retrieve
            results = retriever2.retrieve("brain metastasis segmentation", k=5)
            assert len(results) > 0
            assert results[0].text != ""
            assert results[0].retrieval_method in ("dense", "sparse", "hybrid")

    def test_cross_modal_retrieval(self, mock_open_clip, mock_papers):
        """Test image-to-text cross-modal retrieval."""
        from src.rag.chunking import SemanticChunker
        from src.rag.embeddings import BiomedCLIPEmbedder
        from src.rag.retrieval import HybridRetriever

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            chunker = SemanticChunker()
            chunks = chunker.chunk_corpus(mock_papers)

            embedder = BiomedCLIPEmbedder(device="cpu")
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)

            import chromadb

            db_path = os.path.join(tmpdir, "chromadb_v2")
            client = chromadb.PersistentClient(path=db_path)
            collection = client.create_collection(
                name="literature_chunks_v2",
                metadata={"hnsw:space": "cosine"},
            )

            collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[{"section": c.section, "title": c.title,
                            "authors": c.authors, "journal": c.journal,
                            "year": c.year, "doi": c.doi,
                            "paper_id": c.paper_id} for c in chunks],
            )

            retriever = HybridRetriever(
                db_path=db_path,
                embedder=embedder,
                collection_name="literature_chunks_v2",
            )

            # Simulate MRI volume embedding
            volume = np.random.randn(64, 64, 32).astype(np.float32)
            img_emb = embedder.embed_mri_volume(volume)

            results = retriever.retrieve_by_image(img_emb, k=3)
            assert len(results) > 0
            assert results[0].retrieval_method == "cross_modal"

    def test_paper_chunking_preserves_metadata(self, mock_papers):
        """Verify metadata is preserved through chunking."""
        from src.rag.chunking import SemanticChunker

        chunker = SemanticChunker()

        # Abstract-only paper
        chunks1 = chunker.chunk_paper(mock_papers[0])
        assert all(c.journal == "Neuro-oncology" for c in chunks1)
        assert all(c.year == "2024" for c in chunks1)

        # Full-text paper
        chunks2 = chunker.chunk_paper(mock_papers[1])
        sections = {c.section for c in chunks2}
        assert "introduction" in sections or "abstract" in sections

    def test_retrieval_result_fields(self, mock_open_clip, mock_papers):
        """Verify RetrievalResult contains all expected fields."""
        from src.rag.chunking import SemanticChunker
        from src.rag.embeddings import BiomedCLIPEmbedder
        from src.rag.retrieval import HybridRetriever

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            chunker = SemanticChunker()
            chunks = chunker.chunk_corpus(mock_papers)
            embedder = BiomedCLIPEmbedder(device="cpu")

            import chromadb

            db_path = os.path.join(tmpdir, "chromadb_v2")
            client = chromadb.PersistentClient(path=db_path)
            collection = client.create_collection(
                name="literature_chunks_v2",
                metadata={"hnsw:space": "cosine"},
            )

            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)

            collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[{"section": c.section, "title": c.title,
                            "authors": c.authors, "journal": c.journal,
                            "year": c.year, "doi": c.doi,
                            "paper_id": c.paper_id} for c in chunks],
            )

            bm25_path = os.path.join(tmpdir, "bm25_index.pkl")
            retriever = HybridRetriever(
                db_path=db_path,
                embedder=embedder,
                collection_name="literature_chunks_v2",
            )
            retriever.build_bm25_index(chunks, bm25_path)

            # Reload with BM25
            retriever2 = HybridRetriever(
                db_path=db_path,
                embedder=embedder,
                bm25_index_path=bm25_path,
                collection_name="literature_chunks_v2",
            )

            results = retriever2.retrieve("radiosurgery outcomes", k=3)
            if results:
                r = results[0]
                assert hasattr(r, "chunk_id")
                assert hasattr(r, "text")
                assert hasattr(r, "score")
                assert hasattr(r, "section")
                assert hasattr(r, "title")
                assert hasattr(r, "authors")
                assert hasattr(r, "journal")
                assert hasattr(r, "year")
                assert hasattr(r, "retrieval_method")
