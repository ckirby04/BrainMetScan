"""
Hybrid retrieval engine combining dense (BiomedCLIP) and sparse (BM25) search
with Reciprocal Rank Fusion for brain metastasis literature retrieval.
"""

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np

from .chunking import DocumentChunk


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    chunk_id: str
    text: str
    score: float
    section: str = ""
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    doi: str = ""
    mesh_terms: List[str] = field(default_factory=list)
    retrieval_method: str = "hybrid"  # "dense", "sparse", or "hybrid"


class HybridRetriever:
    """
    Hybrid retrieval combining dense vector search (ChromaDB) with
    sparse keyword search (BM25), fused via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        db_path: str,
        embedder=None,
        bm25_index_path: str = None,
        collection_name: str = "literature_chunks_v2",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.collection_name = collection_name

        # Connect to ChromaDB
        self._client = chromadb.PersistentClient(path=str(db_path))
        try:
            self._collection = self._client.get_collection(collection_name)
        except Exception:
            self._collection = None

        # Load BM25 index if available
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_metadata = {}
        if bm25_index_path and Path(bm25_index_path).exists():
            self._load_bm25_index(bm25_index_path)

    def _load_bm25_index(self, path: str):
        """Load pickled BM25 index and metadata."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_ids = data["ids"]
        self._bm25_metadata = data.get("metadata", {})

    def build_bm25_index(
        self, chunks: List[DocumentChunk], save_path: str
    ):
        """
        Build a BM25 index from document chunks and save to disk.

        Args:
            chunks: List of DocumentChunk objects
            save_path: Path to save the pickled index
        """
        from rank_bm25 import BM25Okapi

        # Tokenize
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]

        bm25 = BM25Okapi(tokenized_corpus)

        ids = [chunk.chunk_id for chunk in chunks]
        metadata = {}
        for chunk in chunks:
            metadata[chunk.chunk_id] = {
                "text": chunk.text,
                "section": chunk.section,
                "title": chunk.title,
                "authors": chunk.authors,
                "journal": chunk.journal,
                "year": chunk.year,
                "doi": chunk.doi,
                "mesh_terms": chunk.mesh_terms,
            }

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {"bm25": bm25, "ids": ids, "metadata": metadata}, f
            )

        self._bm25 = bm25
        self._bm25_ids = ids
        self._bm25_metadata = metadata

    def _dense_search(
        self, query_embedding: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        """Dense vector search via ChromaDB."""
        if self._collection is None:
            return []

        collection_size = self._collection.count()
        if collection_size == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, collection_size),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            # ChromaDB returns distances; convert to similarity for cosine
            distance = results["distances"][0][i] if results.get("distances") else 0
            similarity = 1.0 - distance  # cosine distance -> similarity
            hits.append((chunk_id, similarity))

        return hits

    def _sparse_search(
        self, query_text: str, k: int
    ) -> List[Tuple[str, float]]:
        """Sparse keyword search via BM25."""
        if self._bm25 is None:
            return []

        tokenized_query = query_text.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append((self._bm25_ids[idx], float(scores[idx])))

        return hits

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        k: int,
    ) -> List[Tuple[str, float]]:
        """
        Combine dense and sparse results via weighted RRF.

        score = w_dense * 1/(rank_dense + 1 + k) + w_sparse * 1/(rank_sparse + 1 + k)
        """
        rrf_scores = defaultdict(float)

        for rank, (chunk_id, _score) in enumerate(dense_results):
            rrf_scores[chunk_id] += self.dense_weight * (
                1.0 / (rank + 1 + self.rrf_k)
            )

        for rank, (chunk_id, _score) in enumerate(sparse_results):
            rrf_scores[chunk_id] += self.sparse_weight * (
                1.0 / (rank + 1 + self.rrf_k)
            )

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def retrieve(
        self,
        query_text: str,
        k: int = 10,
        dense_k: int = 30,
        sparse_k: int = 30,
        section_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Main hybrid retrieval: dense + sparse search with RRF fusion.

        Args:
            query_text: Query string
            k: Number of final results
            dense_k: Number of dense candidates
            sparse_k: Number of sparse candidates
            section_filter: Optional section name to filter results

        Returns:
            List of RetrievalResult objects
        """
        # Dense search
        dense_results = []
        if self.embedder and self._collection:
            query_embedding = self.embedder.embed_text(query_text)
            dense_results = self._dense_search(query_embedding, dense_k)

        # Sparse search
        sparse_results = self._sparse_search(query_text, sparse_k)

        # Fusion
        if dense_results and sparse_results:
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k)
            method = "hybrid"
        elif dense_results:
            fused = [(cid, score) for cid, score in dense_results]
            method = "dense"
        elif sparse_results:
            fused = [(cid, score) for cid, score in sparse_results]
            method = "sparse"
        else:
            return []

        # Build results with metadata
        results = []
        for chunk_id, score in fused:
            if len(results) >= k:
                break

            meta = self._get_chunk_metadata(chunk_id)
            if meta is None:
                continue

            # Apply section filter
            if section_filter and meta.get("section", "") != section_filter:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=meta.get("text", ""),
                    score=score,
                    section=meta.get("section", ""),
                    title=meta.get("title", ""),
                    authors=meta.get("authors", ""),
                    journal=meta.get("journal", ""),
                    year=meta.get("year", ""),
                    doi=meta.get("doi", ""),
                    mesh_terms=meta.get("mesh_terms", []),
                    retrieval_method=method,
                )
            )

        return results

    def retrieve_by_image(
        self, image_embedding: np.ndarray, k: int = 10
    ) -> List[RetrievalResult]:
        """
        Cross-modal retrieval: find literature chunks similar to an MRI image.
        Uses BiomedCLIP's shared embedding space.

        Args:
            image_embedding: (512,) image embedding from BiomedCLIPEmbedder
            k: Number of results

        Returns:
            List of RetrievalResult objects
        """
        dense_results = self._dense_search(image_embedding, k)

        results = []
        for chunk_id, score in dense_results:
            meta = self._get_chunk_metadata(chunk_id)
            if meta is None:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=meta.get("text", ""),
                    score=score,
                    section=meta.get("section", ""),
                    title=meta.get("title", ""),
                    authors=meta.get("authors", ""),
                    journal=meta.get("journal", ""),
                    year=meta.get("year", ""),
                    doi=meta.get("doi", ""),
                    mesh_terms=meta.get("mesh_terms", []),
                    retrieval_method="cross_modal",
                )
            )

        return results

    def _get_chunk_metadata(self, chunk_id: str) -> Optional[Dict]:
        """Get metadata for a chunk from BM25 index or ChromaDB."""
        # Try BM25 metadata first (faster)
        if chunk_id in self._bm25_metadata:
            return self._bm25_metadata[chunk_id]

        # Fall back to ChromaDB
        if self._collection is None:
            return None

        try:
            result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if result["ids"]:
                meta = result["metadatas"][0] if result["metadatas"] else {}
                meta["text"] = result["documents"][0] if result["documents"] else ""
                return meta
        except Exception:
            pass

        return None
