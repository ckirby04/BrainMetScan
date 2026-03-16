"""
Corpus builder: orchestrates the full pipeline from PubMed ingestion
through embedding and indexing for the brain metastasis RAG system.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import chromadb
import yaml
from tqdm import tqdm

from ..chunking import SemanticChunker, DocumentChunk
from ..embeddings import BiomedCLIPEmbedder
from ..ingestion import PubMedIngester


class CorpusBuilder:
    """Builds the full RAG corpus from PubMed literature."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent.parent / "configs" / "rag.yaml"
            )
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.embedder = BiomedCLIPEmbedder(config_path=config_path)
        self.ingester = PubMedIngester(config_path=config_path)
        self.chunker = SemanticChunker(config_path=config_path)

    def build(self, output_dir: str) -> Dict:
        """
        Full corpus build pipeline.

        1. Ingest papers from PubMed
        2. Chunk papers into DocumentChunks
        3. Embed all chunks with BiomedCLIP, store in ChromaDB
        4. Build BM25 index
        5. Migrate existing curated knowledge into v2 collections
        6. Save build statistics

        Args:
            output_dir: Output directory for all artifacts

        Returns:
            Dict with build statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {}

        # Step 1: Ingest papers
        print("=" * 60)
        print("Step 1: Ingesting papers from PubMed...")
        print("=" * 60)
        raw_papers_path = output_dir / "raw_papers.json"
        papers = self.ingester.ingest_corpus(output_path=str(raw_papers_path))
        stats["total_papers"] = len(papers)
        stats["fulltext_papers"] = sum(1 for p in papers if p.has_full_text)

        # Step 2: Chunk papers
        print("\n" + "=" * 60)
        print("Step 2: Chunking papers...")
        print("=" * 60)
        chunks = self.chunker.chunk_corpus(papers)
        stats["total_chunks"] = len(chunks)
        print(f"Created {len(chunks)} chunks from {len(papers)} papers")

        # Step 3: Embed chunks and store in ChromaDB
        print("\n" + "=" * 60)
        print("Step 3: Embedding chunks and building ChromaDB...")
        print("=" * 60)
        db_config = self.config.get("chromadb", {})
        db_path = output_dir / db_config.get("path", "chromadb_v2").split("/")[-1]
        collection_name = db_config.get("literature_collection", "literature_chunks_v2")

        client = chromadb.PersistentClient(path=str(db_path))

        # Delete existing collection if present
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed and add in batches
        batch_size = self.embedder.batch_size
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch_chunks = chunks[i : i + batch_size]
            texts = [c.text for c in batch_chunks]
            embeddings = self.embedder.embed_texts(texts)

            ids = [c.chunk_id for c in batch_chunks]
            documents = texts
            metadatas = []
            for c in batch_chunks:
                meta = {
                    "section": c.section,
                    "title": c.title,
                    "authors": c.authors,
                    "journal": c.journal,
                    "year": c.year,
                    "doi": c.doi,
                    "paper_id": c.paper_id,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                }
                # ChromaDB metadata must be str/int/float/bool
                if c.mesh_terms:
                    meta["mesh_terms"] = "; ".join(c.mesh_terms)
                metadatas.append(meta)

            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

        print(f"Stored {collection.count()} chunks in ChromaDB")

        # Step 4: Build BM25 index
        print("\n" + "=" * 60)
        print("Step 4: Building BM25 index...")
        print("=" * 60)
        bm25_path = output_dir / "bm25_index.pkl"

        from ..retrieval import HybridRetriever

        retriever = HybridRetriever(
            db_path=str(db_path),
            embedder=self.embedder,
            collection_name=collection_name,
        )
        retriever.build_bm25_index(chunks, str(bm25_path))
        print(f"BM25 index saved to {bm25_path}")

        # Step 5: Migrate existing curated knowledge
        print("\n" + "=" * 60)
        print("Step 5: Migrating curated knowledge to v2...")
        print("=" * 60)
        knowledge_collection_name = db_config.get(
            "knowledge_collection", "medical_knowledge_v2"
        )
        self._migrate_curated_knowledge(client, knowledge_collection_name)

        # Step 6: Save stats
        stats_path = output_dir / "build_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nBuild stats saved to {stats_path}")

        print("\n" + "=" * 60)
        print("Corpus build complete!")
        print(f"  Papers: {stats['total_papers']}")
        print(f"  Full-text: {stats['fulltext_papers']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Database: {db_path}")
        print("=" * 60)

        return stats

    def _migrate_curated_knowledge(
        self, client: chromadb.ClientAPI, collection_name: str
    ):
        """Migrate existing curated KB + literature into v2 collection with BiomedCLIP embeddings."""
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        all_texts = []
        all_ids = []
        all_metadatas = []

        # Import existing knowledge base facts
        try:
            from ..build_database import build_knowledge_base

            kb_facts = build_knowledge_base()
            for i, fact in enumerate(kb_facts):
                all_ids.append(f"kb_fact_{i}")
                all_texts.append(fact)
                all_metadatas.append({"source": "curated_kb", "type": "fact"})
        except Exception as e:
            print(f"  Warning: Could not import curated KB: {e}")

        # Import existing literature key points
        try:
            from ..add_literature import PAPERS

            for paper in PAPERS:
                for j, point in enumerate(paper["key_points"]):
                    doc_text = (
                        f"{paper['title']} ({paper['authors']}, {paper['year']}): {point}"
                    )
                    all_ids.append(f"lit_{paper['id']}_point_{j}")
                    all_texts.append(doc_text)
                    all_metadatas.append({
                        "source": "curated_literature",
                        "paper_id": paper["id"],
                        "title": paper["title"],
                        "type": "key_point",
                    })
        except Exception as e:
            print(f"  Warning: Could not import curated literature: {e}")

        if all_texts:
            # Embed with BiomedCLIP
            embeddings = self.embedder.embed_texts(all_texts)
            collection.add(
                ids=all_ids,
                embeddings=embeddings.tolist(),
                documents=all_texts,
                metadatas=all_metadatas,
            )
            print(
                f"  Migrated {len(all_texts)} curated items to '{collection_name}'"
            )
        else:
            print("  No curated knowledge to migrate")
