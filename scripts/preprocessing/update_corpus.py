#!/usr/bin/env python3
"""
Monthly incremental corpus update script.
Fetches only new papers since the last update and adds them to the existing index.

Usage:
    python scripts/update_corpus.py --config configs/rag.yaml --output_dir outputs/rag
    python scripts/update_corpus.py --since 2024/01/01 --until 2024/02/01
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.chunking import SemanticChunker
from src.rag.embeddings import BiomedCLIPEmbedder
from src.rag.ingestion import PubMedIngester
from src.rag.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Incremental update of the brain metastasis RAG corpus"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rag.yaml",
        help="Path to RAG configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/rag",
        help="Output directory (must contain existing corpus)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start date for update window (YYYY/MM/DD). Defaults to 30 days ago.",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="End date for update window (YYYY/MM/DD). Defaults to today.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Date range
    if args.until:
        until_date = args.until
    else:
        until_date = datetime.now().strftime("%Y/%m/%d")

    if args.since:
        since_date = args.since
    else:
        since_date = (datetime.now() - timedelta(days=30)).strftime("%Y/%m/%d")

    print(f"Incremental update: {since_date} to {until_date}")

    # Initialize components
    embedder = BiomedCLIPEmbedder(config_path=str(config_path))
    ingester = PubMedIngester(config_path=str(config_path))
    chunker = SemanticChunker(config_path=str(config_path))

    corpus_config = config.get("corpus", {})
    queries = corpus_config.get("pubmed_queries", [])
    fulltext_journals = corpus_config.get("fulltext_journals", [])
    ft_journals_lower = [j.lower() for j in fulltext_journals]

    # Add date filter to queries
    date_filter = f" AND ({since_date}[PDAT] : {until_date}[PDAT])"
    dated_queries = [q + date_filter for q in queries]

    # Search and fetch
    all_pmids = []
    for query in dated_queries:
        pmids = ingester.search_pubmed(query, max_results=500)
        all_pmids.extend(pmids)
        print(f"  Query: {len(pmids)} new PMIDs")

    if not all_pmids:
        print("No new papers found. Corpus is up to date.")
        return

    print(f"Total new PMIDs: {len(all_pmids)}")

    papers = ingester.fetch_abstracts(all_pmids)
    print(f"Papers with abstracts: {len(papers)}")

    # Fetch full text for high-impact journals
    for paper in papers:
        if paper.pmcid and paper.journal.lower() in ft_journals_lower:
            sections = ingester.fetch_pmc_fulltext(paper.pmcid)
            if sections:
                paper.sections = sections
                paper.has_full_text = True

    # Chunk
    new_chunks = chunker.chunk_corpus(papers)
    print(f"New chunks: {len(new_chunks)}")

    if not new_chunks:
        print("No new chunks generated.")
        return

    # Embed and upsert into ChromaDB
    db_config = config.get("chromadb", {})
    db_path = output_dir / db_config.get("path", "chromadb_v2").split("/")[-1]
    collection_name = db_config.get("literature_collection", "literature_chunks_v2")

    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    batch_size = embedder.batch_size
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embedder.embed_texts(texts)

        ids = [c.chunk_id for c in batch]
        metadatas = []
        for c in batch:
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
            if c.mesh_terms:
                meta["mesh_terms"] = "; ".join(c.mesh_terms)
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    print(f"Upserted {len(new_chunks)} chunks. Collection size: {collection.count()}")

    # Rebuild BM25 index (append new chunks)
    bm25_path = output_dir / "bm25_index.pkl"
    if bm25_path.exists():
        # Load existing and append
        with open(bm25_path, "rb") as f:
            existing = pickle.load(f)
        existing_ids = set(existing["ids"])

        # Only add chunks not already in the index
        added = 0
        for chunk in new_chunks:
            if chunk.chunk_id not in existing_ids:
                existing["ids"].append(chunk.chunk_id)
                existing["metadata"][chunk.chunk_id] = {
                    "text": chunk.text,
                    "section": chunk.section,
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "journal": chunk.journal,
                    "year": chunk.year,
                    "doi": chunk.doi,
                    "mesh_terms": chunk.mesh_terms,
                }
                added += 1

        # Rebuild BM25 from all texts
        from rank_bm25 import BM25Okapi

        all_texts = [existing["metadata"][cid]["text"] for cid in existing["ids"]]
        tokenized = [t.lower().split() for t in all_texts]
        existing["bm25"] = BM25Okapi(tokenized)

        with open(bm25_path, "wb") as f:
            pickle.dump(existing, f)
        print(f"Updated BM25 index: added {added} new chunks")
    else:
        # Build fresh
        retriever = HybridRetriever(
            db_path=str(db_path),
            embedder=embedder,
            collection_name=collection_name,
        )
        retriever.build_bm25_index(new_chunks, str(bm25_path))
        print(f"Created new BM25 index with {len(new_chunks)} chunks")

    print("\nIncremental update complete!")


if __name__ == "__main__":
    main()
