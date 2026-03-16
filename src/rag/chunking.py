"""
Semantic chunking for medical literature corpus.
Splits papers into section-aware chunks with rich metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from .ingestion import Paper


@dataclass
class DocumentChunk:
    """A single chunk of a document with metadata for retrieval."""

    chunk_id: str  # Format: {pmid}_{section}_{index}
    text: str
    paper_id: str
    section: str
    title: str
    authors: str
    journal: str
    year: str
    doi: str
    mesh_terms: List[str] = field(default_factory=list)
    citation_count: int = 0
    chunk_index: int = 0
    total_chunks: int = 1


# Section ordering priority
SECTION_ORDER = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusions",
]


class SemanticChunker:
    """Chunks papers into section-aware pieces for RAG retrieval."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        chunking_config = config.get("chunking", {})
        self.max_tokens = chunking_config.get("max_tokens", 512)
        self.overlap_tokens = chunking_config.get("overlap_tokens", 50)

        # Approximate 1 token ~ 4 chars for splitting
        self._max_chars = self.max_tokens * 4
        self._overlap_chars = self.overlap_tokens * 4

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._max_chars,
            chunk_overlap=self._overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_paper(self, paper: Paper) -> List[DocumentChunk]:
        """
        Chunk a single paper into DocumentChunks.

        Full-text papers: one or more chunks per section.
        Abstract-only papers: abstract as a single chunk (split if too long).
        """
        chunks = []
        base_meta = {
            "paper_id": paper.pmid,
            "title": paper.title,
            "authors": paper.authors,
            "journal": paper.journal,
            "year": paper.year,
            "doi": paper.doi,
            "mesh_terms": paper.mesh_terms,
            "citation_count": paper.citation_count,
        }

        if paper.has_full_text and paper.sections:
            # Process sections in standard order, then any remaining
            processed = set()
            ordered_sections = []

            for section_name in SECTION_ORDER:
                if section_name in paper.sections:
                    ordered_sections.append((section_name, paper.sections[section_name]))
                    processed.add(section_name)

            # Add non-standard sections at the end
            for section_name, text in paper.sections.items():
                if section_name not in processed:
                    ordered_sections.append((section_name, text))

            # Always include abstract as first section
            if paper.abstract and "abstract" not in processed:
                ordered_sections.insert(0, ("abstract", paper.abstract))

            for section_name, text in ordered_sections:
                section_chunks = self._split_section(text, paper.pmid, section_name, base_meta)
                chunks.extend(section_chunks)
        else:
            # Abstract-only paper
            if paper.abstract:
                chunks = self._split_section(
                    paper.abstract, paper.pmid, "abstract", base_meta
                )

        # Set total_chunks for each chunk
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_section(
        self,
        text: str,
        pmid: str,
        section: str,
        base_meta: dict,
    ) -> List[DocumentChunk]:
        """Split a section into chunks, applying overflow splitting if needed."""
        if len(text) <= self._max_chars:
            chunk = DocumentChunk(
                chunk_id=f"{pmid}_{section}_0",
                text=text,
                section=section,
                chunk_index=0,
                **base_meta,
            )
            return [chunk]

        # Text exceeds max size, use recursive splitter
        split_texts = self._splitter.split_text(text)
        chunks = []
        for idx, split_text in enumerate(split_texts):
            chunk = DocumentChunk(
                chunk_id=f"{pmid}_{section}_{idx}",
                text=split_text,
                section=section,
                chunk_index=idx,
                **base_meta,
            )
            chunks.append(chunk)

        return chunks

    def chunk_corpus(self, papers: List[Paper]) -> List[DocumentChunk]:
        """Chunk an entire corpus of papers."""
        all_chunks = []
        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)
        return all_chunks
