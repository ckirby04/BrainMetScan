"""Tests for semantic chunking module."""

import pytest
from src.rag.ingestion import Paper
from src.rag.chunking import SemanticChunker, DocumentChunk


@pytest.fixture
def chunker():
    return SemanticChunker()


@pytest.fixture
def abstract_only_paper():
    return Paper(
        pmid="11111111",
        title="Test Paper",
        abstract="Brain metastases occur in 20-40% of cancer patients. "
        "MRI is the gold standard for detection.",
        authors="Smith J, Doe A",
        journal="Neuro-oncology",
        year="2024",
        doi="10.1234/test",
        mesh_terms=["Brain Neoplasms"],
    )


@pytest.fixture
def fulltext_paper():
    return Paper(
        pmid="22222222",
        title="Full Text Paper",
        abstract="This study investigates brain metastasis segmentation.",
        authors="Jones B",
        journal="Radiology",
        year="2024",
        doi="10.5678/full",
        sections={
            "introduction": "Brain metastases are common intracranial tumors.",
            "methods": "We used a U-Net architecture for segmentation.",
            "results": "Our model achieved a Dice score of 0.85.",
            "discussion": "The results demonstrate improved performance.",
            "conclusions": "Deep learning enables accurate segmentation.",
        },
        has_full_text=True,
    )


def test_abstract_only_paper(chunker, abstract_only_paper):
    """Abstract-only papers produce a single chunk."""
    chunks = chunker.chunk_paper(abstract_only_paper)
    assert len(chunks) >= 1
    assert chunks[0].section == "abstract"
    assert chunks[0].paper_id == "11111111"


def test_fulltext_sections(chunker, fulltext_paper):
    """Full-text papers produce chunks for each section."""
    chunks = chunker.chunk_paper(fulltext_paper)
    sections = {c.section for c in chunks}
    assert "abstract" in sections
    assert "introduction" in sections
    assert "methods" in sections
    assert "results" in sections


def test_long_section_splitting(chunker):
    """Sections exceeding max tokens get split into multiple chunks."""
    long_text = "Brain metastasis segmentation is important. " * 500  # ~6000 chars
    paper = Paper(
        pmid="33333333",
        title="Long Paper",
        abstract=long_text,
        authors="Long A",
        journal="Test Journal",
        year="2024",
    )
    chunks = chunker.chunk_paper(paper)
    assert len(chunks) > 1
    # All chunks should belong to abstract section
    assert all(c.section == "abstract" for c in chunks)


def test_chunk_id_format(chunker, abstract_only_paper):
    """Chunk IDs follow {pmid}_{section}_{index} format."""
    chunks = chunker.chunk_paper(abstract_only_paper)
    for chunk in chunks:
        parts = chunk.chunk_id.split("_")
        assert len(parts) >= 3
        assert parts[0] == "11111111"
        assert parts[-1].isdigit()


def test_chunk_metadata(chunker, abstract_only_paper):
    """Chunks preserve paper metadata."""
    chunks = chunker.chunk_paper(abstract_only_paper)
    chunk = chunks[0]
    assert chunk.title == "Test Paper"
    assert chunk.authors == "Smith J, Doe A"
    assert chunk.journal == "Neuro-oncology"
    assert chunk.year == "2024"
    assert chunk.doi == "10.1234/test"


def test_total_chunks_set(chunker, fulltext_paper):
    """Each chunk has total_chunks set to the total number of chunks."""
    chunks = chunker.chunk_paper(fulltext_paper)
    for chunk in chunks:
        assert chunk.total_chunks == len(chunks)


def test_chunk_corpus(chunker, abstract_only_paper, fulltext_paper):
    """chunk_corpus processes multiple papers."""
    chunks = chunker.chunk_corpus([abstract_only_paper, fulltext_paper])
    pmids = {c.paper_id for c in chunks}
    assert "11111111" in pmids
    assert "22222222" in pmids


def test_empty_abstract_paper(chunker):
    """Paper with no abstract and no full text produces no chunks."""
    paper = Paper(pmid="44444444", title="Empty", abstract="")
    chunks = chunker.chunk_paper(paper)
    assert len(chunks) == 0
