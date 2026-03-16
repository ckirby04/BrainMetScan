"""Tests for PubMed ingestion pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from src.rag.ingestion import Paper, PubMedIngester, SECTION_MAP


def _make_medline_record(**kwargs):
    """Create a mock Medline record dict."""
    defaults = {
        "PMID": "12345678",
        "TI": "Brain metastasis segmentation with deep learning",
        "AB": "We present a novel approach to brain met segmentation.",
        "AU": ["Smith J", "Doe A"],
        "JT": "Neuro-oncology",
        "DP": "2023 Jan",
        "AID": ["10.1234/test.2023 [doi]"],
        "PMC": "",
        "MH": ["Brain Neoplasms", "Deep Learning"],
        "OT": ["segmentation", "MRI"],
    }
    defaults.update(kwargs)
    return defaults


class TestPaper:
    def test_paper_creation(self):
        p = Paper(pmid="123", title="Test", abstract="Abstract text")
        assert p.pmid == "123"
        assert p.has_full_text is False
        assert p.sections == {}

    def test_paper_with_sections(self):
        p = Paper(
            pmid="456",
            title="Full paper",
            abstract="Abstract",
            sections={"introduction": "Intro text", "methods": "Methods text"},
            has_full_text=True,
        )
        assert p.has_full_text is True
        assert "introduction" in p.sections


class TestSectionMapping:
    def test_methods_variants(self):
        assert SECTION_MAP["materials and methods"] == "methods"
        assert SECTION_MAP["methodology"] == "methods"
        assert SECTION_MAP["patients and methods"] == "methods"

    def test_intro_variants(self):
        assert SECTION_MAP["background"] == "introduction"
        assert SECTION_MAP["intro"] == "introduction"

    def test_conclusion_variants(self):
        assert SECTION_MAP["conclusion"] == "conclusions"
        assert SECTION_MAP["summary"] == "conclusions"


class TestDeduplication:
    @patch("src.rag.ingestion.PubMedIngester.__init__", return_value=None)
    def test_same_pmid_not_returned_twice(self, mock_init):
        """Verify deduplication across queries."""
        ingester = PubMedIngester.__new__(PubMedIngester)
        ingester._seen_pmids = set()
        ingester._rate_delay = 0
        ingester.Entrez = MagicMock()

        # First search returns PMIDs 1, 2, 3
        ingester.Entrez.esearch.return_value.__enter__ = MagicMock()
        mock_handle = MagicMock()
        ingester.Entrez.esearch.return_value = mock_handle

        ingester.Entrez.read.return_value = {"IdList": ["1", "2", "3"]}
        result1 = ingester.search_pubmed("query1", max_results=10)
        assert result1 == ["1", "2", "3"]

        # Second search returns PMIDs 2, 3, 4 - only 4 should be new
        ingester.Entrez.read.return_value = {"IdList": ["2", "3", "4"]}
        result2 = ingester.search_pubmed("query2", max_results=10)
        assert result2 == ["4"]
        assert "2" not in result2
        assert "3" not in result2


class TestMedlineParsing:
    @patch("src.rag.ingestion.PubMedIngester.__init__", return_value=None)
    def test_parse_medline_record(self, mock_init):
        """Verify Medline record parsing into Paper object."""
        ingester = PubMedIngester.__new__(PubMedIngester)
        ingester._rate_delay = 0
        ingester.Entrez = MagicMock()

        record = _make_medline_record()

        # Mock Entrez.efetch to return a handle that Medline.parse can consume
        from unittest.mock import PropertyMock

        mock_handle = MagicMock()
        ingester.Entrez.efetch.return_value = mock_handle

        with patch("Bio.Medline.parse") as mock_parse:
            mock_parse.return_value = iter([record])
            papers = ingester.fetch_abstracts(["12345678"])

        assert len(papers) == 1
        assert papers[0].pmid == "12345678"
        assert papers[0].journal == "Neuro-oncology"
        assert "Smith J" in papers[0].authors
        assert papers[0].year == "2023"

    @patch("src.rag.ingestion.PubMedIngester.__init__", return_value=None)
    def test_skip_papers_without_abstract(self, mock_init):
        """Papers without abstracts should be excluded."""
        ingester = PubMedIngester.__new__(PubMedIngester)
        ingester._rate_delay = 0
        ingester.Entrez = MagicMock()

        record = _make_medline_record(AB="")
        mock_handle = MagicMock()
        ingester.Entrez.efetch.return_value = mock_handle

        with patch("Bio.Medline.parse") as mock_parse:
            mock_parse.return_value = iter([record])
            papers = ingester.fetch_abstracts(["12345678"])

        assert len(papers) == 0
