"""
PubMed/PMC ingestion pipeline for brain metastasis literature corpus.
Fetches abstracts and full-text articles via NCBI Entrez API.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


@dataclass
class Paper:
    """Standardized representation of a scientific paper."""

    pmid: str
    title: str = ""
    abstract: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    doi: str = ""
    pmcid: str = ""
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    citation_count: int = 0
    sections: Dict[str, str] = field(default_factory=dict)
    has_full_text: bool = False


# Standard section name mapping
SECTION_MAP = {
    "introduction": "introduction",
    "background": "introduction",
    "intro": "introduction",
    "methods": "methods",
    "materials and methods": "methods",
    "methods and materials": "methods",
    "methodology": "methods",
    "experimental": "methods",
    "experimental procedures": "methods",
    "patients and methods": "methods",
    "study design": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusions": "conclusions",
    "conclusion": "conclusions",
    "summary": "conclusions",
}


class PubMedIngester:
    """Fetches and processes papers from PubMed/PMC."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load .env
        try:
            from dotenv import load_dotenv

            env_path = Path(__file__).parent.parent.parent / ".env"
            load_dotenv(dotenv_path=env_path)
        except ImportError:
            pass

        # Set up Entrez
        from Bio import Entrez

        self.Entrez = Entrez
        self.Entrez.email = os.environ.get("NCBI_EMAIL", "")
        api_key = os.environ.get("NCBI_API_KEY", "")
        if api_key:
            self.Entrez.api_key = api_key
            self._rate_delay = 0.11  # 10 requests/sec with API key
        else:
            self._rate_delay = 0.34  # 3 requests/sec without

        self._seen_pmids: Set[str] = set()

    def search_pubmed(self, query: str, max_results: int = 1500) -> List[str]:
        """
        Search PubMed and return deduplicated PMIDs.

        Args:
            query: PubMed search query
            max_results: Maximum results per query

        Returns:
            List of unique PMID strings
        """
        time.sleep(self._rate_delay)
        handle = self.Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
        )
        record = self.Entrez.read(handle)
        handle.close()

        pmids = record.get("IdList", [])

        # Deduplicate across queries
        new_pmids = [p for p in pmids if p not in self._seen_pmids]
        self._seen_pmids.update(new_pmids)

        return new_pmids

    def fetch_abstracts(
        self, pmids: List[str], batch_size: int = 200
    ) -> List[Paper]:
        """
        Batch-fetch paper metadata and abstracts via Entrez efetch.

        Args:
            pmids: List of PubMed IDs
            batch_size: Number of records per API call

        Returns:
            List of Paper objects
        """
        from Bio import Medline

        papers = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            time.sleep(self._rate_delay)

            handle = self.Entrez.efetch(
                db="pubmed",
                id=batch,
                rettype="medline",
                retmode="text",
            )
            records = Medline.parse(handle)

            for record in records:
                # Extract DOI from AID list (entries look like "10.xxx [doi]" or "S1234 [pii]")
                doi = ""
                for aid in record.get("AID", []):
                    if "[doi]" in aid:
                        doi = aid.split(" [doi]")[0].strip()
                        break
                    elif aid.startswith("10."):
                        doi = aid.split(" ")[0].strip()
                        break

                paper = Paper(
                    pmid=record.get("PMID", ""),
                    title=record.get("TI", ""),
                    abstract=record.get("AB", ""),
                    authors=", ".join(record.get("AU", [])),
                    journal=record.get("JT", record.get("TA", "")),
                    year=record.get("DP", "")[:4],
                    doi=doi,
                    pmcid=record.get("PMC", ""),
                    mesh_terms=record.get("MH", []),
                    keywords=record.get("OT", []),
                )

                if paper.abstract:  # Only keep papers with abstracts
                    papers.append(paper)

            handle.close()

        return papers

    def fetch_pmc_fulltext(self, pmcid: str) -> Dict[str, str]:
        """
        Fetch full-text sections from PMC via JATS XML.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            Dict mapping section names to text content
        """
        time.sleep(self._rate_delay)

        try:
            handle = self.Entrez.efetch(
                db="pmc",
                id=pmcid,
                rettype="xml",
            )
            xml_bytes = handle.read()
            handle.close()
            return self._parse_pmc_xml(xml_bytes)
        except Exception:
            return {}

    def _parse_pmc_xml(self, xml_bytes: bytes) -> Dict[str, str]:
        """Parse PMC JATS XML into sections."""
        try:
            import xmltodict
        except ImportError:
            return {}

        try:
            doc = xmltodict.parse(xml_bytes)
        except Exception:
            return {}

        sections = {}
        try:
            article = doc.get("pmc-articleset", {}).get("article", {})
            if isinstance(article, list):
                article = article[0]
            body = article.get("body", {})
            if not body:
                return sections

            secs = body.get("sec", [])
            if isinstance(secs, dict):
                secs = [secs]

            for sec in secs:
                raw_title = sec.get("title", "")
                if isinstance(raw_title, dict):
                    raw_title = raw_title.get("#text", "")
                raw_title = str(raw_title).lower().strip()

                # Map to standard section name
                section_name = SECTION_MAP.get(raw_title, raw_title)

                # Extract text from paragraphs
                paras = sec.get("p", [])
                if isinstance(paras, str):
                    paras = [paras]
                elif isinstance(paras, dict):
                    paras = [paras.get("#text", "")]

                text_parts = []
                for p in paras:
                    if isinstance(p, dict):
                        p = p.get("#text", "")
                    if isinstance(p, str):
                        text_parts.append(p.strip())

                if text_parts:
                    sections[section_name] = " ".join(text_parts)

        except (KeyError, TypeError, AttributeError):
            pass

        return sections

    def ingest_corpus(
        self,
        queries: List[str] = None,
        max_per_query: int = None,
        total_target: int = None,
        fulltext_journals: List[str] = None,
        output_path: str = None,
    ) -> List[Paper]:
        """
        Full ingestion pipeline: search, fetch abstracts, optionally fetch full text.

        Args:
            queries: PubMed search queries (defaults to config)
            max_per_query: Max results per query (defaults to config)
            total_target: Total corpus size target (defaults to config)
            fulltext_journals: Journals for full-text fetching (defaults to config)
            output_path: Path to save raw papers JSON

        Returns:
            List of Paper objects
        """
        corpus_config = self.config.get("corpus", {})
        if queries is None:
            queries = corpus_config.get("pubmed_queries", [])
        if max_per_query is None:
            max_per_query = corpus_config.get("max_per_query", 1500)
        if total_target is None:
            total_target = corpus_config.get("total_target", 8000)
        if fulltext_journals is None:
            fulltext_journals = corpus_config.get("fulltext_journals", [])

        # Normalize journal names for matching
        ft_journals_lower = [j.lower() for j in fulltext_journals]

        # Step 1: Search all queries
        all_pmids = []
        for query in queries:
            pmids = self.search_pubmed(query, max_results=max_per_query)
            all_pmids.extend(pmids)
            print(f"  Query '{query}': {len(pmids)} new PMIDs")

        # Trim to total target
        if len(all_pmids) > total_target:
            all_pmids = all_pmids[:total_target]

        print(f"Total unique PMIDs: {len(all_pmids)}")

        # Step 2: Fetch abstracts
        print("Fetching abstracts...")
        papers = self.fetch_abstracts(all_pmids)
        print(f"Papers with abstracts: {len(papers)}")

        # Step 3: Fetch full text for high-impact journals
        ft_count = 0
        for paper in papers:
            if not paper.pmcid:
                continue
            if paper.journal.lower() in ft_journals_lower:
                sections = self.fetch_pmc_fulltext(paper.pmcid)
                if sections:
                    paper.sections = sections
                    paper.has_full_text = True
                    ft_count += 1

        print(f"Full-text papers fetched: {ft_count}")

        # Step 4: Save raw papers
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump([asdict(p) for p in papers], f, indent=2)
            print(f"Saved {len(papers)} papers to {output_path}")

        return papers
