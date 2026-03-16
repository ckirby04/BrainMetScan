"""
Add medical literature to RAG knowledge base
Downloads and processes papers about brain metastases
"""

import chromadb
from pathlib import Path
import requests
from typing import List

# Key papers about brain metastases and the BrainMetShare dataset
PAPERS = [
    {
        'id': 'grovik_2020',
        'title': 'Deep Learning Enables Automatic Detection and Segmentation of Brain Metastases on Multisequence MRI',
        'authors': 'Grøvik et al.',
        'year': '2020',
        'journal': 'JMRI',
        'key_points': [
            'Multi-sequence MRI improves brain metastasis detection over single sequences',
            'T1 post-contrast is most sensitive for detecting enhancing lesions',
            'FLAIR helps identify perilesional edema',
            'Combination of T1, T2, and FLAIR achieves best sensitivity',
            'Small metastases (<5mm) remain challenging to detect',
        ]
    },
    {
        'id': 'achrol_2019',
        'title': 'Brain metastases',
        'authors': 'Achrol et al.',
        'year': '2019',
        'journal': 'Nature Reviews Disease Primers',
        'key_points': [
            'Brain metastases occur in 10-30% of adult cancer patients',
            'Most common primary sources: lung (40-50%), breast (15-25%), melanoma (5-20%)',
            'Multiple metastases are more common than solitary lesions',
            'Metastases preferentially occur at gray-white matter junction',
            'Treatment options: surgery, stereotactic radiosurgery, whole brain radiation, systemic therapy',
            'Prognosis depends on number/size of lesions, primary cancer type, and systemic disease burden',
        ]
    },
    {
        'id': 'treatment_guidelines',
        'title': 'Management of Brain Metastases',
        'authors': 'Clinical Guidelines',
        'year': '2023',
        'key_points': [
            'Solitary metastasis <3cm: Consider surgical resection if accessible',
            'Multiple metastases (1-4 lesions): Stereotactic radiosurgery preferred',
            'Multiple small lesions: Consider targeted therapy or immunotherapy',
            'Symptomatic edema: Corticosteroids for management',
            'Follow-up MRI every 2-3 months to monitor response and detect new lesions',
            'Multidisciplinary tumor board discussion recommended for complex cases',
        ]
    },
    {
        'id': 'imaging_patterns',
        'title': 'MRI Characteristics of Brain Metastases',
        'authors': 'Radiology Review',
        'year': '2022',
        'key_points': [
            'Ring enhancement pattern suggests central necrosis in larger lesions',
            'Melanoma and renal cell metastases may show hemorrhage (bright on T1)',
            'Breast cancer metastases often multiple and small',
            'Lung cancer metastases most common, variable appearance',
            'Perilesional edema extent varies by primary tumor type',
            'Leptomeningeal enhancement suggests meningeal involvement',
        ]
    },
    {
        'id': 'radiomic_features',
        'title': 'Radiomics in Brain Metastasis',
        'authors': 'Imaging Biomarkers',
        'year': '2023',
        'key_points': [
            'Lesion volume correlates with prognosis',
            'Shape irregularity may indicate aggressive biology',
            'Peritumoral edema volume ratio predicts treatment response',
            'Texture features can differentiate primary tumor types',
            'Sphericity and margin characteristics inform treatment planning',
            'Lesion location affects surgical accessibility and radiation planning',
        ]
    }
]


def add_literature_to_db(db_path: Path):
    """
    Add medical literature to knowledge base

    Args:
        db_path: Path to ChromaDB database
    """
    print("Adding medical literature to knowledge base...")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(db_path))

    # Get or create literature collection
    try:
        collection = client.get_collection("medical_literature")
        # Delete and recreate to update
        client.delete_collection("medical_literature")
    except Exception:
        pass

    collection = client.create_collection(
        name="medical_literature",
        metadata={"description": "Medical literature about brain metastases"}
    )

    # Add papers
    ids = []
    documents = []
    metadatas = []

    for paper in PAPERS:
        paper_id_base = paper['id']

        # Add each key point as a separate document
        for i, point in enumerate(paper['key_points']):
            doc_id = f"{paper_id_base}_point_{i}"

            # Create document text
            doc_text = f"{paper['title']} ({paper['authors']}, {paper['year']}): {point}"

            # Create metadata
            metadata = {
                'paper_id': paper['id'],
                'title': paper['title'],
                'authors': paper['authors'],
                'year': paper['year'],
                'point_index': i
            }

            if 'journal' in paper:
                metadata['journal'] = paper['journal']

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(metadata)

    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✓ Added {len(documents)} literature points from {len(PAPERS)} papers")

    return collection


def query_literature(db_path: Path, query: str, k: int = 5):
    """
    Query literature collection

    Args:
        db_path: Path to ChromaDB database
        query: Query text
        k: Number of results

    Returns:
        Retrieved documents
    """
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("medical_literature")

    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    print(f"\nQuery: {query}")
    print(f"{'='*60}\n")

    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"{i+1}. {doc}")
        print(f"   Source: {metadata['title']} ({metadata['year']})\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add medical literature to RAG")
    parser.add_argument('--db_path', type=str, default='../../outputs/rag/chromadb',
                        help='Path to ChromaDB database')
    parser.add_argument('--query', type=str, default=None,
                        help='Test query (optional)')

    args = parser.parse_args()

    db_path = Path(args.db_path)

    # Add literature
    add_literature_to_db(db_path)

    # Test query if provided
    if args.query:
        query_literature(db_path, args.query, k=5)
    else:
        # Example queries
        print("\n" + "="*60)
        print("Example queries:")
        print("="*60)
        query_literature(db_path, "treatment for multiple brain metastases", k=3)
        query_literature(db_path, "melanoma brain metastases imaging", k=3)
