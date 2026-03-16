"""
Build vector database for RAG retrieval
Processes all cases and stores features + embeddings in ChromaDB
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import pandas as pd

from .feature_extractor import extract_case_features


def build_knowledge_base():
    """Build medical knowledge base about brain metastases"""
    kb = [
        "Brain metastases are the most common intracranial tumors in adults, occurring in 20-40% of cancer patients.",
        "MRI with gadolinium contrast (T1_gd) is the gold standard for detecting brain metastases, appearing as enhancing lesions.",
        "The most common primary cancers that metastasize to the brain are lung cancer (50%), breast cancer (15-20%), and melanoma (10%).",
        "Multiple brain metastases are more common than solitary lesions, with 60-70% of patients having multiple lesions at diagnosis.",
        "Brain metastases typically occur at the gray-white matter junction due to vascular factors and caliber changes.",
        "FLAIR sequences help identify perilesional edema surrounding metastatic lesions, important for treatment planning.",
        "Small cerebellar metastases may indicate hematogenous spread and often require stereotactic radiosurgery.",
        "Lesion size, number, and location are critical factors in determining treatment approach (surgery, radiation, systemic therapy).",
        "Ring enhancement pattern on T1_gd suggests viable tumor periphery with necrotic center, common in larger metastases.",
        "Differential diagnosis includes primary brain tumors (glioblastoma), abscesses, and demyelinating lesions.",
        "Metastases from melanoma and renal cell carcinoma are often hemorrhagic, appearing hyperintense on T1_pre sequences.",
        "Multi-modal MRI analysis combining T1_pre, T1_gd, T2, and FLAIR improves detection sensitivity and characterization."
    ]
    return kb


def process_dataset(
    data_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    device='cuda'
):
    """
    Process all cases and extract features

    Args:
        data_dir: Path to data directory (train or test)
        metadata_path: Path to metadata CSV
        output_dir: Output directory for database
        device: Device for feature extraction
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata['Patient ID'] = metadata['Patient ID'].astype(str)

    # Get all cases
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])
    print(f"Found {len(cases)} cases")

    # Check if masks exist (training data)
    has_masks = (data_dir.name == 'train')

    # Extract features for all cases
    all_features = []

    for case_dir in tqdm(cases, desc="Extracting features"):
        case_id = case_dir.name

        # Determine mask path
        mask_path = None
        if has_masks:
            mask_path = case_dir / "seg.nii.gz"

        try:
            # Extract features
            features = extract_case_features(
                case_dir,
                mask_path=mask_path,
                sequences=['t1_gd'],
                device=device
            )

            # Add metadata
            patient_id = case_id.split('_')[1].lstrip('0') or '0'
            meta_row = metadata[metadata['Patient ID'] == patient_id]

            if len(meta_row) > 0:
                features['primary_cancer'] = meta_row.iloc[0]['Primary cancer type']
            else:
                features['primary_cancer'] = 'unknown'

            all_features.append(features)

        except Exception as e:
            print(f"\nError processing {case_id}: {e}")
            continue

    # Save features to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / 'case_features.json'

    # Convert numpy arrays to lists for JSON serialization
    for features in all_features:
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                features[k] = v.tolist()

    with open(features_path, 'w') as f:
        json.dump(all_features, f, indent=2)

    print(f"\nSaved features to {features_path}")

    return all_features


def build_vector_database(
    features_list,
    kb_facts,
    db_path: Path
):
    """
    Build ChromaDB vector database

    Args:
        features_list: List of feature dictionaries
        kb_facts: List of knowledge base facts
        db_path: Path to database directory
    """
    print("\nBuilding vector database...")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(db_path))

    # Delete existing collection if it exists
    try:
        client.delete_collection("brain_mets_cases")
    except Exception:
        pass

    # Create collection
    collection = client.create_collection(
        name="brain_mets_cases",
        metadata={"description": "Brain metastasis MRI cases", "hnsw:space": "cosine"}
    )

    # Add cases to collection
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for features in features_list:
        case_id = features['case_id']

        # Use image embedding
        if 'image_embedding' in features:
            embedding = features['image_embedding']
        else:
            continue

        # Create metadata (exclude embedding and large arrays)
        metadata = {k: v for k, v in features.items()
                   if k not in ['image_embedding', 'mean_centroid'] and not isinstance(v, list)}

        # Create document text for retrieval
        doc_text = f"Case {case_id}, Primary cancer: {features.get('primary_cancer', 'unknown')}"

        if features.get('num_lesions', 0) > 0:
            doc_text += f", {features['num_lesions']} lesion(s)"
            doc_text += f", total volume: {features['total_volume']:.0f} voxels"
            doc_text += f", mean lesion size: {features['mean_lesion_volume']:.0f} voxels"

        ids.append(case_id)
        embeddings.append(embedding)
        metadatas.append(metadata)
        documents.append(doc_text)

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

    print(f"Added {len(ids)} cases to vector database")

    # Also store knowledge base facts
    try:
        client.delete_collection("medical_knowledge")
    except Exception:
        pass

    kb_collection = client.create_collection(
        name="medical_knowledge",
        metadata={"description": "Medical facts about brain metastases"}
    )

    kb_collection.add(
        ids=[f"fact_{i}" for i in range(len(kb_facts))],
        documents=kb_facts
    )

    print(f"Added {len(kb_facts)} knowledge base facts")

    return collection


def main(args):
    """Main function"""
    print("Building RAG database for brain metastasis cases...\n")

    data_dir = Path(args.data_dir)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)

    # Build knowledge base
    kb_facts = build_knowledge_base()

    # Process dataset and extract features
    features_list = process_dataset(
        data_dir,
        metadata_path,
        output_dir,
        device=args.device
    )

    # Build vector database
    db_path = output_dir / 'chromadb'
    collection = build_vector_database(
        features_list,
        kb_facts,
        db_path
    )

    print(f"\n{'='*60}")
    print(f"Database built successfully!")
    print(f"Total cases: {len(features_list)}")
    print(f"Database location: {db_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG vector database")

    parser.add_argument('--data_dir', type=str, default='../../train',
                        help='Path to data directory')
    parser.add_argument('--metadata_path', type=str, default='../../metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--output_dir', type=str, default='../../outputs/rag',
                        help='Output directory for database')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for feature extraction')

    args = parser.parse_args()

    main(args)
