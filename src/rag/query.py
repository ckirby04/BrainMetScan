"""
RAG Query Interface for Brain Metastasis Analysis
Retrieves similar cases and generates clinical reports
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import chromadb
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables only.")
    print("Install with: pip install python-dotenv")

from feature_extractor import extract_case_features


def retrieve_similar_cases(
    query_features: Dict,
    db_path: Path,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve k most similar cases from vector database

    Args:
        query_features: Features of query case
        db_path: Path to ChromaDB database
        k: Number of similar cases to retrieve

    Returns:
        List of similar case dictionaries
    """
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("brain_mets_cases")

    # Query using image embedding
    query_embedding = query_features.get('image_embedding')
    if not query_embedding:
        raise ValueError("Query case must have image_embedding")

    # Retrieve similar cases
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # Format results
    similar_cases = []
    for i in range(len(results['ids'][0])):
        case = {
            'case_id': results['ids'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None,
            'metadata': results['metadatas'][0][i],
            'document': results['documents'][0][i]
        }
        similar_cases.append(case)

    return similar_cases


def retrieve_knowledge(
    query_text: str,
    db_path: Path,
    k: int = 3
) -> List[str]:
    """
    Retrieve relevant medical knowledge

    Args:
        query_text: Query text
        db_path: Path to ChromaDB database
        k: Number of facts to retrieve

    Returns:
        List of relevant facts
    """
    client = chromadb.PersistentClient(path=str(db_path))

    all_facts = []

    # Try to get from medical_knowledge collection (basic KB)
    try:
        kb_collection = client.get_collection("medical_knowledge")
        results = kb_collection.query(
            query_texts=[query_text],
            n_results=k
        )
        all_facts.extend(results['documents'][0])
    except:
        pass

    # Try to get from medical_literature collection (research papers)
    try:
        lit_collection = client.get_collection("medical_literature")
        results = lit_collection.query(
            query_texts=[query_text],
            n_results=k
        )
        all_facts.extend(results['documents'][0])
    except:
        pass

    # Return top k from combined results
    return all_facts[:k]


def generate_report_local(
    query_features: Dict,
    similar_cases: List[Dict],
    kb_facts: List[str]
) -> str:
    """
    Generate clinical report using template (fallback when no LLM available)

    Args:
        query_features: Features of query case
        similar_cases: Retrieved similar cases
        kb_facts: Retrieved knowledge base facts

    Returns:
        Generated report text
    """
    case_id = query_features.get('case_id', 'Unknown')
    primary_cancer = query_features.get('primary_cancer', 'unknown')
    num_lesions = query_features.get('num_lesions', 0)
    total_volume = query_features.get('total_volume', 0)

    report = f"BRAIN METASTASIS ANALYSIS REPORT\n"
    report += f"{'='*60}\n\n"

    report += f"Case ID: {case_id}\n"
    report += f"Primary Cancer: {primary_cancer}\n\n"

    report += f"FINDINGS:\n"
    report += f"---------\n"

    if num_lesions > 0:
        report += f"Number of lesions detected: {num_lesions}\n"
        report += f"Total lesion volume: {total_volume:.0f} voxels\n"

        if num_lesions == 1:
            report += f"Single metastatic lesion identified.\n"
        else:
            mean_vol = query_features.get('mean_lesion_volume', 0)
            max_vol = query_features.get('max_lesion_volume', 0)
            report += f"Multiple metastatic lesions present.\n"
            report += f"  - Mean lesion volume: {mean_vol:.0f} voxels\n"
            report += f"  - Largest lesion volume: {max_vol:.0f} voxels\n"

        # Location info
        centroid = query_features.get('mean_centroid', [0, 0, 0])
        report += f"  - Approximate center location: ({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})\n"

    else:
        report += f"No lesions detected in segmentation.\n"

    report += f"\nSIMILAR CASES:\n"
    report += f"--------------\n"

    for i, case in enumerate(similar_cases[:3], 1):
        report += f"{i}. {case['document']}\n"

    report += f"\nCLINICAL CONTEXT:\n"
    report += f"-----------------\n"

    for fact in kb_facts:
        report += f"• {fact}\n"

    report += f"\nRECOMMENDATIONS:\n"
    report += f"----------------\n"

    if num_lesions > 0:
        if num_lesions == 1:
            report += f"• Single metastasis may be amenable to surgical resection or stereotactic radiosurgery.\n"
        else:
            report += f"• Multiple metastases detected. Consider stereotactic radiosurgery for eligible lesions.\n"

        report += f"• Correlate findings with clinical history and systemic disease burden.\n"
        report += f"• Recommend multidisciplinary discussion for treatment planning.\n"
        report += f"• Follow-up MRI to assess response to therapy and monitor for new lesions.\n"
    else:
        report += f"• No metastatic lesions identified. Continue surveillance imaging.\n"

    report += f"\n{'='*60}\n"
    report += f"This report was generated using AI-assisted analysis.\n"
    report += f"Clinical correlation and expert review are recommended.\n"
    report += f"{'='*60}\n"

    return report


def generate_report_openai(
    query_features: Dict,
    similar_cases: List[Dict],
    kb_facts: List[str],
    api_key: str
) -> str:
    """
    Generate clinical report using OpenAI GPT

    Args:
        query_features: Features of query case
        similar_cases: Retrieved similar cases
        kb_facts: Retrieved knowledge base facts
        api_key: OpenAI API key

    Returns:
        Generated report text
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build prompt
        prompt = f"""You are a medical imaging AI assistant. Generate a concise clinical analysis report for a brain metastasis case.

Case Information:
- Case ID: {query_features.get('case_id', 'Unknown')}
- Primary Cancer: {query_features.get('primary_cancer', 'unknown')}
- Number of lesions: {query_features.get('num_lesions', 0)}
- Total volume: {query_features.get('total_volume', 0):.0f} voxels

Similar Cases from Database:
"""
        for i, case in enumerate(similar_cases[:3], 1):
            prompt += f"{i}. {case['document']}\n"

        prompt += f"\nRelevant Medical Knowledge:\n"
        for fact in kb_facts:
            prompt += f"• {fact}\n"

        prompt += """
Please generate a structured clinical report with the following sections:
1. FINDINGS: Describe the detected metastases
2. SIMILAR CASES: Brief comparison with retrieved similar cases
3. CLINICAL CONTEXT: Interpretation based on medical knowledge
4. RECOMMENDATIONS: Suggested next steps for clinical management

Keep the report concise, professional, and clinically relevant. Include a disclaimer that this is AI-generated and requires expert review.
"""

        # Call GPT
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=800,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error using OpenAI API: {e}")
        print("Falling back to template-based report...")
        return generate_report_local(query_features, similar_cases, kb_facts)


def generate_report_claude(
    query_features: Dict,
    similar_cases: List[Dict],
    kb_facts: List[str],
    api_key: str,
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """
    Generate clinical report using Anthropic Claude

    Args:
        query_features: Features of query case
        similar_cases: Retrieved similar cases
        kb_facts: Retrieved knowledge base facts
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        Generated report text
    """
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        # Build prompt
        prompt = f"""You are a medical imaging AI assistant specializing in brain metastasis analysis. Generate a concise, clinically accurate report for this case.

Case Information:
- Case ID: {query_features.get('case_id', 'Unknown')}
- Primary Cancer: {query_features.get('primary_cancer', 'unknown')}
- Number of lesions: {query_features.get('num_lesions', 0)}
- Total volume: {query_features.get('total_volume', 0):.0f} voxels

Similar Cases from Database:
"""
        for i, case in enumerate(similar_cases[:3], 1):
            prompt += f"{i}. {case['document']}\n"

        prompt += f"\nRelevant Medical Knowledge:\n"
        for fact in kb_facts:
            prompt += f"• {fact}\n"

        prompt += """
Please generate a structured clinical report with the following sections:
1. FINDINGS: Describe the detected metastases with clinical detail
2. SIMILAR CASES: Brief comparison with retrieved similar cases and their outcomes
3. CLINICAL CONTEXT: Interpretation based on medical knowledge, imaging characteristics, and primary tumor type
4. RECOMMENDATIONS: Evidence-based suggestions for clinical management, imaging follow-up, and multidisciplinary care

Keep the report concise, professional, and clinically relevant. Use appropriate medical terminology. Include a disclaimer that this is AI-generated and requires expert radiologist review.
"""

        # Call Claude
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.3,
            messages=[{'role': 'user', 'content': prompt}]
        )

        return response.content[0].text

    except Exception as e:
        print(f"Error using Anthropic Claude API: {e}")
        print("Falling back to template-based report...")
        return generate_report_local(query_features, similar_cases, kb_facts)


def query_case(
    case_dir: Path,
    db_path: Path,
    output_dir: Path = None,
    k_cases: int = 5,
    k_facts: int = 3,
    use_openai: bool = False,
    use_claude: bool = False,
    device: str = 'cuda'
):
    """
    Query RAG system for a case and generate report

    Args:
        case_dir: Path to case directory
        db_path: Path to ChromaDB database
        output_dir: Output directory for report (optional)
        k_cases: Number of similar cases to retrieve
        k_facts: Number of KB facts to retrieve
        use_openai: Whether to use OpenAI for report generation
        use_claude: Whether to use Anthropic Claude for report generation
        device: Device for feature extraction
    """
    case_id = case_dir.name
    print(f"Analyzing case: {case_id}")

    # Extract features from query case
    print("Extracting features...")
    mask_path = case_dir / "seg.nii.gz" if (case_dir / "seg.nii.gz").exists() else None

    query_features = extract_case_features(
        case_dir,
        mask_path=mask_path,
        sequences=['t1_gd'],
        device=device
    )

    # Retrieve similar cases
    print(f"Retrieving {k_cases} similar cases...")
    similar_cases = retrieve_similar_cases(
        query_features,
        db_path,
        k=k_cases
    )

    # Build query text for knowledge retrieval
    query_text = f"{query_features.get('primary_cancer', 'cancer')} brain metastases"
    if query_features.get('num_lesions', 0) > 1:
        query_text += " multiple lesions"

    # Retrieve knowledge
    print(f"Retrieving medical knowledge...")
    kb_facts = retrieve_knowledge(query_text, db_path, k=k_facts)

    # Generate report
    print("Generating clinical report...")

    if use_claude:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment. Using template.")
            report = generate_report_local(query_features, similar_cases, kb_facts)
        else:
            print("Using Claude (Anthropic) for report generation...")
            report = generate_report_claude(query_features, similar_cases, kb_facts, api_key)
    elif use_openai:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment. Using template.")
            report = generate_report_local(query_features, similar_cases, kb_facts)
        else:
            print("Using OpenAI GPT for report generation...")
            report = generate_report_openai(query_features, similar_cases, kb_facts, api_key)
    else:
        report = generate_report_local(query_features, similar_cases, kb_facts)

    # Print report
    print("\n" + report)

    # Save report if output dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{case_id}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nReport saved to: {report_path}")

        # Also save detailed results as JSON
        results = {
            'case_id': case_id,
            'query_features': {k: v for k, v in query_features.items() if k != 'image_embedding'},
            'similar_cases': similar_cases,
            'knowledge_base_facts': kb_facts
        }

        json_path = output_dir / f"{case_id}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Detailed results saved to: {json_path}")


def main(args):
    """Main function"""
    case_dir = Path(args.case_dir)
    db_path = Path(args.db_path)

    if not case_dir.exists():
        raise ValueError(f"Case directory not found: {case_dir}")

    if not db_path.exists():
        raise ValueError(f"Database not found: {db_path}. Run build_database.py first.")

    query_case(
        case_dir,
        db_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        k_cases=args.k_cases,
        k_facts=args.k_facts,
        use_openai=args.use_openai,
        use_claude=args.use_claude,
        device=args.device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query RAG system for brain metastasis case")

    parser.add_argument('--case_dir', type=str, required=True,
                        help='Path to case directory (e.g., train/Mets_040)')
    parser.add_argument('--db_path', type=str, default='../../outputs/rag/chromadb',
                        help='Path to ChromaDB database')
    parser.add_argument('--output_dir', type=str, default='../../outputs/rag/reports',
                        help='Output directory for reports')

    parser.add_argument('--k_cases', type=int, default=5,
                        help='Number of similar cases to retrieve')
    parser.add_argument('--k_facts', type=int, default=3,
                        help='Number of KB facts to retrieve')

    parser.add_argument('--use_openai', action='store_true',
                        help='Use OpenAI GPT for report generation')
    parser.add_argument('--use_claude', action='store_true',
                        help='Use Anthropic Claude for report generation (recommended)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for feature extraction')

    args = parser.parse_args()

    main(args)
