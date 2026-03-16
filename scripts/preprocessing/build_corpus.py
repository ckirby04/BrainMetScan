#!/usr/bin/env python3
"""
CLI entrypoint for building the full RAG corpus.

Usage:
    python scripts/build_corpus.py --config configs/rag.yaml --output_dir outputs/rag
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.corpus.builder import CorpusBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build the brain metastasis RAG corpus from PubMed"
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
        help="Output directory for corpus artifacts",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    builder = CorpusBuilder(config_path=str(config_path))
    stats = builder.build(output_dir=args.output_dir)

    print(f"\nBuild complete. Stats: {stats}")


if __name__ == "__main__":
    main()
