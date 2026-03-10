#!/usr/bin/env python
"""
Launch the Brain Metastasis Segmentation Demo (Consolidated in v1.23)

Usage:
    python run_demo.py                    # Normal mode (requires GPU + models)
    python run_demo.py --demo             # Demo mode (pre-computed cases, no GPU needed)
    python run_demo.py --api-url http://localhost:8000  # Use remote API for inference
    python run_demo.py --port 8080        # Custom port

Options:
    --demo        Run in demo mode with pre-computed/synthetic cases (no GPU required)
    --api-url     Connect to BrainMetScan API server for inference
    --api-key     API key for authenticated access
    --port        Port to run the Gradio server on (default: 7860)
    --share       Create a public Gradio URL for sharing
"""

import argparse
import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    missing = []

    try:
        import gradio
    except ImportError:
        missing.append('gradio')

    try:
        import torch
    except ImportError:
        missing.append('torch')

    try:
        import nibabel
    except ImportError:
        missing.append('nibabel')

    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Launch Brain Metastasis Segmentation Demo")
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--share', action='store_true', help='Create a public URL')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (pre-computed cases, no GPU required)')
    parser.add_argument('--api-url', type=str, default=None,
                        help='BrainMetScan API URL for remote inference')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for authenticated access')
    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Set environment variables for API connectivity
    if args.api_url:
        os.environ["BRAINMETSCAN_API_URL"] = args.api_url
    if args.api_key:
        os.environ["BRAINMETSCAN_API_KEY"] = args.api_key

    print("=" * 60)
    print("BrainMetScan Demo v1.30")
    print("=" * 60)

    if args.demo:
        print("Mode: DEMO (pre-computed cases, no GPU required)")
        os.environ["DEMO_MODE"] = "true"
    elif args.api_url:
        print(f"Mode: API ({args.api_url})")
    else:
        print("Mode: LOCAL (GPU inference)")

    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("=" * 60)

    from demo.app import create_demo, DEVICE

    if not args.demo:
        from demo.app import load_stacking_model_cached, load_model
        ensemble = load_stacking_model_cached()
        if ensemble is None:
            print("Falling back to single model...")
            try:
                load_model()
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Run with --demo flag for demo mode without models.")
                if not args.demo:
                    print("Continuing without models...")

    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
