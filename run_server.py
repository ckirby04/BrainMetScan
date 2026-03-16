"""
BrainMetScan API server entry point.

Usage:
    python run_server.py
    python run_server.py --port 9000 --host 127.0.0.1
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run BrainMetScan API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        workers=1,  # Single worker for GPU memory management
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
