"""
Structured JSON logging configuration for BrainMetScan.
Provides consistent log format across all components.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields
        for key in ("endpoint", "method", "status_code", "duration_ms",
                     "client_ip", "api_key_id", "job_id", "case_id",
                     "lesion_count", "error_detail"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    root = logging.getLogger("brainmetscan")
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler with JSON formatting
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JSONFormatter())
    root.addHandler(console)

    # File handler if specified
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in ("uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the brainmetscan namespace."""
    return logging.getLogger(f"brainmetscan.{name}")


class RequestTimer:
    """Context manager for timing API requests."""

    def __init__(self):
        self.start_time = None
        self.duration_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
