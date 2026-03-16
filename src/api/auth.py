"""
API key authentication middleware for BrainMetScan.
Supports API key validation via header, rate limiting, and permission checking.
"""

import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .database import Database

# API key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory rate limit tracking: {key_id: [(timestamp, ...)] }
_rate_limit_windows: dict = defaultdict(list)


def _get_db() -> Database:
    """Lazy database initialization."""
    if not hasattr(_get_db, "_instance"):
        _get_db._instance = Database()
    return _get_db._instance


def set_db(db: Database):
    """Set the database instance (for testing or custom configuration)."""
    _get_db._instance = db


async def get_api_key_info(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[dict]:
    """
    Dependency that validates the API key if provided.

    When AUTH_REQUIRED=true (env var), requests without a valid key are rejected.
    When AUTH_REQUIRED=false (default), unauthenticated requests are allowed.
    """
    import os

    auth_required = os.environ.get("AUTH_REQUIRED", "false").lower() == "true"

    if api_key is None or api_key == "":
        if auth_required:
            raise HTTPException(status_code=401, detail="API key required. Provide X-API-Key header.")
        return None

    db = _get_db()
    key_info = db.validate_api_key(api_key)

    if key_info is None:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    # Rate limiting
    _check_rate_limit(key_info["key_id"], key_info["rate_limit_per_minute"])

    return key_info


def _check_rate_limit(key_id: str, limit_per_minute: int):
    """Simple sliding window rate limiter."""
    now = time.time()
    window_start = now - 60

    # Clean old entries
    _rate_limit_windows[key_id] = [
        t for t in _rate_limit_windows[key_id] if t > window_start
    ]

    if len(_rate_limit_windows[key_id]) >= limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {limit_per_minute} requests per minute.",
        )

    _rate_limit_windows[key_id].append(now)


def check_permission(key_info: Optional[dict], permission: str):
    """Check if the API key has a specific permission."""
    if key_info is None:
        # Unauthenticated — only allowed if AUTH_REQUIRED is false
        return

    if permission not in key_info.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail=f"API key does not have '{permission}' permission",
        )
