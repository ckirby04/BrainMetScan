"""Tests for API key authentication."""

import os
import pytest

from src.api.auth import _check_rate_limit, _rate_limit_windows
from src.api.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "auth_test.db")
    return Database(db_path)


class TestRateLimiting:
    def test_allows_within_limit(self):
        _rate_limit_windows.clear()
        # Should not raise
        _check_rate_limit("test_key", 10)

    def test_blocks_over_limit(self):
        from fastapi import HTTPException

        _rate_limit_windows.clear()
        _rate_limit_windows["test_key_2"] = [__import__("time").time()] * 5

        with pytest.raises(HTTPException) as exc_info:
            _check_rate_limit("test_key_2", 5)
        assert exc_info.value.status_code == 429


class TestAuthPermissions:
    def test_check_permission_no_auth(self):
        from src.api.auth import check_permission
        # Should not raise when auth is not required
        check_permission(None, "predict")

    def test_check_permission_valid(self):
        from src.api.auth import check_permission
        key_info = {"permissions": ["predict", "admin"]}
        check_permission(key_info, "predict")

    def test_check_permission_denied(self):
        from fastapi import HTTPException
        from src.api.auth import check_permission
        key_info = {"permissions": ["predict"]}
        with pytest.raises(HTTPException) as exc_info:
            check_permission(key_info, "admin")
        assert exc_info.value.status_code == 403
