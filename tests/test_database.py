"""Tests for the database layer."""

import os
import tempfile

import pytest

from src.api.database import Database


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    return Database(db_path)


class TestAPIKeyManagement:
    def test_create_api_key(self, db):
        result = db.create_api_key("test_key", permissions=["predict", "admin"])
        assert "api_key" in result
        assert result["api_key"].startswith("bms_")
        assert result["name"] == "test_key"
        assert "predict" in result["permissions"]

    def test_validate_api_key(self, db):
        created = db.create_api_key("test_key")
        validated = db.validate_api_key(created["api_key"])
        assert validated is not None
        assert validated["key_id"] == created["key_id"]

    def test_invalid_key_returns_none(self, db):
        result = db.validate_api_key("bms_invalid_key_123")
        assert result is None

    def test_revoke_api_key(self, db):
        created = db.create_api_key("test_key")
        assert db.revoke_api_key(created["key_id"])
        assert db.validate_api_key(created["api_key"]) is None

    def test_list_api_keys(self, db):
        db.create_api_key("key1")
        db.create_api_key("key2")
        keys = db.list_api_keys()
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"key1", "key2"}


class TestPredictionRecords:
    def test_record_and_get_prediction(self, db):
        db.record_prediction(
            job_id="job123",
            case_id="case_001",
            status="completed",
            lesion_count=3,
            total_volume_mm3=1500.5,
            processing_time=2.5,
        )
        pred = db.get_prediction("job123")
        assert pred is not None
        assert pred["case_id"] == "case_001"
        assert pred["lesion_count"] == 3

    def test_list_predictions(self, db):
        db.record_prediction(job_id="j1", case_id="c1", status="completed")
        db.record_prediction(job_id="j2", case_id="c2", status="completed")
        preds = db.list_predictions(limit=10)
        assert len(preds) == 2

    def test_prediction_not_found(self, db):
        assert db.get_prediction("nonexistent") is None


class TestAuditLog:
    def test_log_event(self, db):
        db.log_event(
            event_type="prediction",
            endpoint="/predict",
            details={"job_id": "j1"},
            ip_address="127.0.0.1",
            status_code=200,
        )
        # No exception means success


class TestAnalytics:
    def test_stats_empty_db(self, db):
        stats = db.get_stats(days=30)
        assert stats["total_predictions"] == 0
        assert stats["success_rate"] == 0

    def test_stats_with_data(self, db):
        db.record_prediction(job_id="j1", case_id="c1", status="completed", processing_time=1.5, lesion_count=2)
        db.record_prediction(job_id="j2", case_id="c2", status="completed", processing_time=2.5, lesion_count=4)
        stats = db.get_stats(days=30)
        assert stats["total_predictions"] == 2
        assert stats["successful_predictions"] == 2
        assert stats["success_rate"] == 100.0
        assert stats["average_lesion_count"] == 3.0
