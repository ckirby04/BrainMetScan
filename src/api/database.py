"""
SQLite database layer for persisting predictions, cases, and API keys.
Uses Python's built-in sqlite3 — no ORM required for this scale.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Database:
    """SQLite persistence layer for BrainMetScan."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = str(project_root / "data" / "brainmetscan.db")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    expires_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    permissions TEXT NOT NULL DEFAULT '["predict"]',
                    rate_limit_per_minute INTEGER NOT NULL DEFAULT 60
                );

                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    input_format TEXT NOT NULL DEFAULT 'nifti',
                    metadata TEXT DEFAULT '{}',
                    api_key_id TEXT,
                    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    job_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    status TEXT NOT NULL DEFAULT 'pending',
                    lesion_count INTEGER DEFAULT 0,
                    total_volume_mm3 REAL DEFAULT 0.0,
                    processing_time_seconds REAL DEFAULT 0.0,
                    threshold REAL DEFAULT 0.5,
                    use_tta INTEGER DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    api_key_id TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases(case_id),
                    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    event_type TEXT NOT NULL,
                    api_key_id TEXT,
                    endpoint TEXT,
                    details TEXT DEFAULT '{}',
                    ip_address TEXT,
                    status_code INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_case
                    ON predictions(case_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_created
                    ON predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_event_type
                    ON audit_log(event_type);
            """)

    # --- API Key Management ---

    def create_api_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
        rate_limit: int = 60,
        expires_at: Optional[str] = None,
    ) -> Dict:
        """Create a new API key. Returns the key (only shown once) and metadata."""
        import hashlib

        key = f"bms_{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_id = uuid.uuid4().hex[:12]

        if permissions is None:
            permissions = ["predict"]

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO api_keys (key_id, key_hash, name, permissions, rate_limit_per_minute, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key_id, key_hash, name, json.dumps(permissions), rate_limit, expires_at),
            )

        return {
            "key_id": key_id,
            "api_key": key,
            "name": name,
            "permissions": permissions,
            "rate_limit_per_minute": rate_limit,
        }

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key and return its metadata, or None if invalid."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        with self._connect() as conn:
            row = conn.execute(
                """SELECT key_id, name, permissions, rate_limit_per_minute, expires_at, is_active
                   FROM api_keys WHERE key_hash = ?""",
                (key_hash,),
            ).fetchone()

        if row is None:
            return None

        if not row["is_active"]:
            return None

        if row["expires_at"]:
            if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
                return None

        return {
            "key_id": row["key_id"],
            "name": row["name"],
            "permissions": json.loads(row["permissions"]),
            "rate_limit_per_minute": row["rate_limit_per_minute"],
        }

    def list_api_keys(self) -> List[Dict]:
        """List all API keys (without the actual key values)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key_id, name, created_at, expires_at, is_active, permissions, rate_limit_per_minute FROM api_keys"
            ).fetchall()

        return [
            {
                "key_id": r["key_id"],
                "name": r["name"],
                "created_at": r["created_at"],
                "expires_at": r["expires_at"],
                "is_active": bool(r["is_active"]),
                "permissions": json.loads(r["permissions"]),
                "rate_limit_per_minute": r["rate_limit_per_minute"],
            }
            for r in rows
        ]

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
        return cursor.rowcount > 0

    # --- Case Management ---

    def create_case(
        self, case_id: str, input_format: str = "nifti", metadata: Optional[Dict] = None, api_key_id: Optional[str] = None
    ) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO cases (case_id, input_format, metadata, api_key_id) VALUES (?, ?, ?, ?)",
                (case_id, input_format, json.dumps(metadata or {}), api_key_id),
            )
        return case_id

    def get_case(self, case_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()

        if row is None:
            return None
        return {
            "case_id": row["case_id"],
            "created_at": row["created_at"],
            "input_format": row["input_format"],
            "metadata": json.loads(row["metadata"]),
        }

    # --- Prediction Records ---

    def record_prediction(
        self,
        job_id: str,
        case_id: str,
        status: str = "completed",
        lesion_count: int = 0,
        total_volume_mm3: float = 0.0,
        processing_time: float = 0.0,
        threshold: float = 0.5,
        use_tta: bool = False,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ):
        self.create_case(case_id, api_key_id=api_key_id)

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (job_id, case_id, status, lesion_count, total_volume_mm3,
                    processing_time_seconds, threshold, use_tta, result_json, error, api_key_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, case_id, status, lesion_count, total_volume_mm3,
                    processing_time, threshold, int(use_tta), result_json, error, api_key_id,
                ),
            )

    def get_prediction(self, job_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE job_id = ?", (job_id,)).fetchone()

        if row is None:
            return None
        result = dict(row)
        if result.get("result_json"):
            result["result_json"] = json.loads(result["result_json"])
        return result

    def list_predictions(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, case_id, created_at, status, lesion_count, total_volume_mm3, processing_time_seconds "
                "FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Audit Log ---

    def log_event(
        self,
        event_type: str,
        endpoint: Optional[str] = None,
        api_key_id: Optional[str] = None,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audit_log (event_type, api_key_id, endpoint, details, ip_address, status_code)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event_type, api_key_id, endpoint, json.dumps(details or {}), ip_address, status_code),
            )

    # --- Analytics ---

    def get_stats(self, days: int = 30) -> Dict:
        """Get usage statistics for the last N days."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()["cnt"]

            successful = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE status='completed' AND created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()["cnt"]

            avg_time = conn.execute(
                "SELECT AVG(processing_time_seconds) as avg_t FROM predictions "
                "WHERE status='completed' AND created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()["avg_t"]

            avg_lesions = conn.execute(
                "SELECT AVG(lesion_count) as avg_l FROM predictions "
                "WHERE status='completed' AND created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()["avg_l"]

            daily = conn.execute(
                "SELECT date(created_at) as day, COUNT(*) as cnt "
                "FROM predictions WHERE created_at >= datetime('now', ?) "
                "GROUP BY date(created_at) ORDER BY day",
                (f"-{days} days",),
            ).fetchall()

            unique_cases = conn.execute(
                "SELECT COUNT(DISTINCT case_id) as cnt FROM predictions WHERE created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()["cnt"]

        return {
            "period_days": days,
            "total_predictions": total,
            "successful_predictions": successful,
            "failed_predictions": total - successful,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0,
            "average_processing_time_seconds": round(avg_time or 0, 2),
            "average_lesion_count": round(avg_lesions or 0, 1),
            "unique_cases": unique_cases,
            "daily_predictions": [{"date": r["day"], "count": r["cnt"]} for r in daily],
        }
