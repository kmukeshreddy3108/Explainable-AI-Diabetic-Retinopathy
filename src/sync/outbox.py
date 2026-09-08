"""
Offline-First Store & Forward Outbox Queue.

Stores screening scans and diagnostic reports in a local SQLite outbox database when offline.
Prioritizes high-risk referable DR cases for sync over routine normal scans.
"""

import os
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.utils.config import SYNC_MAX_RETRIES


class OutboxQueue:
    """SQLite-backed persistent outbox queue for offline-first synchronization."""

    def __init__(self, db_path: str = "data/outbox.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    scan_id TEXT UNIQUE NOT NULL,
                    image_path TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1, -- 2=HIGH (Referable), 1=NORMAL
                    status TEXT NOT NULL DEFAULT 'pending', -- pending, syncing, synced, failed
                    specialist_status TEXT DEFAULT 'pending_review', -- pending_review, confirmed_referral, phc_rescreen, recapture_requested
                    specialist_notes TEXT,
                    reviewed_at TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    synced_at TIMESTAMP
                )
                """
            )
            conn.commit()

    def enqueue(
        self,
        patient_id: str,
        scan_id: str,
        image_path: str,
        report_dict: Dict[str, Any],
        is_referable: bool = False,
    ) -> int:
        """
        Add a completed scan and report to the local outbox. High risk/referable cases get Priority=2.
        """
        priority = 2 if is_referable else 1
        report_json = json.dumps(report_dict)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO outbox (patient_id, scan_id, image_path, report_json, priority, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (patient_id, scan_id, image_path, report_json, priority),
            )
            conn.commit()
            return cursor.lastrowid

    def fetch_next_pending(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch pending items ordered by priority (HIGH priority first) then creation time.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM outbox
                WHERE status = 'pending' AND retry_count < ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (SYNC_MAX_RETRIES, limit),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_status(
        self,
        scan_id: str,
        status: str,
        error_message: Optional[str] = None,
    ):
        """Update transaction status (e.g. 'synced', 'failed', 'syncing')."""
        synced_at = datetime.now().isoformat() if status == "synced" else None
        with self._get_connection() as conn:
            if status == "failed":
                conn.execute(
                    """
                    UPDATE outbox
                    SET status = ?, retry_count = retry_count + 1, error_message = ?
                    WHERE scan_id = ?
                    """,
                    (status, error_message, scan_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE outbox
                    SET status = ?, synced_at = ?, error_message = ?
                    WHERE scan_id = ?
                    """,
                    (status, synced_at, error_message, scan_id),
                )
            conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Get summary counts of outbox entries by status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, COUNT(*) as count FROM outbox GROUP BY status
                """
            )
            rows = cursor.fetchall()
            stats = {"pending": 0, "syncing": 0, "synced": 0, "failed": 0}
            for row in rows:
                stats[row["status"]] = row["count"]
            return stats

    def record_specialist_review(
        self, scan_id: str, specialist_status: str, specialist_notes: str
    ):
        """Record human ophthalmologist review decision and notes for PHC feedback loop."""
        reviewed_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE outbox
                SET specialist_status = ?, specialist_notes = ?, reviewed_at = ?
                WHERE scan_id = ?
                """,
                (specialist_status, specialist_notes, reviewed_at, scan_id),
            )
            conn.commit()

    def fetch_all_scans(self) -> List[Dict[str, Any]]:
        """Fetch all outbox records for telemedicine specialist queue."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM outbox ORDER BY priority DESC, created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

