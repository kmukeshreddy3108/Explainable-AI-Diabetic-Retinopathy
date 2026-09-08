"""
Unit tests for Phase 4.5 — Offline-First Sync & Outbox Queue.
"""

import pytest
import os
import sqlite3

from src.sync.outbox import OutboxQueue
from src.sync.sync_worker import SyncWorker


def test_outbox_queue_priority_ordering(tmp_path):
    db_file = str(tmp_path / "test_outbox.db")
    queue = OutboxQueue(db_path=db_file)

    # Enqueue a routine non-referable scan FIRST
    queue.enqueue(
        patient_id="P_ROUTINE",
        scan_id="SCAN_001",
        image_path="/tmp/scan1.jpg",
        report_dict={"grade": 0},
        is_referable=False,
    )

    # Enqueue a high-risk referable scan SECOND
    queue.enqueue(
        patient_id="P_URGENT",
        scan_id="SCAN_002",
        image_path="/tmp/scan2.jpg",
        report_dict={"grade": 3},
        is_referable=True,
    )

    # Fetch next pending item — must return the high-risk referable SCAN_002 FIRST!
    pending = queue.fetch_next_pending(limit=1)
    assert len(pending) == 1
    assert pending[0]["scan_id"] == "SCAN_002"
    assert pending[0]["priority"] == 2


def test_sync_worker_execution(tmp_path):
    db_file = str(tmp_path / "test_worker.db")
    queue = OutboxQueue(db_path=db_file)
    worker = SyncWorker(outbox=queue)

    queue.enqueue("P1", "SCAN_A", "/tmp/a.jpg", {"grade": 1}, is_referable=False)
    queue.enqueue("P2", "SCAN_B", "/tmp/b.jpg", {"grade": 4}, is_referable=True)

    # 1. Offline run
    res_offline = worker.process_outbox_batch(mock_network_available=False)
    assert res_offline["skipped_offline"] == 1
    assert queue.get_stats()["pending"] == 2

    # 2. Online run
    res_online = worker.process_outbox_batch(mock_network_available=True)
    assert res_online["succeeded"] == 2
    assert queue.get_stats()["synced"] == 2
    assert queue.get_stats()["pending"] == 0
