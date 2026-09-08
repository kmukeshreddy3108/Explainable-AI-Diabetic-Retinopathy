"""
Background Sync Worker for Offline-First Store & Forward Architecture.

Transmits pending local outbox scans to central district/cloud server when connectivity is available.
Handles priority routing, retries, and network outage simulation.
"""

from typing import Dict, Any, Optional
import time

from src.sync.outbox import OutboxQueue


class SyncWorker:
    """Worker managing outbox queue processing and network transmission."""

    def __init__(self, outbox: Optional[OutboxQueue] = None):
        self.outbox = outbox or OutboxQueue()

    def process_outbox_batch(
        self,
        batch_size: int = 5,
        mock_network_available: bool = True,
        mock_failure_rate: float = 0.0,
    ) -> Dict[str, int]:
        """
        Process a batch of pending scans from the outbox.

        Args:
            batch_size: Max scans to process in single run
            mock_network_available: If False, simulates offline status (all skipped)
            mock_failure_rate: Probability of transmission failure (0.0 to 1.0)

        Returns:
            Dict with counts of processed, succeeded, failed, skipped.
        """
        if not mock_network_available:
            return {"processed": 0, "succeeded": 0, "failed": 0, "skipped_offline": 1}

        pending_items = self.outbox.fetch_next_pending(limit=batch_size)
        if not pending_items:
            return {"processed": 0, "succeeded": 0, "failed": 0, "skipped_offline": 0}

        succeeded = 0
        failed = 0

        for item in pending_items:
            scan_id = item["scan_id"]
            self.outbox.update_status(scan_id, "syncing")

            # Simulate network transmission check
            if mock_failure_rate > 0.0 and (hash(scan_id) % 100 / 100.0) < mock_failure_rate:
                # Simulated connection error
                self.outbox.update_status(
                    scan_id, "failed", error_message="HTTP 503 Server Unavailable"
                )
                failed += 1
            else:
                # Simulated successful upload
                self.outbox.update_status(scan_id, "synced")
                succeeded += 1

        return {
            "processed": len(pending_items),
            "succeeded": succeeded,
            "failed": failed,
            "skipped_offline": 0,
        }
