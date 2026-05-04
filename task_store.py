"""
task_store.py

SQLite-backed task state persistence for APEX async operations.

Replaces the in-memory dicts (_TASK_STORE, _ingestion_tasks, _training_tasks)
that break under multi-worker Uvicorn.  Uses Python's native ``sqlite3`` --
zero external dependencies.

Usage::

    from task_store import task_db, IngestionProgressTracker, TrainingProgressTracker

    # Inference task
    task_db.insert_task(task_id, "inference", "PENDING", payload={...})
    task_db.update_status(task_id, "PROCESSING")
    task_db.update_result(task_id, "COMPLETED", result_dict)

    # Ingestion task (via tracker)
    tracker = IngestionProgressTracker(task_id, sources, task_db)
    tracker.set_progress(50, "Downloading...")
    tracker.complete(result_dict)

    # Training task (via tracker)
    tracker = TrainingProgressTracker(task_id, task_db)
    tracker.set_phase(3, "Preprocessing", 30)
    tracker.log_epoch(trial=0, epoch=1, ...)
    tracker.complete(result_dict)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from config.paths import TASK_DB_PATH

_DB_PATH: Path = TASK_DB_PATH


# ---------------------------------------------------------------------------
# SQLite state manager
# ---------------------------------------------------------------------------

class TaskStateManager:
    """
    Multi-worker-safe task state persistence backed by SQLite.

    Each method opens a short-lived connection, commits, and closes.
    WAL journal mode allows concurrent readers alongside a single writer.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS TaskState (
                task_id    TEXT PRIMARY KEY,
                task_type  TEXT NOT NULL,
                status     TEXT NOT NULL,
                result     TEXT,
                error      TEXT,
                payload    TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("TaskStateManager: database ready at %s", self.db_path)

    # ------------------------------------------------------------------ #
    # Core CRUD
    # ------------------------------------------------------------------ #

    def insert_task(
        self,
        task_id: str,
        task_type: str,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a new task row."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO TaskState (task_id, task_type, status, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    task_id,
                    task_type,
                    status,
                    json.dumps(payload or {}),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a task by ID.  Returns ``None`` if not found.

        The ``result`` and ``payload`` columns are deserialized from JSON.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM TaskState WHERE task_id = ?", (task_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_dict(row)

    def update_status(self, task_id: str, status: str) -> None:
        """Update only the status column."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE TaskState SET status = ? WHERE task_id = ?",
                (status, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_result(
        self, task_id: str, status: str, result: Dict[str, Any]
    ) -> None:
        """Set status and store a JSON result."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE TaskState SET status = ?, result = ? WHERE task_id = ?",
                (status, json.dumps(result), task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_error(self, task_id: str, status: str, error: str) -> None:
        """Set status and store an error message."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE TaskState SET status = ?, error = ? WHERE task_id = ?",
                (status, error, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_payload(self, task_id: str) -> Dict[str, Any]:
        """Read and deserialize the payload JSON for a task."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload FROM TaskState WHERE task_id = ?", (task_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {}
        return json.loads(row["payload"])

    def merge_payload(
        self, task_id: str, updates: Dict[str, Any]
    ) -> None:
        """Atomic read-modify-write: load payload, dict-update, write back."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload FROM TaskState WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                return
            payload = json.loads(row["payload"])
            payload.update(updates)
            conn.execute(
                "UPDATE TaskState SET payload = ? WHERE task_id = ?",
                (json.dumps(payload), task_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_to_payload_list(
        self, task_id: str, key: str, item: Any
    ) -> None:
        """Atomically append an item to a list field inside the payload JSON."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload FROM TaskState WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                return
            payload = json.loads(row["payload"])
            lst = payload.get(key, [])
            lst.append(item)
            payload[key] = lst
            conn.execute(
                "UPDATE TaskState SET payload = ? WHERE task_id = ?",
                (json.dumps(payload), task_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(row)
        # Deserialize JSON columns
        for col in ("result", "payload"):
            raw = d.get(col)
            if raw is not None:
                try:
                    d[col] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


# ---------------------------------------------------------------------------
# Ingestion progress tracker (SQLite-backed)
# ---------------------------------------------------------------------------

class IngestionProgressTracker:
    """Track ingestion task progress in SQLite."""

    def __init__(
        self,
        task_id: str,
        sources: List[str],
        manager: TaskStateManager,
    ) -> None:
        self.task_id = task_id
        self._mgr = manager
        self._mgr.insert_task(
            task_id=task_id,
            task_type="ingestion",
            status="running",
            payload={
                "progress_pct": 0,
                "message": "Starting ingestion...",
                "total_sources": len(sources),
                "completed_sources": 0,
                "datasets": [],
            },
        )

    def set_progress(self, pct: int, message: str) -> None:
        self._mgr.merge_payload(self.task_id, {
            "progress_pct": pct,
            "message": message,
        })

    def report_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """Report a single dataset's result (success or failure)."""
        self._mgr.append_to_payload_list(self.task_id, "datasets", dataset_info)
        payload = self._mgr.get_payload(self.task_id)
        done = len(payload.get("datasets", []))
        total = payload.get("total_sources", 1)
        pct = int((done / max(total, 1)) * 90) + 5
        self._mgr.merge_payload(self.task_id, {
            "completed_sources": done,
            "progress_pct": pct,
        })

    def complete(self, result: Dict[str, Any]) -> None:
        self._mgr.merge_payload(self.task_id, {
            "progress_pct": 100,
            "message": "Ingestion complete",
        })
        self._mgr.update_result(self.task_id, "completed", result)

    def fail(self, error: str) -> None:
        self._mgr.merge_payload(self.task_id, {
            "message": f"Failed: {error}",
        })
        self._mgr.update_error(self.task_id, "failed", error)


# ---------------------------------------------------------------------------
# Training progress tracker (SQLite-backed)
# ---------------------------------------------------------------------------

class TrainingProgressTracker:
    """Track training task progress in SQLite."""

    def __init__(self, task_id: str, manager: TaskStateManager) -> None:
        self.task_id = task_id
        self._mgr = manager
        self._mgr.insert_task(
            task_id=task_id,
            task_type="training",
            status="running",
            payload={
                # Existing fields (preserved for compatibility)
                "current_phase": 0,
                "current_phase_name": "Initializing",
                "progress_pct": 0,
                "messages": [],
                "epoch_metrics": [],
                "trial_progress": None,
                "data_split": None,
                # New structured live-cockpit fields
                "substage": "init",           # e.g. "jit_profiling", "embedding_text", "trial_running"
                "current_trial": None,        # {number, total, fusion, lr, epochs, current_epoch, max_epoch, status}
                "best_so_far": None,          # {trial, val_loss, val_acc, val_f1, val_auroc, timestamp}
                "trial_events": [],           # ordered list of {trial, event, detail, timestamp}
                "pruning_status": {           # pruning backend availability + counts
                    "available": None,
                    "backend": None,
                    "reason": "",
                    "pruned_count": 0,
                    "completed_count": 0,
                },
                "next_trial_plan": None,      # adaptive override preview for next trial
            },
        )

    def set_phase(
        self, phase_num: int, phase_name: str, progress_pct: int
    ) -> None:
        self._mgr.merge_payload(self.task_id, {
            "current_phase": phase_num,
            "current_phase_name": phase_name,
            "progress_pct": progress_pct,
        })

    def add_message(self, phase: int, msg_type: str, text: str) -> None:
        self._mgr.append_to_payload_list(self.task_id, "messages", {
            "phase": phase,
            "type": msg_type,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })

    def log_epoch(
        self,
        trial: int,
        epoch: int,
        max_epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        train_f1: float = 0.0,
        val_f1: float = 0.0,
        val_auroc: float = 0.0,
        alignment_loss: float = 0.0,
        contrastive_loss: float = 0.0,
        pruned: bool = False,
    ) -> None:
        """Push structured per-epoch metrics for live rendering."""
        metric = {
            "trial": trial,
            "epoch": epoch,
            "max_epoch": max_epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "train_f1": round(train_f1, 4),
            "val_f1": round(val_f1, 4),
            "val_auroc": round(val_auroc, 4),
            "alignment_loss": round(alignment_loss, 5),
            "contrastive_loss": round(contrastive_loss, 5),
            "pruned": pruned,
            "timestamp": datetime.now().isoformat(),
        }
        self._mgr.append_to_payload_list(self.task_id, "epoch_metrics", metric)
        # Recalculate progress within Phase 5 (55-95% range)
        payload = self._mgr.get_payload(self.task_id)
        tp = payload.get("trial_progress") or {}
        total_epochs = max_epoch * tp.get("total", 1)
        done_epochs = (trial * max_epoch) + epoch
        pct = 55 + int(40 * done_epochs / max(total_epochs, 1))
        current_trial = dict(payload.get("current_trial") or {})
        current_trial.update(
            {
                "current_epoch": int(epoch),
                "max_epoch": int(max_epoch),
                "status": "pruned" if pruned else "running",
            }
        )
        self._mgr.merge_payload(
            self.task_id,
            {
                "progress_pct": min(pct, 95),
                "current_trial": current_trial or None,
            },
        )

    def set_trial(self, current: int, total: int) -> None:
        """Update which Optuna trial is running."""
        self._mgr.merge_payload(self.task_id, {
            "trial_progress": {"current": current, "total": total},
        })

    def set_substage(self, substage: str) -> None:
        """Set the current sub-stage within Phase 5 (e.g. 'jit_profiling', 'trial_running')."""
        self._mgr.merge_payload(self.task_id, {"substage": substage})

    def set_current_trial(self, number: int, total: int,
                          fusion: str = "", lr: float = 0.0,
                          epochs: int = 0, status: str = "running",
                          current_epoch: int = 0, max_epoch: int = 0) -> None:
        """Record the current trial's sampled hyperparameters for the cockpit."""
        self._mgr.merge_payload(self.task_id, {
            "current_trial": {
                "number": number, "total": total,
                "fusion": fusion, "lr": lr,
                "epochs": epochs,
                "current_epoch": current_epoch,
                "max_epoch": max_epoch or epochs,
                "status": status,
            },
        })

    def set_best_so_far(self, trial: int, val_loss: float, val_acc: float = 0.0,
                        val_f1: float = 0.0, val_auroc: float = 0.0) -> None:
        """Update the best-trial card whenever a new best is achieved."""
        self._mgr.merge_payload(self.task_id, {
            "best_so_far": {
                "trial": trial,
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 4),
                "val_f1": round(val_f1, 4),
                "val_auroc": round(val_auroc, 4),
                "timestamp": datetime.now().isoformat(),
            },
        })

    def push_trial_event(
        self,
        trial: Optional[int],
        event: str,
        detail: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a structured trial event (start/complete/pruned/new_best) for the timeline."""
        self._mgr.append_to_payload_list(
            self.task_id,
            "trial_events",
            {
                "trial": trial,
                "event": event,
                "detail": detail,
                "data": dict(data or {}),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def set_pruning_status(
        self,
        available: bool,
        backend: str = "",
        reason: str = "",
        pruned_count: int = 0,
        completed_count: int = 0,
    ) -> None:
        """Record pruning backend availability and cumulative trial counts."""
        self._mgr.merge_payload(self.task_id, {
            "pruning_status": {
                "available": available,
                "backend": backend,
                "reason": reason,
                "pruned_count": pruned_count,
                "completed_count": completed_count,
            },
        })

    def set_next_trial_plan(self, overrides: Dict[str, Any]) -> None:
        """Preview the adaptive overrides for the next trial (shown before it starts)."""
        self._mgr.merge_payload(self.task_id, {"next_trial_plan": overrides or None})

    def set_data_split(self, train: int, val: int, total: int) -> None:
        """Record train/val split sizes."""
        self._mgr.merge_payload(self.task_id, {
            "data_split": {"train": train, "val": val, "total": total},
        })

    def complete(self, result: Dict[str, Any]) -> None:
        self._mgr.merge_payload(self.task_id, {
            "progress_pct": 100,
            "current_phase_name": "Complete",
            "substage": "complete",
        })
        self._mgr.update_result(self.task_id, "completed", result)

    def fail(self, error: str) -> None:
        self._mgr.merge_payload(self.task_id, {
            "substage": "failed",
        })
        self._mgr.update_error(self.task_id, "failed", error)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
task_db = TaskStateManager()
