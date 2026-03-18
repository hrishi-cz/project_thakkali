"""
Performance tracking for model monitoring.

Persists to SQLite for durability across restarts.  Supports both
classification (accuracy, precision, recall, F1) and regression
(MSE, MAE, RMSE) metrics with bounded history (max 10 000 entries
per model) and time-windowed trend queries.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DB_DIR = Path("monitoring")
_DB_PATH = _DB_DIR / "performance.db"
_MAX_ENTRIES_PER_MODEL = 10_000


def _get_conn() -> sqlite3.Connection:
    """Open a WAL-mode connection with busy_timeout."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS performance_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id   TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL,
            metric_json TEXT   NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_perf_model_ts
            ON performance_log (model_id, timestamp)
        """
    )
    conn.commit()


class PerformanceTracker:
    """
    Track model prediction performance over time.

    Each ``log_prediction`` call computes and stores metrics.  History is
    persisted to ``monitoring/performance.db`` and capped at
    ``_MAX_ENTRIES_PER_MODEL`` rows per model.
    """

    _instances: Dict[str, "PerformanceTracker"] = {}

    def __new__(cls, model_id: str = "default") -> "PerformanceTracker":
        if model_id not in cls._instances:
            inst = super().__new__(cls)
            inst._initialized = False
            cls._instances[model_id] = inst
        return cls._instances[model_id]

    def __init__(self, model_id: str = "default") -> None:
        if getattr(self, "_initialized", False):
            return
        self.model_id = model_id
        self._initialized = True
        # Ensure the table exists on first use
        try:
            conn = _get_conn()
            _ensure_schema(conn)
            conn.close()
        except Exception as exc:
            logger.warning("PerformanceTracker: DB init failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_metrics(
        prediction: np.ndarray,
        actual: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute appropriate metrics based on shapes.

        - 1-D arrays  → classification if values look discrete, else regression.
        - 2-D arrays  → multi-class classification (argmax comparison).
        """
        import json as _json  # only used here

        if prediction.ndim == 2 and actual.ndim == 2:
            # Multi-class: softmax probabilities or one-hot
            pred_cls = np.argmax(prediction, axis=1)
            true_cls = np.argmax(actual, axis=1)
            return PerformanceTracker._classification_metrics(pred_cls, true_cls)

        # Flatten for 1-D comparison
        pred_flat = prediction.ravel().astype(float)
        true_flat = actual.ravel().astype(float)

        # Heuristic: if unique values <= 50 and all integers → classification
        unique_vals = np.unique(true_flat)
        is_classification = (
            len(unique_vals) <= 50
            and np.allclose(true_flat, true_flat.astype(int))
        )

        if is_classification:
            return PerformanceTracker._classification_metrics(
                pred_flat.astype(int), true_flat.astype(int),
            )
        return PerformanceTracker._regression_metrics(pred_flat, true_flat)

    @staticmethod
    def _classification_metrics(
        pred: np.ndarray, true: np.ndarray,
    ) -> Dict[str, float]:
        """Accuracy, macro precision, macro recall, macro F1."""
        accuracy = float(np.mean(pred == true))
        classes = np.unique(np.concatenate([pred, true]))
        precisions, recalls, f1s = [], [], []
        for cls in classes:
            tp = int(np.sum((pred == cls) & (true == cls)))
            fp = int(np.sum((pred == cls) & (true != cls)))
            fn = int(np.sum((pred != cls) & (true == cls)))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        return {
            "accuracy":        accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall":    float(np.mean(recalls)),
            "macro_f1":        float(np.mean(f1s)),
        }

    @staticmethod
    def _regression_metrics(
        pred: np.ndarray, true: np.ndarray,
    ) -> Dict[str, float]:
        """MSE, MAE, RMSE."""
        mse = float(np.mean((pred - true) ** 2))
        mae = float(np.mean(np.abs(pred - true)))
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def log_prediction(
        self,
        prediction: np.ndarray,
        actual: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> None:
        """Compute metrics and persist to SQLite."""
        import json as _json

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        metrics = self._compute_metrics(prediction, actual)

        try:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO performance_log (model_id, timestamp, metric_json) "
                "VALUES (?, ?, ?)",
                (self.model_id, timestamp, _json.dumps(metrics)),
            )
            # Enforce cap: delete oldest rows beyond the limit
            conn.execute(
                """
                DELETE FROM performance_log
                WHERE model_id = ? AND id NOT IN (
                    SELECT id FROM performance_log
                    WHERE model_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (self.model_id, self.model_id, _MAX_ENTRIES_PER_MODEL),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("PerformanceTracker.log_prediction failed: %s", exc)

    def get_recent_metrics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the last *limit* metric entries for this model."""
        import json as _json

        try:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT timestamp, metric_json FROM performance_log "
                "WHERE model_id = ? ORDER BY id DESC LIMIT ?",
                (self.model_id, limit),
            ).fetchall()
            conn.close()
            return [
                {"timestamp": r["timestamp"], "metrics": _json.loads(r["metric_json"])}
                for r in reversed(rows)
            ]
        except Exception as exc:
            logger.warning("PerformanceTracker.get_recent_metrics failed: %s", exc)
            return []

    def get_metric_trend(
        self, metric_name: str, hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get values for *metric_name* within a time window."""
        import json as _json

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        try:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT timestamp, metric_json FROM performance_log "
                "WHERE model_id = ? AND timestamp >= ? ORDER BY id",
                (self.model_id, cutoff),
            ).fetchall()
            conn.close()
            trend = []
            for r in rows:
                m = _json.loads(r["metric_json"])
                if metric_name in m:
                    trend.append({
                        "timestamp": r["timestamp"],
                        "value": m[metric_name],
                    })
            return trend
        except Exception as exc:
            logger.warning("PerformanceTracker.get_metric_trend failed: %s", exc)
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Aggregate statistics over all history for this model."""
        import json as _json

        try:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT metric_json FROM performance_log "
                "WHERE model_id = ? ORDER BY id",
                (self.model_id,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("PerformanceTracker.get_performance_summary failed: %s", exc)
            return {}

        if not rows:
            return {}

        all_metrics = [_json.loads(r["metric_json"]) for r in rows]
        summary: Dict[str, Any] = {}
        keys = set()
        for m in all_metrics:
            keys.update(m.keys())
        for key in sorted(keys):
            values = [m[key] for m in all_metrics if key in m]
            if values:
                summary[key] = {
                    "mean":  float(np.mean(values)),
                    "std":   float(np.std(values)),
                    "min":   float(np.min(values)),
                    "max":   float(np.max(values)),
                    "count": len(values),
                }
        return summary

    def clear_history(self) -> None:
        """Delete all log entries for this model."""
        try:
            conn = _get_conn()
            conn.execute(
                "DELETE FROM performance_log WHERE model_id = ?",
                (self.model_id,),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("PerformanceTracker.clear_history failed: %s", exc)
