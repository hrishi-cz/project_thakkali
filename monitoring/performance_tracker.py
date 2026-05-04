"""Performance tracking for model monitoring."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import weakref
import numpy as np


class PerformanceTracker:
    """Track model performance over time."""
    
    _instances: "weakref.WeakValueDictionary[str, PerformanceTracker]" = weakref.WeakValueDictionary()
    _lock = threading.Lock()
    
    def __new__(cls, model_id: str = "default", max_history: int = 5000):
        existing = cls._instances.get(model_id)
        if existing is not None:
            return existing
        with cls._lock:
            existing = cls._instances.get(model_id)
            if existing is not None:
                return existing
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model_id] = instance
            return instance
    
    def __init__(self, model_id: str = "default", max_history: int = 5000):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.model_id = model_id
        self.max_history = max(1, int(max_history))
        self.metrics = {}
        self.history = []
        self._initialized = True
    
    def log_prediction(self, prediction: np.ndarray, actual: np.ndarray, timestamp: Optional[str] = None):
        """Log a prediction with actual value."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        prediction = np.asarray(prediction)
        actual = np.asarray(actual)

        if prediction.size == 0 or actual.size == 0:
            raise ValueError("prediction and actual cannot be empty")

        pred_rows = prediction.shape[0] if prediction.ndim > 0 else 1
        actual_rows = actual.shape[0] if actual.ndim > 0 else 1
        if pred_rows != actual_rows:
            raise ValueError(
                f"prediction and actual row counts must match (got {pred_rows} vs {actual_rows})"
            )
        
        # Calculate metrics
        if actual.ndim == 1 and prediction.ndim == 1:  # Regression or binary
            mse = np.mean((prediction - actual) ** 2)
            mae = np.mean(np.abs(prediction - actual))
            metrics = {"mse": float(mse), "mae": float(mae), "rmse": float(np.sqrt(mse))}

            # Binary classification path: use thresholded logits/probabilities.
            unique_vals = np.unique(actual)
            if unique_vals.size and np.all(np.isin(unique_vals, [0, 1])):
                binary_preds = (prediction >= 0.5).astype(int)
                accuracy = np.mean(binary_preds == actual.astype(int))
                metrics["accuracy"] = float(accuracy)
        elif actual.ndim == 1 and prediction.ndim == 2 and prediction.shape[1] == 1:
            pred_flat = prediction.reshape(-1)
            mse = np.mean((pred_flat - actual) ** 2)
            mae = np.mean(np.abs(pred_flat - actual))
            metrics = {"mse": float(mse), "mae": float(mae), "rmse": float(np.sqrt(mse))}
        else:  # Multi-class classification
            if prediction.ndim < 2:
                prediction = prediction.reshape(-1, 1)
            pred_labels = np.argmax(prediction, axis=1)
            if actual.ndim == 1:
                true_labels = actual.astype(int)
            else:
                true_labels = np.argmax(actual, axis=1)
            accuracy = np.mean(pred_labels == true_labels)
            metrics = {"accuracy": float(accuracy)}

        metrics = {
            key: float(value)
            for key, value in metrics.items()
            if np.isfinite(value)
        }
        if not metrics:
            return
        
        entry = {
            "timestamp": timestamp,
            "metrics": metrics,
        }
        self.history.append(entry)
        overflow = len(self.history) - self.max_history
        if overflow > 0:
            del self.history[:overflow]
        self.metrics = metrics
    
    def get_recent_metrics(self, limit: int = 20) -> List[Dict]:
        """Get recent performance metrics."""
        if limit <= 0:
            return []
        return self.history[-limit:] if self.history else []
    
    def get_metric_trend(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get trend for a specific metric over time."""
        if not self.history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=max(1, int(hours)))
        trend = []
        
        for entry in self.history:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
            except (TypeError, ValueError):
                continue
            if entry_time >= cutoff_time and metric_name in entry.get("metrics", {}):
                trend.append({
                    "timestamp": entry["timestamp"],
                    "value": entry["metrics"][metric_name]
                })
        
        return trend
    
    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics."""
        if not self.history:
            return {}
        
        # Calculate averages
        all_metrics = [h["metrics"] for h in self.history]
        summary = {}
        keys = {k for metric_map in all_metrics for k in metric_map.keys()}
        for key in keys:
            values = [float(m[key]) for m in all_metrics if key in m and np.isfinite(m[key])]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        return summary
    
    def update_prediction_distribution(
        self,
        predictions: np.ndarray,
        model_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Track P(ŷ) over time and flag concept drift when the distribution
        shifts significantly relative to the first recorded batch.

        A two-sample KS test compares the *current* batch against the
        reference batch (first call).  Returns None until at least two
        batches have been recorded.

        Parameters
        ----------
        predictions : 1-D or 2-D float array
            Raw model outputs (probabilities or logits).  Multi-class arrays
            are reduced to the max-probability column for the 1-D KS test.
        model_id : str, optional
            Key for per-model history isolation (defaults to self.model_id).

        Returns
        -------
        dict with keys: concept_drift_detected, ks_stat, p_value, n_batches
        or None when not enough data is available.
        """
        from scipy import stats as _stats

        key = f"{model_id or self.model_id}_pred_dist"
        if not hasattr(self, "_pred_dist_history"):
            self._pred_dist_history: dict = {}

        preds = np.asarray(predictions, dtype=np.float64)
        if preds.ndim == 2:
            preds = preds.max(axis=1)
        preds = preds.ravel()

        if key not in self._pred_dist_history:
            self._pred_dist_history[key] = []
        self._pred_dist_history[key].append(preds)

        batches = self._pred_dist_history[key]
        if len(batches) < 2:
            return None

        ref = batches[0]
        cur = batches[-1]
        ks_stat, p_value = _stats.ks_2samp(ref, cur)

        return {
            "concept_drift_detected": bool(p_value < 0.05),
            "ks_stat": float(ks_stat),
            "p_value": float(p_value),
            "n_batches": int(len(batches)),
        }

    def clear_history(self):
        """Clear all history."""
        self.history = []
        self.metrics = {}
        if hasattr(self, "_pred_dist_history"):
            self._pred_dist_history = {}

